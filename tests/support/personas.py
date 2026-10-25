"""tests/support/personas.py - scripted model behaviours ("personas") for loop, provider and golden tests.

A persona is a function that returns a SCRIPT: a ``list[Round]`` when the number of replies is exact, or a
``Sequence`` (a callable script that serves a fixed prefix and then a repeating tail, with a runaway guard) when
the behaviour continues for as long as the loop under test keeps asking. Both go into
``ScriptedProvider(script)`` and into ``ScriptedServer(script)`` unchanged. Personas that model a transport fault
(``flaky_429_then_ok``, ``drops_mid_stream``) return ``ScriptedServer`` items (HTTP status, cut streams) and are only
meaningful against the server. Every persona is registered under its function name in ``PERSONAS`` (``build(name,
**kwargs)``), and ``PROMPTS`` holds a prompt that makes the loop treat the turn as the persona expects.

The five personas added by the review: ``claims_done_without_work``, ``asks_the_user``, ``loops_on_reads``,
``plan_in_other_language``, ``slow_local``.
"""

import copy
import threading
from typing import Any, Callable

from support.scripted_llm import (Round, advance, bad_json, call, calls, empty, malformed_json, say, tick,
                                  think)
from support.server import drop_after, http_error

DOC = "PersonaDoc"
PLAN_TEXT = ("Here is my plan:\n1. Create the document\n2. Model the base plate\n3. Add the mounting holes\n"
             "4. Verify the result and export it\nI will start with these steps.")
DONE_TEXT = "Done. The model is built and verified."

GOLDEN_PI4_PROMPT = ("Build a housing for a Raspberry Pi 4. Bottom shell with cutouts for every I/O port, air vent slots, "
                     "snap-fit lid, ABS. And source the Pi 4 STEP file yourself.")

PROMPTS = {
    "plan_only": "make a plate",
    "tool_then_stop": "Build a Pi 4 housing",
    "cutoff_then_plan": "make a plate",
    "stalls_then_acts": "make a plate",
    "giant_script_truncated": "make a plate with a 20 mm hole",
    "bad_json_args": "make a plate",
    "text_tool_calls": "make a plate",
    "think_tags": "what is the volume of a 10 mm cube",
    "flaky_429_then_ok": "hello",
    "drops_mid_stream": "hello",
    "empty_reply": "hello",
    "tool_loop_repeat": "make a plate",
    "hidden_reasoning": "make a plate",
    "claims_done_without_work": "Build a Pi 4 housing",
    "asks_the_user": "Build a Pi 4 housing",
    "loops_on_reads": "make a plate",
    "plan_in_other_language": "Konstruiere bitte eine Platte mit zwei Bohrungen.",
    "slow_local": "make a plate",
    "golden_pi4": GOLDEN_PI4_PROMPT,
}

PERSONAS: dict[str, Callable[..., Any]] = {}


def persona(fn):
    """Register ``fn`` in ``PERSONAS`` under its name."""
    PERSONAS[fn.__name__] = fn
    return fn


def build(name: str, **kwargs):
    """The script of persona ``name`` (``KeyError`` with the known names when it does not exist)."""
    try:
        factory = PERSONAS[name]
    except KeyError:
        raise KeyError(f"unknown persona {name!r}; known: {sorted(PERSONAS)}") from None
    return factory(**kwargs)


def prompt_for(name: str) -> str:
    """The prompt that goes with persona ``name``."""
    return PROMPTS[name]


class Sequence:
    """A callable script: ``prefix`` rounds in order, then ``tail`` for every further request.

    ``tail`` is a ``Round`` (a fresh deep copy each time), a callable ``tail(i) -> Round`` (``i`` counts from 0
    within the tail, so tool-call ids can be unique) or ``None`` (a request past the prefix raises
    ``AssertionError`` like a strict ``ScriptedProvider``). ``limit`` caps the total number of requests: a loop
    that never terminates fails the test instead of hanging it. Ignores its argument, so it serves
    ``ScriptedProvider`` (receives a ``RequestRecord``) and ``ScriptedServer`` (receives the request JSON) alike."""

    def __init__(self, prefix=(), tail=None, *, limit: int = 60, name: str = "sequence"):
        self.prefix = list(prefix)
        self.tail = tail
        self.limit = limit
        self.name = name
        self.served = 0
        self._lock = threading.Lock()

    def __call__(self, *_args) -> Round:
        with self._lock:
            n = self.served
            self.served += 1
        if n >= self.limit:
            raise AssertionError(f"persona {self.name!r} was asked for more than {self.limit} rounds: "
                                 "the loop under test does not terminate")
        if n < len(self.prefix):
            return copy.deepcopy(self.prefix[n])
        if self.tail is None:
            raise AssertionError(f"unexpected extra provider request #{n + 1} for persona {self.name!r}")
        if callable(self.tail):
            return self.tail(n - len(self.prefix))
        return copy.deepcopy(self.tail)


# -- building blocks --------------------------------------------------------------------------------------

def geo(n: int = 1, doc: str = DOC):
    """A successful-looking geometry call: ``create_object`` of a box (unique id and name per ``n``)."""
    return call("create_object", call_id=f"call_geo{n}", doc_name=doc, obj_type="Part::Box", obj_name=f"Box{n}",
                obj_properties={"Length": 10 + n, "Width": 10, "Height": 10}, include_screenshot=False)


def _new_doc(doc: str = DOC):
    return call("create_document", call_id="call_doc", name=doc)


# -- personas ---------------------------------------------------------------------------------------------

@persona
def plan_only(times: int = 3, plan: str = PLAN_TEXT):
    """Answers with a numbered plan and never calls a tool: ``times`` replies (the first answer plus the gate's
    two NO_WORK nudges). Enables: nudge, then ``outcome == "failed"``, never green."""
    return [say(plan) for _ in range(times)]


@persona
def tool_then_stop(then: str = DONE_TEXT):
    """One geometry call, then "done" while ledger steps are pending, and "done" again after every nudge.
    Enables: the gate nudge that names the pending ids."""
    return Sequence([calls(_new_doc(), geo(1)), say(then)], say(then), name="tool_then_stop")


@persona
def cutoff_then_plan(think_chars: int = 400):
    """A reasoning-only reply cut off at the token limit, then a plan with no tool, then (after the nudge) a
    geometry call and the final answer. Enables: the cutoff retry and the execution nudge both fire."""
    return [think(think_chars), say(PLAN_TEXT), calls(_new_doc(), geo(1)), say(DONE_TEXT)]


@persona
def stalls_then_acts(reasoning_chars: int = 20000, chunk: int = 1000):
    """A huge stream of reasoning (``reasoning_chars`` in ``chunk``-sized events, cut off at the end), then a
    normal tool call and answer. Enables: the watchdog trip within N characters and the retry at
    ``reasoning="off"`` (when it does not trip, the cutoff path takes over and the script still terminates)."""
    pieces = max(1, reasoning_chars // chunk)
    flood = Round([("reasoning", "x" * chunk) for _ in range(pieces)] + [("stop_reason", "length"), ("done", "length")])
    return [flood, calls(_new_doc(), geo(1)), say(DONE_TEXT)]


@persona
def giant_script_truncated(times: int = 2):
    """``execute_code`` with arguments truncated at the token limit, ``times`` times in a row, then a small
    script. Enables: breaker ``truncation`` and "smaller scripts afterwards"."""
    big = '{"code": "import Part\\nimport FreeCAD\\ndoc = FreeCAD.newDocument(\'Big\')\\n' + "box = Part.makeBox(1, 1, 1)\\n" * 3
    return [bad_json("execute_code", raw=big, call_id=f"call_big{i}") for i in range(times)] + [
        calls(call("execute_code", call_id="call_small", code="print('small script ok')")), say(DONE_TEXT)]


@persona
def bad_json_args():
    """A complete but malformed tool-call argument string (not truncated), then a valid call and the answer.
    Enables: the ``args_error`` message wording ("malformed JSON")."""
    return [malformed_json("create_object", raw='{"doc_name": "D", "obj_type": Part::Box}'),
            calls(_new_doc(), geo(1)), say(DONE_TEXT)]


@persona
def text_tool_calls(split: bool = True):
    """Round 1 calls two tools by writing ``<tool_call>`` blocks in plain text (for text-mode providers), the text
    cut into stream chunks inside the tags when ``split``; round 2 is a plain-text answer. Enables: the extractor,
    the stream hold-back, and the text-mode history (the second request must carry the first calls and their
    results as text blocks and no ``tool_calls`` key)."""
    doc = '<tool_call>\n{"name": "create_document", "arguments": {"name": "%s"}}\n</tool_call>' % DOC
    box = ('<tool_call>\n{"name": "create_object", "arguments": {"doc_name": "%s", "obj_type": "Part::Box", '
           '"obj_name": "Box1", "obj_properties": {"Length": 10, "Width": 10, "Height": 10}}}\n</tool_call>' % DOC)
    text = "I will create the document and the box now.\n" + doc + "\n" + box
    if split:
        cuts = [text.index("<tool_call>") + 6, text.index("</tool_call>") + 5, text.rindex("<tool_call>") + 3]
        pieces = [text[a:b] for a, b in zip([0] + cuts, cuts + [len(text)])]
    else:
        pieces = [text]
    return [Round([("text", t) for t in pieces] + [("stop_reason", "stop"), ("done", "stop")]),
            say("The box Box1 now exists in the document.")]


@persona
def think_tags(answer: str = "The volume is 1000 cubic millimetres."):
    """One reply with an inline ``<think>...</think>`` span, the tags split across chunks. Enables: the reasoning
    split (the span must come out as ``reasoning``, never as ``text``, and no tag may leak)."""
    pieces = ["<thi", "nk>The cube has side 10, so ", "the volume is 10*10*10.</th", "ink>", answer]
    return [Round([("text", p) for p in pieces] + [("stop_reason", "stop"), ("done", "stop")])]


@persona
def flaky_429_then_ok(retry_after: float | str | None = 1, text: str = "Recovered after the rate limit."):
    """ScriptedServer items: HTTP 429 (with ``Retry-After``), then a good reply. Enables: retry and backoff."""
    return [http_error(429, retry_after=retry_after), say(text)]


@persona
def drops_mid_stream(first: tuple = ("The housing has ", "two parts: "), full: str = "The housing has two parts: a shell and a lid."):
    """ScriptedServer items: a reply cut after its first chunk(s) with no terminal marker, then the complete reply.
    Enables: retry after ``stream_dropped``, a ``reset`` event and NO duplicated text."""
    partial = Round([("text", t) for t in first] + [("stop_reason", "stop"), ("done", "stop")])
    return [drop_after(partial, chunks=len(first)), say(full)]


@persona
def empty_reply(limit: int = 12):
    """Empty replies (no text, no calls) for as long as it is asked. Enables: the empty-reply messaging."""
    return Sequence([], empty(), limit=limit, name="empty_reply")


@persona
def tool_loop_repeat(code: str = "undefined_name_in_a_failing_script()", limit: int = 40):
    """The same failing call (same name and arguments, a fresh call id each round) for as long as it is asked.
    Enables: breaker ``repeat`` and ``needs_help``."""
    return Sequence([calls(_new_doc())], lambda i: calls(call("execute_code", call_id=f"call_rep{i}", code=code)),
                    limit=limit, name="tool_loop_repeat")


@persona
def hidden_reasoning(silent_s: float = 120.0, step_s: float = 15.0):
    """A long silence with ticks only (no reasoning text, ``advance`` pseudo-events move the injected clock), then
    a tool call. Enables: the ``silent_s`` watchdog. Pass ``ScriptedProvider(advance=clock.advance)``."""
    steps = max(1, int(silent_s // step_s))
    events = []
    for i in range(1, steps + 1):
        events += [advance(step_s), tick(i * step_s, idle_s=i * step_s)]
    events += [("stop_reason", "tool_use"), ("tool_calls", [geo(1)])]
    return [Round(events), calls(geo(2)), say(DONE_TEXT)]


@persona
def claims_done_without_work(step_ids=("S6", "S7", "S8", "S9", "S10"), then: str = "All steps are done."):
    """ONE ``create_object``, then ``plan_update(done)`` for the remaining required steps with only text as
    "evidence", then "done" (again after every nudge). Enables: the ``CLAIM_ONLY`` nudge and the final
    ``unverified`` outcome with "claimed without recorded work"."""
    claims = [call("plan_update", call_id=f"call_claim{i}", id=sid, status="done", note="finished",
                   evidence="I did it") for i, sid in enumerate(step_ids)]
    return Sequence([calls(_new_doc(), geo(1)), calls(*claims), say(then)], say(then), name="claims_done_without_work")


@persona
def asks_the_user(step_id: str = "S2", note: str = "please attach the STEP file",
                  question: str = "Could you attach the STEP file of the board so I can wrap it exactly?"):
    """Marks ``step_id`` blocked with a note and replies with a question; zero geometry. Enables: no NO_WORK
    nudge and the outcome ``needs_input``."""
    return [calls(call("plan_update", call_id="call_block", id=step_id, status="blocked", note=note)), say(question)]


@persona
def loops_on_reads(limit: int = 40):
    """``get_objects`` / ``get_view`` forever (read-only, never a change). Enables: the breaker ``stall`` note after
    six rounds, then ``needs_help``."""
    def read(i):
        if i % 2:
            return calls(call("get_view", call_id=f"call_read{i}", view_name="Isometric"))
        return calls(call("get_objects", call_id=f"call_read{i}", doc_name=DOC))
    return Sequence([calls(_new_doc())], read, limit=limit, name="loops_on_reads")


@persona
def plan_in_other_language():
    """A reply in German with a geometry call; pair it with ``PROMPTS["plan_in_other_language"]`` (a non-English
    prompt the intent classifier files under ``other``). Enables: escalation to ``generic_part`` on the first
    geometry call, the gate active afterwards, and all tools offered in round 1."""
    return [calls(_new_doc(), geo(1), text="Ich erstelle zuerst die Grundplatte."),
            say("Fertig: Die Platte wurde erstellt.")]


@persona
def slow_local(seconds: float = 200.0):
    """No events for ``seconds`` of injected clock time, then a complete tool call. Enables: NO watchdog trip for a
    slow local host (``host_class`` ollama / ``tool_args_stream False``). Pass ``ScriptedProvider(advance=...)``."""
    slow_round = Round([advance(seconds), ("stop_reason", "tool_use"), ("tool_calls", [_new_doc(), geo(1)])])
    return [slow_round, say(DONE_TEXT)]


@persona
def golden_pi4(*, nominal: bool = False, lid_lift: float | None = None):
    """The scripted Pi 4 build: 25 replies, one call each (reply 24 is the report).
    ``nominal`` drops the STEP import and measurement (the run without ``OSCAR_REFPART_LOCAL_DIR``) and adds the
    ``reference`` build stage (nothing imported satisfies it, so the kit builds the nominal board: 24 replies);
    ``lid_lift`` adds a ``set_parameter(lid_lift=...)`` before ``check_fit`` (the variant run)."""
    d = "PiCase"
    n = [0]

    def c(tool, /, **args):
        n[0] += 1
        return calls(call(tool, call_id=f"call_g{n[0]}", **args))

    script = [c("find_reference_part", query="raspberry pi 4 model b"),
              c("source_reference_part", id="rpi4b", reason="reference geometry for the housing"),
              c("create_document", name=d)]
    if not nominal:
        script += [c("import_reference_part", doc_name=d, id="rpi4b"),
                   c("measure_reference_part", doc_name=d, write_params=True)]
    script += [c("enclosure_spec", action="set", spec={"name": d, "material": "ABS"}, doc_name=d),
               c("build_enclosure", doc_name=d, stage="next"), c("build_enclosure", doc_name=d, stage="next")]
    if nominal:   # no imported STEP satisfies the "reference" stage: the kit builds the nominal board itself
        script.append(c("build_enclosure", doc_name=d, stage="next"))
    script += [c("get_view", view_name="Isometric"),
               c("execute_code", code="print(undefined_name_for_the_golden_test)"),
               c("get_objects", doc_name=d)]
    script += [c("build_enclosure", doc_name=d, stage="next") for _ in range(6)]
    script += [c("add_cutout", doc_name=d, part="lid", face="top", u=65, v=45, w=6, h=6, id="status_led"),
               c("set_parameter", doc_name=d, name="wall_t", value="2.4 mm"), c("set_parameter", doc_name=d, name="clearance", value=0.35)]
    if lid_lift is not None:
        script.append(c("set_parameter", doc_name=d, name="lid_lift", value=lid_lift))
    script += [c("validate_model", doc_name=d, deep=True), c("check_fit", doc_name=d),
               c("export_model", doc_name=d, formats=["step", "stl"]), c("export_rebuild_script", run_check=True),
               say("Built the Pi 4 housing: bottom shell with every port opening, vent slots and a snap-fit lid in ABS. "
                   "Checks passed; STEP and STL are exported.")]
    return script
