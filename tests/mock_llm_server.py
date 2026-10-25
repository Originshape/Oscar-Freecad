"""A scripted OpenAI-compatible chat server for end-to-end tests (``gui_smoke.py``).

Since the build-flow contract this is a thin layer over ``support.server.ScriptedServer``: state is per server
instance (no class-level globals) and there are two ways to drive it.

* ORDINAL mode (the original behaviour, kept): ``make_server(standard_rounds())`` answers request N with item N
  (the last item repeats). Brittle by construction: one extra provider request (a gate nudge, a wrap-up, a
  watchdog retry) shifts every later reply.
* REACTIVE mode: ``make_server(reactive=smoke_reactive(stl))`` decides every reply from the LAST EVENT of the
  request body (the tool result the model is answering, or the user's text), so extra requests cannot
  desynchronise it, and the test asserts the number of requests explicitly (``server.request_count``).

``standard_rounds`` and ``attachment_rounds`` still return pre-rendered SSE bodies (bytes). No API key is needed;
the provider under test cannot tell the server apart from a real endpoint (beyond the model name).
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # tests/: the support package

from support.scripted_llm import Round, call, calls, say  # noqa: E402
from support.server import RawReply, ScriptedServer, last_event  # noqa: E402


def _sse_chunk(delta=None, finish=None):
    choice = {"delta": delta or {}, "index": 0}
    if finish:
        choice["finish_reason"] = finish
    return "data: " + json.dumps({"id": "chatcmpl-mock", "choices": [choice]}) + "\n"


def _tool_call_stream(call_id, name, arguments: dict):
    body = _sse_chunk({"role": "assistant", "tool_calls": [
        {"index": 0, "id": call_id, "function": {"name": name}}
    ]})
    body += _sse_chunk({"tool_calls": [
        {"index": 0, "function": {"arguments": json.dumps(arguments)}}
    ]})
    body += _sse_chunk(finish="tool_calls")
    body += "data: [DONE]\n"
    return body.encode()


def _text_stream(text):
    body = _sse_chunk({"role": "assistant", "content": text})
    body += _sse_chunk(finish="stop")
    body += "data: [DONE]\n"
    return body.encode()


def _reasoning_text_stream(reasoning, text):
    """A reasoning-model answer: thinking first, then the visible text."""
    body = _sse_chunk({"role": "assistant", "reasoning_content": reasoning})
    body += _sse_chunk({"role": "assistant", "content": text})
    body += _sse_chunk(finish="stop")
    body += "data: [DONE]\n"
    return body.encode()


def _text_then_tool_stream(call_id, name, arguments, text):
    """A remark before a tool call — the panel folds it into the card."""
    body = _sse_chunk({"role": "assistant", "content": text})
    body += _sse_chunk({"tool_calls": [
        {"index": 0, "id": call_id, "function": {"name": name}}
    ]})
    body += _sse_chunk({"tool_calls": [
        {"index": 0, "function": {"arguments": json.dumps(arguments)}}
    ]})
    body += _sse_chunk(finish="tool_calls")
    body += "data: [DONE]\n"
    return body.encode()


def make_server(rounds=None, port=8765, *, reactive=None, dialect="openai"):
    """Start the scripted server on a daemon thread and return the running ``ScriptedServer`` (``shutdown()``,
    ``port``, ``base_url``, ``requests``, ``request_count`` ...).

    ``rounds``: the items served in order (``bytes`` bodies from ``standard_rounds`` / ``attachment_rounds``,
    ``Round`` objects or strings); the LAST item repeats for any further request. ``reactive``: a callable
    ``script(request_json) -> Round`` that decides every reply from the request (``smoke_reactive``); it
    replaces ``rounds``. ``port=0`` picks a free port."""
    if reactive is not None:
        script = reactive
    else:
        script = [RawReply(r) if isinstance(r, (bytes, bytearray)) else r for r in (rounds or [])]
    return ScriptedServer(script, dialect=dialect, port=port, repeat_last=True)


def standard_rounds():
    return [
        _text_then_tool_stream(
            "call_1", "create_document", {"name": "SmokeDoc"},
            "Setting up a document.",
        ),
        _tool_call_stream(
            "call_2",
            "create_object",
            {
                "doc_name": "SmokeDoc",
                "obj_type": "Part::Box",
                "obj_name": "Box",
                "obj_properties": {"Length": 10, "Width": 10, "Height": 10},
                "include_screenshot": False,
            },
        ),
        # get_view exercised end-to-end (regression guard for the
        # stale-engine-method bug class).
        _tool_call_stream("call_2b", "get_view", {"view_name": "Isometric"}),
        _text_stream("I created a 10x10x10 mm box in SmokeDoc."),
    ]


# the cube the second conversation adds next to the imported part
SPACER = dict(doc_name="SmokeDoc", obj_type="Part::Box", obj_name="Spacer",
              obj_properties={"Length": 5, "Width": 5, "Height": 5}, include_screenshot=False)


def attachment_rounds(stl_path):
    """standard_rounds + a second turn: import an attached part and add a 5 mm cube,
    ending with a reasoning-style answer (thinking + text)."""
    return standard_rounds() + [
        _tool_call_stream(
            "call_3",
            "import_file",
            {"path": stl_path, "include_screenshot": False},
        ),
        _tool_call_stream("call_4", "create_object", SPACER),
        _reasoning_text_stream(
            "The part is imported; a 5 mm cube sits beside it.",
            "I imported the attached part and added a 5 mm cube.",
        ),
    ]


def smoke_reactive(stl_path=""):
    """The reactive script of the GUI smoke test: the same two conversations as ``attachment_rounds`` (build a
    box; import an attached part and add a cube), but every reply is decided from the last event of the request.

    * user text with "attached" -> ``import_file`` (call_3);
    * any other user text -> ``create_document`` (call_1) with a remark;
    * tool result of call_1 -> ``create_object`` (call_2); call_2 -> ``get_view`` (call_2b); call_2b -> the final text;
    * tool result of call_3 -> ``create_object`` (call_4, a 5 mm cube); call_4 -> the final reasoning-style text;
    * anything else (a nudge, an unknown id) -> a short final text, so an unexpected extra request ends the turn
      instead of looping. The test then fails on ``server.request_count``, not by accident."""
    box = dict(doc_name="SmokeDoc", obj_type="Part::Box", obj_name="Box",
               obj_properties={"Length": 10, "Width": 10, "Height": 10}, include_screenshot=False)

    def script(request_json) -> Round:
        kind, call_id, text = last_event(request_json)
        if kind == "user":
            low = text.lower()
            if "attached" in low:
                return calls(call("import_file", "call_3", path=stl_path, include_screenshot=False))
            return calls(call("create_document", "call_1", name="SmokeDoc"), text="Setting up a document.")
        if kind == "tool":
            if call_id == "call_1":
                return calls(call("create_object", "call_2", **box))
            if call_id == "call_2":
                return calls(call("get_view", "call_2b", view_name="Isometric"))
            if call_id == "call_2b":
                return say("I created a 10x10x10 mm box in SmokeDoc.")
            if call_id == "call_3":
                return calls(call("create_object", "call_4", **SPACER))
            if call_id == "call_4":
                return say("I imported the attached part and added a 5 mm cube.",
                           reasoning="The part is imported; a 5 mm cube sits beside it.")
        return say("Nothing more to do.")

    return script
