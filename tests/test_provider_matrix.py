"""Capability-tier matrix: the SAME scripted build must end identically through every model tier.

The model is a ``ScriptedServer`` (a real local HTTP server), the provider is the REAL ``OpenAICompatProvider`` /
``AnthropicProvider``, the tools are the REAL ``run_tool`` with a REAL Project and REAL FreeCAD documents. One logical
script (create a document, a plate, a cylinder by script, then a one-line answer) is rendered per tier:

* native tools        - ordinary tool-call chunks
* text tools          - the same calls as ``<tool_call>`` blocks in plain text (``caps.tools == "text"``)
* no vision           - ``caps.vision False``: no image may travel
* reasoning with knob - a reasoning stream plus an effort parameter on the wire
* reasoning, no knob  - a reasoning stream and NO reasoning parameter (nothing the model cannot take)
* small context       - an 8k window: lite prompt and lite tool list
* anthropic dialect   - the native tools tier through the other wire format

What must not depend on the tier: the turn outcome, the ledger's step statuses, the recorded steps, the objects in the
document, the Build Summary's structure. What may differ (and is asserted per tier): the wire shape.
"""

import json
import os
import sys
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

from oscar import params, prompt, providers, tools  # noqa: E402
from oscar.providers import Message  # noqa: E402
from support.realflow import RealFlowCase  # noqa: E402
from support.scripted_llm import Round, call, calls, say  # noqa: E402
from support.server import ScriptedServer  # noqa: E402

CYLINDER = "o = d.addObject('Part::Feature', 'Cyl')\no.Shape = Part.makeCylinder(3, 10)\nd.recompute()"
PROMPT = "make a plate and a cylinder"
FINAL = "Built a plate and a cylinder."

#: tier name -> (caps_override, wire dialect)
TIERS = {
    "native": ({}, "openai"),
    "text": ({"tools": "text"}, "openai"),
    "no_vision": ({"vision": False}, "openai"),
    "reasoning_knob": ({"reasoning": "hidden", "reasoning_control": "openai_effort"}, "openai"),
    "reasoning_no_knob": ({"reasoning": "stream", "reasoning_control": "none"}, "openai"),
    "small_context": ({"context_window": 8192, "max_output": 1024}, "openai"),
    "anthropic": ({}, "anthropic"),
}


def logical_script(doc: str) -> list[list]:
    """The build as rounds of (tool, arguments) pairs; the last round is the answer."""
    cyl = f"import FreeCAD as App\nimport Part\nd = App.getDocument('{doc}')\n{CYLINDER}\n"
    return [
        [("create_document", {"name": doc})],
        [("create_object", {"doc_name": doc, "obj_type": "Part::Box", "obj_name": "Plate", "include_screenshot": False}),
         ("execute_code", {"code": cyl, "include_screenshot": False})],
    ]


def text_block(tool: str, arguments: dict) -> str:
    return "<tool_call>\n" + json.dumps({"name": tool, "arguments": arguments}) + "\n</tool_call>"


def render(rounds: list[list], tier: str) -> list[Round]:
    """The server script for ``tier``: native tool-call rounds, or text blocks for the text tier; then the answer."""
    out: list[Round] = []
    for i, pairs in enumerate(rounds):
        reasoning = "Thinking about the next step. " * 3 if tier.startswith("reasoning") else ""
        if tier == "text":
            text = "Next.\n" + "\n".join(text_block(t, a) for t, a in pairs)
            out.append(Round([("text", text), ("stop_reason", "stop"), ("done", "stop")]))
        else:
            out.append(calls(*[call(t, call_id=f"call_m{i}_{k}", **a) for k, (t, a) in enumerate(pairs)], reasoning=reasoning))
    out.append(say(FINAL, reasoning="Done." if tier.startswith("reasoning") else ""))
    return out


class MatrixCase(RealFlowCase):
    """Runs the logical build through one tier and keeps everything observable on ``self``."""

    def run_tier(self, tier: str):
        override, dialect = TIERS[tier]
        self.doc = self.doc_name("Mx")
        self.project = self.new_project(f"matrix {tier}", chat_id=f"20261002-101010-{uuid.uuid4().hex[:6]}")
        self.server = ScriptedServer(render(logical_script(self.doc), tier), dialect=dialect)
        self.addCleanup(self.server.shutdown)
        # a model name no other test (or tier) shares, so nothing learned in one run leaks into another
        profile = params.Profile(name=f"mx-{tier}", provider="anthropic" if dialect == "anthropic" else "openai_compat",
                                 base_url=self.server.base_url, model=f"mx-{tier}-{uuid.uuid4().hex[:8]}", api_key="key",
                                 max_tokens=1024, caps_override=dict(override))
        provider = providers.make_provider(profile)
        result = self.turn(PROMPT, project=self.project, provider=provider, profile=profile, resolve_caps=True)
        return result

    def signature(self, result) -> dict:
        """Everything that must be the same through every tier."""
        steps = self.project.read_steps()
        ledger = self.worker.ledger
        return {
            "outcome": result.outcome,
            "ok": result.ok,
            "ledger_steps": [(s.id, s.status) for s in ledger.steps] if ledger is not None else [],
            "recorded_tools": [s.tool for s in steps],
            "recorded_ok": [s.ok for s in steps],
            "objects": sorted(o.Name for o in App.getDocument(self.doc).Objects),
            "summary_headings": [h for h in ("## Build Summary", "### Model tree", "### Requirements", "### Parameters",
                                             "### Provenance", "### Validated", "### NOT validated") if h in result.text],
            "validated_cyl": "Cyl" in self.section(result.text, "Model tree"),
        }

    def system_text(self, request) -> str:
        body = request.json
        if body.get("system"):
            return body["system"] if isinstance(body["system"], str) else json.dumps(body["system"])
        first = (body.get("messages") or [{}])[0]
        return first.get("content", "") if first.get("role") == "system" else ""


class EveryTierTest(MatrixCase):
    """Each tier completes the build on its own."""

    def test_each_tier_completes_the_build_and_validates(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                result = self.run_tier(tier)
                self.assertEqual(result.outcome, "validated", (tier, result.text[:300]))
                self.assertTrue(result.ok)
                self.server.assert_no_unexpected()
                self.assertEqual(self.signature(result)["objects"], ["Cyl", "Plate"])


class SignatureTest(MatrixCase):
    """The same ledger outcome, dossier and document through every tier."""

    def test_signatures_are_identical_across_tiers(self):
        sigs = {}
        for tier in TIERS:
            result = self.run_tier(tier)
            sigs[tier] = self.signature(result)
        base = sigs["native"]
        self.assertEqual(base["outcome"], "validated")
        self.assertEqual(base["recorded_tools"], ["create_document", "create_object", "execute_code"])
        for tier, sig in sigs.items():
            with self.subTest(tier=tier):
                self.assertEqual(sig, base)

    def test_the_final_text_starts_with_the_model_commentary_in_every_tier(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                result = self.run_tier(tier)
                self.assertTrue(result.text.startswith(FINAL), result.text[:120])
                self.assertIn("## Build Summary", result.text)

    def test_every_tier_records_a_snapshot_or_a_reason_and_a_checkpoint_per_geometry_step(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                self.run_tier(tier)
                for s in self.project.read_steps():
                    if s.geometry:
                        self.assertTrue(s.snapshot or s.snapshot_note)
                        self.assertTrue(s.checkpoint)


class WireShapeTest(MatrixCase):
    """What differs between the tiers is the request, never the result."""

    def test_native_tier_sends_a_tools_array_and_text_tier_sends_none(self):
        self.run_tier("native")
        self.assertTrue(all("tools" in r.json for r in self.server.requests))
        self.run_tier("text")
        for r in self.server.requests:
            self.assertNotIn('"tools"', json.dumps(r.json))
            self.assertIn("TOOL CALLING PROTOCOL", self.system_text(r))

    def test_text_tier_carries_the_calls_and_results_as_text_blocks_in_the_next_request(self):
        self.run_tier("text")
        later = json.dumps(self.server.requests[-1].json)
        self.assertNotIn('"tool_calls"', later)
        self.assertIn("<tool_call>", later)
        self.assertIn("<tool_result", later)

    def test_no_vision_tier_never_sends_an_image(self):
        self.run_tier("no_vision")
        for r in self.server.requests:
            self.assertNotIn("image_url", json.dumps(r.json))
            self.assertNotIn('"type": "image"', json.dumps(r.json))

    def test_the_reasoning_knob_tier_sends_an_effort_and_the_other_does_not(self):
        self.run_tier("reasoning_knob")
        self.assertTrue(any("reasoning_effort" in r.json for r in self.server.requests), "no effort parameter on the wire")
        self.run_tier("reasoning_no_knob")
        for r in self.server.requests:
            self.assertFalse({"reasoning_effort", "reasoning", "thinking"} & set(r.json), r.json.keys())

    def test_the_reasoning_text_never_becomes_the_answer(self):
        for tier in ("reasoning_knob", "reasoning_no_knob"):
            with self.subTest(tier=tier):
                result = self.run_tier(tier)
                self.assertNotIn("Thinking about the next step", result.text)

    def test_the_small_context_tier_gets_the_lite_prompt_and_the_lite_tool_list(self):
        result = self.run_tier("small_context")
        self.assertEqual(result.outcome, "validated")
        lite = {s.name for s in tools.specs(lite=True)}
        for r in self.server.requests:
            self.assertTrue(set(r.tool_names) <= lite, set(r.tool_names) - lite)
        self.assertLess(len(self.system_text(self.server.requests[0])), len(prompt.CORE))

    def test_the_system_message_is_the_same_on_every_round_of_a_tier(self):
        for tier in ("native", "text", "small_context"):
            with self.subTest(tier=tier):
                self.run_tier(tier)
                self.assertEqual(len({self.system_text(r) for r in self.server.requests}), 1)

    def test_the_anthropic_dialect_puts_the_tools_in_its_own_shape(self):
        self.run_tier("anthropic")
        first = self.server.requests[0].json
        self.assertIn("tools", first)
        self.assertTrue(all("input_schema" in t for t in first["tools"]))
        self.assertTrue(self.server.requests[0].tool_names)

    def test_each_tier_makes_exactly_one_request_per_scripted_round(self):
        for tier in TIERS:
            with self.subTest(tier=tier):
                self.run_tier(tier)
                self.assertEqual(self.server.request_count, 3)


class TierFailureTest(MatrixCase):
    """A weak tier that misbehaves is still handled the same way: the failure is visible, not a success."""

    def test_a_text_tier_that_never_writes_a_valid_block_ends_without_a_validated_outcome(self):
        override, _ = TIERS["text"]
        self.doc = self.doc_name("Mx")
        self.project = self.new_project("matrix bad text", chat_id=f"20261002-101010-{uuid.uuid4().hex[:6]}")
        plan = Round([("text", "I will build it now: create_document then create_object."), ("stop_reason", "stop"), ("done", "stop")])
        self.server = ScriptedServer([plan] * 6)
        self.addCleanup(self.server.shutdown)
        profile = params.Profile(name="mx-bad", provider="openai_compat", base_url=self.server.base_url,
                                 model=f"mx-bad-{uuid.uuid4().hex[:8]}", api_key="key", max_tokens=1024,
                                 caps_override=dict(override))
        result = self.turn(PROMPT, project=self.project, provider=providers.make_provider(profile), profile=profile, resolve_caps=True)
        self.assertNotEqual(result.outcome, "validated")
        self.assertFalse(App.listDocuments().get(self.doc))   # nothing was built
        self.assertTrue(self.notes("gate") or self.notes("cutoff") or self.notes("breaker"))


if __name__ == "__main__":
    unittest.main()
