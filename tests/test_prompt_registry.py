"""oscar.prompt against the REAL tool registry (every ``tools_*`` module, kit tools included) and the lite persona.

* the full-registry lint: every tool a prompt text names is registered, visible to the intent the text is for, and the
  tool lists / sets the prompt module assumes (the 19 lite tools, the kit-hidden tools) are the ones the registry applies;
* ``lite_prompt_wanted`` conditions map to the 19-tool lite set (a request built from those caps offers exactly them);
* the enclosure sequence the prompts teach is the sequence the kit supports (stage names, order, tool visibility per phase);
* the lite prompt ALONE can drive an enclosure build: a scripted model that calls only the tools the LITE text names (through a
  real local server, the real provider, real ``run_tool``, a real Project and real FreeCAD documents) finishes validated.
"""

import json
import os
import re
import sys
import unittest
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

import FreeCAD as App  # noqa: E402

from oscar import context, flow, models, params, prompt, providers, toolrun, tools  # noqa: E402
from oscar.kit import spec as kit_spec  # noqa: E402
from support.realflow import RealFlowCase  # noqa: E402
from support.scripted_llm import Round, call, calls, say  # noqa: E402
from support.server import ScriptedServer  # noqa: E402

BACKTICKED = re.compile(r"`([a-z][a-z0-9_]*)(?:\([^`]*\))?`")
OPTIONAL = frozenset({"web_search", "fetch_url"})
KIT_INTENTS = ("enclosure", "fixture", "bracket")
STAGE_NAMES = ("shell", "standoffs", "ports", "lid", "vents", "hooks", "assembly")


def named(text: str) -> list[str]:
    """Tool names in backticks, in order of first appearance (names the prompt knows)."""
    out: list[str] = []
    for name in BACKTICKED.findall(text):
        if name in prompt.KNOWN_TOOLS and name not in out:
            out.append(name)
    return out


def texts() -> dict[str, str]:
    found = {"CORE": prompt.CORE, "LITE": prompt.LITE}
    found.update({f"PLAYBOOK {k}": v for k, v in prompt.PLAYBOOKS.items()})
    found.update({f"PHASE {k}": v for k, v in prompt.PHASE_GUIDANCE.items()})
    return found


def registry() -> set[str]:
    return {t.name for t in tools.get_tools()}


class FullRegistryLintTest(unittest.TestCase):
    def test_every_non_optional_tool_of_the_matrix_is_registered(self):
        missing = (prompt.KNOWN_TOOLS - OPTIONAL) - registry()
        self.assertFalse(missing, f"the matrix names tools that are not registered: {sorted(missing)}")

    def test_the_kit_tools_are_registered(self):
        for name in ("enclosure_spec", "build_enclosure", "add_cutout", "add_vents", "set_parameters", "get_parameters",
                     "bind_expression", "set_parameter"):
            self.assertIn(name, registry())

    def test_every_tool_a_prompt_text_names_is_registered_or_optional(self):
        reg = registry()
        for label, text in texts().items():
            unknown = {n for n in named(text) if n not in reg and n not in OPTIONAL}
            self.assertFalse(unknown, f"{label} names unregistered tools: {sorted(unknown)}")
        for key, text in prompt.TEMPLATES.items():
            self.assertFalse(named(text), f"template {key} names a tool")

    def test_the_registry_does_not_know_a_name_the_prompt_lacks(self):
        """KNOWN_TOOLS (the conditional-fragment universe) covers the whole registry: a tool added without it would
        never get its sentence dropped when it is hidden."""
        self.assertFalse(registry() - prompt.KNOWN_TOOLS, sorted(registry() - prompt.KNOWN_TOOLS))

    def test_the_lite_set_is_the_nineteen_tools_the_prompt_assumes(self):
        self.assertEqual(len(toolrun.LITE_TOOLS), 19)
        self.assertEqual({s.name for s in tools.specs(lite=True)}, set(toolrun.LITE_TOOLS))
        self.assertLessEqual(set(named(prompt.LITE)), set(toolrun.LITE_TOOLS))

    def test_the_kit_intents_hide_what_the_prompt_never_names_for_them(self):
        hidden = set(toolrun.INTENT_HIDDEN["enclosure"])
        for intent in KIT_INTENTS:
            self.assertEqual(set(toolrun.INTENT_HIDDEN[intent]), hidden)
        text = prompt.build_system_prompt(lite=False, tool_names={s.name for s in tools.specs(intent="enclosure")})
        for name in hidden:
            self.assertNotIn(name, text, name)
        self.assertLessEqual(context.estimate_tokens_text(text), 1100)

    def test_every_rendering_per_intent_and_lite_stays_in_budget_and_names_only_offered_tools(self):
        for intent in ("enclosure", "bracket", "generic_part", "fem", "question"):
            for lite in (False, True):
                offered = {s.name for s in tools.specs(intent=intent, lite=lite)}
                text = prompt.build_system_prompt(lite=lite, tool_names=offered,
                                                  playbooks=("fem", "parts_library", "structured_part"))
                extra = {n for n in named(text) if n not in offered}
                self.assertFalse(extra, f"{intent} lite={lite}: {sorted(extra)}")
                base = prompt.build_system_prompt(lite=lite, tool_names=offered)
                self.assertLessEqual(context.estimate_tokens_text(base), 500 if lite else 1100, (intent, lite))

    def test_ultra_lite_keeps_the_enclosure_story(self):
        text = prompt.build_system_prompt(lite=True, tool_names=context.ULTRA_LITE_TOOLS)
        self.assertIn("`enclosure_spec`", text)
        self.assertIn("`build_enclosure`", text)
        for stage in STAGE_NAMES:
            self.assertIn(stage, text)
        self.assertNotIn("validate_model", text)   # not offered in the 9-tool floor set


class LiteSelectionTest(unittest.TestCase):
    """``lite_prompt_wanted`` (small window, text mode, no parallel calls) maps to the 19-tool lite request."""

    CASES = {
        "small window": models.ModelCaps(context_window=16384),
        "text tool mode": models.ModelCaps(tools="text"),
        "no parallel calls": models.ModelCaps(parallel_calls=False),
    }

    def test_each_condition_selects_the_lite_prompt_and_the_lite_tool_array(self):
        for label, caps in self.CASES.items():
            with self.subTest(label):
                self.assertTrue(models.lite_prompt_wanted(caps))
                self.assertEqual(prompt.build_system_prompt(caps), prompt.LITE)
                self.assertEqual({s.name for s in tools.specs(lite=models.lite_prompt_wanted(caps))}, set(toolrun.LITE_TOOLS))

    def test_a_capable_model_gets_the_full_prompt(self):
        for caps in (models.ModelCaps(context_window=16385), models.ModelCaps(context_window=200000)):
            self.assertFalse(models.lite_prompt_wanted(caps))
            self.assertEqual(prompt.build_system_prompt(caps), prompt.CORE)

    def test_the_lite_prompt_names_every_kit_tool_the_lite_set_can_use(self):
        names = set(named(prompt.LITE))
        for needed in ("find_reference_part", "source_reference_part", "enclosure_spec", "build_enclosure", "validate_model",
                       "check_fit", "set_parameter", "execute_code", "get_view", "get_objects"):
            self.assertIn(needed, names)


class EnclosureSequenceTest(unittest.TestCase):
    """The prompts teach the sequence the kit supports, and only with tools visible when it is taught."""

    def test_the_stages_the_prompts_list_are_the_kit_stages_in_order(self):
        kit_order = [s for s in kit_spec.STAGES if s in STAGE_NAMES]
        self.assertEqual(tuple(kit_order), STAGE_NAMES)   # the kit builds params and reference first (next)
        for label in ("CORE", "LITE", "PHASE BUILD"):
            text = texts()[label]
            start = text.index("`build_enclosure` once per stage")
            positions = []
            for stage in STAGE_NAMES:
                m = re.search(rf"\b{stage}\b", text[start:])
                self.assertIsNotNone(m, f"{label}: stage {stage} is not listed")
                positions.append(m.start())
            self.assertEqual(positions, sorted(positions), label)

    def test_core_order_is_spec_then_stages_then_checks_then_export(self):
        text = prompt.CORE
        order = ["`enclosure_spec`", "`build_enclosure`", "`validate_model`", "`check_fit`", "`export_model`"]
        positions, start = [], text.index("Enclosure order:")
        for o in order:
            start = text.index(o, start)
            positions.append(start)
        self.assertEqual(positions, sorted(positions))
        self.assertLess(text.index("`find_reference_part`"), text.index("`enclosure_spec`"))
        self.assertLess(text.index("`set_parameters`"), text.index("`validate_model`"))
        self.assertIn("execute_code", text)   # free-form remains for what the kit does not cover

    def test_phase_guidance_walks_the_sequence(self):
        guide = prompt.PHASE_GUIDANCE
        self.assertIn("`find_reference_part`", guide["SOURCE"])
        self.assertIn("`source_reference_part`", guide["SOURCE"])
        self.assertIn("`enclosure_spec`", guide["PLAN"])
        self.assertIn("`set_parameters`", guide["PARAMS"])
        self.assertIn("`build_enclosure`", guide["PARAMS"])
        self.assertIn("`build_enclosure`", guide["BUILD"])
        self.assertIn("`execute_code`", guide["BUILD"])   # free-form for non-enclosure parts
        self.assertIn("`validate_model`", guide["VERIFY"])
        self.assertIn("`check_fit`", guide["VERIFY"])
        self.assertIn("`export_model`", guide["DELIVER"])
        self.assertIn("`record_note`", guide["DELIVER"])
        for word in ("validated", "NOT validated"):
            self.assertIn(word, guide["DELIVER"])

    def test_guidance_names_only_tools_the_enclosure_intent_is_offered(self):
        for phase, text in prompt.PHASE_GUIDANCE.items():
            offered = {s.name for s in tools.specs(intent="enclosure")}
            self.assertLessEqual(set(named(text)), offered, f"{phase}: {sorted(set(named(text)) - offered)}")

    def test_guidance_names_a_tool_before_its_phase_only_for_tools_already_visible(self):
        """Every tool a phase's guidance names is visible at that phase for the enclosure intent (cumulative masks)."""
        for phase in flow.PHASES:
            visible = {s.name for s in tools.specs(phases=_visible(phase), intent="enclosure")}
            names = set(named(prompt.PHASE_GUIDANCE[phase]))
            self.assertLessEqual(names, visible, f"{phase}: {sorted(names - visible)}")

    def test_guidance_marks_tools_outside_the_lite_set_as_conditional(self):
        """A lite model is not offered ``set_parameters`` and friends: they sit behind 'if offered'."""
        for phase, text in prompt.PHASE_GUIDANCE.items():
            for name in named(text):
                if name in toolrun.LITE_TOOLS:
                    continue
                before = text[:text.index(f"`{name}`")].lower()
                self.assertIn("if offered", before, f"{phase}: {name} is not lite and not conditional")

    def test_the_final_report_template_demands_validated_vs_not_validated_from_recorded_checks(self):
        template = prompt.FINAL_REPORT_TEMPLATE
        self.assertIn("Validated vs NOT validated", template)
        self.assertIn("Never claim a check you did not run", template)
        self.assertIn("Only recorded checks count", template)
        self.assertFalse(named(template))
        self.assertLessEqual(len(template.splitlines()), 20)
        # the same wording for the lite model
        self.assertIn("validated vs NOT validated", prompt.LITE)
        self.assertIn("only checks you ran", prompt.LITE)

    def test_the_enclosure_template_is_the_pi4_housing_request(self):
        text = prompt.TEMPLATES["enclosure_intake"]
        for part in ("Raspberry Pi 4", "snap-fit lid", "I/O port", "vent", "STEP", "parametric", "<ABS>"):
            self.assertIn(part, text)
        self.assertEqual(flow.classify_intent(text), "enclosure")

    def test_the_panel_menu_labels_every_template(self):
        try:
            from oscar import panel
        except ImportError:
            self.skipTest("the panel needs Qt")
        self.assertLessEqual(set(prompt.TEMPLATES), set(panel.OscarPanel._TEMPLATE_LABELS))
        self.assertIn("enclosure_intake", panel.OscarPanel._TEMPLATE_LABELS)


def _visible(phase: str) -> set[str]:
    index = flow.PHASES.index(phase)
    phases = set(flow.PHASES[: index + 1])
    if index >= flow.PHASES.index("PLAN"):
        phases.add("BUILD")
    return phases


# ------------------------------------------------------------------------------------------------------------------
# the lite persona
# ------------------------------------------------------------------------------------------------------------------

LITE_REQUEST = "Build a housing for a Raspberry Pi 4: bottom shell with port cutouts, vent slots and a snap-fit lid, ABS."


def lite_persona(doc: str) -> list[Round]:
    """A model that follows ONLY the lite prompt: the tools it names, in the order it names them. The prompt says: source
    with ``find_reference_part`` then ``source_reference_part``; ``enclosure_spec``; ``build_enclosure`` once per stage until all are
    built; ``validate_model``; ``check_fit``. (A document comes from ``create_document``, a tool of the lite array.)"""
    order = named(prompt.LITE)
    chain = [t for t in order if t in ("find_reference_part", "source_reference_part", "enclosure_spec", "build_enclosure",
                                       "validate_model", "check_fit")]
    args = {
        "find_reference_part": {"query": "raspberry pi 4 model b"},
        "source_reference_part": {"id": "rpi4b", "reason": "reference geometry for the housing"},
        "enclosure_spec": {"action": "set", "spec": {"name": doc, "material": "ABS"}, "doc_name": doc},
        "build_enclosure": {"stage": "next", "doc_name": doc},
        "validate_model": {"doc_name": doc},
        "check_fit": {"doc_name": doc},
    }
    rounds: list[Round] = []
    n = [0]

    def step(tool: str, **extra):
        n[0] += 1
        rounds.append(calls(call(tool, call_id=f"call_l{n[0]}", **{**args.get(tool, {}), **extra})))

    for tool in chain:
        if tool == "enclosure_spec":
            step("create_document", name=doc)
        if tool == "build_enclosure":
            for _ in range(len(kit_spec.STAGES)):   # "once per stage until all are built": next walks the nine stages
                step(tool)
        else:
            step(tool)
    rounds.append(say("Built the Pi 4 housing: shell, vents and a snap-fit lid in ABS. Validated: the model checks I ran."))
    return rounds


class LitePersonaTest(RealFlowCase):
    """A model that only knows the lite prompt finishes an enclosure build with the lite tools alone."""

    def run_lite(self):
        """One enclosure turn against the lite persona; returns the result (document name on ``self.doc``)."""
        self.doc = self.doc_name("LitePi")
        self.project = self.new_project("lite persona", chat_id=f"20261002-111111-{uuid.uuid4().hex[:6]}")
        self.server = ScriptedServer(lite_persona(self.doc))
        self.addCleanup(self.server.shutdown)
        profile = params.Profile(name="lite-persona", provider="openai_compat", base_url=self.server.base_url,
                                 model=f"lite-{uuid.uuid4().hex[:8]}", api_key="key", max_tokens=1024,
                                 caps_override={"context_window": 16384})
        provider = providers.make_provider(profile)
        return self.turn(LITE_REQUEST, project=self.project, provider=provider, profile=profile, resolve_caps=True)

    def test_the_persona_script_uses_only_lite_tools_named_by_the_lite_prompt(self):
        names = [c.name for r in lite_persona("X") for c in _round_calls(r)]
        self.assertTrue(names)
        used = set(names)
        self.assertLessEqual(used - {"create_document"}, set(named(prompt.LITE)))
        self.assertLessEqual(used, set(toolrun.LITE_TOOLS))
        self.assertEqual(names.count("build_enclosure"), len(kit_spec.STAGES))

    def test_the_lite_prompt_alone_drives_the_build_to_a_validated_finish(self):
        result = self.run_lite()
        self.server.assert_no_unexpected()
        self.assertEqual(result.outcome, "validated", result.text[:400])
        self.assertTrue(result.ok)
        # every request carried the LITE prompt (never the full one) and only lite tools
        self.assertTrue(self.server.requests)
        offered: set = set()
        for request in self.server.requests:
            system = _system_text(request)
            self.assertIn("EXECUTION CONTRACT", system)
            self.assertNotIn("HOUSING/ENCLOSURE AROUND A REAL PART", system)
            self.assertLessEqual(set(request.tool_names), set(toolrun.LITE_TOOLS))
            offered |= set(request.tool_names)
        # the phase-scoped array grows over the build and ends with the kit and verify tools
        self.assertGreaterEqual(offered, {"enclosure_spec", "build_enclosure", "validate_model", "check_fit"})
        recorded = [s.tool for s in self.project.read_steps()]
        self.assertEqual(recorded.count("build_enclosure"), len(kit_spec.STAGES))
        names = {o.Name for o in App.getDocument(self.doc).Objects}
        self.assertLessEqual({"Params", "Part_Shell", "Part_Lid", "Assembly"}, names)
        self.check_structure(result, self.project)
        self.assertIn("NOT validated", result.text)


def _round_calls(round_: Round) -> list:
    """The ``ToolCall`` objects of a scripted round."""
    found = []
    for event in round_.events:
        if event[0] == "tool_calls":
            found.extend(event[1])
    return found


def _system_text(request) -> str:
    body = request.json
    first = (body.get("messages") or [{}])[0]
    return first.get("content", "") if first.get("role") == "system" else json.dumps(body.get("system", ""))


if __name__ == "__main__":
    unittest.main()
