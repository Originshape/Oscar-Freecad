"""oscar.prompt: the static prompt texts, their lint rules and the conditional fragments."""

import math
import os
import re
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import context, flow, models, prompt  # noqa: E402

# The tool matrix (41 required + the two optional net tools). Kept here on purpose: the prompt's own
# KNOWN_TOOLS must equal it, so an edit on one side without the other fails.
CONTRACT_TOOLS = frozenset("""
list_documents get_objects get_object get_view plan_set plan_update record_note create_document reload_document
import_file download_file get_parts_list insert_part_from_library create_object edit_object delete_object execute_code
execute_code_async execute_code_headless get_async_status
apply_feature_plan reapply_plan_feature feature_tree_status run_fem_analysis find_reference_part source_reference_part
import_reference_part measure_reference_part enclosure_spec build_enclosure add_cutout add_vents set_parameters
get_parameters bind_expression set_parameter validate_model check_fit save_document export_model
export_rebuild_script web_search fetch_url
""".split())
OPTIONAL_TOOLS = frozenset({"web_search", "fetch_url"})
LITE_TOOLS = frozenset("""
plan_set plan_update record_note list_documents get_objects get_object get_view create_document create_object
edit_object execute_code import_file find_reference_part source_reference_part build_enclosure enclosure_spec
set_parameter validate_model check_fit
""".split())
KIT_HIDDEN = frozenset("""
execute_code_async get_async_status execute_code_headless apply_feature_plan reapply_plan_feature
insert_part_from_library get_parts_list reload_document feature_tree_status download_file save_document
""".split())
# snake_case identifiers the prompts may put in backticks that are NOT tools (none today)
NOT_TOOLS = frozenset()

BACKTICKED = re.compile(r"`([^`]+)`")
TOOL_LIKE = re.compile(r"^([a-z][a-z0-9]*(?:_[a-z0-9]+)+)(?:\([^`]*\))?$")


def est(text: str) -> int:
    return math.ceil(len(text) / 3.5)


def referenced_tools(text: str) -> set[str]:
    """Identifiers in backticks that look like a tool name (snake_case, optionally with an argument list)."""
    found = set()
    for chunk in BACKTICKED.findall(text):
        m = TOOL_LIKE.match(chunk.strip())
        if m and m.group(1) not in NOT_TOOLS:
            found.add(m.group(1))
    return found


def mentions(text: str, name: str) -> bool:
    return re.search(rf"(?<![a-z0-9_]){re.escape(name)}(?![a-z0-9_])", text) is not None


def all_texts() -> dict[str, str]:
    texts = {"CORE": prompt.CORE, "LITE": prompt.LITE}
    texts.update({f"PLAYBOOK {k}": v for k, v in prompt.PLAYBOOKS.items()})
    texts.update({f"TEMPLATE {k}": v for k, v in prompt.TEMPLATES.items()})
    texts.update({f"PHASE {k}": v for k, v in prompt.PHASE_GUIDANCE.items()})
    return texts


def numbered_lists(text: str) -> list[list[int]]:
    """The numbered lists of ``text`` (lines starting ``N.`` or ``N)``): a list restarts whenever a number is not previous + 1."""
    lists: list[list[int]] = []
    for line in text.splitlines():
        m = re.match(r"^\s*(\d+)[.)]\s", line)
        if not m:
            continue
        n = int(m.group(1))
        if lists and n == lists[-1][-1] + 1:
            lists[-1].append(n)
        else:
            lists.append([n])
    return lists


class LintTest(unittest.TestCase):
    def test_numbering_is_unique_and_sequential(self):
        for label, text in (("CORE", prompt.CORE), ("LITE", prompt.LITE)):
            lists = numbered_lists(text)
            for numbers in lists:
                self.assertEqual(numbers, list(range(1, len(numbers) + 1)), f"{label}: {numbers}")
            # at most one numbered list per text: a second list would reuse the numbers 1.. (the old duplicate "13.")
            self.assertLessEqual(len(lists), 1, f"{label}: more than one numbered list: {lists}")

    def test_budget_in_estimated_tokens(self):
        self.assertLessEqual(context.estimate_tokens_text(prompt.CORE), 1100)
        self.assertLessEqual(context.estimate_tokens_text(prompt.LITE), 500)
        self.assertEqual(context.estimate_tokens_text(prompt.CORE), est(prompt.CORE))
        for name, text in prompt.PLAYBOOKS.items():
            self.assertLessEqual(context.estimate_tokens_text(text), 400, name)

    def test_budget_holds_for_every_rendering_variant(self):
        for names in (None, LITE_TOOLS, CONTRACT_TOOLS - KIT_HIDDEN, CONTRACT_TOOLS - OPTIONAL_TOOLS):
            self.assertLessEqual(est(prompt.build_system_prompt(lite=False, tool_names=names)), 1100)
            self.assertLessEqual(est(prompt.build_system_prompt(lite=True, tool_names=names)), 500)

    def test_known_tools_match_the_contract_matrix(self):
        self.assertEqual(prompt.KNOWN_TOOLS, CONTRACT_TOOLS)
        self.assertEqual(len(CONTRACT_TOOLS), 43)

    def test_tool_names_exist(self):
        referenced = set()
        for label, text in all_texts().items():
            names = referenced_tools(text)
            unknown = names - CONTRACT_TOOLS
            self.assertFalse(unknown, f"{label} names tools the contract does not know: {sorted(unknown)}")
            referenced |= names
        # the registry check is strict once every non-optional tool of the matrix is registered; until
        # then the CONTRACT_TOOLS list is the universe (the registry of an unfinished tree lacks some tools)
        try:
            from oscar import tools
        except ImportError:   # pure-Python run without FreeCAD
            self.skipTest("oscar.tools needs FreeCAD; the contract-list check above already ran")
        registry = {t.name for t in tools.get_tools()}
        if (CONTRACT_TOOLS - OPTIONAL_TOOLS) <= registry:
            missing = (referenced - OPTIONAL_TOOLS) - registry
            self.assertFalse(missing, f"the prompts name tools that are not registered: {sorted(missing)}")
            optional = referenced & OPTIONAL_TOOLS
            self.assertTrue(optional <= registry or optional <= OPTIONAL_TOOLS)

    def test_lite_names_only_lite_tools(self):
        extra = referenced_tools(prompt.LITE) - LITE_TOOLS
        self.assertFalse(extra, f"LITE names tools the lite tool set does not offer: {sorted(extra)}")
        try:
            from oscar import toolrun
        except ImportError:
            return
        self.assertEqual(toolrun.LITE_TOOLS, LITE_TOOLS)

    def test_every_phase_has_short_guidance_with_an_exit_criterion(self):
        self.assertEqual(set(prompt.PHASE_GUIDANCE), set(flow.PHASES))
        for phase, text in prompt.PHASE_GUIDANCE.items():
            self.assertLessEqual(len(text.split()), 80, phase)
            self.assertIn("Exit:", text, phase)
            self.assertTrue(text.strip().endswith("."), phase)

    def test_forbidden_phrases_are_absent_everywhere(self):
        forbidden = ("think and plan internally", "without deliberating", "Thinking counts against")
        shouting = ("in ONE response", "ONE feature")   # the old batch-vs-one-per-call pair, matched case-sensitively
        for label, text in all_texts().items():
            for phrase in forbidden:
                self.assertNotIn(phrase.lower(), text.lower(), f"{label}: {phrase!r}")
            for phrase in shouting:
                self.assertNotIn(phrase, text, f"{label}: {phrase!r}")

    def test_required_literals_are_present_in_core(self):
        for literal in ("EXECUTION CONTRACT", "never end the turn at", "HOUSING/ENCLOSURE", "download_file"):
            self.assertIn(literal, prompt.CORE)
        self.assertIn("about 300 words of thinking per round", prompt.CORE)
        self.assertIn("under 40 lines", prompt.CORE)
        self.assertIn("[governor]", prompt.CORE)
        self.assertIn("<workspace_state>", prompt.CORE)
        self.assertIn("[health]", prompt.CORE)
        self.assertIn("Never claim a check you did not run", prompt.CORE)
        for source in ("user_requirement", "datasheet", "step_measurement", "standard", "derived", "assumption"):
            self.assertIn(source, prompt.CORE)

    def test_core_sections_come_in_the_contract_order(self):
        headings = ["HOW THE WORK RUNS", "EXECUTION CONTRACT", "HOUSING/ENCLOSURE AROUND A REAL PART",
                    "PARAMETERS AND NAMING", "TOOLS AND VERIFICATION", "FINAL REPORT"]
        positions = [prompt.CORE.index(h) for h in headings]
        self.assertEqual(positions, sorted(positions))
        self.assertTrue(prompt.CORE.startswith("You are Oscar"))   # the ROLE section has no heading and comes first

    def test_lite_keeps_the_contract_core(self):
        for literal in ("EXECUTION CONTRACT", "never end the turn at", "HOUSING/ENCLOSURE", "[governor]", "under 40 lines"):
            self.assertIn(literal, prompt.LITE)
        self.assertNotIn("download_file", prompt.LITE)   # not in the 19 lite tools

    def test_final_report_template(self):
        self.assertLessEqual(len(prompt.FINAL_REPORT_TEMPLATE.splitlines()), 20)
        self.assertIn(prompt.FINAL_REPORT_TEMPLATE, prompt.CORE)
        self.assertEqual(numbered_lists(prompt.FINAL_REPORT_TEMPLATE), [[1, 2, 3, 4, 5]])
        self.assertIn("Build Summary", prompt.FINAL_REPORT_TEMPLATE)

    def test_build_system_prompt_is_deterministic_and_static(self):
        self.assertEqual(prompt.build_system_prompt(), prompt.build_system_prompt())
        self.assertEqual(prompt.build_system_prompt(), prompt.CORE)
        for text in (prompt.CORE, prompt.LITE):
            self.assertNotRegex(text, r"\b\d{4}-\d{2}-\d{2}\b")   # no date / clock value
            self.assertNotRegex(text, r"[0-9a-f]{8}-[0-9a-f]{4}")   # no uuid

    def test_prompt_never_embeds_a_document_or_object_name(self):
        try:
            import FreeCAD
        except ImportError:
            self.skipTest("FreeCAD not importable")
        before = prompt.build_system_prompt()
        doc = FreeCAD.newDocument("UniqueDocName9137")
        try:
            doc.addObject("Part::Box", "UniqueObjectName9137")
            after = prompt.build_system_prompt()
        finally:
            FreeCAD.closeDocument(doc.Name)
        self.assertEqual(before, after)
        self.assertNotIn("UniqueDocName9137", after)
        self.assertNotIn("UniqueObjectName9137", after)

    def test_playbook_keys_and_headings(self):
        self.assertEqual(set(prompt.PLAYBOOKS), {"fem", "parts_library", "structured_part"})
        for key, text in prompt.PLAYBOOKS.items():
            self.assertTrue(text.startswith(f"PLAYBOOK {key}"), key)

    def test_contradictions_of_the_old_prompt_are_gone(self):
        # AUDIT F-21: batch vs one-per-call, two modelling doctrines, a doubled rule 13, a garbled 11a
        text = prompt.CORE
        self.assertIn("independent calls inside that step may be batched", text)
        self.assertEqual(text.count("Never claim a check you did not run"), 1)   # said once, in the report template
        self.assertNotIn("PartDesign::Body, positioned", text)
        self.assertEqual(len(re.findall(r"\bnever fuse\b", text)), 1)
        self.assertNotRegex(text, r"\n13\.")

    def test_templates_classify_as_their_intent(self):
        expected = {"enclosure_intake": "enclosure", "parametric_part": "generic_part",
                    "bracket": "bracket"}
        self.assertEqual(set(prompt.TEMPLATES), set(expected))
        for key, intent in expected.items():
            self.assertEqual(flow.classify_intent(prompt.TEMPLATES[key]), intent, key)
            self.assertFalse(referenced_tools(prompt.TEMPLATES[key]), key)


class ConditionalFragmentsTest(unittest.TestCase):
    def test_default_is_the_full_text(self):
        self.assertEqual(prompt.build_system_prompt(), prompt.CORE)
        self.assertEqual(prompt.build_system_prompt(tool_names=CONTRACT_TOOLS), prompt.CORE)
        self.assertEqual(prompt.build_system_prompt(lite=True), prompt.LITE)

    def test_lite_text_renders_whole_for_the_lite_tool_set(self):
        self.assertEqual(prompt.build_system_prompt(lite=True, tool_names=LITE_TOOLS), prompt.LITE)

    def test_a_tool_that_is_not_offered_is_never_named(self):
        offered_sets = {
            "kit intents": CONTRACT_TOOLS - KIT_HIDDEN,
            "no optional tools": CONTRACT_TOOLS - OPTIONAL_TOOLS,
            "lite": LITE_TOOLS,
            "tiny": frozenset({"get_objects", "create_object"}),
            "empty": frozenset(),
        }
        for label, offered in offered_sets.items():
            for lite in (False, True):
                text = prompt.build_system_prompt(lite=lite, tool_names=offered,
                                                  playbooks=("fem", "parts_library", "structured_part"))
                for name in CONTRACT_TOOLS - offered:
                    self.assertFalse(mentions(text, name), f"{label} lite={lite}: {name} is named but not offered")

    def test_download_file_follows_its_availability(self):
        self.assertIn("download_file", prompt.build_system_prompt(tool_names=CONTRACT_TOOLS))
        self.assertNotIn("download_file", prompt.build_system_prompt(tool_names=CONTRACT_TOOLS - KIT_HIDDEN))

    def test_optional_tools_appear_only_when_registered(self):
        without = prompt.build_system_prompt(tool_names=CONTRACT_TOOLS - OPTIONAL_TOOLS)
        self.assertNotIn("web_search", without)
        self.assertNotIn("fetch_url", without)
        self.assertIn("web_search", prompt.build_system_prompt(tool_names=CONTRACT_TOOLS))

    def test_a_section_without_any_offered_sentence_disappears_with_its_heading(self):
        text = prompt.build_system_prompt(tool_names={"get_objects"})
        self.assertNotIn("HOUSING/ENCLOSURE", text)
        self.assertIn("EXECUTION CONTRACT", text)
        self.assertIn("FINAL REPORT", text)

    def test_render_does_not_depend_on_argument_order_or_container(self):
        a = prompt.build_system_prompt(tool_names=sorted(LITE_TOOLS), playbooks=["parts_library", "fem"])
        b = prompt.build_system_prompt(tool_names=tuple(reversed(sorted(LITE_TOOLS))), playbooks=("fem", "parts_library", "parts_library"))
        c = prompt.build_system_prompt(tool_names=iter(sorted(LITE_TOOLS)), playbooks=iter(["fem", "parts_library"]))
        self.assertEqual(a, b)
        self.assertEqual(a, c)

    def test_playbooks_are_appended_when_active_only(self):
        base = prompt.build_system_prompt()
        with_fem = prompt.build_system_prompt(playbooks=["fem"])
        self.assertTrue(with_fem.startswith(base))
        self.assertIn(prompt.PLAYBOOKS["fem"], with_fem)
        self.assertNotIn("PLAYBOOK parts_library", with_fem)
        self.assertEqual(prompt.build_system_prompt(playbooks=["nope"]), base)   # unknown keys are ignored

    def test_playbook_sentences_follow_tool_availability(self):
        text = prompt.build_system_prompt(playbooks=["fem"], tool_names=CONTRACT_TOOLS - {"run_fem_analysis"})
        self.assertNotIn("run_fem_analysis", text)
        self.assertIn("PLAYBOOK fem", text)

    def test_lite_choice_from_caps(self):
        small = models.ModelCaps(context_window=8192)
        text_only = models.ModelCaps(tools="text")
        no_parallel = models.ModelCaps(parallel_calls=False)
        big = models.ModelCaps(context_window=200000)
        for caps in (small, text_only, no_parallel):
            self.assertEqual(prompt.build_system_prompt(caps), prompt.LITE, caps)
        self.assertEqual(prompt.build_system_prompt(big), prompt.CORE)
        self.assertEqual(prompt.build_system_prompt(None), prompt.CORE)
        self.assertEqual(prompt.build_system_prompt(small, lite=False), prompt.CORE)   # explicit beats caps
        self.assertEqual(prompt.build_system_prompt(big, lite=True), prompt.LITE)

    def test_unusable_caps_fall_back_to_core(self):
        self.assertEqual(prompt.build_system_prompt(object()), prompt.CORE)


class PhaseGuidanceTest(unittest.TestCase):
    def test_phase_guidance_lookup(self):
        for phase in flow.PHASES:
            self.assertEqual(prompt.phase_guidance(phase), prompt.PHASE_GUIDANCE[phase])
        self.assertEqual(prompt.phase_guidance("build"), prompt.PHASE_GUIDANCE["BUILD"])
        self.assertEqual(prompt.phase_guidance(" Verify "), prompt.PHASE_GUIDANCE["VERIFY"])
        self.assertEqual(prompt.phase_guidance("NOPE"), "")
        self.assertEqual(prompt.phase_guidance(""), "")
        self.assertEqual(prompt.phase_guidance(None), "")

    def test_flow_uses_the_prompt_guidance(self):
        for phase in flow.PHASES:
            self.assertEqual(flow.phase_guidance(phase), prompt.PHASE_GUIDANCE[phase])

    def test_guidance_names_only_contract_tools(self):
        for phase, text in prompt.PHASE_GUIDANCE.items():
            self.assertLessEqual(referenced_tools(text), CONTRACT_TOOLS, phase)


class StructuredPartPlaybookTest(unittest.TestCase):
    """The structured_part playbook: prefer a schema-2 plan for prismatic parts (plan v2 wiring)."""

    TEXT = prompt.PLAYBOOKS["structured_part"]

    def test_heading_budget_and_tools(self):
        self.assertTrue(self.TEXT.startswith("PLAYBOOK structured_part"))
        self.assertLessEqual(context.estimate_tokens_text(self.TEXT), 400)
        self.assertEqual(referenced_tools(self.TEXT), {"apply_feature_plan", "reapply_plan_feature", "validate_model",
                                                      "execute_code"})

    def test_it_says_the_things_the_model_needs(self):
        for needle in ("parameters first", "one feature per entry", "schema_version 2", "never by index",
                       "never fused", "stays available"):
            self.assertIn(needle, self.TEXT)
        for op in ("fillet", "chamfer", "thickness", "linear_pattern", "polar_pattern", "mirror", "boss"):
            self.assertIn(op, self.TEXT)
        for rule in ("all_outer", "vertical", "top_outer", "bottom_outer", "edges_of_feature:<id>"):
            self.assertIn(rule, self.TEXT)

    def test_every_op_and_rule_it_names_exists_in_the_schema(self):
        from oscar.plan import _schema_v2, schema
        for op in schema.FEATURE_OPS_V2:
            self.assertIn(op, self.TEXT, op)
        for rule in _schema_v2.EDGE_RULES:
            self.assertIn(rule, self.TEXT, rule)

    def test_sentences_follow_tool_availability(self):
        offered = CONTRACT_TOOLS - {"apply_feature_plan", "reapply_plan_feature"}
        text = prompt.build_system_prompt(playbooks=["structured_part"], tool_names=offered)
        self.assertNotIn("apply_feature_plan", text)
        self.assertNotIn("reapply_plan_feature", text)
        self.assertIn("PLAYBOOK structured_part", text)
        self.assertIn("never fused", text)
        full = prompt.build_system_prompt(playbooks=["structured_part"], tool_names=CONTRACT_TOOLS)
        self.assertIn("`apply_feature_plan`", full)

    def test_the_playbook_never_pushes_the_core_over_budget(self):
        base = prompt.build_system_prompt(tool_names=CONTRACT_TOOLS)
        both = prompt.build_system_prompt(playbooks=["structured_part"], tool_names=CONTRACT_TOOLS)
        self.assertTrue(both.startswith(base))
        self.assertLessEqual(est(base), 1100)

    def test_it_is_not_in_the_lite_prompt(self):
        self.assertNotIn("structured_part", prompt.LITE)
        self.assertNotIn("apply_feature_plan", prompt.LITE)


if __name__ == "__main__":
    unittest.main()
