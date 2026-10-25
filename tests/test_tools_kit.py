"""The enclosure tools (oscar/tools_kit.py, tools_kit_params.py, tools_kit_base.py): registration, then every
handler through ``tools.run_tool`` with a real Project, a real Ledger and real FreeCAD documents under freecadcmd.

Covered: matrix flags and schemas, ``enclosure_spec`` (get / set / validate / defaults, merge, invalid values, the third-call rescue with
``SPEC DEFAULTS APPLIED`` per turn, Project persistence, ``spec_ok`` / ``param_names`` through ``ToolOutcome.extra``), ``build_enclosure``
(one stage per call, ``next`` resolved into ``outcome.args``, out-of-order and unknown stages, the defaults applied by the tool itself and
recorded as an assumption, a spec patch, up-to-date stages), a full nine-stage Pi 4 run (shell and lid separate closed valid solids, every
port window contains its connector, standoffs / vents / hooks exist, parameters bound), ``add_cutout`` / ``add_vents`` (idempotent per id,
rebuild of the built stages, rollback), ``set_parameters`` / ``get_parameters`` / ``bind_expression`` / ``set_parameter`` (edit, unknown
name, derived rows, spec-rule refusal, count parameters rebuild their stage, expression-bound edits change geometry) and the ``kit`` / ``P``
script helpers.
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
from FreeCAD import Vector as V  # noqa: E402

from oscar import tools, tools_kit, tools_kit_base  # noqa: E402
from oscar.cad import scripting  # noqa: E402
from oscar.flow import Ledger  # noqa: E402
from oscar.kit import build, params  # noqa: E402
from oscar.kit import geometry as G  # noqa: E402
from oscar.kit import spec as kspec  # noqa: E402
from oscar.toolctx import ToolContext, use_context  # noqa: E402
from support.fakes import temp_project  # noqa: E402

REQUEST = "Design a snap-fit enclosure for the Raspberry Pi 4 Model B with port cutouts, vent slots and a snap-fit lid in ABS."
KIT_TOOLS = ("enclosure_spec", "build_enclosure", "add_cutout", "add_vents", "set_parameters", "get_parameters", "bind_expression",
             "set_parameter")
STAGES = kspec.STAGES


def tip_size(doc, role):
    bb = G.shape_facts(G.tip(doc, role))["bbox"]
    return [round(bb[3] - bb[0], 2), round(bb[4] - bb[1], 2), round(bb[5] - bb[2], 2)]


def inside(obj, x, y, z):
    return G.global_shape(obj).isInside(V(x, y, z), 1e-6, True)


class Base(unittest.TestCase):
    """A real Project and Ledger, a fresh document and a context for ``run_tool``."""

    with_project = True

    def setUp(self):
        tools_kit_base.forget_specs()
        tools_kit.reset_failure_counters()
        self.project = temp_project(title="Pi 4 case") if self.with_project else None
        self.ledger = Ledger.from_request(REQUEST)
        self.turn = "T" + uuid.uuid4().hex[:8]
        self.state = {}
        self.docs = []
        self.addCleanup(self.close_docs)
        self.doc = self.new_doc()

    def close_docs(self):
        for name in self.docs:
            if name in App.listDocuments():
                App.closeDocument(name)

    def new_doc(self, prefix="PiCase"):
        doc = App.newDocument(f"{prefix}{uuid.uuid4().hex[:5]}")
        self.docs.append(doc.Name)
        return doc

    def ctx(self, **kw):
        values = dict(project=self.project, ledger=self.ledger, turn_id=self.turn, state=self.state)
        values.update(kw)
        return ToolContext(**values)

    def call(self, tool_name, ctx=None, **args):
        """``(text, is_error)`` of ``tools.run_tool``; ``self.outcome`` keeps the ToolOutcome."""
        ctx = ctx or self.ctx()
        if tool_name != "enclosure_spec":
            args.setdefault("doc_name", self.doc.Name)
        with use_context(ctx):
            text, _img, is_error = tools.run_tool(tool_name, args)
        self.outcome = ctx.outcome
        return text, is_error

    def ok(self, tool_name, **args):
        text, is_error = self.call(tool_name, **args)
        self.assertFalse(is_error, text)
        return text

    def fails(self, tool_name, *needles, **args):
        text, is_error = self.call(tool_name, **args)
        self.assertTrue(is_error, text)
        for n in needles:
            self.assertIn(n, text)
        return text

    def build_to(self, stage):
        """Build ``next`` until ``stage`` (inclusive)."""
        for _ in range(len(STAGES)):
            text = self.ok("build_enclosure", stage="next")
            if f"Stage '{stage}' built" in text:
                return
        self.fail(f"never reached {stage}")

    def build_all(self):
        for _ in range(len(STAGES)):
            text = self.ok("build_enclosure", stage="next")
            if "All 9 stages are built" in text:
                return text
        self.fail("the nine stages did not finish")


# --------------------------------------------------------------------------------------------------------------------
# registration
# --------------------------------------------------------------------------------------------------------------------

class RegistrationTest(unittest.TestCase):
    MATRIX = {   # name: (category, mutating, geometry, phases)
        "enclosure_spec": ("plan", True, False, ("PLAN", "PARAMS", "BUILD")),
        "build_enclosure": ("build", True, True, ("BUILD",)),
        "add_cutout": ("build", True, True, ("BUILD", "VERIFY")),
        "add_vents": ("build", True, True, ("BUILD", "VERIFY")),
        "set_parameters": ("params", True, True, ("PARAMS", "BUILD", "VERIFY")),
        "get_parameters": ("params", False, False, ("PLAN", "PARAMS", "BUILD", "VERIFY", "DELIVER")),
        "bind_expression": ("params", True, True, ("PARAMS", "BUILD", "VERIFY")),
        "set_parameter": ("params", True, True, ("BUILD", "VERIFY", "DELIVER")),
    }

    def test_all_eight_tools_are_registered_with_the_matrix_flags(self):
        for name, (category, mutating, geometry, phases) in self.MATRIX.items():
            entry = tools._HANDLERS.get(name)
            self.assertIsNotNone(entry, name)
            self.assertEqual((entry.category, entry.mutating, entry.geometry, tuple(entry.phases)), (category, mutating, geometry, phases), name)

    def test_schemas_follow_6_6(self):
        spec_of = {n: tools._HANDLERS[n].parameters for n in KIT_TOOLS}
        self.assertEqual(spec_of["enclosure_spec"]["properties"]["action"]["enum"], ["get", "set", "validate", "defaults"])
        self.assertEqual(spec_of["build_enclosure"]["properties"]["stage"]["enum"], ["next", *STAGES])
        self.assertEqual(spec_of["build_enclosure"]["required"], ["doc_name"])
        self.assertEqual(spec_of["add_cutout"]["required"], ["doc_name", "part", "face", "u", "v", "w", "h"])
        self.assertEqual(spec_of["add_vents"]["required"], ["doc_name", "part", "face", "count", "slot_w", "slot_l", "pitch"])
        self.assertEqual(spec_of["set_parameters"]["properties"]["params"]["items"]["properties"]["source"]["enum"], list(params.SOURCES))
        self.assertEqual(spec_of["set_parameter"]["required"], ["doc_name", "name", "value"])
        for name, schema in spec_of.items():
            self.assertNotIn("oneOf", json.dumps(schema), name)
            self.assertNotIn("anyOf", json.dumps(schema), name)

    def test_script_helpers_kit_and_P_are_registered(self):
        helpers = scripting.registered_helpers()
        self.assertIn("kit", helpers)
        self.assertIn("P", helpers)
        self.assertTrue(callable(helpers["P"]))
        self.assertTrue(hasattr(helpers["kit"], "spec"))
        tools_kit._register_script_helpers()          # idempotent: registering again replaces, never raises
        again = scripting.registered_helpers()
        self.assertTrue(callable(again["P"]) and hasattr(again["kit"], "build"))

    def test_P_is_the_params_of_the_active_document(self):
        doc = App.newDocument("PHelper")
        self.addCleanup(App.closeDocument, doc.Name)
        P = scripting.registered_helpers()["P"]
        table = P(doc)
        table.set("wall_t", 2.4)
        self.assertAlmostEqual(table.get("wall_t"), 2.4)
        self.assertAlmostEqual(P(doc).get("wall_t"), 2.4)        # the same sheet, not a second one
        self.assertEqual([o.Name for o in doc.Objects if o.TypeId == "Spreadsheet::Sheet"], ["Params"])


# --------------------------------------------------------------------------------------------------------------------
# enclosure_spec
# --------------------------------------------------------------------------------------------------------------------

class EnclosureSpecTest(Base):
    def test_get_without_a_spec_is_a_report_not_an_error(self):
        text = self.ok("enclosure_spec", action="get")
        self.assertIn("No enclosure spec", text)
        self.assertIn("spec_ok: False", text)

    def test_get_is_the_default_action(self):
        self.assertIn("No enclosure spec", self.ok("enclosure_spec"))

    def test_defaults_store_a_valid_spec_in_the_project(self):
        text = self.ok("enclosure_spec", action="defaults", board="rpi4b")
        self.assertIn("SPEC DEFAULTS APPLIED", text)
        self.assertTrue(self.outcome.extra["spec_ok"])
        stored = self.project.read_spec()
        self.assertEqual(stored["board"], "rpi4b")
        self.assertTrue(kspec.validate(stored)[0])
        self.assertTrue(any("defaults" in n["text"] for n in self.project.read_notes() if n["kind"] == "assumption"))

    def test_defaults_work_for_every_catalogue_board(self):
        for board in kspec._known_board_ids():
            tools_kit_base.forget_specs()
            text, err = self.call("enclosure_spec", action="defaults", board=board, ctx=self.ctx(turn_id="T" + board))
            self.assertFalse(err, f"{board}: {text}")
            self.assertEqual(self.outcome.extra["spec_ok"], True, board)
            self.assertTrue(kspec.validate(self.project.read_spec())[0], board)
            self.assertEqual(self.project.read_spec()["board"], board)

    def test_unknown_board_lists_the_known_ones(self):
        text = self.fails("enclosure_spec", "rpi4b", action="defaults", board="rpi44")
        self.assertIn("enclosure_spec", text)

    def test_a_board_name_or_alias_resolves_to_the_catalogue_id(self):
        """A weak model writes 'Raspberry Pi 4 Model B', not 'rpi4b': the exact name / alias is accepted wherever a board is named."""
        self.ok("enclosure_spec", action="defaults", board="Raspberry Pi Pico")
        self.assertEqual(self.project.read_spec()["board"], "rpi_pico")
        self.ok("enclosure_spec", action="defaults", board="Raspberry Pi 4 Model B")
        self.assertEqual(self.project.read_spec()["board"], "rpi4b")
        self.ok("enclosure_spec", action="set", spec={"board": "RPi 4", "wall_t": 2.4})
        self.assertEqual((self.project.read_spec()["board"], self.project.read_spec()["wall_t"]), ("rpi4b", 2.4))
        text = self.ok("enclosure_spec", action="validate", board="Raspberry Pi 4 Model B")
        self.assertIn("board 'rpi4b'", text)
        self.fails("enclosure_spec", "not in the board library", action="set", spec={"board": "Raspberry Pi"})   # a partial name is never guessed

    def test_build_enclosure_resolves_a_board_name_in_its_spec_patch(self):
        self.ok("enclosure_spec", action="set", spec={"name": "PiCase"})
        text = self.ok("build_enclosure", stage="params", spec={"board": "Raspberry Pi 4 Model B", "wall_t": 2.4})
        self.assertIn("Spec updated", text)
        self.assertEqual((self.project.read_spec()["board"], self.project.read_spec()["wall_t"]), ("rpi4b", 2.4))

    def test_set_merges_and_validates_and_saves(self):
        self.ok("enclosure_spec", action="defaults")
        text = self.ok("enclosure_spec", action="set", spec={"wall_t": 2.4, "hooks": {"count": 2}})
        self.assertIn("Spec saved (2 changes", text)
        self.assertIn("wall_t 2 -> 2.4", text)
        stored = self.project.read_spec()
        self.assertEqual((stored["wall_t"], stored["hooks"]["count"], stored["hooks"]["w"]), (2.4, 2, 6.0))   # merged, not replaced
        self.assertTrue(self.outcome.extra["spec_ok"])

    def test_set_without_an_earlier_spec_merges_onto_the_board_defaults(self):
        text = self.ok("enclosure_spec", action="set", spec={"material": "pla"})
        self.assertIn("no earlier spec", text)
        self.assertEqual(self.project.read_spec()["material"], "PLA")

    def test_set_accepts_strings_with_units(self):
        self.ok("enclosure_spec", action="set", spec={"wall_t": "2.6 mm"})
        self.assertEqual(self.project.read_spec()["wall_t"], 2.6)

    def test_invalid_set_is_an_error_that_changes_nothing(self):
        self.ok("enclosure_spec", action="defaults")
        before = self.project.read_spec()
        text = self.fails("enclosure_spec", "not valid", "wall_t", action="set", spec={"wall_t": 0.5})
        self.assertIn("1.2", text)
        self.assertEqual(self.project.read_spec(), before)
        self.assertIn("more failing call", text)

    def test_unknown_key_is_reported_with_a_suggestion(self):
        self.ok("enclosure_spec", action="defaults")
        text = self.fails("enclosure_spec", "wal_t", action="set", spec={"wal_t": 2.4})
        self.assertIn("wall_t", text)
        self.assertEqual(self.project.read_spec()["wall_t"], 2.0)

    def test_bad_argument_shapes_are_listed_in_one_error(self):
        text = self.fails("enclosure_spec", "'action'", "'spec'", action="explode", spec="wall_t=2")
        self.assertIn("enclosure_spec:", text)

    def test_validate_checks_without_storing(self):
        self.ok("enclosure_spec", action="defaults")
        before = self.project.read_spec()
        text = self.ok("enclosure_spec", action="validate", spec={"wall_t": 3.0})
        self.assertIn("Spec valid", text)
        self.assertIn("nothing was stored", text)
        self.assertEqual(self.project.read_spec(), before)
        self.fails("enclosure_spec", "wall_t", action="validate", spec={"wall_t": 0.2})

    def test_third_failing_call_applies_the_defaults_and_keeps_the_valid_part(self):
        self.ok("enclosure_spec", action="defaults")
        bad = {"wall_t": 0.2, "clearance": 0.4}
        self.fails("enclosure_spec", action="set", spec=bad)
        self.fails("enclosure_spec", action="validate", spec=bad)
        text = self.ok("enclosure_spec", action="set", spec=bad)
        self.assertIn("SPEC DEFAULTS APPLIED", text)
        self.assertIn("clearance", text)             # the valid leaf survived
        self.assertIn("wall_t", text)                # the invalid one is named as dropped
        stored = self.project.read_spec()
        self.assertEqual((stored["wall_t"], stored["clearance"]), (2.0, 0.4))
        self.assertTrue(self.outcome.extra["spec_ok"])
        self.assertTrue(any("defaults of board" in n["text"] for n in self.project.read_notes() if n["kind"] == "assumption"))

    def test_calls_outside_a_turn_never_add_up_to_the_rescue(self):
        """Replay and the bridge call tools without a turn id: their failures must not be pooled into one silent defaults rescue."""
        for _ in range(5):
            text, is_error = self.call("enclosure_spec", ctx=self.ctx(turn_id=""), action="set", spec={"wall_t": -5})
            self.assertTrue(is_error, text)
            self.assertNotIn("SPEC DEFAULTS APPLIED", text)

    def test_the_failure_count_is_per_turn_and_resets_on_success(self):
        bad = {"wall_t": 0.2}
        self.fails("enclosure_spec", action="set", spec=bad, ctx=self.ctx(turn_id="A"))
        self.fails("enclosure_spec", action="set", spec=bad, ctx=self.ctx(turn_id="A"))
        self.fails("enclosure_spec", action="set", spec=bad, ctx=self.ctx(turn_id="B"))      # another turn: still the first failure there
        self.ok("enclosure_spec", action="set", spec={"wall_t": 2.2}, ctx=self.ctx(turn_id="A"))   # success clears A
        self.fails("enclosure_spec", action="set", spec=bad, ctx=self.ctx(turn_id="A"))
        self.fails("enclosure_spec", action="set", spec=bad, ctx=self.ctx(turn_id="A"))
        text = self.ok("enclosure_spec", action="set", spec=bad, ctx=self.ctx(turn_id="A"))
        self.assertIn("SPEC DEFAULTS APPLIED", text)

    def test_get_returns_the_stored_json_and_status(self):
        self.ok("enclosure_spec", action="set", spec={"wall_t": 2.4})
        text = self.ok("enclosure_spec", action="get")
        self.assertIn("valid.", text)
        self.assertIn('"wall_t":2.4', text)
        self.assertTrue(self.outcome.extra["spec_ok"])

    def test_param_names_are_reported_when_the_document_has_a_sheet(self):
        self.ok("enclosure_spec", action="defaults")
        self.ok("build_enclosure", stage="params")
        self.ok("enclosure_spec", action="set", spec={"wall_t": 2.4}, doc_name=self.doc.Name)
        self.assertIn("wall_t", self.outcome.extra["param_names"])

    def test_without_a_project_the_spec_lives_in_the_process_cache(self):
        ctx = ToolContext(turn_id="T1", state={})
        self.ok("enclosure_spec", action="set", spec={"wall_t": 2.4}, ctx=ctx)
        text = self.ok("enclosure_spec", action="get", ctx=ToolContext(turn_id="T2", state={}))
        self.assertIn('"wall_t":2.4', text)


# --------------------------------------------------------------------------------------------------------------------
# build_enclosure
# --------------------------------------------------------------------------------------------------------------------

class BuildEnclosureTest(Base):
    def test_first_call_applies_the_defaults_and_builds_params(self):
        text = self.ok("build_enclosure", stage="next")
        self.assertIn("No enclosure spec was set", text)
        self.assertIn("Stage 'params' built (1 of 9)", text)
        self.assertIn("Next stage: reference.", text)
        self.assertEqual(self.outcome.args["stage"], "params")                    # 'next' resolved for the ledger
        self.assertTrue(self.outcome.extra["spec_ok"])
        self.assertIn("wall_t", self.outcome.extra["param_names"])
        self.assertTrue(any("defaults for board" in n["text"] for n in self.project.read_notes() if n["kind"] == "assumption"))
        self.assertTrue(kspec.validate(self.project.read_spec())[0])
        self.assertEqual(self.outcome.title, "Build enclosure: params")

    def test_one_stage_per_call_in_the_documented_order(self):
        seen = []
        for _ in range(9):
            text = self.ok("build_enclosure")
            seen.append(self.outcome.args["stage"])
            if len(seen) < 9:
                self.assertIn("Next stage:", text)
        self.assertEqual(seen, list(STAGES))
        self.assertIn("All 9 stages are built. Next: validate_model, then check_fit.", text)
        self.assertEqual(build.next_stage(self.doc, kspec.default_spec()), None)

    def test_a_tenth_next_call_has_nothing_to_do(self):
        self.build_all()
        text = self.ok("build_enclosure")
        self.assertIn("All 9 stages are built and up to date", text)

    def test_stage_result_line_has_objects_solids_and_volume(self):
        self.build_to("reference")
        text = self.ok("build_enclosure", stage="shell")
        self.assertIn("Stage 'shell' built (3 of 9). Objects: ", text)
        self.assertIn("Shell", text)
        self.assertIn("(1 solid,", text)
        self.assertIn("mm3)", text)
        self.assertIn("Size: Shell 94.1 x 61.94 x 24 mm", text)
        self.assertIn("Next stage: standoffs.", text)

    def test_explicit_stage_out_of_order_is_a_clear_error_and_changes_nothing(self):
        self.ok("build_enclosure", stage="params")
        before = sorted(o.Name for o in self.doc.Objects)
        text = self.fails("build_enclosure", "needs stage", stage="lid")
        self.assertIn("shell", text)
        self.assertEqual(sorted(o.Name for o in self.doc.Objects), before)

    def test_stage_before_params_names_the_missing_sheet(self):
        self.fails("build_enclosure", "Params", stage="shell")

    def test_unknown_stage_and_bad_patch_are_listed_together(self):
        text = self.fails("build_enclosure", "'stage'", "'spec'", stage="roof", spec=5)
        self.assertIn("params", text)

    def test_unknown_document_lists_the_open_ones(self):
        text = self.fails("build_enclosure", "not open", doc_name="Nope")
        self.assertIn(self.doc.Name, text)

    def test_spec_patch_is_merged_validated_and_stored(self):
        text = self.ok("build_enclosure", stage="params", spec={"wall_t": 2.4})
        self.assertIn("Spec updated: wall_t -> 2.4", text)
        self.assertEqual(self.project.read_spec()["wall_t"], 2.4)
        self.assertAlmostEqual(params.Params(self.doc).get("wall_t"), 2.4)
        text = self.fails("build_enclosure", "makes the spec invalid", stage="next", spec={"wall_t": 0.1})
        self.assertEqual(self.project.read_spec()["wall_t"], 2.4)

    def test_an_invalid_stored_spec_blocks_the_build_with_the_fix(self):
        self.ok("enclosure_spec", action="defaults")
        data = self.project.read_spec()
        data["wall_t"] = 0.4
        self.project.save_spec(data)
        tools_kit_base.forget_specs()
        self.fails("build_enclosure", "stored spec is not valid", "enclosure_spec")

    def test_an_explicit_stage_after_a_spec_edit_pushes_the_spec_into_the_sheet_first(self):
        self.build_to("shell")
        self.ok("enclosure_spec", action="set", spec={"wall_t": 2.4})
        self.assertAlmostEqual(params.Params(self.doc).get("wall_t"), 2.0)       # not pushed yet
        text = self.ok("build_enclosure", stage="shell")
        self.assertIn("The Params rows followed the spec first: wall_t", text)
        self.assertAlmostEqual(params.Params(self.doc).get("wall_t"), 2.4)
        self.assertEqual(tip_size(self.doc, "shell")[0], 94.9)

    def test_an_up_to_date_stage_is_skipped_unless_rebuild(self):
        self.ok("build_enclosure", stage="params")
        self.assertIn("already built and up to date", self.ok("build_enclosure", stage="params"))
        text = self.ok("build_enclosure", stage="params", rebuild=True)
        self.assertIn("Stage 'params' built", text)

    def test_rebuild_of_a_middle_stage_replaces_it_and_the_later_ones(self):
        self.build_to("lid")
        text = self.ok("build_enclosure", stage="shell", rebuild=True)
        self.assertIn("Replaced", text)
        self.assertEqual(self.outcome.args["stage"], "shell")
        self.assertEqual(list(build.built_stages(self.doc, "enc1")), ["params", "reference", "shell"])

    def test_next_with_rebuild_and_nothing_left_is_an_error(self):
        self.build_all()
        self.fails("build_enclosure", "nothing left to build", stage="next", rebuild=True)

    def test_the_ledger_counts_stages_as_they_are_built(self):
        self.build_to("shell")
        steps = {s.id: s for s in self.ledger.steps}
        shell_step = next(s for s in self.ledger.steps if "shell" in getattr(s, "stages", ()))
        self.assertIn("shell", shell_step.stages_done)
        self.assertIsNotNone(steps)

    def test_failed_stage_is_an_error_and_the_model_is_unchanged(self):
        self.build_to("reference")
        self.project.save_spec(self.project.read_spec())
        original = G.stage_objects
        before = sorted(o.Name for o in self.doc.Objects)
        try:
            from oscar.kit import _stages

            real = _stages.STAGE_FUNCS["shell"]

            def boom(ctx):
                real(ctx)
                raise RuntimeError("forced failure")

            _stages.STAGE_FUNCS["shell"] = boom
            text = self.fails("build_enclosure", "forced failure", stage="shell")
        finally:
            _stages.STAGE_FUNCS["shell"] = real
        self.assertIn("rolled back", text)
        self.assertEqual(sorted(o.Name for o in self.doc.Objects), before)
        self.assertIs(G.stage_objects, original)

    def test_screenshot_flag_is_accepted_headless(self):
        text = self.ok("build_enclosure", stage="params", include_screenshot=True, view_name="Front")
        self.assertIn("Stage 'params' built", text)


# --------------------------------------------------------------------------------------------------------------------
# the whole Pi 4 enclosure
# --------------------------------------------------------------------------------------------------------------------

class FullPi4RunTest(Base):
    @classmethod
    def setUpClass(cls):
        cls._shared = None

    def setUp(self):
        super().setUp()
        self.build_all()

    def test_shell_and_lid_are_separate_closed_valid_single_solids(self):
        shell, lid = G.tip(self.doc, "shell"), G.tip(self.doc, "lid")
        self.assertIsNot(shell, lid)
        for obj in (shell, lid):
            self.assertEqual(G.single_solid(obj), (True, ""), obj.Name)
        self.assertEqual(tip_size(self.doc, "shell"), [94.1, 61.94, 24.0])
        self.assertLessEqual(G.common_volume(shell, lid), 0.05)
        self.assertEqual([o.Name for o in self.doc.Objects if "Invalid" in o.State], [])

    def test_ports_standoffs_vents_hooks_and_assembly_exist(self):
        names = {o.Name for o in self.doc.Objects}
        self.assertTrue({"Part_Shell", "Part_Lid", "Assembly", "Params"} <= names)
        self.assertTrue(any(n.startswith("Shell_StandoffPost") for n in names))
        self.assertTrue(any(n.startswith("Shell_Port_") for n in names))
        self.assertTrue(any(n.startswith("Lid_Hook") and n.endswith("_Beam") for n in names))
        self.assertTrue(any(n.startswith("Shell_Pocket") for n in names))
        self.assertIn("Lid_Vents", names)

    def test_every_port_window_contains_its_connector(self):
        from oscar.kit import checks

        rows = {r["id"]: r for r in checks.fit_rows(self.doc, kspec.default_spec(), "catalog-nominal")}
        self.assertEqual(rows["containment.ports"]["status"], "PASS", rows["containment.ports"])
        self.assertEqual(rows["alignment.standoffs"]["status"], "PASS", rows["alignment.standoffs"])
        self.assertEqual(rows["hooks.engagement"]["status"], "PASS", rows["hooks.engagement"])
        self.assertEqual(rows["hooks.clear_of_cutouts"]["status"], "PASS", rows["hooks.clear_of_cutouts"])

    def test_parameters_are_bound_to_the_geometry(self):
        rows = {r["name"]: r for r in params.Params(self.doc).table()}
        for name in ("wall_t", "floor_t", "standoff_h", "lid_t", "lid_fit", "lip_t", "corner_r"):
            self.assertGreater(rows[name]["bound"], 0, name)

    def test_set_parameter_edit_changes_the_geometry(self):
        text = self.ok("set_parameter", name="wall_t", value="2.4")
        self.assertIn("wall_t: 2 mm -> 2.4 mm (source: user_requirement)", text)
        self.assertRegex(text, r"changed: Shell, Lid")
        self.assertEqual(tip_size(self.doc, "shell")[0], 94.9)
        self.assertEqual(tip_size(self.doc, "lid")[0], 97.8)

    def test_set_parameter_wall_and_clearance_gives_the_editability_bbox(self):
        self.ok("set_parameter", name="wall_t", value=2.4)
        self.ok("set_parameter", name="clearance", value="0.35 mm")
        got = tip_size(self.doc, "shell")
        for g, want in zip(got, (94.9, 62.79, 24.0)):
            self.assertAlmostEqual(g, want, delta=0.05)
        for obj in (G.tip(self.doc, "shell"), G.tip(self.doc, "lid")):
            self.assertEqual(G.single_solid(obj), (True, ""))
        self.assertEqual(self.ok("get_parameters").count("wall_t"), 1)

    def test_a_setexpression_bound_parameter_edit_moves_a_custom_object(self):
        box = self.doc.addObject("Part::Box", "CustomBox")
        self.doc.recompute()
        self.ok("bind_expression", object="CustomBox", property="Length", expression="wall_t*10")
        self.assertAlmostEqual(box.Length.Value, 20.0)
        self.ok("set_parameter", name="wall_t", value="2.4")
        self.assertAlmostEqual(box.Length.Value, 24.0)
        self.assertEqual(self.outcome.name, "set_parameter")

    def test_set_parameter_syncs_the_spec_and_keeps_the_stages_current(self):
        self.ok("set_parameter", name="wall_t", value="2.4")
        self.assertEqual(self.project.read_spec()["wall_t"], 2.4)
        sp = kspec.from_dict(self.project.read_spec())
        self.assertIsNone(build.next_stage(self.doc, sp))                     # nothing looks stale
        self.assertIn("All 9 stages are built and up to date", self.ok("build_enclosure"))
        self.ok("add_vents", part="lid", face="top", count=5, slot_w=2, slot_l=20, pitch=5)
        self.assertAlmostEqual(params.Params(self.doc).get("wall_t"), 2.4)     # a later spec push does not undo the edit

    def test_check_fit_through_the_tool_has_no_failures_on_the_default_model(self):
        text, err = self.call("check_fit")
        self.assertFalse(err, text)
        for row_id in ("interference.shell_lid", "clearance.pcb_to_wall_min", "clearance.lid_to_tallest", "retention.lid_lift",
                       "containment.ports", "alignment.standoffs", "hooks.engagement", "hooks.clear_of_cutouts"):
            self.assertRegex(text, rf"PASS {row_id.replace('.', r'[.]')}", row_id)
        self.assertIn(" 0 FAIL", text)


# --------------------------------------------------------------------------------------------------------------------
# add_cutout / add_vents
# --------------------------------------------------------------------------------------------------------------------

class CutoutTest(Base):
    def test_cutout_before_the_shell_exists_is_stored_and_built_later(self):
        text = self.ok("add_cutout", part="shell", face="+y", u=40, v=6, w=10, h=5, id="fan")
        self.assertIn("Cut opening fan in Shell (+y wall)", text)
        self.assertIn("not built yet", text)
        self.assertEqual(self.project.read_spec()["cutouts"][0]["id"], "fan")
        self.build_all()
        shell = G.tip(self.doc, "shell")
        wall_y = shell.Shape.BoundBox.YMax - 1.0
        self.assertFalse(inside(shell, 40.0, wall_y, 6.0))                   # open
        self.assertTrue(inside(shell, 20.0, wall_y, 6.0))                    # the rest of the wall is there

    def test_cutout_on_a_built_model_rebuilds_ports_and_stays_valid(self):
        self.build_all()
        text = self.ok("add_cutout", part="shell", face="+y", u=40, v=6, w=10, h=5, r=1, margin=0.2, id="fan")
        self.assertIn("Cut opening fan in Shell (+y wall): u=40 v=6 w=10 h=5 mm", text)
        self.assertIn("Rebuilt stages ports", text)
        shell = G.tip(self.doc, "shell")
        self.assertEqual(G.single_solid(shell), (True, ""))
        self.assertFalse(inside(shell, 40.0, shell.Shape.BoundBox.YMax - 1.0, 6.0))
        self.assertEqual(build.next_stage(self.doc, kspec.from_dict(self.project.read_spec())), None)
        self.assertEqual(self.outcome.title, tools.derive_title("add_cutout", self.outcome.args))

    def test_same_id_replaces_the_cutout(self):
        self.ok("add_cutout", part="shell", face="+y", u=40, v=6, w=10, h=5, id="fan")
        text = self.ok("add_cutout", part="shell", face="+y", u=20, v=6, w=6, h=5, id="fan")
        self.assertIn("Replaced opening fan", text)
        cuts = self.project.read_spec()["cutouts"]
        self.assertEqual(len(cuts), 1)
        self.assertEqual(cuts[0]["u"], 20.0)

    def test_generated_ids_do_not_collide(self):
        self.ok("add_cutout", part="shell", face="+y", u=40, v=6, w=10, h=5)
        text = self.ok("add_cutout", part="shell", face="+y", u=20, v=6, w=6, h=5)
        self.assertIn("cutout2", text)
        self.assertEqual([c["id"] for c in self.project.read_spec()["cutouts"]], ["cutout1", "cutout2"])

    def test_lid_cutout_rebuilds_the_vents_stage(self):
        self.build_all()
        text = self.ok("add_cutout", part="lid", face="top", u=10, v=45, w=8, h=8, id="led")
        self.assertIn("Cut opening led in Lid (top face)", text)
        self.assertIn("vents", text)
        lid = G.tip(self.doc, "lid")
        self.assertEqual(G.single_solid(lid), (True, ""))
        self.assertFalse(inside(lid, 10.0, 45.0, lid.Shape.BoundBox.ZMax - 0.5))

    def test_argument_problems_are_listed_in_one_error(self):
        text = self.fails("add_cutout", "no top face", "'id'", part="shell", face="top", u=1, v=1, w=3, h=5, id="1bad")
        self.assertIn("add_cutout:", text)
        self.fails("add_cutout", "'w'", part="shell", face="+y", u=1, v=1, w=0.01, h=5)         # the schema minimum, reported by the registry
        self.fails("add_cutout", "no bottom face", part="lid", face="bottom", u=1, v=1, w=3, h=3)
        self.fails("add_cutout", "'part'", part="roof", face="+y", u=1, v=1, w=3, h=3)
        self.fails("add_cutout", "must be a number", part="shell", face="+y", u="x", v=1, w=3, h=3)

    def test_an_opening_that_breaks_a_spec_rule_is_refused_and_nothing_is_stored(self):
        self.build_all()
        before = self.project.read_spec()
        text = self.fails("add_cutout", "does not lie on the +y wall", "spans u", part="shell", face="+y", u=500, v=6, w=10, h=5, id="far")
        self.assertIn("'far'", text)
        self.assertEqual(self.project.read_spec(), before)
        self.fails("add_cutout", "does not lie on the top face of the lid", part="lid", face="top", u=40, v=300, w=10, h=5, id="off")
        self.fails("add_cutout", "does not lie", part="shell", face="-x", u=30, v=40, w=10, h=5, id="high")       # v above the wall
        self.assertEqual(self.project.read_spec(), before)

    def test_a_failed_rebuild_restores_the_previous_spec_and_model(self):
        self.build_all()
        before = self.project.read_spec()
        names = sorted(o.Name for o in self.doc.Objects)
        from oscar.kit import _stages

        real = _stages.STAGE_FUNCS["ports"]
        calls = {"n": 0}

        def flaky(ctx):
            calls["n"] += 1
            if calls["n"] == 1:
                raise RuntimeError("forced ports failure")
            return real(ctx)

        _stages.STAGE_FUNCS["ports"] = flaky
        try:
            self.fails("add_cutout", "forced ports failure", "previous model and spec were kept", part="shell", face="+y", u=40, v=6, w=10, h=5, id="x")
        finally:
            _stages.STAGE_FUNCS["ports"] = real
        self.assertEqual(self.project.read_spec(), before)
        self.assertEqual(sorted(o.Name for o in self.doc.Objects), names)
        self.assertEqual(G.single_solid(G.tip(self.doc, "shell")), (True, ""))


class VentTest(Base):
    def test_vents_are_stored_before_the_lid_exists(self):
        text = self.ok("add_vents", part="lid", face="top", count=5, slot_w=2, slot_l=20, pitch=4)
        self.assertIn("Cut 5 vent slots in Lid (top face): slot 2 x 20 mm, pitch 4 mm", text)
        self.assertIn("not built yet", text)
        self.assertEqual(self.project.read_spec()["vents"]["count"], 5)

    def test_vents_on_a_built_model_rebuild_the_vents_stage(self):
        self.build_all()
        before = G.shape_facts(G.tip(self.doc, "lid"))["volume"]
        text = self.ok("add_vents", part="lid", face="top", count=5, slot_w=2, slot_l=20, pitch=5)
        self.assertIn("Rebuilt stages vents, hooks, assembly", text)
        lid = G.tip(self.doc, "lid")
        self.assertEqual(G.single_solid(lid), (True, ""))
        self.assertGreater(G.shape_facts(lid)["volume"], before + 1.0)       # fewer, shorter slots remove less material
        self.assertEqual(round(params.Params(self.doc).get("vent_count")), 5)  # the sheet followed the spec
        self.assertEqual(self.outcome.title, tools.derive_title("add_vents", self.outcome.args))

    def test_round_holes_and_id_note(self):
        text = self.ok("add_vents", part="lid", face="top", count=6, slot_w=3, slot_l=3, pitch=6, pattern="holes", id="vents2")
        self.assertIn("round holes", text)
        self.assertIn("ONE vent array", text)

    def test_too_tight_a_pitch_is_refused_with_the_rule(self):
        before = None
        self.ok("enclosure_spec", action="defaults")
        before = self.project.read_spec()
        self.fails("add_vents", "does not fit", part="lid", face="top", count=9, slot_w=2, slot_l=24, pitch=2.5)
        self.assertEqual(self.project.read_spec(), before)

    def test_argument_problems(self):
        text = self.fails("add_vents", "'center'", "no top face", part="shell", face="top", count=3, slot_w=2, slot_l=20, pitch=4, center=[1])
        self.assertIn("add_vents:", text)
        self.fails("add_vents", "'count'", part="lid", face="top", count=0, slot_w=2, slot_l=20, pitch=4)       # the schema minimum
        self.fails("add_vents", "no bottom face", part="lid", face="bottom", count=2, slot_w=2, slot_l=20, pitch=4)


# --------------------------------------------------------------------------------------------------------------------
# parameter tools
# --------------------------------------------------------------------------------------------------------------------

class OtherDocumentTest(Base):
    """A Project keeps ONE spec for the chat; another document with its own ``Params`` sheet must not be held to it (review finding:
    a bracket's ``wall_t`` row was refused by the enclosure's minimum wall and, when it passed, written into the enclosure spec)."""

    BRACKET = [{"name": "wall_t", "value": "1.0", "unit": "mm", "source": "user_requirement"},
               {"name": "plate_w", "value": "30", "unit": "mm", "source": "user_requirement"}]

    def check_bracket_is_free(self, bracket):
        before = dict(self.project.read_spec())
        self.ok("set_parameters", doc_name=bracket.Name, params=self.BRACKET)               # 1.0 is below the enclosure minimum of 1.2
        self.ok("set_parameter", doc_name=bracket.Name, name="wall_t", value="3.0")
        self.ok("set_parameter", doc_name=bracket.Name, name="wall_t", value="0.8")
        self.assertEqual(self.project.read_spec(), before, "an edit in another document changed the enclosure spec")
        self.assertAlmostEqual(params.Params(bracket).get("wall_t"), 0.8)

    def test_a_bracket_document_is_not_held_to_a_built_enclosures_rules(self):
        self.build_all()
        self.check_bracket_is_free(self.new_doc("Bracket"))
        self.fails("set_parameter", "minimum 1.2 mm", name="wall_t", value="1.0")           # the enclosure itself still is
        self.assertEqual(tip_size(self.doc, "shell")[0], 94.1)

    def test_the_same_holds_before_the_first_build(self):
        self.ok("enclosure_spec", action="set", spec={"name": "PiCase"}, doc_name=self.doc.Name)
        self.ok("set_parameters", params=[{"name": "wall_t", "value": "2.0", "unit": "mm", "source": "user_requirement"}])
        self.check_bracket_is_free(self.new_doc("Bracket"))
        self.fails("set_parameters", "enclosure spec rules", params=[{"name": "wall_t", "value": "0.5", "source": "user_requirement"}])

    def test_the_owner_is_the_document_the_spec_was_stored_for(self):
        from oscar import tools_kit_base as base

        self.ok("enclosure_spec", action="set", spec={"name": "PiCase"}, doc_name=self.doc.Name)
        self.assertEqual(base.spec_owner(self.ctx()), self.doc.Name)
        self.assertEqual(base.spec_owner(ToolContext()), "")                                  # no project: no owner


class SetParametersTest(Base):
    def test_creates_the_sheet_and_rows_and_returns_the_table(self):
        text = self.ok("set_parameters", params=[
            {"name": "arm_len", "value": "40 mm", "source": "user_requirement", "note": "from the sketch"},
            {"name": "arm_w", "value": 12, "source": "assumption"},
            {"name": "arm_area", "value": "=arm_len*arm_w", "unit": "mm", "source": "derived"}])
        self.assertIn("Set 3 parameters", text)
        self.assertIn("PARAMETERS (sheet 'Params', 3 rows", text)
        table = params.Params(self.doc)
        self.assertAlmostEqual(table.get("arm_area"), 480.0)
        self.assertEqual(table.sources()["arm_len"], "user_requirement")
        self.assertEqual(self.outcome.extra["param_names"], ["arm_len", "arm_w", "arm_area"])
        self.assertTrue(self.outcome.geometry)

    def test_all_problems_are_listed_and_nothing_is_written(self):
        text = self.fails("set_parameters", "params[0]", "'b_c'", "source", params=[
            {"name": "A", "value": "1", "source": "assumption"},
            {"name": "b_c", "value": "", "source": "nonsense"}])
        self.assertIn("set_parameters:", text)
        self.assertIsNone(self.doc.getObject("Params"))

    def test_duplicate_names_and_empty_list_are_errors(self):
        self.fails("set_parameters", "appears twice", params=[{"name": "wall_t", "value": "2", "source": "assumption"},
                                                             {"name": "wall_t", "value": "3", "source": "assumption"}])
        self.fails("set_parameters", "non-empty list", params=[])

    def test_a_cell_that_cannot_be_read_aborts_the_whole_call(self):
        self.ok("set_parameters", params=[{"name": "wall_t", "value": "2", "source": "assumption"}])
        self.fails("set_parameters", "nothing was changed", params=[
            {"name": "good_one", "value": "5", "source": "assumption"},
            {"name": "bad_one", "value": "2,5 mm", "source": "assumption"}])
        self.assertNotIn("good_one", params.Params(self.doc).names())

    def test_a_value_that_breaks_the_spec_rules_is_rolled_back(self):
        self.build_all()
        text = self.fails("set_parameters", "spec rules", "nothing was changed",
                          params=[{"name": "wall_t", "value": "0.5", "source": "user_requirement"}])
        self.assertIn("wall_t", text)
        self.assertAlmostEqual(params.Params(self.doc).get("wall_t"), 2.0)
        self.assertEqual(tip_size(self.doc, "shell")[0], 94.1)

    def test_count_parameters_rebuild_their_stage(self):
        self.build_all()
        text = self.ok("set_parameters", params=[{"name": "vent_count", "value": "5", "source": "user_requirement"}])
        self.assertIn("Rebuilt stages vents", text)
        self.assertEqual(G.single_solid(G.tip(self.doc, "lid")), (True, ""))
        self.assertEqual(round(params.Params(self.doc).get("vent_count")), 5)

    def test_other_sheet_names_work(self):
        self.ok("set_parameters", sheet="Dims", params=[{"name": "arm_len", "value": "40", "source": "assumption"}])
        self.assertIsNotNone(self.doc.getObject("Dims"))
        self.assertIn("sheet 'Dims'", self.ok("get_parameters", sheet="Dims"))


class GetParametersTest(Base):
    def test_missing_sheet_is_a_report_with_the_way_forward(self):
        text = self.ok("get_parameters")
        self.assertIn("no sheet 'Params'", text)
        self.assertEqual(self.outcome.extra["param_names"], [])

    def test_table_shows_values_units_sources_and_bound_counts(self):
        self.build_all()
        text = self.ok("get_parameters")
        self.assertIn("PARAMETERS (sheet 'Params'", text)
        self.assertRegex(text, r"wall_t\s+2\s+mm\s+assumption\s+[1-9]")
        self.assertRegex(text.splitlines()[-1], r"^\+\d+ more$")          # more than 40 rows: the tail line
        self.assertGreater(text.count("\n"), 40)

    def test_get_parameters_is_not_mutating(self):
        self.ok("get_parameters")
        self.assertFalse(self.outcome.mutating)


class BindExpressionTest(Base):
    def setUp(self):
        super().setUp()
        self.ok("set_parameters", params=[{"name": "wall_t", "value": "2", "source": "assumption"}])
        self.box = self.doc.addObject("Part::Box", "Box")
        self.doc.recompute()

    def test_bare_names_get_the_sheet_prefix_and_the_object_follows(self):
        text = self.ok("bind_expression", object="Box", property="Length", expression="wall_t*5")
        self.assertRegex(text, r"Bound Box\.Length = Params\.wall_t ?\* ?5")
        self.assertAlmostEqual(self.box.Length.Value, 10.0)
        self.ok("set_parameter", name="wall_t", value="3")
        self.assertAlmostEqual(self.box.Length.Value, 15.0)
        rows = {r["name"]: r for r in params.Params(self.doc).table()}
        self.assertEqual(rows["wall_t"]["bound"], 1)

    def test_dotted_property_and_label_lookup(self):
        self.box.Label = "Plate"
        self.ok("bind_expression", object="Plate", property="Placement.Base.z", expression="Params.wall_t*2")
        self.assertAlmostEqual(self.box.Placement.Base.z, 4.0)

    def test_unknown_object_and_property_suggest_names(self):
        self.fails("bind_expression", "'Bix' is not an object", "similar: Box", object="Bix", property="Length", expression="wall_t")
        self.fails("bind_expression", "no property 'Lenght'", "similar: Length", object="Box", property="Lenght", expression="wall_t")

    def test_an_expression_that_fails_the_recompute_changes_nothing(self):
        before = self.box.Length.Value
        text = self.fails("bind_expression", "nothing was changed", object="Box", property="Length", expression="wal_t*2")
        self.assertIn("bind_expression:", text)
        self.assertEqual(self.box.ExpressionEngine, [])
        self.assertAlmostEqual(self.box.Length.Value, before)
        self.assertEqual([o.Name for o in self.doc.Objects if "Invalid" in o.State], [])

    def test_empty_arguments_are_listed(self):
        self.fails("bind_expression", "'object'", "'property'", "'expression'", object="", property="", expression="")

    def test_no_sheet_is_a_clear_error(self):
        other = self.new_doc("Plain")
        box = other.addObject("Part::Box", "Box")
        other.recompute()
        self.fails("bind_expression", "no sheet", doc_name=other.Name, object=box.Name, property="Length", expression="wall_t")


class SetParameterTest(Base):
    def test_unknown_name_lists_parameters_and_suggests(self):
        self.build_to("params")
        text = self.fails("set_parameter", "'wal_t' is not a parameter in sheet 'Params'", "similar: wall_t", name="wal_t", value="2.4")
        self.assertIn("Parameters: ", text)

    def test_no_sheet(self):
        self.fails("set_parameter", "no 'Params' sheet", name="wall_t", value="2")

    def test_argument_problems(self):
        self.fails("set_parameter", "'name'", "'value'", name="", value="")

    def test_derived_rows_refuse_a_plain_number(self):
        self.build_to("params")
        self.fails("set_parameter", "derived", "shell_len", name="shell_len", value="100")

    def test_a_formula_may_replace_a_derivation(self):
        self.build_to("params")
        self.ok("set_parameter", name="lid_gap", value="=wall_t*0.5")
        self.assertAlmostEqual(params.Params(self.doc).get("lid_gap"), 1.0)

    def test_value_that_breaks_a_spec_rule_is_refused_and_restored(self):
        self.build_all()
        text = self.fails("set_parameter", "wall_t", "stays 2 mm", name="wall_t", value="0.5")
        self.assertIn("1.2", text)
        self.assertAlmostEqual(params.Params(self.doc).get("wall_t"), 2.0)
        self.assertEqual(tip_size(self.doc, "shell")[0], 94.1)

    def test_garbage_value_is_a_params_error_and_nothing_changes(self):
        self.build_to("params")
        self.fails("set_parameter", "cannot read", "stays 2 mm", name="wall_t", value="two")
        self.assertAlmostEqual(params.Params(self.doc).get("wall_t"), 2.0)

    def test_count_parameter_rebuilds_the_vents_and_reports_it(self):
        self.build_all()
        before = G.shape_facts(G.tip(self.doc, "lid"))["volume"]
        text = self.ok("set_parameter", name="vent_count", value="5")
        self.assertIn("Rebuilt stages vents", text)
        self.assertIn("changed: Lid", text)
        self.assertGreater(G.shape_facts(G.tip(self.doc, "lid"))["volume"], before)
        self.assertEqual(G.single_solid(G.tip(self.doc, "lid")), (True, ""))
        self.assertEqual(self.outcome.extra["param_names"].count("vent_count"), 1)

    def test_hook_count_rebuilds_the_hooks(self):
        self.build_all()
        beams = lambda: sorted(o.Name for o in self.doc.Objects if o.Name.startswith("Lid_Hook") and o.Name.endswith("_Beam"))  # noqa: E731
        self.assertEqual(len(beams()), 4)
        self.ok("set_parameter", name="hook_count", value="2")
        self.assertEqual(len(beams()), 2)
        self.assertEqual(G.single_solid(G.tip(self.doc, "shell")), (True, ""))

    def test_recompute_false_only_edits_the_cell(self):
        self.build_all()
        text = self.ok("set_parameter", name="wall_t", value="2.4", recompute=False)
        self.assertIn("Not recomputed", text)
        self.assertAlmostEqual(params.Params(self.doc).get("wall_t"), 2.4)

    def test_count_parameter_without_a_spec_explains_the_missing_rebuild(self):
        self.build_all()
        tools_kit_base.forget_specs()
        self.project.save_spec({})         # an empty spec dict counts as no spec
        text = self.ok("set_parameter", name="vent_count", value="5", ctx=ToolContext(turn_id="x", state={}))
        self.assertIn("no enclosure spec is stored", text)

    def test_fit_recheck_line_and_flag(self):
        self.build_all()
        text = self.ok("set_parameter", name="wall_t", value="2.2")
        self.assertIn("KIT FIT:", text)
        text = self.ok("set_parameter", name="wall_t", value="2.0", recheck=False)
        self.assertNotIn("KIT FIT:", text)

    def test_health_runs_after_the_edit(self):
        self.build_all()
        self.ok("set_parameter", name="wall_t", value="2.4")
        self.assertIsNotNone(self.outcome.health)
        self.assertTrue(self.outcome.health["valid"])
        self.assertTrue(self.outcome.geometry)
        self.assertGreaterEqual(self.outcome.step_index, 0)

    def test_the_edit_is_recorded_with_a_readable_title(self):
        self.build_to("params")
        self.ok("set_parameter", name="wall_t", value="2.4")
        self.assertEqual(self.outcome.title, tools.derive_title("set_parameter", {"name": "wall_t", "value": "2.4"}))


if __name__ == "__main__":
    unittest.main()
