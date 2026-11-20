# SPDX-License-Identifier: Apache-2.0
"""The parametric contract (#24): a request that asks for a parametric, spreadsheet-driven model must END as one.

Deterministic pieces, no LLM involved (that is the point: the execution layer enforces the contract):

* ``flow_intent.parametric_requested`` decides from the typed text (language-tolerant core words);
* ``Ledger.parametric`` carries the flag for the whole turn (derived from the stored request, JSON round-trips);
* ``playbooks_for`` adds the ``structured_part`` playbook for a parametric request on ANY intent that offers the
  plan tool, not only bracket / fixture / generic_part;
* ``run_validation`` enforces the contract on the REAL document when the ledger says parametric: a
  ``Params`` / ``Parameters`` (or plan-stamped) spreadsheet with at least one expression-bound feature, the
  ±10 % editability proof (self-deepened), and no silent no-op boolean / loose fragments.

The reproduced failure is the report's tree: a ``Bracket`` of loose primitives and cutters
(``Solid`` = Outer/Body/FlangeL/FlangeR, ``Tools`` = Inner/Slot/HoleL/HoleR, inputs hidden, no spreadsheet) —
geometrically a sound bracket, reported "validated", never editable.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

import FreeCAD as App  # noqa: E402
import verify_support as S  # noqa: E402

from oscar import prompt, tools_verify as TV  # noqa: E402
from oscar.agent_ctx import playbooks_for  # noqa: E402
from oscar.cad import health, health_eval  # noqa: E402
from oscar.flow import AgentConfig, Ledger, TurnState, finish_gate, parametric_requested  # noqa: E402
from oscar.toolctx import ToolContext  # noqa: E402

V = App.Vector

# The exact wordings of the issue (Test B, Test C) and the Spanish failure case.
FLANGE_PROMPT = ("Create a parametric 80 mm flange, 8 mm thick, with six M6 bolt holes equally spaced on a 60 mm "
                 "bolt circle. Keep all dimensions editable from a parameter spreadsheet.")
FLANGE_REWORD = ("Create an 80 mm diameter circular flange with an 8 mm thickness. Add six equally spaced M6 "
                 "mounting holes on a 60 mm pitch circle. All dimensions must be editable through a spreadsheet.")
SPANISH_PROMPT = "Crear un soporte paramétrico de 8 mm, editable desde una hoja de cálculo."
CUBE_PROMPT = "Make a 30 mm cube"

PLAN_TOOLS = frozenset({"apply_feature_plan", "reapply_plan_feature", "create_object", "execute_code",
                        "validate_model", "set_parameters"})
LITE_NAMES = frozenset({"create_document", "create_object", "edit_object", "execute_code", "set_parameter",
                        "validate_model", "get_objects", "get_view"})


class ParametricRequestedTest(unittest.TestCase):
    """``parametric_requested``: the deterministic text signal behind the whole contract."""

    def test_the_issue_wordings_all_count(self):
        for text in (FLANGE_PROMPT, FLANGE_REWORD, SPANISH_PROMPT,
                     "Model a flanged bushing as a parametric part: put every dimension in a parameter table.",
                     "Keep the dimensions parametric",
                     "Make a mounting plate driven by parameters.",
                     "A parameter-driven housing for the board.",
                     "Criar uma peça paramétrica com uma planilha de parâmetros.",
                     "Создай параметрическую деталь.",
                     "创建一个参数化的法兰，尺寸放在电子表格里。"):
            self.assertTrue(parametric_requested(text), text)

    def test_plain_requests_do_not_count(self):
        for text in ("", "Make a 30 mm cube", "change the width parameter to 40 mm",
                     "Export the model as STEP", "thanks"):
            self.assertFalse(parametric_requested(text), text)


class LedgerFlagTest(unittest.TestCase):
    """``Ledger.parametric``: derived from the stored request, alive across the JSON round trip."""

    def test_the_flag_follows_the_request(self):
        self.assertTrue(Ledger.from_request(FLANGE_PROMPT).parametric)
        self.assertFalse(Ledger.from_request(CUBE_PROMPT).parametric)

    def test_the_flag_survives_the_json_round_trip(self):
        ledger = Ledger.from_request(FLANGE_PROMPT)
        again = Ledger.from_dict(json.loads(ledger.to_json()))
        self.assertTrue(again.parametric)


class PlaybookTest(unittest.TestCase):
    """A parametric request gets the structured_part playbook even when the intent regexes miss it."""

    def test_a_parametric_request_on_any_intent_gets_the_playbook(self):
        self.assertIn("structured_part", playbooks_for("other", SPANISH_PROMPT, PLAN_TOOLS))
        self.assertIn("structured_part", playbooks_for("generic_part", FLANGE_PROMPT, PLAN_TOOLS))

    def test_a_plain_other_request_still_gets_no_playbook(self):
        self.assertEqual(playbooks_for("other", "hello there", PLAN_TOOLS), ())

    def test_without_the_plan_tool_there_is_no_playbook(self):
        # a lite profile is never offered apply_feature_plan: the playbook cannot push what it cannot call
        self.assertEqual(playbooks_for("generic_part", FLANGE_PROMPT, LITE_NAMES), ())


class PromptContractTest(unittest.TestCase):
    """The doctrine: the plan is the way, the scripted fallback must be declared."""

    def test_structured_part_mandates_the_plan_and_declares_the_fallback(self):
        text = prompt.PLAYBOOKS["structured_part"]
        self.assertIn("not a script", text)
        self.assertIn("say in the final report that this piece is not parametric", text)

    def test_the_lite_prompt_states_the_parametric_limitation(self):
        self.assertIn("what is not parametric", prompt.LITE)


class VerifyCase(unittest.TestCase):
    """A fresh FreeCAD document per helper call; everything is closed afterwards."""

    def setUp(self):
        health_eval.clear_cache()
        self._made = []
        self.addCleanup(self._close)

    def _close(self):
        for name in self._made:
            if name in App.listDocuments():
                App.closeDocument(name)
        health_eval.clear_cache()

    def new_doc(self, name="PC"):
        doc = App.newDocument(name)
        self._made.append(doc.Name)
        return doc

    def pctx(self, request=FLANGE_PROMPT, intent="generic_part", **state):
        """A context whose ledger says parametric (the finish-time wiring does exactly this)."""
        return ToolContext(ledger=Ledger.from_request(request, intent=intent), intent=intent, state=dict(state))


class GateRulesTest(VerifyCase):
    """``run_validation`` enforces the parametric contract on the real document."""

    def loose_bracket(self, doc):
        """The report's failure tree: groups Solid/Tools, loose primitives and chained cuts, inputs hidden, no sheet."""
        solid = doc.addObject("App::DocumentObjectGroup", "Solid")
        tools = doc.addObject("App::DocumentObjectGroup", "Tools")
        outer = doc.addObject("Part::Box", "Outer")
        outer.Length, outer.Width, outer.Height = 80, 60, 8
        flange_l = doc.addObject("Part::Box", "FlangeL")
        flange_l.Length, flange_l.Width, flange_l.Height = 12, 60, 20
        flange_l.Placement.Base = V(-4, 0, 0)                  # overlaps the plate: one clean solid
        flange_r = doc.addObject("Part::Box", "FlangeR")
        flange_r.Length, flange_r.Width, flange_r.Height = 12, 60, 20
        flange_r.Placement.Base = V(76, 0, 0)
        body = doc.addObject("Part::MultiFuse", "Body")
        body.Shapes = [outer, flange_l, flange_r]
        for obj in (outer, flange_l, flange_r, body):
            solid.addObject(obj)

        def cutter(kind, name, **kw):
            obj = doc.addObject(kind, name)
            for prop, value in kw.items():
                setattr(obj, prop, value)
            tools.addObject(obj)
            return obj

        bore = cutter("Part::Cylinder", "Inner", Radius=20, Height=12)
        bore.Placement.Base = V(40, 30, -2)
        slot = cutter("Part::Box", "Slot", Length=10, Width=30, Height=12)
        slot.Placement.Base = V(35, -5, -2)                    # an edge slot: the part stays ONE solid
        hole_l = cutter("Part::Cylinder", "HoleL", Radius=3, Height=12)
        hole_l.Placement.Base = V(40, 9, -2)
        hole_r = cutter("Part::Cylinder", "HoleR", Radius=3, Height=12)
        hole_r.Placement.Base = V(40, 51, -2)
        doc.recompute()
        base = body
        for tool_obj in (bore, slot, hole_l, hole_r):
            cut = doc.addObject("Part::Cut", f"Cut_{tool_obj.Name}")
            cut.Base, cut.Tool = base, tool_obj
            base.Visibility = False
            base = cut
        bracket = base
        bracket.Label = "Bracket"
        root = doc.addObject("App::DocumentObjectGroup", "Bracket")
        root.addObject(solid)
        root.addObject(tools)
        root.addObject(bracket)
        doc.recompute()
        return bracket

    def test_the_reported_solid_cutter_tree_is_not_validated(self):
        doc = self.new_doc("EJ633")
        bracket = self.loose_bracket(doc)
        self.assertEqual(len(bracket.Shape.Solids), 1)          # the trap: it IS a sound single solid
        result = TV.run_validation(doc.Name, ctx=self.pctx(), fit=False)
        self.assertFalse(result.ok, result.text)
        codes = [e.get("code") for e in result.gate_errors]
        self.assertIn("PARAMETRIC_UNMET", codes)
        message = next(e for e in result.gate_errors if e.get("code") == "PARAMETRIC_UNMET")["message"]
        self.assertIn("spreadsheet", message)

    def test_a_sheet_without_bindings_is_still_not_parametric(self):
        doc = self.new_doc("SheetOnly")
        S.params(doc, [("diameter", "=80 mm", "mm", "user_requirement")])
        box = doc.addObject("Part::Box", "Plate")
        box.Length, box.Width, box.Height = 80, 80, 8
        doc.recompute()
        result = TV.run_validation(doc.Name, ctx=self.pctx(), fit=False)
        self.assertFalse(result.ok, result.text)
        message = next(e["message"] for e in result.gate_errors if e["code"] == "PARAMETRIC_UNMET")
        self.assertIn("bound", message)

    def test_a_plan_stamped_sheet_counts_even_under_another_name(self):
        doc = self.new_doc("Stamped")
        S.build_parametric(doc)
        doc.Params.Label2 = "Oscar plan build"                  # the plan builder's stamp
        doc.Params.Label = "Dimensions"
        doc.recompute()
        result = TV.run_validation(doc.Name, ctx=self.pctx(), fit=False)
        self.assertTrue(result.ok, result.gate_errors)

    def test_a_binding_on_a_user_object_does_not_satisfy_the_gate(self):
        doc = self.new_doc("Ride")
        S.build_parametric(doc)                                 # the user's own bound block, built before the turn
        plate = doc.addObject("Part::Box", "Plate")             # this turn's loose work
        plate.Length, plate.Width, plate.Height = 80, 80, 8
        doc.recompute()
        result = TV.run_validation(doc.Name, ctx=self.pctx(touched={doc.Name: {"Plate"}}), fit=False)
        self.assertFalse(result.ok, result.text)
        self.assertTrue(any(e["code"] == "PARAMETRIC_UNMET" for e in result.gate_errors), result.gate_errors)

    def test_a_binding_on_the_work_of_this_task_satisfies_the_gate(self):
        doc = self.new_doc("Own")
        objs = S.build_parametric(doc)
        result = TV.run_validation(doc.Name, ctx=self.pctx(touched={doc.Name: {objs["base"].Name}}), fit=False)
        self.assertTrue(result.ok, result.gate_errors)

    def test_an_expression_driven_model_passes(self):
        doc = self.new_doc("Bound")
        S.build_parametric(doc)                                 # Params sheet + expression-bound features
        result = TV.run_validation(doc.Name, ctx=self.pctx(), fit=False)
        self.assertTrue(result.ok, result.gate_errors)
        self.assertNotIn("editability", result.not_validated)   # self-deepened: the proof ran

    def test_without_the_parametric_words_the_same_tree_still_validates(self):
        doc = self.new_doc("Plain")
        self.loose_bracket(doc)
        result = TV.run_validation(doc.Name, ctx=self.pctx(request=CUBE_PROMPT), fit=False)
        self.assertTrue(result.ok, result.gate_errors)          # the contract only binds parametric requests

    def test_the_editability_proof_is_owed_by_a_parametric_request(self):
        doc = self.new_doc("Fragile")
        S.build_parametric(doc, slot=9.5)                       # +10% slot cuts the block in two
        result = TV.run_validation(doc.Name, ctx=self.pctx(), fit=False)   # deep defaults to False
        self.assertFalse(result.ok, result.text)
        self.assertIn("editability", result.ran)
        self.assertTrue(any(e["code"] == "EDITABILITY" for e in result.gate_errors))

    def test_a_no_op_boolean_is_counted_on_a_parametric_turn(self):
        doc = self.new_doc("Noop")
        base = doc.addObject("Part::Box", "Base")
        base.Length, base.Width, base.Height = 20, 20, 5
        far = doc.addObject("Part::Cylinder", "FarTool")
        far.Radius, far.Height = 2, 5
        far.Placement.Base = V(60, 0, 0)                        # touches nothing: the cut removes no material
        cut = doc.addObject("Part::Cut", "Hole")
        cut.Base, cut.Tool = base, far
        doc.recompute()
        result = TV.run_validation(doc.Name, ctx=self.pctx(), fit=False)
        self.assertFalse(result.ok, result.text)
        self.assertTrue(any(e["code"] == "NO_GEOMETRIC_CHANGE" for e in result.gate_errors),
                        result.gate_errors)
        plain = TV.run_validation(doc.Name, ctx=self.pctx(request=CUBE_PROMPT), fit=False)
        self.assertTrue(any(f["code"] == "NO_GEOMETRIC_CHANGE" and f["severity"] == "warning"
                            for f in plain.health["findings"]))  # without the words it stays a warning

    def test_loose_fragments_are_counted_on_a_parametric_turn(self):
        import Part

        doc = self.new_doc("Frag")
        S.feature(doc, "Stray", Part.makeCompound([S.box(0, 0, 0, 50, 50, 10), S.box(80, 0, 0, 2, 2, 2)]))
        S.params(doc, [("diameter", "=80 mm", "mm", "user_requirement")])
        doc.recompute()
        result = TV.run_validation(doc.Name, ctx=self.pctx(), fit=False)
        self.assertFalse(result.ok, result.text)
        self.assertTrue(any(e["code"] == "FRAGMENTS" for e in result.gate_errors), result.gate_errors)
        plain = TV.run_validation(doc.Name, ctx=self.pctx(request=CUBE_PROMPT), fit=False)
        self.assertTrue(any(f["code"] == "FRAGMENTS" and f["severity"] == "warning"
                            for f in plain.health["findings"]))  # without the words it stays a warning


class FinishGateTest(unittest.TestCase):
    """A PARAMETRIC_UNMET gate error reaches the model as a HEALTH_FAIL nudge, then an unverified end."""

    def ledger_with_geometry(self):
        ledger = Ledger.from_request(FLANGE_PROMPT)
        ledger.mark_delivered()
        return ledger

    def test_the_gate_nudges_and_names_the_parametric_problem(self):
        ledger = self.ledger_with_geometry()
        state = TurnState(is_work_turn=True, geometry_ok=3,
                          validation={"gate_errors": [{"severity": "error", "code": "PARAMETRIC_UNMET",
                                                        "object": "", "message": "no Params spreadsheet"}]})
        verdict = finish_gate(ledger, state, AgentConfig())
        self.assertFalse(verdict.ok)
        self.assertIn("HEALTH_FAIL", verdict.codes)
        self.assertIn("no Params spreadsheet", verdict.nudge)

    def test_the_ledger_records_the_open_problem(self):
        ledger = self.ledger_with_geometry()
        ledger.apply_validation({"gate_errors": [{"severity": "error", "code": "PARAMETRIC_UNMET", "object": "",
                                                  "message": "no Params spreadsheet"}], "ran": ["health"]})
        self.assertFalse(ledger.health_ok)
        self.assertTrue(any("PARAMETRIC_UNMET" in p or "Params spreadsheet" in p for p in ledger.open_problems),
                        ledger.open_problems)


if __name__ == "__main__":
    unittest.main()
