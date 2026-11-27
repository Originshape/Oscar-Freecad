"""The capability vocabulary the model sees is GENERATED from the op registry
(``schema.op_reference``), so the apply_feature_plan hint and the
structured_part playbook can never drift from the ops the schema accepts; the
failure discipline (cause + alternatives, bounded attempts) and the
``rebuild_later`` repair flag are part of the same contract. Pure Python: no
FreeCAD needed (FreeCADGui is stubbed the way test_tool_registry does)."""

import os
import sys
import types
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar

if "FreeCADGui" not in sys.modules:
    try:
        import FreeCADGui  # noqa: F401
    except Exception:
        sys.modules["FreeCADGui"] = types.ModuleType("FreeCADGui")

from oscar import prompt, tools  # noqa: E402
from oscar.plan import schema  # noqa: E402


class OpReferenceTest(unittest.TestCase):
    """``schema.op_reference``: the one generated capability summary."""

    def test_reference_lists_every_registered_op(self):
        text = schema.op_reference()
        for name in schema.OPS:
            self.assertIn(name, text, f"op {name!r} missing from op_reference()")

    def test_reference_names_each_ops_required_fields(self):
        text = schema.op_reference()
        for name, spec in schema.OPS.items():
            for fld in spec.required:
                self.assertIn(fld, text,
                              f"required field {name}.{fld} missing from op_reference()")

    def test_reference_covers_the_profile_shapes_and_edge_rules(self):
        text = schema.op_reference()
        for shape in schema.SKETCH_SHAPES:
            self.assertIn(shape, text)
        for rule in ("all_outer", "vertical", "top_outer", "bottom_outer",
                     "edges_of_feature"):
            self.assertIn(rule, text)

    def test_reference_states_the_ordering_and_reference_rules(self):
        text = schema.op_reference()
        self.assertIn("sketch", text)
        self.assertIn("earlier", text)          # of / edges_of_feature name an EARLIER feature
        self.assertIn("parameter name", text)   # every dimension is a parameter name

    def test_reference_is_deterministic_and_bounded(self):
        first = schema.op_reference()
        self.assertEqual(first, schema.op_reference())
        self.assertLess(len(first), 2000, "the reference must stay prompt-sized")

    def test_the_profile_shapes_are_generated_not_hardcoded(self):
        # SKETCH_SHAPES is the validator's own tuple: the reference must follow it
        original = schema.SKETCH_SHAPES
        try:
            schema.SKETCH_SHAPES = tuple(original) + ("probe_shape",)
            self.assertIn("probe_shape", schema.op_reference(),
                          "the profile list must come from SKETCH_SHAPES, not a literal")
        finally:
            schema.SKETCH_SHAPES = original
        self.assertNotIn("probe_shape", schema.op_reference())

    def test_a_new_registry_op_cannot_stay_unlisted(self):
        # the drift guard the hint relies on: whatever the registry knows, the
        # reference says (remove the op again after the probe)
        spec = schema.OpSpec("probe_op", "solid", "additive", ("id", "op", "probe_len"),
                             required=("probe_len",), dims={"probe_len": "positive"})
        try:
            schema.register_op(spec)
            self.assertIn("probe_op", schema.op_reference())
        finally:
            del schema.OPS["probe_op"]


class HintSyncTest(unittest.TestCase):
    """The apply_feature_plan hint carries the generated reference: the op
    list the model reads is the one the schema enforces."""

    @classmethod
    def setUpClass(cls):
        cls.hint = tools._PLAN_SCHEMA_HINT
        cls.playbook = prompt.PLAYBOOKS["structured_part"]

    def test_the_hint_lists_every_registered_op(self):
        for name in schema.OPS:
            self.assertIn(name, self.hint,
                          f"op {name!r} missing from the apply_feature_plan hint")

    def test_the_hint_is_the_generated_reference(self):
        self.assertIn(schema.op_reference(), self.hint)

    def test_the_playbook_lists_every_registered_op(self):
        for name in schema.OPS:
            self.assertIn(name, self.playbook,
                          f"op {name!r} missing from the structured_part playbook")


class FailureDisciplineTest(unittest.TestCase):
    """The playbook tells the model how to use a failed build: the named
    field, the cause line, bounded attempts, the repair tools."""

    @classmethod
    def setUpClass(cls):
        cls.text = prompt.PLAYBOOKS["structured_part"]

    def test_the_playbook_bounds_repair_attempts(self):
        self.assertIn("two attempts", self.text,
                      "the playbook must bound repair attempts per cause")

    def test_the_reapply_tool_offers_the_suffix_rebuild(self):
        tool_obj = tools._HANDLERS["reapply_plan_feature"]
        properties = tool_obj.parameters["properties"]
        self.assertIn("rebuild_later", properties)
        self.assertEqual(properties["rebuild_later"].get("type"), "boolean")
        self.assertIn("rebuild_later", tool_obj.description)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
