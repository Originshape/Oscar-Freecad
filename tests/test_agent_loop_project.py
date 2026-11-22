"""oscar.agent against a REAL ``Project`` and REAL documents (the provider is the only fake).

Every test drives ``AgentWorker.run_turn`` with the real ``tools.run_tool`` (so the Project hook, the health hook and
the ledger tick run for real), a scripted provider and a project folder under the test sandbox. They pin what only a
real pipeline can show: the health baseline of the turn, the document the
validator is asked about, ``Project.finalize`` and what the turn leaves on disk, and the ledger that survives into the
next turn.
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

from oscar import agent, params  # noqa: E402
from oscar.cad import health as cad_health  # noqa: E402
from oscar.flow import Ledger  # noqa: E402
from oscar.models import ModelCaps  # noqa: E402
from oscar.project import Project  # noqa: E402
from oscar.providers import Message  # noqa: E402
from support.fakes import fake_validation, temp_project, tiny_png_snapshot  # noqa: E402
from support.scripted_llm import ScriptedProvider, call, calls, say  # noqa: E402
from support.signal_recorder import SignalRecorder  # noqa: E402


def profile():
    return params.Profile(name="t", provider="openai_compat", model="scripted", base_url="http://127.0.0.1:9/v1")


def unique(prefix: str) -> str:
    return f"{prefix}{uuid.uuid4().hex[:6]}"


class RealCase(unittest.TestCase):
    """A worker with the real run_tool, a real Project and documents that are closed again afterwards."""

    def setUp(self):
        self.docs: list[str] = []
        self.addCleanup(self.close_docs)

    def close_docs(self):
        for name in self.docs:
            if name in App.listDocuments():
                App.closeDocument(name)

    def new_doc(self, prefix: str = "AgProj") -> str:
        name = unique(prefix)
        App.newDocument(name)
        self.docs.append(name)
        return name

    def worker(self, prompt, script, *, project=None, history=None, **kw):
        self.history = history if history is not None else [Message(role="user", content=prompt)]
        self.provider = ScriptedProvider(script)
        kw.setdefault("caps", ModelCaps())
        worker = agent.AgentWorker(self.history, provider=self.provider, profile=profile(), project=project, **kw)
        self.rec = SignalRecorder(worker)
        return worker

    def box(self, doc: str, name: str = "Plate"):
        return call("create_object", doc_name=doc, obj_type="Part::Box", obj_name=name, include_screenshot=False)


class WholeTurnTest(RealCase):
    def test_a_real_build_leaves_the_full_dossier_and_the_build_summary(self):
        doc = unique("AgPlate")
        self.docs.append(doc)
        project = temp_project(chat_id="20261002-101010-aaaaaa", title="Plate")
        script = [calls(call("create_document", name=doc), self.box(doc)), say("Modelled the plate.")]
        worker = self.worker("make a plate", script, project=project, validator=lambda d, c: fake_validation(True))
        result = worker.run_turn()
        self.assertEqual((result.ok, result.outcome), (True, "validated"))
        self.assertEqual(result.project_dir, project.root)
        self.assertIn("## Build Summary", result.text)
        self.assertTrue(result.text.startswith("Modelled the plate."))
        steps = project.read_steps()
        self.assertEqual([s.tool for s in steps], ["create_document", "create_object"])
        self.assertTrue(steps[1].geometry and steps[1].ok)
        for name in ("project.json", "steps.jsonl", "ledger.json", "NOTES.md", "rebuild.py", "model.FCStd"):
            self.assertTrue(os.path.isfile(project.path(name)), name)
        ledger = project.read_ledger()
        self.assertEqual(ledger["intent"], "generic_part")
        self.assertEqual(project.manifest["finalized"]["outcome"], "validated")
        self.assertEqual(self.rec.of("turn_outcome")[0][0], "validated")
        self.assertIn("## Build Summary", self.rec.of("turn_outcome")[0][1])

    def test_the_validator_is_asked_about_the_document_the_turn_changed(self):
        quiet = self.new_doc("AgQuiet")                 # the user's other open document must never be judged
        doc = unique("AgChanged")
        self.docs.append(doc)
        asked = []

        def validator(name, ctx):
            asked.append((name, dict(ctx.state.get("touched", {}))))
            return fake_validation(True)

        script = [calls(call("create_document", name=doc), self.box(doc)), say("Done.")]
        result = self.worker("make a plate", script, validator=validator).run_turn()
        self.assertEqual(result.outcome, "validated")
        self.assertEqual(asked[0][0], doc)
        self.assertNotEqual(asked[0][0], quiet)
        self.assertEqual({k: sorted(v) for k, v in asked[0][1].items()}, {doc: ["Plate"]})

    def test_the_open_documents_are_recorded_in_the_turn_start_line(self):
        opened = self.new_doc("AgOpenDoc")
        project = temp_project(chat_id="20261002-101017-bbbbb2")
        self.worker("make a plate", [say("x")], project=project)._begin_turn()
        with open(project.path("steps.jsonl"), encoding="utf-8") as f:
            starts = [json.loads(line) for line in f if '"turn_start"' in line]
        self.assertIn(opened, starts[0]["docs"])

    def test_a_chat_turn_creates_no_project_folder(self):
        project = temp_project(chat_id="20261002-101011-bbbbbb", create=False)
        result = self.worker("what is a fillet?", [say("A rounded edge.")], project=project).run_turn()
        self.assertEqual(result.outcome, "answered")
        self.assertFalse(os.path.isdir(project.root))
        self.assertEqual(self.rec.of("artifacts_changed"), [])

    def test_a_work_turn_materializes_the_project_before_the_first_call(self):
        project = temp_project(chat_id="20261002-101012-cccccc", create=False)
        worker = self.worker("make a plate", [say("x")], project=project)
        worker._begin_turn()
        try:
            self.assertTrue(os.path.isdir(project.root))
        finally:
            worker._teardown("failed", None)


class BaselineIntegrationTest(RealCase):
    """the gate judges what the turn created or changed, not the user's pre-existing broken objects."""

    def broken_doc(self) -> str:
        doc = self.new_doc("AgBroken")
        d = App.getDocument(doc)
        d.addObject("Part::Cut", "UsersBrokenCut")           # an empty Part::Cut: invalid before the turn starts
        d.recompute()
        return doc

    def break_it(self, doc: str, name: str = "MyCut"):
        """A script that leaves an invalid ``Part::Cut`` behind (the call itself succeeds; the health hook flags it)."""
        code = (f"import FreeCAD as App\nd = App.getDocument('{doc}')\nd.addObject('Part::Cut', '{name}')\nd.recompute()\n")
        return call("execute_code", code=code, include_screenshot=False)

    def judge(self, calls_seen):
        """What ``tools_verify.run_validation`` must do with the state the agent hands over, on real health."""

        def validator(name, ctx):
            report = cad_health.health_via_dispatch(name, strict=True)
            baseline = {tuple(k) for k in ctx.state.get("baseline", {}).get(name, [])}
            touched = set(ctx.state.get("touched", {}).get(name, ()))
            gate = [f.to_dict() for f in report.findings if f.severity == "error"
                    and ((f.code, f.object) not in baseline or f.object in touched)]
            calls_seen.append({"doc": name, "baseline": sorted(baseline), "touched": sorted(touched),
                               "errors": [f.object for f in report.findings if f.severity == "error"], "gate": gate})
            result = fake_validation(not gate)
            result.gate_errors = gate
            result.doc = name
            return result

        return validator

    def test_the_baseline_holds_the_preexisting_findings_of_every_open_document(self):
        broken = self.broken_doc()
        seen = []
        script = [calls(self.box(broken, "NewBox")), say("Added a box.")]
        worker = self.worker("make a plate", script, validator=self.judge(seen))
        worker.run_turn()
        baseline = worker._state["baseline"]
        self.assertIn(("BOOLEAN_INPUT_MISSING", "UsersBrokenCut"), baseline[broken])
        self.assertIn(("RECOMPUTE_ERROR", "UsersBrokenCut"), baseline[broken])

    def test_the_users_broken_object_does_not_block_the_turn(self):
        broken = self.broken_doc()
        seen = []
        script = [calls(self.box(broken, "NewBox")), say("Added a box.")]
        result = self.worker("make a plate", script, validator=self.judge(seen)).run_turn()
        self.assertEqual(seen[0]["touched"], ["NewBox"])
        self.assertIn("UsersBrokenCut", seen[0]["errors"])           # still visible in the full report ...
        self.assertEqual(seen[0]["gate"], [])                         # ... but never a gate error
        self.assertEqual((result.ok, result.outcome), (True, "validated"))

    def test_an_error_the_turn_introduces_does_block_it(self):
        doc = self.new_doc("AgNew")
        seen = []
        script = [calls(self.break_it(doc)), say("Done."), say("Done again."), say("Done!")]
        result = self.worker("make a plate", script, validator=self.judge(seen)).run_turn()
        self.assertTrue(seen and seen[0]["gate"], seen)
        self.assertEqual({e["object"] for e in seen[0]["gate"]}, {"MyCut"})
        self.assertNotEqual(result.outcome, "validated")

    def test_a_document_created_during_the_turn_has_no_baseline(self):
        doc = unique("AgFresh")
        self.docs.append(doc)
        seen = []
        script = [calls(call("create_document", name=doc), self.box(doc)), say("Done.")]
        worker = self.worker("make a plate", script, validator=self.judge(seen))
        worker.run_turn()
        self.assertNotIn(doc, worker._state["baseline"])
        self.assertEqual(seen[0]["baseline"], [])

    def test_the_lazy_baseline_of_an_escalated_turn_predates_its_first_geometry_call(self):
        broken = self.broken_doc()
        seen = []
        script = [calls(call("get_objects", doc_name=broken)), calls(self.box(broken, "Late")), say("Added a box.")]
        worker = self.worker("show me three views", script, validator=self.judge(seen))
        worker.run_turn()
        self.assertIn(("BOOLEAN_INPUT_MISSING", "UsersBrokenCut"), worker._state["baseline"][broken])
        self.assertEqual(seen[0]["gate"], [])


class RealValidatorTest(RealCase):
    """The default validator is ``tools_verify.run_validation`` fed with the state the agent builds."""

    def setUp(self):
        super().setUp()
        try:
            from oscar import tools_verify  # noqa: F401
        except ModuleNotFoundError:
            self.skipTest("oscar.tools_verify is not installed yet")

    def test_the_default_validator_validates_a_clean_build(self):
        doc = unique("AgReal")
        self.docs.append(doc)
        project = temp_project(chat_id="20261002-101016-aaaaa1")
        script = [calls(call("create_document", name=doc), self.box(doc)), say("Done.")]
        result = self.worker("make a plate", script, project=project).run_turn()
        self.assertEqual((result.ok, result.outcome), (True, "validated"))
        self.assertEqual(result.validation["doc"], doc)
        self.assertTrue(result.validation["ok"])
        self.assertEqual(project.read_validation()["doc"], doc)

    def test_the_users_broken_object_is_a_warning_for_the_gate_not_an_error(self):
        broken = self.new_doc("AgRealBroken")
        App.getDocument(broken).addObject("Part::Cut", "UsersBrokenCut")
        App.getDocument(broken).recompute()
        script = [calls(self.box(broken, "NewBox")), say("Added a box.")]
        result = self.worker("make a plate", script).run_turn()
        self.assertEqual((result.ok, result.outcome), (True, "validated"))
        self.assertEqual(result.validation["gate_errors"], [])
        self.assertIn("UsersBrokenCut", json.dumps(result.validation["health"]))

    def test_an_error_the_turn_introduces_is_not_validated(self):
        doc = self.new_doc("AgRealNew")
        code = (f"import FreeCAD as App\nd = App.getDocument('{doc}')\nd.addObject('Part::Cut', 'MyCut')\nd.recompute()\n")
        script = [calls(call("execute_code", code=code, include_screenshot=False)),
                  say("Done."), say("Done again."), say("Done!"), say("Done!!")]
        result = self.worker("make a plate", script).run_turn()
        self.assertNotEqual(result.outcome, "validated")
        if result.validation is not None:
            self.assertTrue(result.validation["gate_errors"])


class ElidedScriptTest(RealCase):
    def test_an_old_script_in_the_view_points_at_the_file_the_project_saved(self):
        doc = unique("AgElide")
        self.docs.append(doc)
        project = temp_project(chat_id="20261002-101018-cccccc")
        pad = "# " + "padding to make this script long " * 40 + "\n"

        def code(i):
            return (f"import FreeCAD as App\nimport Part\n{pad}d = App.getDocument('{doc}')\n"
                    f"o = d.addObject('Part::Box', 'ScriptBox{i}')\nd.recompute()\n")

        script = [calls(call("create_document", name=doc))] + [
            calls(call("execute_code", code=code(i), include_screenshot=False)) for i in range(8)] + [say("Done.")]
        worker = self.worker("make eight boxes", script, project=project, validator=False,
                             caps=ModelCaps(context_window=8192, max_output=1024))
        worker.run_turn()
        elided = [c.arguments["code"] for r in self.provider.requests for m in r.messages for c in m.tool_calls
                  if c.name == "execute_code" and c.arguments.get("code", "").startswith("[script elided")]
        self.assertTrue(elided, "no script was elided: the window was not small enough for the test")
        self.assertTrue(all("saved as scripts/" in text for text in elided), elided)
        saved = [s.script for s in project.read_steps() if s.tool == "execute_code"]
        self.assertTrue(all(os.path.isfile(project.path(path)) for path in saved))
        self.assertIn(saved[0].split("/")[-1], elided[0])
        stored = [c.arguments["code"] for m in self.history for c in m.tool_calls if c.name == "execute_code"]
        self.assertTrue(all(code_.startswith("import FreeCAD") for code_ in stored))       # the stored history kept every script


class ContinuationTest(RealCase):
    def test_the_ledger_on_disk_resumes_the_next_turn(self):
        doc = unique("AgResume")
        self.docs.append(doc)
        root = isolation.fresh_dir("proj_")
        project = Project.for_chat("20261002-101013-dddddd", "Resume", projects_root=root, snapshot_fn=tiny_png_snapshot)
        first = self.worker("make a plate", [calls(call("create_document", name=doc), self.box(doc)), say("Made the plate."),
                                             say("Made the plate."), say("Made the plate."), say("Made the plate.")],
                            project=project, validator=lambda d, c: fake_validation(False))
        result = first.run_turn()
        self.assertEqual(result.outcome, "unverified")                         # validation failed: the plan stays open
        reopened = Project.for_chat("20261002-101013-dddddd", "Resume", projects_root=root, snapshot_fn=tiny_png_snapshot)
        self.assertEqual(reopened.root, project.root)                          # one chat, one project folder
        saved = reopened.read_ledger()
        self.assertEqual(saved["request"], "make a plate")
        self.assertTrue(any(s["status"] not in ("done", "skipped", "blocked", "changed") and s["required"] for s in saved["steps"]))
        carried = Ledger.from_dict(saved)
        history = [Message(role="user", content="continue")]
        second = self.worker("continue", [say("Nothing left to do.")] * 5, project=reopened, history=history, ledger=carried)
        second.run_turn()
        self.assertEqual(second.ledger.request, "make a plate")              # resumed, not re-seeded from "continue"
        self.assertEqual(second.ledger.intent, "generic_part")
        self.assertIn("PLAN (intent generic_part", history[0].content)


class ModelFreeTurnTest(RealCase):
    def test_a_plain_question_touches_nothing_in_the_project_or_the_gui(self):
        project = temp_project(chat_id="20261002-101014-eeeeee", create=False)
        from unittest import mock

        with mock.patch.object(cad_health, "snapshot_all_via_dispatch", side_effect=AssertionError("GUI touched")):
            result = self.worker("hello", [say("Hi.")], project=project).run_turn()
        self.assertEqual(result.outcome, "answered")
        self.assertFalse(os.path.isdir(project.root))

    def test_the_project_summary_json_roundtrips(self):
        doc = unique("AgJson")
        self.docs.append(doc)
        project = temp_project(chat_id="20261002-101015-ffffff")
        script = [calls(call("create_document", name=doc), self.box(doc)), say("Done.")]
        result = self.worker("make a plate", script, project=project, validator=lambda d, c: fake_validation(True)).run_turn()
        with open(project.path("ledger.json"), encoding="utf-8") as f:
            ledger = json.load(f)
        self.assertEqual(ledger["intent"], "generic_part")
        self.assertEqual(result.ledger["request"], ledger["request"])


if __name__ == "__main__":
    unittest.main()
