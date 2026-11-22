"""Connect mode in the panel (oscar.connect_gui): a job from the fake OriginShape workspace runs as a REAL panel turn.

The worker runs on a real QThread with a scripted provider and fake tools, exactly as in test_panel_e2e; the connect
loop runs on its own thread against the fake workspace. Covered: turning the setting on starts the loop; a job opens
a "Workspace: ..." chat seeded with the task's earlier messages, runs on the job's model, and its steps, preview,
exports and reply reach the server; the same task reuses its chat; a model sent from the browser opens as a new
document (and a file that holds no model is reported, leaving none); files attached in the browser are downloaded and
sent with the turn like files attached in FreeCAD (an image as an image, a CAD file by path), and earlier messages'
files are named in the seeded history; a busy panel takes no work; a send the panel blocks (waitlisted) is answered
to the browser. The turn report: a question asked at this computer reaches the running turn as one step that
changes, the feature tree mirrors a real document (every part, the model before a hidden reference, the parameter
sheet), the preview is one GLB node per visible part placed as in FreeCAD, a job runs on the task's model (Origin1
Super too), and a model with several parts is exported per part plus as a whole. A revert from the browser undoes
the task's turns with the panel's own revert, newest first and without asking, and stops at a turn typed in FreeCAD
after them or one from before a restart. Offscreen QApplication, no network beyond 127.0.0.1. Skipped when PySide / FreeCAD are unavailable (plain app
python).
"""

import os
import sys
import threading
import time
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

_IMPORT_ERROR = ""
try:
    from PySide import QtCore
    from oscar import agent, connect, connect_gui, models, panel, params
    from oscar.auth import account, discovery, keystore, session
    from oscar.auth.transport import TokenSet
    from oscar.cad import dispatch
    from oscar.toolctx import ToolOutcome, current_context
    from support.fake_workspace import FakeWorkspace
    from test_panel_flow import FlowTestCase, png
    from test_panel_logic import GOOD_SNAPSHOT
except Exception as exc:   # pragma: no cover - plain interpreter without FreeCAD / Qt
    _IMPORT_ERROR = str(exc)

    class _Unavailable(unittest.TestCase):
        """Base class that skips every test of a module whose Qt / FreeCAD imports failed."""

        @classmethod
        def setUpClass(cls):
            raise unittest.SkipTest(f"Qt / FreeCAD unavailable: {_IMPORT_ERROR}")

    FlowTestCase = _Unavailable

FAST = {"FLUSH_S": 0.02, "KEEPALIVE_S": 0.1, "PREVIEW_INTERVAL_S": 0.0, "HEARTBEAT_S": 0.1,
        "BACKOFF_S": (0.02, 0.02, 0.02), "SIGNED_OUT_RETRY_S": 0.05}


class ConnectPanelTest(FlowTestCase):
    def setUp(self):
        super().setUp()
        for reset in (discovery.clear_cache, session.reset_sessions, keystore.reset_memory, account.clear_model_cache):
            reset()
            self.addCleanup(reset)
        for name, value in FAST.items():
            fast = mock.patch.object(connect, name, value)
            fast.start()
            self.addCleanup(fast.stop)
        env = mock.patch.dict(os.environ, {keystore.ENV_STORE: "memory"})
        env.start()
        self.addCleanup(env.stop)

        self.fake = FakeWorkspace()
        self.addCleanup(self.fake.close)
        self.session = session.get_session(self.fake.url)
        data = self.fake.issue_tokens(device_name="Test Box")
        self.session.complete_sign_in(TokenSet(data["access_token"], data["refresh_token"], data["expires_in"],
                                               data["scope"], received_at=time.monotonic()))

        engine = mock.Mock()
        engine.begin_turn_snapshot.return_value = dict(GOOD_SNAPSHOT)
        engine.revert_turn.return_value = {"success": True, "undone": {"D": 1}, "closed_docs": []}
        self.engine = engine

        steps = iter(range(1, 1000))   # the project's step indexes grow across turns, as in a real chat

        def fake_run_tool(name, args):
            ctx = current_context()
            geometry = name == "create_object"
            shot = png(os.path.join(self.tmp, f"{name}_{int(time.monotonic() * 1000)}.png")) if geometry else ""
            outcome = ToolOutcome(
                name=name, call_id=ctx.call_id, title=args.get("step_title") or f"Step {name}", ok=True,
                handler_ok=True, text=f"{name} done", mutating=True, geometry=geometry,
                step_index=next(steps) if geometry else -1, snapshot_path=shot, args=dict(args),
                delta={"added": ["Box"], "removed": [], "changed": []} if geometry else {})
            ctx.outcome = outcome
            if ctx.ledger is not None:
                ctx.ledger.on_tool_result(outcome)
            return f"{name} done", None, False

        dispatch.init()
        self.addCleanup(dispatch.shutdown)
        self.models: list[str] = []
        self.worker_files: list[list[str]] = []

        def factory(history, **kw):
            from support.fakes import fake_validation
            from support.scripted_llm import ScriptedProvider, call, calls, say

            self.models.append(kw.pop("profile").model)
            self.worker_files.append(list(kw.get("attachments") or []))
            provider = ScriptedProvider([
                calls(call("create_object", doc_name="D", obj_type="Part::Box", step_title="Make the box")),
                say("I built a 10 mm box.")])
            profile = params.Profile(name="t", provider="openai_compat", model="scripted")
            return agent.AgentWorker(history, provider=provider, profile=profile,
                                     validator=lambda doc, ctx: fake_validation(True), **kw)

        for patcher in (
            mock.patch("oscar.cad.get_engine", return_value=engine),
            mock.patch.object(agent._tools, "run_tool", fake_run_tool),
            mock.patch.object(panel, "AgentWorker", factory),
            mock.patch.object(self.p, "_send_blocker", return_value=""),
            mock.patch.object(connect_gui.WorkspaceConnector, "_session", staticmethod(lambda: self.session)),
            mock.patch.object(connect_gui, "_active_doc_name", return_value="D"),
            mock.patch.object(connect_gui, "_export",
                              side_effect=lambda doc, formats, tol, per_part=False: [
                                  (f"D.{f}", f, f"{f} of {doc}".encode()) for f in formats]),
            mock.patch.object(connect_gui, "_fcstd", return_value=("model.FCStd", "fcstd", b"PK")),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self.shutdown)

    def shutdown(self):
        params.set_bool(connect.PREF, False)
        if self.p._workspace is not None:
            self.p._workspace.shutdown()
        worker = self.p.worker
        if worker is not None:
            worker.stop()
            worker.wait(5000)

    def connect_on(self):
        params.set_bool(connect.PREF, True)
        connect_gui.setting_changed()

    def wait_until(self, condition, seconds=30):
        deadline = time.monotonic() + seconds
        while not condition() and time.monotonic() < deadline:
            self.app.processEvents(QtCore.QEventLoop.AllEvents, 50)
            time.sleep(0.01)
        self.app.processEvents()
        return condition()

    def test_the_connector_follows_the_setting(self):
        connector = self.p._workspace
        self.assertIs(connect_gui.current(), connector)
        self.assertEqual(connector.state[0], "off")
        self.connect_on()
        self.assertTrue(self.wait_until(lambda: connector.state[0] == "connected"))
        self.assertEqual(self.fake.hellos[0]["oscar_version"], connect.__version__)
        params.set_bool(connect.PREF, False)
        connect_gui.setting_changed()
        self.assertEqual(connector.state[0], "off")

    def test_a_workspace_job_runs_as_a_panel_turn(self):
        self.fake.add_job("Build a box", task_id="t1", title="Box", model="origin1",
                          history=[("user", "Hello"), ("assistant", "Hi, what shall we build?")])
        self.connect_on()
        self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished), "the job was not finished")

        text, ok = self.fake.finished["t1"]
        self.assertTrue(ok)
        self.assertTrue(text.startswith("I built a 10 mm box."), text[:80])
        self.assertIn("Build Summary", text)   # the app's own summary of the turn goes to the web too
        self.assertEqual(self.models, ["origin1"])
        progress = [e["data"] for e in self.fake.events["t1"] if e["kind"] == "progress"]
        self.assertIn({"title": "Make the box", "ok": True}, progress)
        uploads = [(u["kind"], u["format"]) for u in self.fake.uploads]
        self.assertEqual(uploads[0], ("preview", "stl"))
        self.assertEqual(sorted(uploads[1:]), [("export", "fcstd"), ("export", "step"), ("export", "stl")])

        chat = self.p._current_session()
        self.assertEqual(chat["title"], "Workspace: Box")
        texts = [(m.role, m.content if isinstance(m.content, str) else "") for m in chat["history"]]
        self.assertEqual(texts[:2], [("user", "Hello"), ("assistant", "Hi, what shall we build?")])
        self.assertIn("Build a box", texts[2][1])

        # The next message of the same task continues in the same chat.
        self.fake.add_job("Now make it 20 mm", task_id="t1", title="Box", turn=2)
        self.fake.running.add("t1")
        self.fake.finished.pop("t1")
        self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished))
        self.assertIs(self.p._current_session(), chat)
        self.assertEqual(len([s for s in self.p.sessions if s["title"] == "Workspace: Box"]), 1)

        # Renamed in the workspace: the chat follows the task's title...
        self.fake.add_job("And a lid", task_id="t1", title="Box with lid", turn=3)
        self.fake.running.add("t1")
        self.fake.finished.pop("t1")
        self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished))
        self.assertIs(self.p._current_session(), chat)
        self.assertEqual(chat["title"], "Workspace: Box with lid")
        self.assertEqual(self.p.chat_combo.currentText(), "Workspace: Box with lid")
        # ...unless the user named the chat in FreeCAD
        chat["title"], chat["named"] = "My box", True
        self.fake.add_job("Thicker walls", task_id="t1", title="Box v2", turn=4)
        self.fake.running.add("t1")
        self.fake.finished.pop("t1")
        self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual(chat["title"], "My box")

    def test_each_task_builds_in_its_own_new_document(self):
        import FreeCAD as App

        before = set(App.listDocuments())
        self.addCleanup(lambda: [App.closeDocument(n) for n in set(App.listDocuments()) - before])
        App.newDocument("UserPart")   # the user's own model is open and active
        self.fake.add_job("Build a flange", task_id="t1", title="An 80 mm flange with six bolt holes")
        self.connect_on()
        self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished), "the job was not finished")
        first = connect_gui._load_docs()["t1"]["doc"]
        self.assertEqual(first, "FlangeSixBolt")
        self.assertEqual(App.getDocument(first).Label, "An 80 mm flange with six bolt holes")
        self.assertEqual(App.ActiveDocument.Name, first)
        self.assertEqual(len(App.getDocument("UserPart").Objects), 0)   # never built into

        self.fake.add_job("Build a box", task_id="t2", title="Box")
        self.assertTrue(self.wait_until(lambda: "t2" in self.fake.finished))
        second = connect_gui._load_docs()["t2"]["doc"]
        self.assertNotIn(second, (first, "UserPart"))

        # A later message of the first task goes back to its document, whatever is active meanwhile.
        App.setActiveDocument("UserPart")
        self.fake.add_job("Make it 10 mm thick", task_id="t1", title="An 80 mm flange with six bolt holes", turn=2)
        self.fake.running.add("t1")
        self.fake.finished.pop("t1")
        self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished))
        self.assertEqual(App.ActiveDocument.Name, first)
        self.assertEqual(connect_gui._load_docs()["t1"]["doc"], first)

    def test_a_model_sent_from_the_browser_opens_as_a_new_document(self):
        import FreeCAD as App
        import Part

        before = set(App.listDocuments())
        self.addCleanup(lambda: [App.closeDocument(n) for n in set(App.listDocuments()) - before])
        path = os.path.join(self.tmp, "sent.step")
        Part.makeBox(10, 20, 30).exportStep(path)
        with open(path, "rb") as handle:
            step = handle.read()
        App.newDocument("UserPart")   # the user's own model stays as it is
        good = self.fake.add_model(step, title="An 80 mm flange")
        bad = self.fake.add_model(b"ISO-10303-21;\nnot really a model\n", title="Broken")
        self.connect_on()
        self.assertTrue(self.wait_until(lambda: good in self.fake.model_reports and bad in self.fake.model_reports))
        self.assertIn(connect.OPEN_MODEL, self.fake.hellos[-1]["capabilities"])

        self.assertEqual(self.fake.model_reports[good], {"ok": True, "message": "", "document": "An 80 mm flange"})
        opened = [App.getDocument(name) for name in set(App.listDocuments()) - before - {"UserPart"}]
        self.assertEqual([doc.Label for doc in opened], ["An 80 mm flange"], "the broken file left no document")
        solids = [obj for obj in opened[0].Objects if getattr(getattr(obj, "Shape", None), "Solids", None)]
        self.assertAlmostEqual(sum(obj.Shape.Volume for obj in solids), 6000.0, places=3)
        self.assertEqual(len(App.getDocument("UserPart").Objects), 0)

        report = self.fake.model_reports[bad]
        self.assertFalse(report["ok"])
        self.assertTrue(report["message"].startswith("FreeCAD couldn’t open the model"), report["message"])
        self.assertNotIn(self.tmp, report["message"])   # no local paths leave this computer

    def test_document_names_come_from_the_title(self):
        self.assertEqual(connect_gui._doc_name_for("Raspberry Pi 4 housing with vents"), "RaspberryHousingVents")
        self.assertEqual(connect_gui._doc_name_for("80 x 40 mm"), "Task")
        self.assertEqual(connect_gui._doc_name_for(""), "Task")

    def test_files_attached_in_the_browser_go_with_the_turn(self):
        with open(png(os.path.join(self.tmp, "sketch.png")), "rb") as handle:
            sketch = handle.read()
        step = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;\n"
        self.fake.add_job("", task_id="t1", title="Bracket", attachments=[
            ("sketch.png", "image", sketch), ("bracket.step", "cad", step), ("notes.txt", "text", b"Bore 30 H7")],
            history=[("user", "Here is the old part", ["old.step"]), ("assistant", "Got it.")])
        self.connect_on()
        self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished), "the job was not finished")
        self.assertTrue(self.fake.finished["t1"][1])

        folder = os.path.join(connect_gui.files_dir(), "t1", "m-1")
        expected = [os.path.join(folder, name) for name in ("sketch.png", "bracket.step", "notes.txt")]
        self.assertEqual(self.worker_files, [expected])
        with open(expected[1], "rb") as handle:
            self.assertEqual(handle.read(), step)

        chat = self.p._current_session()
        history = chat["history"]
        self.assertIn("[attached: old.step]", history[0].content)
        sent = history[2]
        self.assertEqual(len(sent.images), 1)              # the sketch reaches the model as an image
        self.assertIn(expected[1], sent.content)           # the STEP file by path, for import_file
        self.assertIn("Bore 30 H7", sent.content)          # the text file inline
        self.assertEqual(self.p.pending, [])               # FreeCAD's own attachment bar is left alone

    def test_a_question_at_this_computer_reaches_the_running_turn(self):
        from oscar import netguard

        connector = self.p._workspace
        turn = connect_gui.GuiTurn(connect.Job.parse({"task_id": "t1", "message": {"id": "m1", "text": "x"}}))
        connector._turn = turn
        info = {"id": 9, "op": "net_permission", "title": "Oscar — network access",
                "text": "Download a file from github.com. It is saved in the project."}
        netguard.on_approval({**info, "state": "waiting"})
        netguard.on_approval({**info, "state": "approved"})
        waiting, approved = (event["data"] for event in turn.events.drain())
        self.assertEqual((waiting["key"], waiting["state"]), ("approval-9", "waiting"))
        self.assertEqual(waiting["detail"], "Network access: Download a file from github.com")
        self.assertEqual((approved["key"], approved["state"], approved["ok"]), ("approval-9", "approved", True))
        turn._on_finished(True, "done")
        netguard.on_approval({**info, "id": 10, "state": "waiting"})   # a finished turn hears nothing more
        self.assertEqual(turn.events.drain(), [])

    def test_a_question_is_offered_to_the_browser_only_with_the_setting_on(self):
        from oscar import netguard

        turn = connect_gui.GuiTurn(connect.Job.parse({"task_id": "t1", "message": {"id": "m1", "text": "x"}}))
        self.p._workspace._turn = turn
        info = {"id": 21, "op": "code_permission", "title": "Oscar — run a script?", "text": "Run a Python script.",
                "detailed": "print('hi')", "choices": [{"key": "run", "label": "Run", "role": "YesRole"},
                                                      {"key": "deny", "label": "Don't run", "role": "RejectRole"}]}
        self.addCleanup(params.set_bool, connect.REMOTE_PREF, False)
        params.set_bool(connect.REMOTE_PREF, False)
        netguard.on_approval({**info, "state": "waiting"})
        (off,) = (event["data"] for event in turn.events.drain())
        self.assertEqual(off["title"], "Waiting for your approval in FreeCAD")
        self.assertNotIn("choices", off)
        self.assertFalse(turn.awaiting_answer())
        with mock.patch.object(netguard, "answer_remote") as answer_remote:
            turn.answer("approval-21", "run")
        answer_remote.assert_not_called()

        params.set_bool(connect.REMOTE_PREF, True)
        netguard.on_approval({**info, "id": 22, "state": "waiting"})
        (on,) = (event["data"] for event in turn.events.drain())
        self.assertEqual((on["title"], on["script"]), ("Waiting for your approval", "print('hi')"))
        self.assertEqual([(c["key"], c["tone"]) for c in on["choices"]], [("run", "approve"), ("deny", "decline")])
        self.assertTrue(turn.awaiting_answer())
        with mock.patch.object(netguard, "answer_remote", return_value=True) as answer_remote:
            turn.answer("approval-21", "run")     # not offered to the browser
            turn.answer("approval-22", "run")
        answer_remote.assert_called_once_with(22, "run")
        netguard.on_approval({**info, "id": 22, "state": "approved", "by": "browser"})
        (done,) = (event["data"] for event in turn.events.drain())
        self.assertEqual((done["title"], done["by"]), ("Approved in the browser", "browser"))
        self.assertFalse(turn.awaiting_answer())

    def test_the_feature_tree_mirrors_the_document(self):
        import FreeCAD as App

        doc = App.newDocument("TreeProbe")
        self.addCleanup(App.closeDocument, doc.Name)
        box = doc.addObject("Part::Box", "Box")
        cyl = doc.addObject("Part::Cylinder", "Hole")
        cyl.Radius, cyl.Height = 2, 20
        cyl.Placement.Base = App.Vector(5, 5, 0)   # through the middle of the box
        cut = doc.addObject("Part::Cut", "Body")
        cut.Base, cut.Tool = box, cyl
        cut.Label = "Drilled block"
        doc.addObject("Spreadsheet::Sheet", "Params")
        doc.recompute()
        tree = connect_gui._tree_of(doc)
        self.assertEqual(tree["document"], "TreeProbe")
        self.assertEqual(tree["roots"], ["Body", "Params"])
        node = {item["name"]: item for item in tree["objects"]}
        self.assertEqual(sorted(node["Body"]["children"]), ["Box", "Hole"])
        self.assertEqual(node["Body"]["label"], "Drilled block")
        self.assertEqual((node["Body"]["type"], node["Body"]["solids"]), ("Part::Cut", 1))
        self.assertAlmostEqual(node["Body"]["volume"], 1000 - 3.14159265 * 4 * 10, places=1)
        self.assertEqual(node["Body"]["bbox"], [10.0, 10.0, 10.0])
        self.assertNotIn("bbox", node["Params"])
        self.assertFalse(tree["truncated"])

    def test_the_tree_lists_every_part_and_carries_the_parameters(self):
        import FreeCAD as App

        doc = App.newDocument("PartsProbe")
        self.addCleanup(App.closeDocument, doc.Name)
        ref = doc.addObject("Part::Box", "Ref_Board")   # the hidden reference comes first in the document
        ref.Visibility = False
        sheet = doc.addObject("Spreadsheet::Sheet", "Params")
        for cell, value in (("A2", "wall_t"), ("B2", "2.4"), ("C2", "mm"), ("D2", "default"), ("E2", "Wall"),
                            ("A3", "lid_t"), ("B3", "0"), ("C3", "mm"), ("D3", "derived")):
            sheet.set(cell, value)
        sheet.setAlias("B2", "wall_t")
        sheet.setAlias("B3", "lid_t")
        sheet.set("B3", "=wall_t + 1")
        part = doc.addObject("App::Part", "Part_Shell")
        outer = doc.addObject("Part::Box", "Shell_Outer")
        cavity = doc.addObject("Part::Box", "Shell_Cavity")
        cavity.Length = cavity.Width = cavity.Height = 6
        shell = doc.addObject("Part::Cut", "Shell_Body")
        shell.Base, shell.Tool, shell.Label = outer, cavity, "Shell"
        part.addObjects([outer, cavity, shell])
        lid = doc.addObject("Part::Box", "Lid")
        doc.recompute()

        tree = connect_gui._tree_of(doc)
        node = {item["name"]: item for item in tree["objects"]}
        self.assertEqual(tree["roots"], ["Params", "Part_Shell", "Lid", "Ref_Board"])   # the model before the reference
        origin = part.Origin.Name
        self.assertEqual(node["Part_Shell"]["children"], [origin, "Shell_Body"])   # as FreeCAD's tree shows it
        self.assertEqual(node["Shell_Body"]["children"], ["Shell_Outer", "Shell_Cavity"])
        self.assertEqual(len(node[origin]["children"]), len(part.Origin.OriginFeatures))
        self.assertEqual(tree["params"]["sheet"], "Params")
        self.assertEqual([(r["name"], r["value"], r["unit"], r["expr"], r["source"], r["note"])
                          for r in tree["params"]["rows"]],
                         [("wall_t", 2.4, "mm", "", "default", "Wall"),
                          ("lid_t", 3.4, "mm", "=wall_t + 1", "derived", "")])

        with mock.patch.object(connect_gui, "TREE_LIMIT", 4):   # a cut tree keeps the model, not the reference
            cut = connect_gui._tree_of(doc)
        self.assertTrue(cut["truncated"])
        self.assertNotIn("Ref_Board", [item["name"] for item in cut["objects"]])

    def test_the_preview_has_one_node_per_visible_part(self):
        import json
        import struct

        import FreeCAD as App

        doc = App.newDocument("PreviewProbe")
        self.addCleanup(App.closeDocument, doc.Name)
        part = doc.addObject("App::Part", "Part_Lid")
        part.Placement.Base = App.Vector(0, 0, 30)   # the lid sits on top: its mesh must be placed there
        lid = doc.addObject("Part::Box", "Lid_Plate")
        lid.Label = "Lid"
        part.addObject(lid)
        box = doc.addObject("Part::Box", "Box")
        hole = doc.addObject("Part::Cylinder", "Hole")
        hole.Radius, hole.Height = 2, 20
        shell = doc.addObject("Part::Cut", "Shell")
        shell.Base, shell.Tool = box, hole
        ref = doc.addObject("Part::Box", "Ref_Board")
        ref.Visibility = False
        doc.recompute()
        box.Visibility = hole.Visibility = False   # as FreeCAD hides a boolean's inputs

        parts = connect_gui._preview_parts(doc)
        self.assertEqual(sorted(p.name for p in parts), ["Lid_Plate", "Shell"])
        by_name = {p.name: p for p in parts}
        self.assertEqual(by_name["Lid_Plate"].label, "Lid")
        zs = by_name["Lid_Plate"].positions[2::3]
        self.assertEqual((min(zs), max(zs)), (30.0, 40.0))

        data = connect.preview_glb(parts, doc.Name)
        length = struct.unpack("<I", data[12:16])[0]
        gltf = json.loads(data[20:20 + length])
        self.assertEqual(sorted(n["name"] for n in gltf["nodes"][1:]), ["Lid_Plate", "Shell"])
        with mock.patch.object(connect_gui, "_glb_preview", return_value=data):
            turn = connect_gui.GuiTurn(connect.Job.parse({"task_id": "t1", "message": {"id": "m1", "text": "x"}}))
            turn._doc, turn._preview_due = doc.Name, 4
            self.assertEqual(turn.take_preview(), (4, data, "glb"))

    def test_a_job_runs_on_the_tasks_model(self):
        from oscar.providers import Message, make_provider

        labels = []
        real_show = connect_gui._show_model
        with mock.patch.object(connect_gui, "_show_model",
                               side_effect=lambda panel, profile: (labels.append(params.model_display_name(profile)),
                                                                   real_show(panel, profile))):
            self.fake.add_job("Build a box", task_id="t1", title="Box", model="origin1-super")
            self.connect_on()
            self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished), "the job was not finished")
        self.assertEqual(self.models, ["origin1-super"])
        # The panel names the task's model while the turn runs, then the active profile's again.
        self.assertEqual(labels[0], "Origin1 Super")
        self.assertEqual(labels[-1], params.model_display_name(params.active_profile()))
        self.assertEqual(self.p.model_label.text(), labels[-1])
        # The request to the gateway names the task's model.
        profile = params.account_profile()
        profile.model = "origin1-super"
        body = make_provider(profile)._body("system", [Message(role="user", content="Build a box")], [])
        self.assertEqual(body["model"], "origin1-super")

    def test_a_model_the_server_added_after_this_oscar_is_used_and_named(self):
        # The browser may pick a model this Oscar was released without: the job runs on it (not on the profile's
        # model), and its name and capabilities come from the server's list, fetched when the job names it.
        env = mock.patch.dict(os.environ, {"OSCAR_API_URL": self.fake.url})   # the server the panel names models by
        env.start()
        self.addCleanup(env.stop)
        with self.fake.lock:
            self.fake.models["data"].append(
                {"id": "extra-model", "display_name": "Extra Model",
                 "context_window": 200000, "max_output_tokens": 32000, "capabilities": {"vision": False}})
        labels = []
        real_show = connect_gui._show_model
        with mock.patch.object(connect_gui, "_show_model",
                               side_effect=lambda panel, profile: (labels.append(params.model_display_name(profile)),
                                                                   real_show(panel, profile))):
            self.fake.add_job("Build a box", task_id="t1", title="Box", model="extra-model")
            self.connect_on()
            self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished), "the job was not finished")
        self.assertEqual(self.models, ["extra-model"])
        self.assertEqual(labels[0], "Extra Model")
        profile = params.account_profile()
        profile.model = "extra-model"
        self.assertFalse(models.resolve_caps(profile).vision)   # no screenshots for a model that reads text only

    def test_a_model_the_list_never_names_is_asked_about_once(self):
        # The server's list is fetched for a job's unknown model at most once per MODELS_REFETCH_S: a model the list
        # never names (or a list that can't be fetched) doesn't hold up every job with a request.
        for task_id in ("t1", "t2"):
            self.fake.add_job("Build a box", task_id=task_id, title="Box", model="origin1-unlisted")
        self.connect_on()
        self.assertTrue(self.wait_until(lambda: {"t1", "t2"} <= set(self.fake.finished)), "the jobs were not finished")
        self.assertEqual(self.models, ["origin1-unlisted", "origin1-unlisted"])   # still the task's model
        self.assertEqual(len(self.fake.calls_to("/v1/models")), 1)

    def test_several_parts_are_exported_one_by_one_and_whole(self):
        def export(doc, formats, tol, per_part=False):
            if per_part:
                return [(f"D-{part}.{f}", f, b"x") for part in ("Shell", "Lid") for f in formats]
            return [(f"D.{f}", f, b"whole") for f in formats]

        turn = connect_gui.GuiTurn(connect.Job.parse({"task_id": "t1", "message": {"id": "m1", "text": "x"}}))
        turn._doc, turn._geometry = "D", True
        with mock.patch.object(connect_gui, "_export", side_effect=export):
            names = [name for name, _fmt, _data in turn.exports()]
        self.assertEqual(names, ["D-Shell.step", "D-Shell.stl", "D-Lid.step", "D-Lid.stl", "D-assembly.stl",
                                 "model.FCStd"])

    def test_a_closed_turn_is_answered_from_the_tasks_chat(self):
        from oscar import chats
        from oscar.providers import Message

        connector = self.p._workspace
        session = chats.new_session(title="Workspace: Bracket")
        session["history"].extend([Message(role="user", content="A bracket"),
                                   Message(role="assistant", content="Built the bracket, 50 x 62 x 50 mm.")])
        chats.save(session)
        connect_gui._save_map({"t1": session["id"], "t2": "no-such-chat"})
        self.assertEqual(connector._recover("t1", "m1"), ("found", True, "Built the bracket, 50 x 62 x 50 mm."))
        self.assertEqual(connector._recover("t2", "m2")[0], "none")
        self.assertEqual(connector._recover("t3", "m3")[0], "none")   # a task this computer never ran
        session["history"].append(Message(role="user", content="Now 5 mm"))
        chats.save(session)
        self.assertEqual(connector._recover("t1", "m1")[0], "none")   # the chat ends with an unanswered message

    def test_a_busy_panel_takes_no_work(self):
        self.fake.add_job("Build a box", task_id="t1")
        self.p.worker = mock.Mock()   # a turn the user started in FreeCAD
        self.connect_on()
        self.assertTrue(self.wait_until(lambda: len(self.fake.hellos) >= 3))
        self.assertEqual(self.fake.next_calls, 0)
        self.p.worker = None
        self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished))

    def test_a_blocked_send_is_answered(self):
        self.p._send_blocker.return_value = "waitlisted"
        self.fake.add_job("Build a box", task_id="t1")
        self.connect_on()
        self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished))
        text, ok = self.fake.finished["t1"]
        self.assertFalse(ok)
        self.assertIn("waitlist", text)


    # --- reverts from the browser ----------------------------------------------------

    def run_turns(self, *texts):
        """Each text as a workspace job of task t1, run to its end: their message ids, oldest first."""
        ids = []
        for number, text in enumerate(texts, 1):
            self.fake.add_job(text, task_id="t1", title="Box", turn=number)
            ids.append(self.fake.jobs[-1]["message"]["id"])
            if number == 1:
                self.connect_on()
            self.fake.finished.pop("t1", None)
            self.assertTrue(self.wait_until(lambda: "t1" in self.fake.finished), f"{text!r} was not finished")
        return ids

    def revert(self, messages):
        revert_id = self.fake.add_revert(messages, task_id="t1")
        self.assertTrue(self.wait_until(lambda: revert_id in self.fake.revert_reports), "no revert report")
        return self.fake.revert_reports[revert_id]

    def test_a_revert_from_the_browser_undoes_the_turns_newest_first(self):
        tree = {"document": "D", "label": "D", "objects": [{"name": "Box", "label": "Box"}]}
        first, second = self.run_turns("Build a box", "Now make it 20 mm")
        cards = [b for b in self.p.blocks if b.kind == panel.Block.TOOL and b.workspace_message]
        self.assertEqual([card.workspace_message for card in cards], [first, second])
        self.assertTrue(all(card.revertible for card in cards))

        with mock.patch.object(connect_gui, "_glb_preview", return_value=b"glTF-after"), \
                mock.patch.object(connect_gui, "_feature_tree", return_value=tree), \
                mock.patch.object(self.p, "_confirm_revert", side_effect=AssertionError("asked")), \
                mock.patch.object(self.p, "_confirm_checkpoint_restore", side_effect=AssertionError("asked")):
            report = self.revert([second, first])
        self.assertEqual(report, {"ok": True, "reverted": [second, first], "message": "", "tree": tree})
        self.assertEqual(self.engine.revert_turn.call_count, 2)
        self.assertTrue(all(card.reverted for card in cards))
        [upload] = self.fake.revert_uploads
        self.assertEqual((upload["format"], upload["data"]), ("glb", b"glTF-after"))
        self.assertTrue(any("reverted every CAD change" in note for note in self.p._pending_notes),
                        "the model hears of it with the next message")

    def test_a_turn_typed_in_freecad_after_them_stops_the_revert(self):
        [first] = self.run_turns("Build a box")
        typed = self.p._append(panel.Block.TOOL, "")   # a later turn typed in FreeCAD's panel
        typed.revertible, typed.snapshot = True, dict(GOOD_SNAPSHOT)
        report = self.revert([first])
        self.assertEqual(report, {"ok": False, "reverted": [], "message": connect_gui.REVERT_LATER_TURN})
        self.engine.revert_turn.assert_not_called()

    def test_turns_from_before_a_restart_are_not_reverted(self):
        [first] = self.run_turns("Build a box")
        self.p._current_session()["_workspace_turns"].discard(first)   # as after a restart: the run forgot it
        report = self.revert([first])
        self.assertEqual(report, {"ok": False, "reverted": [], "message": connect_gui.REVERT_RESTARTED})
        self.engine.revert_turn.assert_not_called()

if __name__ == "__main__":
    unittest.main()
