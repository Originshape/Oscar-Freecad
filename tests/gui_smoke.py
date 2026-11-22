"""End-to-end GUI smoke test for the Oscar panel.

Run inside the FreeCAD GUI:

    /path/to/FreeCAD tests/gui_smoke.py

What it verifies, in the real application:
1. The panel exists and the localhost bridge can be started.
2. A chat profile pointing at a local mock LLM endpoint drives the agent:
   the panel sends a message, the mock streams tool calls, the agent
   executes them through the runtime, and a real document + box appear.
   The mock is REACTIVE (it decides every reply from the last event of the request) and the test asserts the
   number of provider requests and the outcome of each turn explicitly.
3. The final assistant text renders in the panel; the plan strip, the outcome badge, the Build Summary card (once)
   and the widgets located by object name are there.
4. The localhost bridge answers an external XML-RPC ping + get_objects.
5. A screenshot of the panel is written for manual inspection.

NOTE: the bridge listens on port 9876 unless OSCAR_BRIDGE_PORT says otherwise. Set it (scripts/test.sh gui
and the CI job do) to run beside the user's live Oscar; the settings, projects, logs and caches are isolated
through the environment, and the bridge probe below follows the configured port. The mock LLM listens on
OSCAR_SMOKE_MOCK_PORT (default 8765).

Prints OSCAR_SMOKE_OK on success and force-exits (this is a test instance of
the application; os._exit keeps teardown from muddying the result).
"""

import json
import os
import sys
import tempfile
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Isolation: the smoke run must never write into the user's real project folders, logs or
# settings. oscar.paths / oscar.turnlog / oscar.params read these variables at CALL time, so setting them here,
# before the first `import oscar`, is enough. HOME is deliberately not touched.
_SMOKE_ROOT = tempfile.mkdtemp(prefix="oscar_smoke_env_")
for _key, _sub in (("OSCAR_PROJECTS_DIR", "projects"), ("OSCAR_LOG_DIR", "logs"),
                   ("OSCAR_CACHE_DIR", "cache"), ("OSCAR_DOWNLOAD_DIR", "downloads")):
    os.environ[_key] = os.path.join(_SMOKE_ROOT, _sub)
os.environ.setdefault("OSCAR_AI_SETTINGS", os.path.join(_SMOKE_ROOT, "settings.json"))
# The OriginShape account: never the user's keychain or credential file, never a real server (the smoke profile is a
# local mock provider anyway). Scripts run without the approval dialog: the run checks execute_code.
os.environ.setdefault("OSCAR_CREDENTIAL_STORE", "memory")
os.environ.setdefault("OSCAR_API_URL", "http://127.0.0.1:9")
os.environ.setdefault("OSCAR_ALLOW_CODE", "1")

RESULTS = []
FAILURES = []
_LOG = open(os.environ.get("OSCAR_SMOKE_LOG", "/tmp/oscar_smoke_result.log"), "w")


def _report(line):
    # FreeCAD routes script stdout to the internal console; stderr reaches
    # the terminal, so emit through both plus a log file.
    print(line, file=sys.stderr)
    _LOG.write(line + "\n")
    _LOG.flush()


def check(name, condition, detail=""):
    RESULTS.append(name)
    status = "ok" if condition else "FAIL"
    _report(f"[smoke] {name}: {status} {detail}")
    if not condition:
        FAILURES.append(name)


def _wait_for_turn(content, seconds=90):
    """Process events until the panel's worker slot empties (turn done)."""
    from PySide import QtCore, QtWidgets

    loop = QtCore.QEventLoop()
    poll = QtCore.QTimer()
    poll.setInterval(100)
    poll.timeout.connect(lambda: content.worker is None and loop.quit())
    timeout = QtCore.QTimer()
    timeout.setSingleShot(True)
    timeout.timeout.connect(loop.quit)
    timeout.start(seconds * 1000)
    poll.start()
    loop.exec()
    poll.stop()
    timeout.stop()
    QtWidgets.QApplication.processEvents()


def _ignoring_governor(script):
    """Wrap a reactive mock script so the agent's per-round ``[governor] ...`` status line (a user message that
    follows every batch of tool results) is not mistaken for the last event: the mock answers the tool result
    BEFORE it, exactly as the real model does."""

    from support.server import _text_of   # the message text whether the wire carries a string or a list of parts

    def wrapped(request_json):
        messages = list((request_json or {}).get("messages") or [])
        while messages and messages[-1].get("role") == "user" \
                and _text_of(messages[-1].get("content")).lstrip().startswith("[governor]"):
            messages.pop()
        return script(dict(request_json or {}, messages=messages))

    return wrapped


def main():
    import FreeCAD
    import FreeCADGui as Gui

    # ---------------------------------------------------------------- setup
    # isolate task-chat storage so smoke runs don't inherit sessions
    import oscar.chats as _smoke_chats

    _smoke_chats.chats_dir = lambda: os.path.join(
        tempfile.mkdtemp(prefix="oscar_smoke_chats_"))
    from oscar import bridge, params, panel as oscar_panel

    # The test writes a mock profile into the REAL user settings; back the
    # settings file up and restore it before exiting (bottom of file).
    global _SETTINGS_BACKUP
    _SETTINGS_BACKUP = params.backup_state()

    # Profile pointing at the in-process mock LLM.
    mock_port = int(os.environ.get("OSCAR_SMOKE_MOCK_PORT") or 8765)   # CI / scripts/test.sh pick a private port
    params.save_profiles([
        params.Profile(
            name="mock",
            provider="openai_compat",
            base_url=f"http://127.0.0.1:{mock_port}/v1",
            model="mock-model",
            api_key="",
            max_tokens=512,
        )
    ])
    params.set_active_profile("mock")

    from mock_llm_server import make_server, smoke_reactive

    # STL part the attachment turn imports (a 2x3x4 box mesh, 12 triangles;
    # the object lands as 'smoke_part': Mesh.insert names objects after the file stem).
    import Part

    fixture_stl = os.path.join(
        tempfile.mkdtemp(prefix="oscar_smoke_attach_"), "smoke_part.stl"
    )
    Part.makeBox(2, 3, 4).exportStl(fixture_stl)

    reactive_script = _ignoring_governor(smoke_reactive(fixture_stl))
    server = make_server(reactive=reactive_script, port=mock_port)

    # Open the panel (the addon opens it at startup only when PanelAutoOpen is on); show_panel() is idempotent.
    from PySide import QtWidgets

    oscar_panel.show_panel()
    mw = Gui.getMainWindow()
    dock = mw.findChild(QtWidgets.QDockWidget, oscar_panel.DOCK_OBJECT_NAME)
    check("panel dock exists", dock is not None)
    content = dock.widget() if dock else None
    check("panel content is OscarPanel", isinstance(content, oscar_panel.OscarPanel))
    if content is not None:
        for _name in ("oscarLog", "oscarPlanStrip", "oscarArtifactsStrip", "oscarHeartbeat", "oscarStopButton",
                      "oscarActNowButton", "oscarAccountBadge"):
            check(f"widget {_name} exists", content.findChild(QtWidgets.QWidget, _name) is not None)

    if not bridge.is_running():
        # The bridge is off by default (BridgeAutoStart); start it for the ping below.
        print("[smoke] starting the bridge (off by default)")
        bridge.start()
    check("bridge running", bridge.is_running(), bridge.status_line())


    # ------------------------------------------------------- drive one turn
    content.input.setPlainText("Build the smoke test box.")
    content._send()
    _wait_for_turn(content)

    check("agent turn completed", content.worker is None)
    # create_document, create_object, get_view, final text: a gate nudge or a wrap-up would show up here
    check("turn 1 used exactly 4 provider requests", server.request_count == 4, f"requests={server.request_count}")
    requests_after_turn1 = server.request_count
    if server.request_count != 4:   # say WHY: the notes the app injected (gate / watchdog / breaker ...) name the reason
        for _note in [m for m in content.history if getattr(m, "synthetic", False)][:3]:
            _report(f"[smoke] injected {_note.tag} note: {_note.content[:400]!r}")
        from support.server import last_event
        for _rec in list(server.requests)[:8]:   # what the mock saw as the last event of each request
            _kind, _cid, _text = last_event(_rec.json)
            _report(f"[smoke] request {_rec.n}: last event {_kind} {_cid} {_text[:160]!r}")
        if len(server.requests) > 1:
            _tm = [m for m in (server.requests[1].json or {}).get("messages", []) if m.get("role") == "tool"]
            _report(f"[smoke] request 2 tool ids: {[m.get('tool_call_id') for m in _tm]}; the script answers: "
                    f"{reactive_script(server.requests[1].json)!r}"[:600])
        for _rec in list(server.requests)[3:6]:
            _last = ((_rec.json or {}).get("messages") or [{}])[-1]
            _report(f"[smoke] request {_rec.n} ends with {_last.get('role')}: {str(_last.get('content'))[:700]!r}")
        _tail = ((server.requests[3].json or {}).get("messages") or [])[-4:] if len(server.requests) > 3 else []
        _report("[smoke] request 4 tail: " + repr([(m.get("role"), str(m.get("content"))[:140]) for m in _tail]))
        _msgs = (server.requests[1].json or {}).get("messages", []) if len(server.requests) > 1 else []
        _report("[smoke] request 2 messages: " + repr([(m.get("role"), type(m.get("content")).__name__, str(m.get("content"))[:60]) for m in _msgs]))

    # ------------------------------------------------------------ assertions
    docs = FreeCAD.listDocuments()
    check("document created via chat", "SmokeDoc" in docs, f"docs={list(docs)}")
    box = None
    if "SmokeDoc" in docs:
        box = docs["SmokeDoc"].getObject("Box")
    check("box created via chat", box is not None)
    try:
        from oscar.cad.views import fit_view

        check("view fit command resolves", fit_view(),
              "Std_ViewFitAll/Std_ViewFit/ViewFit all missing")
    except Exception:
        check("view fit command resolves", False, traceback.format_exc(limit=2))
    if box is not None:
        shape = getattr(box, "Shape", None)
        check(
            "box has geometry",
            shape is not None and abs(shape.Volume - 1000.0) < 1e-6,
            f"volume={shape.Volume if shape else None}",
        )

    kinds = [b.kind for b in content.blocks]
    check("tool activity logged", oscar_panel.Block.TOOL in kinds, f"kinds={kinds}")
    assistant_texts = [b.text for b in content.blocks if b.kind == oscar_panel.Block.ASSISTANT]
    check(
        "final answer rendered",
        any("box" in t.lower() for t in assistant_texts),
        f"last={assistant_texts[-1][:60] if assistant_texts else None!r}",
    )
    cards = [b for b in content.blocks if b.kind == oscar_panel.Block.TOOL and b.steps]
    check(
        "turn summary card",
        len(cards) == 1 and cards[0].text.startswith("✻ Ran 3 steps") and not cards[0].expanded,
        f"card={cards[0].text if cards else None!r}",
    )
    tool_rows = [s for s in cards[0].steps if not s.plain and not s.is_thought] if cards else []
    check("turn 1 shows exactly 3 step rows (plans and validation never add rows)", len(tool_rows) == 3,
          f"rows={len(tool_rows)}")
    check("turn 1 outcome is validated (badge after the legacy header)",
          bool(cards) and cards[0].text.endswith("✓ validated"), f"card={cards[0].text if cards else None!r}")
    check("plan strip pinned after turn 1", not content.plan_strip.isHidden()
          and "Plan" in content.plan_strip.toPlainText(), content.plan_strip.toPlainText()[:80])
    check("Build Summary rendered once",
          content.log.toPlainText().count("Build Summary") == 1
          and len([b for b in content.blocks if b.kind == oscar_panel.Block.SUMMARY]) <= 1,
          f"count={content.log.toPlainText().count('Build Summary')}")
    check("turn 1 saved the chat's build state",
          bool((content._current_session() or {}).get("ledger")), "session ledger missing")
    for _row in tool_rows:      # a thumbnail that was decoded must still resolve after another render
        if _row.view is not None and _row.view.snapshot_path:
            content._render()
            check("thumbnail resolvable after a second render", content.log.has_resource(_row.call_id),
                  _row.call_id)
    round1_results = [
        tr.content for m in content.history for tr in (m.tool_results or [])
    ]
    check(
        "get_view returns a screenshot",
        any("Screenshot captured" in t for t in round1_results),
        f"n={len(round1_results)}",
    )
    plain_notes = [s for s in cards[0].steps if getattr(s, "plain", False)]
    check(
        "pre-tool remark folded into the card",
        len(plain_notes) == 1 and "Setting up" in plain_notes[0].summary,
        f"notes={len(plain_notes)}",
    )
    check(
        "pre-tool remark not duplicated as a message",
        not any("Setting up" in t for t in assistant_texts),
        f"assistant_texts={assistant_texts!r}",
    )
    check(
        "final answer kept in the conversation history",
        any(m.role == "assistant" and "10x10x10" in (m.content or "") for m in content.history),
    )
    card_pos = next(i for i, b in enumerate(content.blocks) if b is cards[0])
    answers_after = [
        i for i, b in enumerate(content.blocks)
        if b.kind == oscar_panel.Block.ASSISTANT and b.text.strip() and i > card_pos
    ]
    check(
        "answers render after the process card",
        bool(answers_after) and cards[0].text.startswith("✻ Ran 3 steps"),
        f"answers_after={answers_after} summary={cards[0].text!r}",
    )

    # --------------------------------------------------- attachment turn
    content.input.setPlainText("Import the attached part and add a 5 mm cube.")
    content.add_attachments([fixture_stl])
    content._send()
    _wait_for_turn(content)

    check("attachment turn completed", content.worker is None)
    # import_file, then create_object (the build step of the plan), then the final text: 3 requests and no nudge
    check("turn 2 used exactly 3 provider requests", server.request_count - requests_after_turn1 == 3,
          f"requests={server.request_count - requests_after_turn1}")
    user_msgs = [m for m in content.history if m.role == "user"]
    check(
        "attachment path reached the prompt",
        any(fixture_stl in (m.content or "") for m in user_msgs),
        f"msgs={len(user_msgs)}",
    )
    check(
        "attachment shown in chat log",
        any("smoke_part.stl" in b.text for b in content.blocks if b.kind == oscar_panel.Block.USER),
    )
    smoke_doc = FreeCAD.listDocuments().get("SmokeDoc", None)
    part_obj = smoke_doc.getObject("smoke_part") if smoke_doc else None
    faces = len(part_obj.Mesh.Facets) if part_obj is not None else 0
    check("attached STL imported via chat", part_obj is not None and faces >= 12, f"faces={faces}")
    check("cube added next to it", smoke_doc is not None and smoke_doc.getObject("Spacer") is not None)
    # Thoughts are steps inside the turn's process card; the reasoning
    # round of turn 2 is its last step, closed with a duration.
    cards = [b for b in content.blocks if b.kind == oscar_panel.Block.TOOL and b.steps]
    thought_cards = [c for c in cards if any(s.is_thought for s in c.steps)]
    check(
        "thought recorded as a process step",
        bool(thought_cards)
        and thought_cards[-1].steps[-1].is_thought
        and thought_cards[-1].steps[-1].dur > 0,
        f"cards={len(thought_cards)}",
    )
    check(
        "card summarizes thoughts and steps",
        "Thought" in thought_cards[-1].text and "Ran 2 steps" in thought_cards[-1].text,
        f"summary={thought_cards[-1].text!r}",
    )
    check("turn 2 outcome is validated", thought_cards[-1].text.endswith("✓ validated"),
          f"summary={thought_cards[-1].text!r}")
    card_idx = max(
        i for i, b in enumerate(content.blocks)
        if b.kind == oscar_panel.Block.TOOL and b.steps
    )
    answer_idx = max(
        i for i, b in enumerate(content.blocks)
        if b.kind == oscar_panel.Block.ASSISTANT and b.text.strip()
    )
    check("process card above the response", card_idx < answer_idx,
          f"card={card_idx} answer={answer_idx}")

    # ------------------------------------------------ turn revert (real GUI)
    # Turn 2's card must offer revert; reverting pops SmokeDoc's undo stack
    # back to turn-2 start (smoke_part and Spacer disappear, turn-1 Box survives).
    turn2_card = thought_cards[-1]
    check("card offers revert", turn2_card.revertible and not turn2_card.reverted)
    content._revert_turn(turn2_card)
    reverted_doc = FreeCAD.listDocuments().get("SmokeDoc")
    turn2_gone = reverted_doc is None or (reverted_doc.getObject("smoke_part") is None
                                         and reverted_doc.getObject("Spacer") is None)
    box_alive = reverted_doc is not None and reverted_doc.getObject("Box") is not None
    check("revert removed the turn's objects", turn2_gone, f"objects={[o.Name for o in reverted_doc.Objects] if reverted_doc else None}")
    check("revert kept earlier turns", box_alive)
    check("card marked reverted", turn2_card.reverted)

    # ------------------------------------------- direct engine path (Draft)
    # The Draft factories need the workbench's Python modules, which only
    # fully load in the application — exercise them here, outside the mock.
    from oscar import tools as _tools

    text, _image, is_error = _tools.run_tool("create_object", {
        "doc_name": "SmokeDoc",
        "obj_type": "Draft::Rectangle",
        "obj_name": "Plate",
        "obj_properties": {"Length": 30, "Height": 12},
        "include_screenshot": False,
    })
    check("draft factory works in GUI", not is_error and "Failed" not in text, text[:120])
    text, _image, is_error = _tools.run_tool("create_object", {
        "doc_name": "SmokeDoc",
        "obj_type": "Draft::Polygon",
        "obj_name": "ToothProfile",
        "obj_properties": {"FacesNumber": 8, "Radius": 6},
        "include_screenshot": False,
    })
    check("draft polygon factory works in GUI", not is_error and "Failed" not in text, text[:120])
    text, _image, is_error = _tools.run_tool("execute_code", {
        "code": "print('area check', 30 * 12)", "include_screenshot": False,
    })
    check("execute_code via engine", not is_error and "360" in text, text[:120])


    # ------------------------------------------------- external MCP client
    # Real MCP clients call from their own process/thread; a synchronous call
    # from THIS (GUI) thread would deadlock the dispatch queue — the handler
    # thread would wait for the GUI thread that is waiting on the response.
    import threading
    import xmlrpc.client

    bridge_results = {}

    def _bridge_probes():
        try:
            proxy = xmlrpc.client.ServerProxy(f"http://127.0.0.1:{bridge.configured_port()}", allow_none=True)
            bridge_results["ping"] = proxy.ping()
            bridge_results["objects"] = proxy.get_objects("SmokeDoc")
        except Exception as e:  # noqa: BLE001
            bridge_results["error"] = e

    probe_thread = threading.Thread(target=_bridge_probes)
    probe_thread.start()

    from PySide import QtCore

    deadline = QtCore.QElapsedTimer()
    deadline.start()
    while probe_thread.is_alive() and deadline.elapsed() < 30_000:
        QtWidgets.QApplication.processEvents()
        QtCore.QThread.msleep(50)
    check("bridge probes finished", "error" not in bridge_results, str(bridge_results.get("error", ""))[:120])
    check("bridge ping", bridge_results.get("ping") is True)
    objects = bridge_results.get("objects") or []
    check(
        "bridge get_objects sees the box",
        any(o.get("Name") == "Box" for o in objects),
        f"n={len(objects)}",
    )

    # ------------------------------------- task chats (new/switch/clear)
    try:
        chat_combo = getattr(content, "chat_combo", None)
        check("chat switcher exists", chat_combo is not None)
        if content is not None and chat_combo is not None:
            before_count = len(content.sessions)
            content._new_chat()
            check("new chat creates a session",
                  len(content.sessions) == before_count + 1
                  and len(content.history) == 0,
                  f"sessions={len(content.sessions)}")
            # simulate a message in the new chat, then switch away+back
            from oscar.providers import Message as pMsg

            content.history.append(pMsg(role="user", content="task two"))
            content._title_current_from("task two")
            content._persist_current()
            other = next(s for s in content.sessions
                         if s["history"] is not content.history)
            content._activate_session(other)
            check("switch swaps history",
                  content.history is other["history"])
            content._activate_session(
                next(s for s in content.sessions
                     if s["title"] == "task two"))
            check("switch back restores context",
                  any(getattr(m, "content", "") == "task two"
                      for m in content.history))
            # _clear_chat asks "Clear all messages?" in a modal box: answer Yes, or the test waits for a click forever
            _ask = QtWidgets.QMessageBox.question
            QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
            try:
                content._clear_chat()
            finally:
                QtWidgets.QMessageBox.question = _ask
            check("clear empties the chat",
                  len(content.history) == 0
                  and content.chat_combo.currentText() == "New chat")
            # delete must fall back to the last-used REAL chat, not to an empty "New chat" (the reported bug), and
            # no duplicate empties from ＋
            from oscar.providers import Message as pMsg2

            base = content.sessions[0]
            base["history"].append(pMsg2(role="user", content="keep me"))
            base["title"] = "real chat"
            content._activate_session(base)
            content._new_chat()          # empty new chat on top
            dup = content.sessions[0]
            content._new_chat()          # second ＋ must NOT duplicate
            check("plus does not duplicate empty chat",
                  sum(1 for s in content.sessions
                      if not s["history"] and s["title"] == "New chat") == 1)
            content._activate_session(dup)   # switch INTO the empty chat
            content._activate_session(base)   # then make 'real chat' MRU
            content._activate_session(dup)    # back into the empty one…
            _ask = QtWidgets.QMessageBox.question   # _delete_chat asks "Delete the chat?": answer Yes
            QtWidgets.QMessageBox.question = staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
            try:
                content._delete_chat()            # …and delete IT
            finally:
                QtWidgets.QMessageBox.question = _ask
            check("delete falls back to last-used chat",
                  content.chat_combo.currentText() == "real chat"
                  and any(m.content == "keep me" for m in content.history))
    except Exception:
        check("task chats flow ran", False, traceback.format_exc(limit=3))

    # ------------------------------------- prompt templates
    try:
        check("templates button exists",
              getattr(content, "templates_btn", None) is not None)
        if content is not None and hasattr(content, "PARAMPT"):
            content.input.setPlainText("")
            content.input.setPlainText(content.PARAMPT)
            check("parametric template copyable",
                  "PartDesign Body" in content.input.toPlainText()
                  and "spreadsheet" in content.input.toPlainText())
            check("mutating-tool set for live drawing",
                  "apply_feature_plan" in content._MUTATING_TOOLS
                  and "execute_code" in content._MUTATING_TOOLS)
    except Exception:
        check("templates flow ran", False, traceback.format_exc(limit=3))

    # ------------------------------------- Project: real snapshot, restore (the GUI half)
    try:
        import FreeCAD as App
        from PySide import QtGui as _QtGui
        from oscar.project import Project

        smoke_doc = App.getDocument("SmokeDoc")
        App.setActiveDocument("SmokeDoc")
        gdoc = Gui.getDocument("SmokeDoc")
        view = gdoc.ActiveView
        view.setCameraOrientation(App.Rotation(App.Vector(0, 1, 0), 33))

        def _camera_state():
            q = view.getCameraOrientation().Q
            position = view.getCameraNode().position.getValue().getValue()
            # rounded: the view re-serialises the camera with sub-1e-3 noise
            return (view.getCameraType(), tuple(round(float(x), 2) for x in position), tuple(round(float(x), 3) for x in q))

        camera_before = _camera_state()
        proj = Project(os.path.join(tempfile.mkdtemp(prefix="oscar_smoke_project_"), "p"),
                       chat_id="20261002-000000-5a0e01", title="Smoke project")
        rel, note = proj.snapshot("SmokeDoc", "smoke snapshot", 1)
        check("project snapshot is a real picture", bool(rel), note)
        if rel:
            image = _QtGui.QImage(proj.path(rel))
            colors = {image.pixel(x * image.width() // 16, y * image.height() // 12)
                      for x in range(16) for y in range(12)} if not image.isNull() else set()
            check("project snapshot is not blank", len(colors) > 1, f"{image.width()}x{image.height()}, {len(colors)} colours")
        _after_now = _camera_state()
        QtWidgets.QApplication.processEvents()
        _report(f"[smoke] camera before {camera_before} / right after {_after_now} / after events {_camera_state()}")
        check("snapshot leaves the user's camera alone", _camera_state() == camera_before, f"{camera_before} -> {_camera_state()}")

        box = smoke_doc.getObject("Box")
        box.ViewObject.ShapeColor = (0.9, 0.1, 0.1)
        ckpt = proj.checkpoint("SmokeDoc", "smoke")
        extra = smoke_doc.addObject("Part::Box", "RestoreProbe")
        smoke_doc.recompute()
        restored = proj.restore("SmokeDoc", ckpt)
        check("project restore succeeded", restored.ok, restored.error)
        check("restore removed the later object", smoke_doc.getObject("RestoreProbe") is None)
        check("restore keeps the document active (App)", App.ActiveDocument is not None and App.ActiveDocument.Name == "SmokeDoc")
        check("restore keeps the document active (Gui)",
              Gui.ActiveDocument is not None and Gui.ActiveDocument.Document.Name == "SmokeDoc")
        color = tuple(round(c, 2) for c in smoke_doc.getObject("Box").ViewObject.ShapeColor[:3])
        check("restore keeps ViewObject colours", color == (0.9, 0.1, 0.1), str(color))
    except Exception:
        check("project snapshot / restore flow ran", False, traceback.format_exc(limit=3))

    # ---------------------------------------------------------- screenshot
    shot_path = os.path.abspath("oscar_panel_smoke.png")
    if content is not None:
        content.grab().save(shot_path)
        check("panel screenshot saved", os.path.exists(shot_path), shot_path)

    server.shutdown()

    passed = len(RESULTS) - len(FAILURES)
    _report(f"[smoke] {passed}/{len(RESULTS)} checks passed")
    if FAILURES:
        _report(f"[smoke] FAILURES: {FAILURES}")
        return 1
    _report("OSCAR_SMOKE_OK")
    return 0


def _restore_user_config():
    """Put the user's real settings back after the test's mock overwrote them."""
    try:
        from oscar import params

        params.restore_state(_SETTINGS_BACKUP)
    except Exception as e:  # noqa: BLE001 - best effort cleanup
        _report(f"[smoke] warning: could not restore settings: {e}")


_SETTINGS_BACKUP = ""
_EXIT_CODE = 2
try:
    _EXIT_CODE = main()
except Exception:
    traceback.print_exc()
    _report("OSCAR_SMOKE_FAILED")
    _EXIT_CODE = 2
finally:
    _restore_user_config()
    os._exit(_EXIT_CODE)
