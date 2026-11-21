# SPDX-License-Identifier: Apache-2.0
"""Backend demo of the ask_user interaction protocol: a REAL AgentWorker turn pauses on decision
cards and resumes with the answers — no credits, no live model, no panel.

``tests/support``'s ScriptedProvider plays the model. The scripted model calls the registered
``ask_user`` tool twice (a unit decision, then a five-question wizard). A timer plays the user by
resolving each open request through the SAME ``askuser.REGISTRY`` the panel resolves through, and
the turn continues from the exact point it blocked. Every controller signal is printed.

Run under freecadcmd (the bundled interpreter has PySide):

    freecadcmd scripts/demo_askuser.py

Success prints ``OSCAR_ASKUSER_DEMO_OK``.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))       # repo root
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tests"))  # tests/support

from support import isolation  # noqa: E402  (sandbox OSCAR_* before importing oscar)

isolation.isolate_process()
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide import QtCore, QtWidgets  # noqa: E402

from oscar import agent, askuser, tools_ask  # noqa: E402
from oscar.providers import Message  # noqa: E402
from support.scripted_llm import ScriptedProvider, call, calls, say  # noqa: E402

CHAT = "demo-chat-1"

UNIT = {"title": "Which unit is Base.STL in?",
        "subtitle": "The imported geometry measures 185 x 12 x 200 in the current unit interpretation.",
        "options": [{"label": "Millimetres", "description": "185 x 12 x 200 mm", "likeliest": True},
                    {"label": "Centimetres", "description": "1.85 x 0.12 x 2 m"},
                    {"label": "Inches", "description": "7.3 x 0.47 x 7.9 in"}],
        "custom_value": {"label": "Another value", "unit": "mm"},
        "allow_free_text": True}

WIZARD = {"wizard": [
    UNIT,
    {"title": "Approve the rounded dimensions?", "subtitle": "Allowed tolerance: +/-0.1 mm",
     "options": [{"label": "Approve all values", "likeliest": True}, {"label": "Review individually"}]},
    {"title": "Wall thickness measured 2.97 mm — nominal?", "subtitle": "Difference 0.03 mm",
     "options": [{"label": "3 mm", "likeliest": True}, {"label": "Keep 2.97 mm"}]},
    {"title": "Model the bottom edge as scanned (14.9 deg incline) or keep it flat?",
     "options": [{"label": "Model as scanned", "likeliest": True}, {"label": "Keep flat"}]},
    {"title": "Accept the remaining 0.2 mm deviation at the window chamfer?",
     "options": [{"label": "Accept", "likeliest": True}, {"label": "Propose a correction"}]},
]}


def script():
    """The scripted model turn: ask (units) → resume → ask (wizard) → resume → deliver."""
    return [
        calls(call("ask_user", **UNIT), text="The mesh is watertight; the unit is ambiguous, so I will ask."),
        calls(call("ask_user", wizard=WIZARD["wizard"]),
              text="Units confirmed. Five related build decisions remain — one wizard card."),
        say("Units confirmed and all five decisions approved. Building the parametric model from the "
            "approved nominal dimensions next."),
    ]


def user_answers(request):
    """The scripted user: likeliest everywhere, a custom 'Metres' on the unit question."""
    if request.wizard:
        return {"answers": [{"answer": q["options"][0]["label"], "index": 0}
                            for q in request.payload["wizard"]]}
    return {"answer": "Metres", "index": -1, "value": "m"}


def main() -> int:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv[:1])
    from oscar.cad import dispatch as _dispatch

    _dispatch.init()          # the wake bridge is thread-affine: born HERE, on the GUI thread
    state = {"tool_results": 0, "ok": False}

    def surface(request):
        title = (request.payload.get("wizard") or [request.payload])[0].get("title", "?")
        print(f"[card] {'wizard of ' + str(len(request.payload['wizard'])) + ' questions: ' if request.wizard else ''}{title}")

    tools_ask.surface = surface

    # strict=False: the finish gate may add one more round after the final reply (this demo builds
    # no geometry, so the gate asks for work); the interaction under test is already complete by then.
    provider = ScriptedProvider(script(), strict=False)

    def trace(name):
        def slot(*args):
            if name in ("tool_started", "tool_finished", "turn_outcome", "turn_finished", "text_delta"):
                print(f"[signal] {name} {' '.join(str(a)[:70] for a in args)}")
        return slot

    worker = agent.AgentWorker([Message(role="user", content="Rebuild Base.STL as a native parametric model.")],
                               provider=provider, request_text="Rebuild Base.STL as a native parametric model.",
                               session_id=CHAT)
    for name in ("text_delta", "tool_started", "tool_finished", "turn_outcome", "turn_finished"):
        getattr(worker, name).connect(trace(name))

    def on_tool_finished(_name, ok, _text):
        if ok:
            state["tool_results"] += 1

    worker.tool_finished.connect(on_tool_finished)

    def finished(ok, message):
        answered = sum(1 for m in worker.history for r in m.tool_results
                       if r.name == "ask_user" and not r.is_error)
        # The PROTOCOL is the verdict: both decisions asked, answered and resumed. The turn's own
        # ok flag may still be False — this demo builds no geometry, so the finish gate honestly
        # reports NO_WORK; that is the product behaving correctly, not an interaction failure.
        state["ok"] = answered == 2 and not askuser.REGISTRY.pending()
        print(f"[demo] turn_finished ok={ok} ask_user results={answered} "
              f"gate={str(message)[:80]!r}")
        if state["ok"]:
            print("OSCAR_ASKUSER_DEMO_OK")
        app.quit()

    worker.turn_finished.connect(finished)

    def user_timer():          # the 'user': answer whatever decision card is open right now
        for pending in askuser.REGISTRY.pending(conversation_id=CHAT):
            ok, err = askuser.REGISTRY.resolve(pending.id, user_answers(pending),
                                               conversation_id=CHAT, run_id=pending.run_id)
            print(f"[user ] answered {pending.id[:8]} -> {'ok' if ok else err}")

    ticker = QtCore.QTimer()
    ticker.timeout.connect(user_timer)
    ticker.start(50)

    watchdog = QtCore.QTimer()
    watchdog.setSingleShot(True)
    watchdog.timeout.connect(lambda: (print("[demo] TIMEOUT"), app.quit()))
    watchdog.start(60_000)

    worker.start()
    app.exec_()
    askuser.REGISTRY.shutdown()
    return 0 if state["ok"] else 1


# This file is RUN (freecadcmd -c "exec(open(...).read())"), never imported: the gui_smoke.py
# pattern — run at module level and leave through os._exit so no Qt loop keeps the process alive.
_EXIT_CODE = 2
try:
    _EXIT_CODE = main()
except BaseException as exc:   # noqa: BLE001 - a demo reports, never crashes the interpreter
    print(f"OSCAR_ASKUSER_DEMO_FAILED: {type(exc).__name__}: {exc}")
import os as _os

sys.stdout.flush()
sys.stderr.flush()
_os._exit(_EXIT_CODE)
