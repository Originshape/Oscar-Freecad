"""Approval for model-written Python: the gate of ``execute_code``, ``execute_code_async``, ``execute_code_headless``.

Scripts run with the user's rights and can read and change any file, so they are OFF by default:

* the Settings switch ``AllowCodeExecution`` (default off) turns them on; while it is off the three tools are not
  even offered to the model (``hidden_tools``) and a call that arrives anyway is refused (``OFF_TEXT``);
* when it is on, every script is shown to the user first (Run / Run all scripts this turn / Don't run), through the
  same dialog machinery as the network permission (``netguard.ask_user``). A script the user declined is refused
  again without a second dialog for the rest of the turn; "Run all scripts this turn" answers the rest of the turn;
* ``OSCAR_ALLOW_CODE=1`` turns scripts on WITHOUT asking (tests and unattended runs, like ``OSCAR_ALLOW_NET``).

Console sessions have nobody to ask, so they refuse unless the environment override is set. The network gate
(``netguard.gate_code``) still runs after this one: allowing a script never allows its network use.

PURE PYTHON apart from the dialog, which netguard builds on the GUI thread; ``params`` is imported lazily.
"""

import hashlib
import os

from . import netguard
from .toolctx import current_context

PREF = "AllowCodeExecution"
ENV_ALLOW = "OSCAR_ALLOW_CODE"
SCRIPT_TOOLS = frozenset({"execute_code", "execute_code_async", "execute_code_headless"})

# per-turn keys in ToolContext.state (the agent creates one state dict per turn)
_TURN_KEY = "code_approval"        # "turn": the user chose "Run all scripts this turn"
_DECLINED_KEY = "code_declined"    # {script hash: refusal text} of the scripts declined (or not answered) this turn

_STRUCTURED = ("create_object, edit_object, apply_feature_plan, set_parameter and the kit tools")

OFF_TEXT = ("Running Python scripts is switched off in Oscar's settings (Settings > Scripts), so nothing ran. "
            f"Build with the structured tools ({_STRUCTURED}). Do not call execute_code, execute_code_async or "
            "execute_code_headless again in this chat.")
DECLINED_TEXT = ("The user declined to run this script, so nothing ran and the model is unchanged. Do not send the same "
                 "script again; continue with the structured tools, or say what the script would do and ask the user to "
                 "allow it.")
NO_ANSWER_TEXT = ("Nobody answered the request to run this script, so nothing ran and the model is unchanged. Do not "
                  "send the same script again in this turn; continue with the structured tools, or ask the user "
                  "whether they want the script run.")
NO_HUMAN_TEXT = ("Running a Python script needs the user's approval and nobody can be asked in this session, so nothing "
                 f"ran. Build with the structured tools ({_STRUCTURED}).")


def _params():
    """``oscar.params`` or None when FreeCAD (and so the settings store) is unavailable."""
    try:
        from . import params
    except Exception:  # noqa: BLE001 - ImportError of FreeCAD, anything else at import
        return None
    return params


def env_override() -> bool:
    """True when ``OSCAR_ALLOW_CODE=1``: scripts run without asking (read at call time)."""
    return os.environ.get(ENV_ALLOW, "").strip() == "1"


def enabled() -> bool:
    """True when scripts may run at all: the environment override, else the ``AllowCodeExecution`` preference.
    A preference that cannot be read counts as off."""
    if env_override():
        return True
    params = _params()
    if params is None:
        return False
    try:
        return bool(params.get_bool(PREF, False))
    except Exception:  # noqa: BLE001 - an unreadable settings file must never switch scripts on
        return False


def hidden_tools() -> frozenset:
    """The tool names to keep away from the model: the three script tools while scripts are off, else none."""
    return frozenset() if enabled() else SCRIPT_TOOLS


def script_hash(code: str) -> str:
    """A short stable id of a script (logged and remembered instead of the code itself)."""
    return hashlib.sha256((code or "").encode("utf-8", "replace")).hexdigest()[:16]


def _turn_state() -> dict:
    """The per-turn scratch dict of the bound tool context (a throwaway dict when none is bound)."""
    try:
        state = current_context().state
    except Exception:  # noqa: BLE001
        state = None
    return state if isinstance(state, dict) else {}


def _question(code: str, what: str) -> tuple[str, str]:
    """(text, informative) of the approval dialog."""
    count = len((code or "").splitlines()) or 1
    text = (f"Oscar wants to run a Python script it wrote ({what}, {count} line{'s' if count != 1 else ''}).\n\n"
            "Scripts run with your user rights and can read and change your files. "
            "The code is under Show Details.")
    verdict = netguard.analyze(code or "")
    informative = ""
    if verdict["network"]:
        reasons = ", ".join(verdict["reasons"][:3])
        informative = (f"The script also looks like it uses the network ({reasons}); Oscar asks about that separately "
                       "before it runs.")
    return text, informative


def gate_execution(code: str, what: str) -> tuple[bool, str]:
    """Pre-execution gate of the script tools: ``(allowed, message)``; on a refusal the message is model-facing.

    Never shows the code in a log: the ``code_permission`` event carries the tool, the decision, the line count and
    ``script_hash`` only."""
    from .turnlog import log_event

    lines = len((code or "").splitlines())
    digest = script_hash(code)

    def log(decision: str) -> None:
        log_event("code_permission", tool=what, decision=decision, lines=lines, script_hash=digest)

    if env_override():
        log("allow_env_override")
        return True, ""
    if not enabled():
        log("off")
        return False, OFF_TEXT
    state = _turn_state()
    if state.get(_TURN_KEY) == "turn":
        log("allow_cached_turn")
        return True, ""
    refused = state.get(_DECLINED_KEY)
    if isinstance(refused, dict) and digest in refused:
        log("deny_cached_turn")
        return False, refused[digest]
    text, informative = _question(code, what)
    answer = netguard.ask_user(
        "Oscar — run a script?", text, informative=informative, detailed=code or "",
        choices=(netguard.Choice("run", "Run", "YesRole"),
                 netguard.Choice("turn", "Run all scripts this turn", "AcceptRole"),
                 netguard.Choice("deny", "Don't run", "RejectRole")),
        default="deny", object_name="oscarScriptApproval", op="code_permission")
    if answer in (netguard.NO_QT, netguard.NO_HUMAN):
        log("deny_console")
        return False, NO_HUMAN_TEXT
    if answer in ("run", "turn"):
        if answer == "turn":
            state[_TURN_KEY] = "turn"
        log("allow" if answer == "run" else "allow_turn")
        return True, ""
    if answer == "deny":
        log("deny")
        message = DECLINED_TEXT
    else:
        log("deny_no_answer" if answer == netguard.NO_ANSWER else "deny_dispatch_failure")
        message = NO_ANSWER_TEXT
    if not isinstance(refused, dict):
        refused = state[_DECLINED_KEY] = {}
    refused[digest] = message     # the same script is refused again without a second dialog this turn
    return False, message
