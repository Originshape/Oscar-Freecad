# Developing Oscar

Oscar is a pure-Python FreeCAD addon: `package.xml`, `InitGui.py` and the `oscar/` package, running inside the
official FreeCAD 1.1.x builds. Nothing is compiled. Its job: turn a chat request into an editable parametric
FreeCAD model, with every step recorded and checked, through any LLM provider.

**Contents:** [Get set up](#get-set-up) · [How a request becomes a part](#how-a-request-becomes-a-part) ·
[Code map](#code-map) · [Add a tool](#add-a-tool) · [Add a provider](#add-a-provider) ·
[The OriginShape account](#the-originshape-account) · [Connect mode](#connect-mode-the-originshape-workspace) ·
[Projects](#projects) ·
[Settings, files and environment](#settings-files-and-environment) · [Logs](#logs) · [MCP bridge](#mcp-bridge) ·
[Tests](#tests) · [Release](#release)

## Get set up

```bash
git clone https://github.com/Originshape/Oscar-Freecad && cd Oscar-Freecad
scripts/launch.sh                 # FreeCAD with this checkout, in a throw-away profile (.devhome/)
python3 scripts/dev_install.py    # or link this checkout into your normal FreeCAD profile
scripts/test.sh fast              # quick tests (about a minute)
```

`scripts/launch.sh` and `scripts/test.sh` find FreeCAD at `/Applications/FreeCAD.app` on macOS and as
`freecad` / `freecadcmd` on the `PATH` elsewhere. Override with `FREECAD` (GUI, for `launch.sh`), `FREECAD_GUI` and
`FREECADCMD` (for `test.sh`), or `FREECAD_APP` (a macOS app bundle).

## How a request becomes a part

```text
panel (chat)  ──>  agent loop (QThread)  ──>  provider (any LLM)
                        │  ▲
                        ▼  │ tool results, screenshots
                   tool registry (43 tools)  ──>  CAD engine (GUI thread, undo transactions)
                        │                              │
                        ▼                              ▼
                 Build Flow ledger              FreeCAD document  ──>  Project folder
```

1. **Panel** (`panel.py` and `panel_*.py`) takes the message and starts an `AgentWorker` on a QThread.
2. **Agent loop** (`agent.py`) classifies the request (`flow_intent.py`: enclosure, bracket, generic part, …), seeds a
   plan, builds the system prompt (`prompt.py`) and calls the provider (`providers.py`).
3. **Tools** (`tools*.py`) run through one pipeline (`toolrun.py` / `toolpipe.py`): argument repair, the handler, a
   model health check, the project step record and the plan bookkeeping. CAD work hops to the GUI thread through
   `cad/dispatch.py`.
4. **Build Flow** (`flow.py`) enforces the loop in code, so the same flow works with a weak or strong, cloud or
   local model:
   - **Phases:** intake → source → plan → parameters → build → verify → deliver.
   - **Turn gate:** a turn ends only when the plan is resolved or a step is visibly blocked.
   - **Build Summary:** the final answer ends with an app-generated summary whose "Validated" and "NOT validated"
     lists come from recorded checks only.
   - **Safety nets:** a finish gate, a round governor, a reasoning watchdog and a failure breaker. `OSCAR_FLOW=0`
     switches them off; recording stays on.
5. **Project** (`project.py`) records every step on disk (see [Projects](#projects)).

**At FreeCAD start-up**, `InitGui.py` does only two things: it arms the GUI-thread dispatcher and registers the
**Oscar** workbench and its commands (`Oscar_ShowPanel` on `Ctrl+Shift+O`, `Oscar_Settings`, `Oscar_ToggleBridge`,
and in the menu `Oscar_SignIn` and `Oscar_Account` for the OriginShape account). Everything heavier (the tool registry, the engine, FreeCAD's Part / Sketcher / Spreadsheet
modules) loads when the panel opens or the bridge starts. FreeCAD runs `InitGui.py` at every start, so keep it light.

## Code map

```text
InitGui.py            start-up: duplicate-install guard, dispatcher, workbench and commands
Init.py               empty on purpose (console sessions load nothing)
oscar/
├── panel.py          the dock panel (chat, plan strip, step rows, snapshots, pause / steer / stop)
├── panel_*.py        anchors (links), blocks, follow-the-build, render (pure, unit-tested), revert, rewind,
│                     session (chats), turn (what the worker's signals do to the screen)
├── agent.py          the tool-calling loop; agent_ctx.py (wiring), agent_legacy.py (the OSCAR_FLOW=0 path),
│                     stopreason.py, loopcore.py (shared loop helpers)
├── flow.py           plan ledger, phase machine, finish gate (pure); flow_intent.py (request classification),
│                     flow_guard.py (watchdog, breaker), flow_run.py (governor and outcome texts)
├── prompt.py         the phase-structured system prompt, playbooks, the panel's templates (pure)
├── context.py        pinned task memory, request view, token budget
├── providers.py      Anthropic and OpenAI-compatible providers; provider_ollama.py (native Ollama);
│                     provider_originshape.py (the OriginShape account);
│                     provider_*.py: args, chat, check (connection test), probe, reasoning, stream, text, transport
├── auth/             the OriginShape sign-in (pure): config (server URL), discovery, pkce, loopback (browser),
│                     device (code), keystore (keychain / file / memory), session (tokens, /v1/me), account
├── connect.py        connect mode (pure): the /v1/connect client, the event batch, the polling Loop;
│                     connect_gui.py: its FreeCAD side (starts workspace turns in the panel)
├── models.py         ModelCaps: what a model can do (API metadata, a seed table, runtime learning)
├── tools.py          the registry and the core CAD tools; tools_flow, tools_project, tools_kit*, tools_ref,
│                     tools_verify*, tools_web: the other tool groups
├── toolrun.py        the registry API and the run_tool pipeline; toolpipe.py (pure helpers); toolctx.py
├── plan/             FeaturePlan: schema and builder (Params spreadsheet + Sketcher + PartDesign)
├── kit/              enclosure kit: spec, Params sheet, geometry, build stages, checks, fit, materials
├── refparts/         reference-part catalog, board data (boards/*.json), fetch, identity, import, measure
├── cad/              the CAD engine (derived from freecad-mcp, see NOTICE): dispatch, engine, properties,
│                     serialize, views, scripting, headless runs, bridge server, model health, imported files
│                     (measured; meshes made solids on import_file's as_solid)
├── project*.py       the per-chat project folder: checkpoints, docs (NOTES.md, rebuild.py), export, replay, rewind
├── netguard.py       network permission gate and the audit hook for model-written scripts; codeguard.py: the
│                     approval of model-written scripts (off by default)
├── params.py         settings file, profiles and provider presets; settings_model.py / settings_ui.py: the dialog
│                     and the AccountHub (the account state the panel and the dialog share)
├── panel_markup.py   the pen: draw on a picture of the 3D view, pick the marks in the model; markup.py: the region
│                     line
├── panel_chats.py    rename and search the chats (⋯ menu); chatsearch.py: the matching rule
├── updates.py        is there a newer Oscar? (pure: GET /v1/plugin once a day; Oscar → Check for Updates…)
└── chats.py  turnlog.py  attachments.py  bridge.py  commands.py  paths.py  fsretry.py
tests/                unit and integration tests, scripted-model scenarios, gui_smoke.py
scripts/              launch.sh, dev_install.py, test.sh
```

The structured-part tools are `apply_feature_plan` (build a whole FeaturePlan), `reapply_plan_feature` (rebuild
one feature) and `feature_tree_status`.

## Add a tool

1. **Write the handler** in the `tools_*.py` module it belongs to (or a new one). A handler returns
   `(text, png_base64_or_None)` and raises `ToolError` with a message the model can act on.

   ```python
   from .tools import ToolError, _P, tool

   @tool(
       "measure_distance",
       "Measure the shortest distance in mm between two objects of a document.",
       {"type": "object",
        "properties": {"doc_name": {"type": "string"}, "a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["doc_name", "a", "b"]},
       category="read",          # read, build, params, source, verify, deliver, plan, session, fem, net
       phases=_P("----XX-"),     # offered from BUILD on (INTAKE SOURCE PLAN PARAMS BUILD VERIFY DELIVER)
   )
   def _measure_distance(doc_name: str, a: str, b: str):
       ...
       return f"{distance:.3f} mm", None
   ```

   Set `mutating=True` when it changes the document (the call is recorded as a project step and the turn becomes
   revertible) and `geometry=True` when it changes geometry (a snapshot and a model health check follow). Do the
   document work on the GUI thread inside an undo transaction, as the engine does: `cad.dispatch.dispatch(...)`
   with `cad.engine.txn_task(task, "Oscar: <tool>", doc_name)`. Then a failed call rolls back and revert works.
2. **Register a new module** in `_TOOL_MODULES` in `oscar/tools.py`. Existing modules need nothing.
3. **Give it a category** in `_TOOL_CATEGORY` in `oscar/flow.py`. The plan ledger reads it to decide which step
   the call counts for; a tool missing there never ticks a plan step.
4. **Name it in the prompt's tool set**, `KNOWN_TOOLS` in `oscar/prompt.py`. A prompt sentence that names a tool in
   backticks is shown only when that tool is offered.
5. **Update the tool lists the tests pin**: `MATRIX` in `tests/test_tool_registry.py`, `CONTRACT_TOOLS` in
   `tests/test_prompt.py`, and the expected set and count in `tests/test_tools_and_params.py`. Then add tests for
   the tool itself and run `scripts/test.sh full`.

## Add a provider

Most providers speak the OpenAI chat-completions protocol. For those, adding a **preset** is enough:

1. Add an entry to `PROVIDER_PRESETS` in `oscar/params.py`:

   ```python
   {"label": "Example AI", "provider": "openai_compat", "base_url": "https://api.example.ai/v1",
    "default_model": "example-large", "needs_key": True, "temperature": 0.2, "max_tokens": 8192},
   ```

2. Optional: teach `oscar/models.py` the host (`host_class`) and what its models can do (the seed table): context
   window, vision, reasoning controls. Oscar also learns these at run time from the provider's answers.
3. Test it from the settings dialog (**Test connection** checks a reply, a tool call and an image), and run
   `scripts/test.sh full test_providers test_provider_catalogue test_models`.

A provider with its own wire protocol needs a class next to `AnthropicProvider` and `OpenAICompatProvider` in
`oscar/providers.py` (or its own module, like `provider_ollama.py`) and a branch in `make_provider`.

**Announcing a provider before it works.** List it in `COMING_SOON_PROVIDERS` in `oscar/params.py` (label, kind,
tooltip note) and add its kind to `COMING_SOON_KINDS` in `oscar/providers.py`. The settings dialog shows it after the
presets, greyed out and not selectable, and `make_provider` refuses a profile of that kind with a clear message. Both
lists are empty today (the OriginShape account, announced this way before it worked, now has its provider class, its
`make_provider` branch and the first preset); to open an announced provider, give it a provider class and a branch
in `make_provider`, add a preset, and remove it from both lists.

**A preset with `"account": True`** (only OriginShape's) has no Base URL, API key or model listing in the editor:
those rows are hidden, the model is one of the preset's `models`, and "Add" starts from Anthropic instead.

## The OriginShape account

Oscar can use an OriginShape account instead of an API key of your own.

- **Where it connects**: one address, `https://app.originshape.design` (`auth.config.DEFAULT_API_URL`): the
  sign-in pages, the tokens, the account and the workspace are all on the origin the user signs in on. The server's
  sign-in metadata must name that origin as its issuer and token endpoints, or Oscar refuses to sign in
  (`auth/discovery.py`).
- **Signing in** (`oscar/auth/`, pure): `LoopbackSignIn` (the browser, OAuth with PKCE, a one-shot listener on
  `127.0.0.1:<free port>`), `DeviceSignIn` (a code typed on the website, RFC 8628) and personal API keys
  (`os_live_…`). `session.get_session()` keeps one `AuthSession` per server: the refresh token or the key in the
  keystore, the access token in memory, renewed single-flight before it expires.
- **The provider**: the `OriginShape` profile (`provider "originshape"`, models `origin1` "Origin1" /
  `origin1-super` "Origin1 Super"; the retired `oscar-fast` / `oscar-default` still work and are read as those two)
  builds `OriginShapeProvider` (`provider_originshape.py`), which sends the model's name with the account's bearer
  token and the `x-oscar-session` / `x-oscar-project` / `x-oscar-turn` headers (from `AgentWorker(session_id=...)`
  and the turn id). What a model can do (thinking, vision, context window, output cap) is what the server declares
  for it: Oscar knows the account's models by their names only. Account errors are `ProviderError`s of kind `auth` /
  `account`.
- **The Qt side**: `settings_ui.AccountHub` (`account_hub()`) is the one place that holds the account state for the
  panel and the dialog. `oscar.auth.session.on_change` points at it; it refreshes `GET /v1/me` on one daemon thread at
  a time. The settings dialog's **OriginShape account** group runs every sign-in, the API key check and sign-out as
  background jobs. The panel shows the badge (`oscarAccountBadge`), gates a send on local facts only (no sign-in, a
  fresh waitlisted snapshot), and turns the worker's `account_problem` signal into an `ACCOUNT` card
  (`panel_render.account_card_html`): signed out, waitlisted, no credits. When the first request of a turn failed
  that way, the message goes back into the input.
- **Links**: `oscar://account/signin|device|buy|subscribe|manage|waitlist|download|refresh` and `oscar://settings` work in
  any chat text, also the model's: the pages they open come from the account (`/v1/me` links or the server's
  metadata), never from the link, and none of them signs out or spends anything.
- **Against another server**: in **⚙ → OriginShape account → Developer: server address**, turn on **Use a custom
  server** and set **Server** (plain http is accepted for loopback hosts only, such as `http://localhost:8210`), or
  start FreeCAD with `OSCAR_API_URL`, which wins. Turn the switch off to go back to `https://app.originshape.design`;
  the Server field is kept for next time but is ignored while the switch is off, so a server left in it never takes
  over a sign-in. A server without an AI model configured answers HTTP 501 `llm_not_configured`, which Oscar shows as
  "The OriginShape server has no AI model configured." and never retries.
- **Tests**: `tests/support/fake_originshape.py` and `tests/support/fake_workspace.py` stand in for the service on
  `127.0.0.1`.

## Connect mode (the OriginShape workspace)

Oscar runs tasks started in the OriginShape workspace on the website (`/studio` on `app.originshape.design`).

- **The setting**: ⚙ → OriginShape account → **Connect to the OriginShape workspace** (`params` boolean
  `ConnectWorkspace`, `connect.PREF`, off by default; applied at once). Only an OriginShape **sign-in** (OAuth)
  connects, never an API key. The line under it shows the loop's state.
- **The loop** (`connect.Loop`, one daemon thread, pure): it says hello, then long-polls for work, but **only while
  the panel is idle**: a turn the user started always comes first, and while busy only the hello every 20 s keeps
  this computer online. Backoff on network errors; a signed-out or waitlisted account waits.
- **A job** (`connect_gui.WorkspaceConnector`, GUI thread) opens the task's chat (*Workspace: <title>*; the task id ->
  chat id map is `workspace-tasks.json` in the chats folder; a computer new to the task seeds the chat with the
  task's earlier messages; the chat follows the task's title when it is renamed in the workspace, unless the user
  named the chat in FreeCAD, `session["named"]`) and sends the message through `OscarPanel._send(workspace=WorkspaceSend(...))`: the job's
  text, the account profile with the job's model alias, the input box untouched. Everything else (project, ledger,
  persistence, the blockers) is the normal send path.
- **One document per task** (`connect_gui.open_task_document`, before the send): a new task activates the Oscar
  workbench and creates a new document through `CadEngine.create_document` (so it counts as Oscar's), named from
  the title (`_doc_name_for`: "An 80 mm flange with bolt holes" -> `FlangeBoltHoles`, label = the whole title). A
  later message activates the task's document again, reopening its file when it was saved and closed. The task id ->
  `{"doc", "file"}` map is `workspace-docs.json` in the chats folder, updated when each turn ends (only with an open
  document). A task from before this map keeps building in the active document.
- **Attachments**: the job's `message.attachments` (the user's files, a PDF followed by its page images) are
  downloaded by the loop BEFORE the turn starts, with
  `AuthSession.authorized_download` -> `transport.download`: no redirects, streamed to a temp file beside the target,
  stopped past the announced size, kept only when the sha256 matches. They land in
  `workspace-files/<task>/<message>/` next to the chats folder (`connect.safe_filename` names, clashes numbered), and
  `WorkspaceSend.attachments` hands the paths to `_send` as its pending attachments, so `oscar/attachments.py` treats
  them exactly like files attached in the panel. `import_file` tells the model each imported object's kind, size
  and position, and makes a mesh a solid on `as_solid` (`cad/imported.py`). A file that fails is an `error` event and the turn goes on; a
  files-only message whose files all fail is finished without a turn. Folders untouched for 30 days are pruned when
  the loop starts. Earlier messages' files appear in seeded history as `[attached: ...]`.
- **What it sends** (`connect.GuiTurn` -> `EventBatch`, posted every 0.5 s by the loop's thread): one `progress` event
  per finished step (title, ok, a short failure detail), the reply as it is written (`reply`, the whole text each
  time), account problems (`error`); a **preview** after a geometry step (at most every 3 s): a GLB with one node
  per visible part, named by its FreeCAD object Name like the tree (`connect_gui._preview_parts` meshes
  `project_gui.visible_tips` face by face in the global frame, `connect.preview_glb` writes the GLB), or a binary STL
  of the whole model (`project_export.export_tips`) when that fails; when the turn ends, the **exports** (STEP, STL, `model.FCStd`) and the reply with its
  Build Summary (`finish`). An empty post every 10 s is the keep-alive and carries the browser's **Stop**
  (`cancel: true` -> `worker.stop()`). Uploads use `authorized_request(content=...)`, only to the API origin.
- **Answers always arrive**: `Loop._report` skips any failing step (logged) and always reaches
  `finish`, which carries the turn's `message_id`. The answer is written to `connect.Outbox`
  (`workspace-outbox.json` beside the chats) before anything else can fail, and removed once the server took it.
  While idle, `Loop._sync` (every minute, and right after a reconnect) delivers what is left there, then answers
  the turns the website closed without an answer from the outbox or the task's chat
  (`WorkspaceConnector._recover`). Outgoing text goes through `project_paths.redact` ("Oscar project", never the
  project folder); the panel shows the same paths as an "Oscar project" link (`project_paths.install`).
- **Approvals stay here**: a workspace turn has exactly the panel's permissions. Scripts need codeguard's approval
  (off by default) and network access its own, both answered at this computer. `netguard.ask_user` reports each
  question to `netguard.on_approval` (wrapped by the connector) when it is asked and when it is answered; the browser
  shows one step that changes in place: waiting, then approved, declined or no answer (`connect.approval_event`).
- **The turn report**: `connect.step_event` decides what the browser sees of a step: plan
  bookkeeping is hidden, untitled tools get readable titles, a failed `check_fit` / `validate_model` becomes "N
  problems found" with each failure in plain words (`issue: design`), other failures say `tool` or `permission`.
  After a geometry step and at the end of the turn, `GuiTurn.take_tree` sends the document's feature tree (children
  from `claimChildren()`, as FreeCAD's tree shows them; rebuilt the same way without a GUI), the model before any
  hidden `Ref_*` reference, with the Params sheet (`project_gui.param_table`) as its `params`. Exports are per part plus the whole model.
- **Reverts from the browser**: with `revert=` the loop's hello says it can revert, and the server may hand it a
  revert job (`connect.RevertJob`: the task and the prompts to undo, newest first). `WorkspaceConnector._revert_on_gui` makes the task's chat current and calls the panel's own
  `_revert_turn(card, remote=True)` for each prompt's turn card, newest first: no dialog (`remote` skips
  `_confirm_revert`, and the checkpoint fallback restores nothing when a document changed after the turn). Each card
  knows its prompt (`Block.workspace_message`, from `_turn_sent["workspace_message"]`), and the chat knows which
  prompts this FreeCAD run holds (`session["_workspace_turns"]`, runtime only). A turn without a card, or that
  changed nothing, has nothing to undo; the revert stops at a turn typed in FreeCAD after them (`REVERT_LATER_TURN`)
  or one from before a restart (`REVERT_RESTARTED`). `Loop._revert` then uploads the preview and
  reports the prompts undone, why not all, and the tree (sent again without the tree if the server refuses it).
- **Tests**: `tests/test_connect.py` (pure: the client, the batch, the loop, attachment downloads and revert jobs
  against `support/fake_workspace.py`) and `tests/test_connect_panel.py` (FreeCAD: a job, with and without
  attachments, runs as a real panel turn with a scripted provider; a revert job undoes real turn cards).

## Projects

Every chat that changes a model owns a folder:

```text
~/Documents/Oscar/Projects/<slug>-<id>/    (OSCAR_PROJECTS_DIR overrides the root)
├── model.FCStd         the working model
├── steps.jsonl         one line per mutating tool call: arguments, result, object delta, health, snapshot, checkpoint
├── scripts/            every executed script, verbatim
├── snapshots/          isometric PNGs, one per geometry step
├── checkpoints/        FCStd restore points for rewind (at most 60, 600 MB; older ones are pruned)
├── params.json         mirror of the Params spreadsheet: name, value, unit, source, note
├── provenance.json     every sourced fact: source, sha256, status
├── ledger.json  spec.json  validation.json  project.json
├── NOTES.md            generated: request, requirements, plan, parameters, provenance, validation, steps
├── rebuild.py          headless replay of the recorded steps (prints REBUILD_OK on success)
├── exports/            STEP / STL from export_model
└── refs/               downloaded or attached reference parts
```

If the default root is not writable, Oscar falls back to `<FreeCAD user folder>/Oscar/Projects`.

## Settings, files and environment

Settings live in `oscar_settings.json` in the FreeCAD user folder (atomic writes), never in FreeCAD's own
parameters. Chats are stored in `<FreeCAD user folder>/Oscar/chats`.

| Variable | Meaning | Default |
|---|---|---|
| `OSCAR_AI_SETTINGS` | settings file | `<FreeCAD user folder>/oscar_settings.json` |
| `OSCAR_PROJECTS_DIR` | projects root | `~/Documents/Oscar/Projects` |
| `OSCAR_LOG_DIR` | folder of `agent.log` | `<FreeCAD user folder>/Oscar/logs` |
| `OSCAR_CACHE_DIR` | cache root (`scripts/`, `headless/`, `reload/`) | `~/.cache/oscar` |
| `OSCAR_DOWNLOAD_DIR` | where downloads are saved | `~/Downloads/Oscar` |
| `OSCAR_ALLOW_NET` | `1` allows network access without the permission dialog (unattended runs) | unset |
| `OSCAR_ALLOW_CODE` | `1` runs model-written scripts without the approval dialog (tests, unattended runs) | unset |
| `OSCAR_API_URL` | the OriginShape server (wins over the account settings' Use a custom server switch and Server field, `OriginShapeCustomServer` / `OriginShapeServer`); plain `http://` only for loopback hosts. The app serves the API's paths too, so this is the app's address | `https://app.originshape.design` |
| `OSCAR_UPDATE_CHECK` | `0` turns off the daily check for a newer Oscar (`updates.py`; also the `CheckForUpdates` setting, Settings → Tools & permissions → Updates) | on |
| `OSCAR_CREDENTIAL_STORE` | where the OriginShape sign-in is kept: `auto` (the system keychain, else the credential file), `keychain`, `file` or `memory` (this process only) | `auto` |
| `OSCAR_CREDENTIAL_FILE` | the credential file (0600) | `oscar_credentials.json` next to the settings file |
| `OSCAR_NET_AUDIT` | `0` disables the audit hook that blocks network calls from model-written scripts | `1` |
| `OSCAR_DNS_PIN` | `0` disables pinning a checked host to its validated public addresses during a download | `1` |
| `OSCAR_FLOW` | `0` disables ledger, gate, governor, watchdog, breaker and run_tool hooks | `1` |
| `OSCAR_BRIDGE_PORT` | MCP bridge port (an invalid value is ignored) | `9876` |
| `OSCAR_SEARCH_URL` | search backend for `web_search` (else the Settings value) | unset |
| `OSCAR_REFPART_LOCAL_DIR` | offline mirror for reference parts: `<dir>/<catalog id>.step` | unset |
| `OSCAR_HEALTH_BUDGET_S` | wall-clock budget of one model health check | `4.0` |
| `OSCAR_REBUILD_SET` | `name=value,...` parameter overrides for a `rebuild.py` run | unset |
| `OSCAR_TEST_MODULES` | comma-separated test modules for `tests/run.py` | all |
| `OSCAR_TEST_REAL_KEYCHAIN` | `1` lets `test_auth_keystore` write a test entry to your real macOS login keychain (and remove it) | unset |

`tests/support/isolation.py` gives every test process `OSCAR_ALLOW_CODE=1`, `OSCAR_CREDENTIAL_STORE=memory`,
`OSCAR_UPDATE_CHECK=0` and `OSCAR_API_URL=http://127.0.0.1:9` (nothing listens there), so no test can reach your
keychain or a real server (the tests of the update check turn it on against the fake server).

## Logs

`agent.log` in `OSCAR_LOG_DIR` holds one JSON event per line, for example:
- turn and round events: `turn_start`, `round`, `turn_end` (with the outcome);
- tool events: `tool_start`, `tool`;
- safety nets: `watchdog_trip`, `gate_nudge`, `breaker_trip`;
- `net_permission` and `provider_retry`.

It never contains API keys or code bodies, and rotates at 5 MB, keeping three old files.

## MCP bridge

An XML-RPC server on `127.0.0.1` (port 9876, `OSCAR_BRIDGE_PORT`). It exposes the same CAD engine the chat uses:
documents, objects, code execution, screenshots, import, FEM and turn snapshots. It refuses requests that:
- carry an `Origin` header;
- have a `Host` that is not the listener itself;
- have a body that is not `text/xml`.

There is no token, so any program of the same user can reach it while it runs. It is off by default, and port 9876
is also the default of the standalone freecad-mcp server.

## Tests

Always run the suites through `scripts/test.sh`. It gives every run a throw-away FreeCAD user home and its own
settings, projects, logs, cache and download folders, so a test never touches your real data or imports another
copy of the addon.

```bash
scripts/test.sh fast                  # the fast set in one process (flow, agent loop, providers, tools, ...)
scripts/test.sh full                  # every tests/test_*.py, one FreeCAD process per module
scripts/test.sh full test_flow        # only the named modules
scripts/test.sh gui                   # GUI smoke test against a scripted model; needs a display (Xvfb on Linux)
OSCAR_TEST_ALLOW_REAL_HOME=1 python3 tests/run.py --pure   # no FreeCAD needed: plain Python 3.11 + requests
```

Every flow test uses a scripted model or a local scripted HTTP server, never a live LLM. The OriginShape account is
tested against `tests/support/fake_originshape.py`, a stand-in for the service (sign-in, tokens, the account, chat)
on `127.0.0.1`. New `tests/test_*.py` files are found automatically, and a guard test fails on a file the runner
cannot see; a module that passes without FreeCAD belongs in `PURE_MODULES` in `tests/run.py` (keep the file ASCII).

**Known failures:**
- `test_golden_pi4` case 42 (the ±10 % editability rebuild) fails on a slow or busy machine.
  `tools_verify_edit.BUDGET_S` is 20 s, and one edit of the enclosure can take about 4 s, so the parameters the test
  expects end up "not checked".
- `test_fix_plan_core` intermittently crashes FreeCAD 1.1.3 with SIGSEGV inside `App::Document::undo` while its
  undo and revert cases run (about one run in four on macOS arm64). The crash is in FreeCAD's undo, not in the test.

## Release

1. Set the version in `package.xml` (with its `<date>`) and `oscar/__init__.py` (`__version__`), following
   [Semantic Versioning](https://semver.org/); the pure suite fails when they differ (`test_auth_basics`). Oscar
   reports `__version__` to the server (`hello`, the user agent).
2. Move the `[Unreleased]` notes in `CHANGELOG.md` under the new version, with the date.
3. Tag `v<version>` on `main` and push the tag. `release.yml` checks that the tag and `package.xml` agree, and
   publishes a GitHub release with a ZIP of the addon (`Oscar-v<version>.zip`, a `Mod/Oscar` folder).
4. Once the release is out, the OriginShape server announces it as the latest version: older builds then offer the
   update (the panel header, the account menu, Oscar → Check for Updates…), signed in or not. A minimum version is
   raised only when the server stops working with older builds: those builds then refuse to chat or connect until
   updated.

`tests/`, `scripts/` and `.github/` are left out of the ZIP.

CI (`.github/workflows/ci.yml`) runs on every push:
1. the pure suite;
2. the full suite on the official FreeCAD 1.1.3 builds for Linux, macOS and Windows;
3. the GUI smoke test on Linux and macOS.
