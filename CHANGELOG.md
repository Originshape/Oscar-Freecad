# Changelog

All notable changes to Oscar for FreeCAD. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/): the version is in `package.xml` and
`oscar/__init__.py`, and each release is a `v<version>` tag with a ZIP attached.

## [Unreleased]

## [0.2.1] - 2026-10-28

The first public release: 0.2.0 (released internally) with these fixes.

### Fixed

- **Brackets and fixtures are built from a FeaturePlan again**: `apply_feature_plan` and `reapply_plan_feature` were
  hidden for those requests along with the enclosure kit's tools, so the structured-part playbook never switched on
  and the model fell back to scripts. Enclosures still build with `build_enclosure`.
- **The parameter tools work on FeaturePlan parts**: `set_parameter`, `set_parameters`, `get_parameters` and
  `bind_expression` found only the `Params` sheet, but a part built by `apply_feature_plan` keeps its parameters in
  `Parameters` (0.2.0 said every dimension lives in one `Params` sheet; that holds for enclosures and scripted parts).
  `set_parameters` no longer creates a `Params` sheet beside the plan's that drives nothing, and a name in both sheets
  is changed where the model is bound to it. A value given with a unit ("6 mm") goes into a plan sheet as a plain
  number, as the plan writes it. A sheet of your own named `Parameters` is never edited.
- **A bracket or fixture plan completes its steps**: one `apply_feature_plan` call builds the whole part, so it now
  counts for the holes, fillets and locating steps whose features the plan contains, instead of leaving them to a bare
  "done" the app does not accept. The build steps of these templates suggest the plan first, then scripts.

With the OriginShape account:

- **A dropped connection to the model is retried**: the server's "connection to the model was interrupted" (and
  Anthropic's own `api_error`) ended the turn instead of being sent again.
- **Clearer account messages**: a long chat that needs more credits set aside than are left says so ("start a new
  chat") instead of "no credits left"; the daily usage limit says when it resets instead of "rate limited, try again in
  41234 s"; one model that isn't available is named instead of "no AI model configured"; other server errors show
  their message, not raw JSON; a reply the server shortened because credits are low says that instead of blaming the
  model's output budget.
- **Credits as the website shows them**: the balance, with what a running request holds, instead of the available
  credits (FreeCAD could say 0 while the website said 12). A trial account is offered **Choose a plan**, not **Buy
  credits** (credit packs are for subscribers).
- **Two FreeCADs no longer sign each other out**: they shared one stored refresh token and the second sent an
  already-used copy, which the server ends the sign-in for. A renewal now waits for the other FreeCAD and uses the
  token it stored.
- **A browser sign-in waits 10 minutes**, as long as the server keeps the request (it gave up after 5).
- **A suspended account is told so**, instead of "sign in again" (and a browser or code sign-in no longer waits out
  its timeout).
- **A model's output cap and a retired model name** (`oscar-default`, `oscar-fast`) follow the server's model list:
  a retired name got a 32k window, 4096 output tokens and no caching. Only `os_live_` keys are taken as API keys.

In the OriginShape workspace (tasks from the browser):

- **A long download or export no longer costs the turn its answer**: FreeCAD went quiet while it downloaded attachments
  or made previews and exports, and after 2 minutes the server gave the turn to FreeCAD again (built twice, the first
  answer lost). FreeCAD now keeps saying it's there.
- **An export the server was too busy for is sent again**, and the browser is told if one still couldn't be uploaded.
- **Reverting past a prompt FreeCAD refused** no longer fails with "FreeCAD was restarted".
- **Stop closes an approval that is waiting**, instead of after two minutes.

## [0.2.0] - 2026-10-25

An internal release. Everything it added is in the first public release, 0.2.1.

### Added

- **A chat panel inside FreeCAD** that builds parametric parts: every dimension in one `Params` spreadsheet, with
  sketches, pads, pockets, holes and fillets bound to it. The panel shows the plan, one row per step with a snapshot,
  and a summary of what was checked and what was not. Pause, steer, stop, revert a turn or rewind to any step.
- **43 CAD tools**: documents and objects, sketches and features, structured FeaturePlans, enclosures for boards
  (Raspberry Pi, Arduino, ESP32, Jetson Nano), parameters, reference parts, validation and fit checks, FEM, web
  lookup, saving and export (STEP, STL).
- **A project folder per chat**: the model, every step and script, snapshots, notes, exports and a `rebuild.py`.
- **Any AI provider**: an OriginShape account (no API key), or your own key for Anthropic, OpenAI, Google Gemini,
  OpenRouter, Mistral, DeepSeek, Groq, xAI and more, or a local model through Ollama or LM Studio.
- **The OriginShape account**: sign in on [app.originshape.design](https://app.originshape.design) in the browser
  (the tab then shows how the sign-in ended) or with a code, credits in the panel header, and the account's models
  (listed by the server, so new ones work without a new Oscar). Sign-in always goes to app.originshape.design; to test
  against another OriginShape server, turn on Use a custom server in Settings → Account → Developer: server address.
- **The OriginShape workspace**: tasks started at app.originshape.design/studio run in this FreeCAD, with live steps,
  previews, exports, files attached in the browser, approvals answerable in the browser, models sent from the browser
  opened here, and reverts.
- **Updates**: Oscar checks once a day whether a newer version is out (Oscar → Check for Updates… at any time; off in
  Settings). When the server needs a newer Oscar, the account pauses with an "Update Oscar" card while your own
  providers keep working.
- **The pen**: draw on the 3D view to show where to change the model. Rename and search your chats.
- **Safety**: model-written Python scripts off by default and approved one by one; network access asked for; the
  OriginShape sign-in kept in the system keychain; an MCP bridge for other AI apps, off by default.

[Unreleased]: https://github.com/Originshape/Oscar-Freecad/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/Originshape/Oscar-Freecad/releases/tag/v0.2.1
[0.2.0]: https://github.com/Originshape/Oscar-Freecad/releases/tag/v0.2.0
