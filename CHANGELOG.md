## [Unreleased]
### Added
- **AI model settings synced with your OriginShape account**: signed in with OriginShape, whether you use the account
  or your own provider, your provider profiles (name, provider, base URL, model, reasoning, Advanced overrides)
  and the one in use are the same in FreeCAD and on the website (Settings → AI model). Oscar syncs when the panel
  opens, after you sign in, when you open its settings, after Save (and a profile picked in the panel), and before a
  chat when the last check is more than a minute old, always in the background. Each setting is merged on its own; when
  both sides changed the same one, the website's wins on a fetch and yours on a Save. **API keys never leave this
  computer** (the website only learns whether a key is saved here), nor does a base URL with a user, password, query or
  fragment in it, nor the profiles' notes. A profile made on the website, or whose base URL or provider the website
  changed, is not used (nor its key sent) until you confirm its address in Oscar's settings; a website pick of such a
  profile, or of one whose key is on another computer, doesn't switch chat until then. A profile with a key that the
  website removed stays until you remove or keep it here, and a sync that would remove more than half of your profiles
  is not applied. Keys, and pieces of keys providers echo in errors, are masked in everything sent. A setting
  OriginShape refuses is named in the settings, with the profile. `OSCAR_SETTINGS_SYNC=0` keeps the settings local.
- **Test in FreeCAD**: with Connect on, the website can ask FreeCAD to test a profile; the test runs here with the key
  saved here, on the latest settings, and only its result comes back, scrubbed of the key.
- **A mesh with holes points at the OriginShape workspace's Repair**: when `import_file` can't make a solid of an open
  mesh, the model is told that the user can repair it in the workspace (Repair, then "Use the repaired copy"), as the
  way to go for a task from the workspace and as an option for a chat in FreeCAD. A workspace attachment's
  `repaired_from` (the original's name) is read.
### Changed
- **Every provider profile has a stable id** in `oscar_settings.json`, and the profile in use is remembered by it, so a
  rename never switches profiles. Older settings files get their ids when they are read (nothing is rewritten until
  the next save), and an older Oscar can still read and save the file.

### Fixed
- **The source mesh no longer appears as a second model after a rebuild** (#5): when a finished turn
  rebuilds an imported STL/OBJ into a solid body, the mesh it traced over is hidden at the end of the
  turn (the body is the deliverable; the mesh stays in the tree as a reference and the saved model
  carries the hidden state). Only meshes Oscar itself imported are hidden, and only in the document the
  turn built the deliverable in — a scan imported for a fit check or measurement is never touched, a
  capped or unverified build keeps its mesh visible, and a later turn does not hide a mesh the user
  showed again on purpose. Reverting the turn brings the mesh back: the engine's turn snapshot now
  records mesh visibility and the revert restores it.
- **Chats are independent** (#2): a turn writes only to the chat that started it. Its transcript, plan,
  artifacts, project folder, attachments, undelivered images, unfinished composer text and turn record are
  kept per chat, and switching chats (the picker, New chat, Chat search) is allowed while a turn runs — a
  message sent in another chat then waits in that chat instead of steering the running turn.
- A queued message whose send was blocked (signed out, new Terms, no key…) no longer disappears together
  with what the user was typing: it keeps waiting in its chat and the composer is left alone.
- Picking the chat that is already shown no longer wipes the message being typed.
- A message queued in the chat on screen goes out when the panel becomes free, even when other chats
  queued messages before it.
- The pause, stop and Esc controls act on the running turn only while its chat is shown; in another chat
  the send button sends (queuing there), Esc belongs to the application again, and the running turn's
  status line, heartbeat and Act-now button do not show.
- A turn whose first request failed for an account reason gives its message back to its own chat also when
  another chat is on screen: the message returns to that chat's input (it used to stay in its history
  unanswered), and the same now holds for an undelivered steered message.
- A queued message that is still waiting when its chat is opened later asks before it is sent.

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
## [Unreleased]
### Added
- **AI model settings synced with your OriginShape account**: signed in with OriginShape, whether you use the account
  or your own provider, your provider profiles (name, provider, base URL, model, reasoning, Advanced overrides)
  and the one in use are the same in FreeCAD and on the website (Settings → AI model). Oscar syncs when the panel
  opens, after you sign in, when you open its settings, after Save (and a profile picked in the panel), and before a
  chat when the last check is more than a minute old, always in the background. Each setting is merged on its own; when
  both sides changed the same one, the website's wins on a fetch and yours on a Save. **API keys never leave this
  computer** (the website only learns whether a key is saved here), nor does a base URL with a user, password, query or
  fragment in it, nor the profiles' notes. A profile made on the website, or whose base URL or provider the website
  changed, is not used (nor its key sent) until you confirm its address in Oscar's settings; a website pick of such a
  profile, or of one whose key is on another computer, doesn't switch chat until then. A profile with a key that the
  website removed stays until you remove or keep it here, and a sync that would remove more than half of your profiles
  is not applied. Keys, and pieces of keys providers echo in errors, are masked in everything sent. A setting
  OriginShape refuses is named in the settings, with the profile. `OSCAR_SETTINGS_SYNC=0` keeps the settings local.
- **Test in FreeCAD**: with Connect on, the website can ask FreeCAD to test a profile; the test runs here with the key
  saved here, on the latest settings, and only its result comes back, scrubbed of the key.
- **A mesh with holes points at the OriginShape workspace's Repair**: when `import_file` can't make a solid of an open
  mesh, the model is told that the user can repair it in the workspace (Repair, then "Use the repaired copy"), as the
  way to go for a task from the workspace and as an option for a chat in FreeCAD. A workspace attachment's
  `repaired_from` (the original's name) is read.
### Changed
- **Every provider profile has a stable id** in `oscar_settings.json`, and the profile in use is remembered by it, so a
  rename never switches profiles. Older settings files get their ids when they are read (nothing is rewritten until
  the next save), and an older Oscar can still read and save the file.

The reconstruction-accuracy stack (Issues 003 + the reconstruction backlog): mesh-to-
parametric-CAD with honest evidence at every stage.
### Added
- Noise-adaptive recognition (K1): sigma from leave-one-out plane residuals; every gate
  derives from the measured noise; scan-quality report (G5) with completeness
  (outer-rim / hole-opening / dropout classification, through-pairing), integrity
  (doubled surfaces via exact raycasts, spikes with isolation guards) and plausibility
  (mm/inch, minimum detectable hole radius) gating the reconstruction outcome.
- Validated repair (K2/G6): copy-based repair with per-operation trial validation; the
  loop-aware fill policy fills only tiny flat pinholes after authorization, records
  `missing_regions`, and never closes a genuine opening.
- Provenance ledger (K3): original/modified/synthetic/removed categories, coordinate-
  keyed and verified; analysis evidence files; the accuracy report `ACCURACY.md` (V-4).
- Units (K4): 3MF source-unit recovery, mm/inch grid evidence, user-confirmed
  conversions on a copy.
- Primitive fitting (G2/G12): sphere/cone/torus LM fitters with nested F-test gates;
  region absorb/merge/upgrade. Fillet/chamfer recognition (G7) feeding measured dress
  ops. Nominal dimension snapping (G8) with the 2-sigma evidence band and drill-size
  suggestions (never forced on hole diameters).
- Routes: prismatic (R-2 section outlines, R-3 blind/counterbore/countersink holes) and
  rotational (R-4: revolution profiles from recognized coaxial cylinders). Datums and
  symmetry proposals (G10) in analysis.
- Validation: exact worst-point deviation and signed missing/excess clusters (V-1);
  export re-import equality verdicts (V-2); the headless rebuild check now produces the
  `rebuild` validation id (V-3).
- The mesh workflow router (W-1), the measured-only deviation mask (G6), benchmark
  corpus and runner (B-1, `scripts/benchmark_recon.py`), plan macro export (S8), and
  the 100k-facet performance tier check (S8).
- CI runs the reconstruction suites on FreeCAD 1.1.3 and 1.1.4 (K5).
### Changed
- The source mesh is hidden at finalize only after a VERIFIED outcome (W-2) and shown
  again when an outcome loses verification; it is never deleted.
- `step_collision` removed from the validation catalog (V-3): its evidence is the
  reference interference rows under `fit.interference`.
### Fixed
- Interactive decisions answered from the question bar actually reach the waiting agent
  (the "answer accepted by the UI, then denied after 120 s" bug): a QuestionsBar click for
  an open netguard question (permission / mesh-repair approval) now settles that question's
  waiting worker through the same path a card click takes, instead of silently becoming a
  queued chat message while the deadline denied the request. Bar answers are routed by
  tracked question ownership, are consumed exactly once, are refused for another chat's
  question, and report "no longer open" when the deadline already closed the request —
  they are never sent into the chat as if accepted. Typed bar answers for choice-only
  questions are refused the same way, and a timed-out decision card shows its real
  timeout instead of the renderer's 120 s default.
- The mesh-intent router is wired into the turn pipeline (W-1): any scan request runs
  the doctrine regardless of phrasing.
- Filleted parts reconstruct reliably: the section loop fitter's smoothing no longer
  drags corners inward (the root cause of the +7 % padded outline) — corner vertices
  keep their raw positions.
- Dome caps reconstruct with measured meridian arcs instead of overfilling staircase
  approximations; cross-holes are drilled as native PartDesign features on rotational
  parts; flat scan dropouts are bypassed without filling under an evidence-aware gate
  (never verified, always masked and reported), while one-sided scans, unseen openings
  and curved dropouts keep honest per-loop refusals.
- The GUI smoke test writes its log as UTF-8 (Windows cp1252 crashed it) and its
  thought-step check matches its name; 69/69 checks pass on the real GUI.

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
