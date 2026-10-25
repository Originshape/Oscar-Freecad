# Changelog

All notable changes to Oscar for FreeCAD. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and versions follow [Semantic Versioning](https://semver.org/): the version is in `package.xml` and
`oscar/__init__.py`, and each release is a `v<version>` tag with a ZIP attached.

## [Unreleased]

## [0.2.0] - 2026-10-25

The first public release.

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

[Unreleased]: https://github.com/originshape/oscar-freecad/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/originshape/oscar-freecad/releases/tag/v0.2.0
