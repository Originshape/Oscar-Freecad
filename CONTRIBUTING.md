# Contributing

Thanks for helping. Oscar is a FreeCAD copilot: it talks to an AI provider and builds parametric CAD models from a
chat. Changes that make that path more reliable are the most welcome: the Build Flow, the tools, the part builders,
the providers and the panel.

Two rules:

1. **Sign off every commit** (`git commit -s`). It certifies the
   [Developer Certificate of Origin](https://developercertificate.org/): you wrote the change or have the right to
   submit it under this project's license (Apache-2.0).
2. **Run the tests** before opening a pull request: `scripts/test.sh fast` at least, and `scripts/test.sh full` for
   changes to tools, the CAD engine or the panel. [DEVELOPMENT.md](DEVELOPMENT.md) explains the suites and how to
   add a tool or a provider.

Also:

- Put an SPDX line (`# SPDX-License-Identifier: Apache-2.0`) at the top of new source files.
- Never commit API keys, customer files or models you may not share.
- The `.gitignore` keeps documents out of the repository except the top-level `README.md`, `DEVELOPMENT.md`,
  `CONTRIBUTING.md`, `SECURITY.md` and `CHANGELOG.md`.
