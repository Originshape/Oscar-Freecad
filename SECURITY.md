# Security

Report vulnerabilities privately to hello@originshape.design (or through a GitHub private security advisory),
not in public issues. We reply within five working days.

What Oscar can do on your machine, and how it is limited:

- **Python scripts written by the model are off by default.** `execute_code`, `execute_code_async` and
  `execute_code_headless` run scripts the model writes, with your user rights, and file access from those scripts is
  not restricted. They are not offered to the model until you turn on *Let Oscar run Python scripts it writes* in
  Settings > Scripts. Even then Oscar shows you each script before it runs (*Run*, *Run all scripts this turn* or
  *Don't run*); nobody can answer that dialog in a console session, so scripts are refused there. Read a script
  before you run it, and keep backups of important files. `OSCAR_ALLOW_CODE=1` runs scripts without asking; it
  exists for tests and unattended runs only.
- **Network access is asked for.** Oscar's download and web tools, and scripts it runs, may reach the network only
  after you allow it (once for the turn, or always for one tool and host). Saved permissions can be removed in
  Settings. For scripts the check also covers threads a script starts, callbacks it leaves behind, host name lookups,
  other programs it starts, and the separate process of `execute_code_headless`; imports that could get around the
  check (computed module names, Qt networking, `ctypes`, `multiprocessing`) are asked about before the script runs.
  This is a safeguard, not a sandbox: code written to get around it can.
- **What leaves your machine:** your prompts, the tool results and small view images go to the model provider you
  configure (with an OriginShape account: OriginShape's server, which forwards them to the AI model). Choose Ollama,
  LM Studio or another local endpoint to keep everything on your machine. Once a day Oscar also asks the OriginShape
  server for the latest Oscar version number (`GET /v1/plugin`, nothing about you is sent); Settings > Updates, or
  `OSCAR_UPDATE_CHECK=0`, turns that off.
- **API keys** of your own providers are stored in plain text in Oscar's settings file (`oscar_settings.json` in your
  FreeCAD user folder) and in its backups next to it (`oscar_settings.json.bak`, and `oscar_settings.json.corrupt-*`
  after a damaged file was replaced).
- **Your OriginShape sign-in** (a refresh token or a personal API key) is kept in the system keychain (macOS Keychain,
  Windows Credential Manager, the Secret Service on Linux), else in `oscar_credentials.json` next to the settings file,
  readable by your user account only. It is only ever sent to the OriginShape server set in Oscar's settings (https,
  or plain http to this computer only). *Sign out* revokes this computer's sign-in; revoke a personal API key on the
  website.
- **The MCP bridge is off by default.** When on, it listens on 127.0.0.1 only and refuses browser requests, but any
  local program can then drive FreeCAD through it, including running Python. Scripts sent through the bridge do not
  go through Oscar's script switch or approval dialog (your MCP client's own approval applies). Turn it on only when
  you need it.
