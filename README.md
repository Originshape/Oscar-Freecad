# Oscar for FreeCAD

**Make CAD from a prompt, inside FreeCAD.** Describe the part you need in plain words, and Oscar builds it for you
as a real, editable FreeCAD model, step by step, while you watch.

```text
You:   Make an 80 mm flange, 8 mm thick, with six M6 bolt holes on a 60 mm circle.
Oscar: plans the part → creates a parameter spreadsheet → sketches → pads → drills the holes
       → checks the model → saves the project and shows you what it validated
```

> **Version 0.2.1** for FreeCAD 1.1. See [CHANGELOG.md](CHANGELOG.md), and read [Safety](#safety) before using it on
> real work.

**Contents:** [What Oscar does](#what-oscar-does) · [Install](#install) · [Connect an AI model](#connect-an-ai-model) ·
[Your first part](#your-first-part) · [Updates](#updates) · [Safety](#safety) · [Troubleshooting](#troubleshooting) ·
[Uninstall](#uninstall) · [Develop](#develop)

---

## What Oscar does

- **Builds parametric parts from a chat.** Every dimension goes into one parameter spreadsheet (`Params`, or
  `Parameters` for a part built from a plan), and the features (sketches, pads, pockets, holes, fillets) are bound
  to it. Change a cell and the part follows.
- **Shows its work.** The panel shows the plan, one row per step with a snapshot, and a summary at the end that
  says what was checked and what was not.
- **Lets you stay in control.** Pause, steer (type while it works), stop, revert a whole turn, or rewind to any
  earlier step.
- **Lets you point.** The pen ✏️ next to the paperclip lets you draw on the 3D view where Oscar should change the
  model; the picture and a line naming the part, side and region go with your message.
- **Keeps your chats findable.** Rename a chat or search all of them (titles, your prompts, Oscar's answers) from
  the panel's ⋯ menu.
- **Checks the result.** It runs a model health check after every change, and validation and fit checks before
  it says it is done.
- **Keeps a project folder per chat.** The model, every step and script, snapshots, notes, exports (STEP, STL) and
  a `rebuild.py` that rebuilds the part from scratch.
- **Has builders for common jobs.** Plates, flanges, brackets and fixtures; enclosures for boards such as the
  Raspberry Pi, Arduino, ESP32 and Jetson Nano, fitted around the board's real dimensions.
- **Works with your choice of AI.** Sign in with an OriginShape account and chat with no API key of your own (the
  account pays for the model with credits). Or bring your own provider: Anthropic, OpenAI, Google Gemini,
  OpenRouter, Mistral, DeepSeek, Groq, xAI and more, or a model on your own computer through Ollama or LM Studio,
  with no API key and no data leaving your machine.

Oscar has 43 tools in all, covering documents and objects, sketches and features, structured parts, enclosures
and parameters, reference parts, validation, FEM, web lookup, saving and export.

## Install

You need **FreeCAD 1.1 or newer**, the official build from [freecad.org](https://www.freecad.org/downloads.php).
Nothing else is required: Oscar is pure Python and uses only what FreeCAD already ships.

### Option A: from FreeCAD's Addon Manager

1. Open **Edit → Preferences → Addon Manager**.
2. Under **Custom repositories**, add `https://github.com/Originshape/Oscar-Freecad` with branch `main`.
3. Open the Addon Manager (**Tools → Addon manager**), find **Oscar**, click **Install**.
4. Restart FreeCAD.

### Option B: from a release ZIP

1. Download `Oscar-v<version>.zip` from the [latest release](https://github.com/Originshape/Oscar-Freecad/releases/latest)
   (or the repository as a ZIP) and unzip it. To check the download, compare its SHA-256 with the one in
   `Oscar-v<version>.zip.sha256` on the same release: `shasum -a 256 Oscar-v<version>.zip` on macOS or Linux,
   `certutil -hashfile Oscar-v<version>.zip SHA256` on Windows.
2. Rename the folder to `Oscar`. It must directly contain `package.xml` and `InitGui.py`.
3. Move the `Oscar` folder into FreeCAD's `Mod` folder:

   | System | `Mod` folder |
   |---|---|
   | macOS | `~/Library/Application Support/FreeCAD/v1-1/Mod/` |
   | Windows | `%APPDATA%\FreeCAD\v1-1\Mod\` |
   | Linux | `~/.local/share/FreeCAD/v1-1/Mod/` |

   Create the `Mod` folder if it does not exist. FreeCAD can tell you the exact location: in its Python console
   type `FreeCAD.getUserAppDataDir()`.
4. Restart FreeCAD.

### Option C: for development (a live link to your checkout)

```bash
git clone https://github.com/Originshape/Oscar-Freecad
cd Oscar-Freecad
python3 scripts/dev_install.py      # links this folder into FreeCAD's Mod folder; --uninstall removes the link
```

### Check that it worked

Open the workbench list (the drop-down in the toolbar) and choose **Oscar**, or press **Ctrl+Shift+O** anywhere
(**⇧⌘O** on macOS).
The Oscar panel opens on the right.

## Connect an AI model

### With an OriginShape account (no API key)

Create an account at [app.originshape.design](https://app.originshape.design), then:

1. In the Oscar panel, click **Sign in** in the header (or **⚙ → OriginShape account**, or **Oscar → Sign in to
   OriginShape…**).
2. Choose how:
   - **Sign in with OriginShape**: your browser opens the sign-in page on app.originshape.design; allow *Oscar for
     FreeCAD* there and the browser hands the result back to Oscar, then shows
     how the sign-in ended on app.originshape.design. You can close that tab.
   - **Sign in with a code**: Oscar shows a short code such as `BCDF-GHJK`; enter it on the OriginShape website, on
     any device. Use this when the browser cannot reach FreeCAD.
   - **Use an API key**: paste a personal OriginShape API key (`os_live_…`, created on the website).
3. The header badge then shows your credits (for example *42 credits*); its menu has **Buy credits…**, **Manage
   account…**, **Refresh** and **Sign out**. Signing in makes the **OriginShape** profile the active one; pick its
   model in Settings → AI model, from the models your account offers (such as **Origin1**, recommended for most
   work, and **Origin1 Super**, the most capable model for CAD work), each with a line on what it is for.

Every request goes to OriginShape's server, which forwards it to the AI model and counts the credits it costs. A new
account may start on the waitlist: Oscar then shows a waitlist card instead of starting a chat turn, and you can use
your own provider meanwhile. When the credits run out, Oscar shows a card with **Buy credits**; when that happens
before the turn did anything, your message goes back in the input.

### Run tasks from the OriginShape workspace

The workspace on the OriginShape website ([app.originshape.design/studio](https://app.originshape.design/studio)) is a chat with Oscar next to a 3D view of the part. The part is
built here, in your FreeCAD:

1. Sign in with your OriginShape account (above).
2. In **⚙ → OriginShape account**, turn on **Connect to the OriginShape workspace**. The line under it shows the
   connection (*Connected: tasks from the workspace run here*).
3. Start a task in the workspace. Oscar picks it up within a few seconds and runs it as a chat named
   *Workspace: <task>* in this panel. The website shows its steps and reply as they happen, a preview of the model
   after each change, and the STEP, STL and FreeCAD files when it finishes. **Stop** in the browser stops the turn.
4. Files attached in the workspace (photos, sketches, PDF drawings, STEP, IGES, STL, DXF and other CAD files, text
   notes) are downloaded to a `workspace-files` folder next to Oscar's chats and sent with the message, as if you had
   attached them here. Oscar checks each file's checksum before using it. A mesh (STL, OBJ, 3MF, PLY) is imported as
   a solid when the design changes it, so Oscar can cut, fillet and add to it. Downloads older than 30 days are
   removed.

Oscar never takes a workspace task while it is busy with one of yours; the task waits. Everything a workspace task
does stays behind the same approvals as your own chats: scripts and network access ask **here**, at this computer.
**Approve scripts and network access from the workspace** is on by default once you connect: the website then
offers the same question, with the script to read first, and an answer there is clicked in this dialog (which stays
answerable here; *Always allow* is only ever offered here). With it on, anyone signed in to your OriginShape account
can run code on this computer through Oscar, so turn it off if others use your account or you want to answer only
at this computer. Turn **Connect to the OriginShape workspace** off to stop taking tasks.

**Revert from the workspace.** In the workspace, a prompt's **Revert** (or ⌘Z / Ctrl+Z for the latest one) takes the
task back to just before that prompt. Oscar undoes those turns here, newest first, the way **revert changes** on a
turn card does (FreeCAD's undo, else the checkpoint taken before the turn), and reports back. It asks nothing here
(the website asked), and it stops instead of throwing anything away: a turn you typed in FreeCAD after those prompts
has to be reverted here first, a document you changed by hand after the turn is left alone, and turns from before
FreeCAD was last restarted can only be undone here (Edit > Undo, or reopen the file).

### With your own provider

1. Click **⚙** and open **Advanced: use your own provider** (it is open already when you use one).
2. Click **Add**, choose a **Provider** from the list, and pick or type a **Model**.
3. Paste your **API key** (not needed for local models).
4. Click **Test connection** to check it, then **Save**.

| Provider | What you need |
|---|---|
| **OriginShape (your account)** | A sign-in (above). No key, no server address |
| Anthropic (Claude), OpenAI, Google Gemini, OpenRouter, xAI (Grok), DeepSeek, Groq, Mistral, Together, Fireworks, Cerebras, Perplexity, Z.ai, Qwen | An API key from that provider |
| **Ollama (local)** | [Ollama](https://ollama.com) running, and a model pulled, for example `ollama pull qwen2.5:7b`. No key; nothing leaves your machine |
| **LM Studio (local)** | [LM Studio](https://lmstudio.ai) with its local server started. No key |
| Custom | Any OpenAI-compatible endpoint: its base URL, model name and key |

You can keep several profiles (say, your account, a cloud model and a local one) and switch between them in the panel
header. Requests with your own provider go straight to it: they never pass through OriginShape and cost no credits.
Larger models give better results; small local models work best on simple parts.

## Your first part

1. Open the Oscar panel.
2. Type a request and press **Enter**, for example:
   - *"Make an 80 mm flange, 8 mm thick, six M6 bolt holes on a 60 mm circle."*
   - *"Design a mounting bracket: 60 x 40 x 4 mm base plate with four M4 holes, a 30 mm upright, 3 mm fillets."*
   - *"Build a snap-fit enclosure for a Raspberry Pi 4 in ABS, with cutouts for every port and vent slots."*

   Or click **📋** in the input row for ready-made templates (a parametric part, a mounting bracket, a board
   enclosure).
3. Watch it build. Each step appears as a row; click a row to see what it did, or **↶ rewind** to go back to it.
4. When it finishes, read the summary: what was built, what was validated, what was not, and anything it needs
   from you.
5. Change a dimension later by asking ("make the plate 6 mm thick") or by editing the parameter spreadsheet
   (`Params` or `Parameters`) yourself.

While Oscar works, the send button becomes **⏸** (pause) and **Esc** stops the turn. Type a message while it
works to steer it.

**Where your work goes:** every chat that builds something gets a folder under `~/Documents/Oscar/Projects/`, with
the model (`model.FCStd`), every step and script, snapshots, `NOTES.md`, exports and `rebuild.py`.

## Updates

Oscar asks the OriginShape server once a day whether a newer version is out (only the version number comes back;
nothing about you or your work is sent). When one is, the panel header says so, with a link to the download page.
**Oscar → Check for Updates…** asks at any time, and **⚙ → Tools & permissions → Updates** turns the daily check
off.

To update: in the **Addon Manager**, select **Oscar** and click **Update** (an install from the Addon Manager), or
replace the `Mod/Oscar` folder with the new release ZIP (a ZIP install); then restart FreeCAD. Your chats, projects and settings
stay. If the server needs a newer Oscar than yours (after a change in how they talk), the OriginShape account pauses
with an **Update Oscar** card until you update; your own AI providers keep working meanwhile.

## Safety

- **Python scripts are off by default.** Oscar builds with its CAD tools. If you switch on **⚙ → Scripts → Let
  Oscar run Python scripts it writes**, the model may also write scripts, and Oscar shows you each one and asks
  before it runs (with your user rights). Oscar blocks network access from those scripts unless you allow it, but
  that block is a safeguard, not a sandbox, and file access is not restricted. Use a provider and model you trust,
  save your work, and keep backups.
- **What is sent to the AI provider:** your messages, the results of Oscar's tools and small images of the 3D view.
  With an OriginShape account this goes to OriginShape's server, which forwards it to the AI model. Use Ollama or LM
  Studio to keep everything on your computer.
- **API keys** of your own providers are stored in plain text in `oscar_settings.json` in your FreeCAD user folder,
  and in its backups next to it (`oscar_settings.json.bak`, and `oscar_settings.json.corrupt-*` after a damaged file
  was replaced).
- **Your OriginShape sign-in** is kept in the system keychain (macOS Keychain, Windows Credential Manager, the
  Secret Service on Linux). Where none can be used, it is kept in `oscar_credentials.json` next to the settings file,
  readable by your user account only, and Oscar tells you so once. **Sign out** revokes this computer's sign-in
  (a personal API key is only removed from this computer: revoke it on the website).
- **Network access** (downloading a board's STEP model, reading a datasheet) is asked for first, once per turn or
  always for one tool and site. Saved permissions can be removed in **⚙ → Network permissions**.
- **The daily update check** asks the OriginShape server for the latest version number, with nothing about you
  or your work; turn it off in **⚙ → Tools & permissions → Updates**.
- **MCP bridge** (the **Toggle Oscar MCP bridge** button) lets other AI apps on your computer drive FreeCAD. It is
  off by default, listens only on `127.0.0.1:9876`, and any local program can use it while it is on.

To report a vulnerability, see [SECURITY.md](SECURITY.md).

## Troubleshooting

| Problem | What to do |
|---|---|
| No **Oscar** workbench after a restart | For a ZIP install, check that the folder is `Mod/Oscar` and contains `package.xml` directly (not `Mod/Oscar/Oscar-Freecad-main/...`); the Addon Manager names its folder itself. The Report view (**View → Panels → Report view**) shows load errors |
| "No API key" or an HTTP 401 error | Open **⚙**, select your profile, paste the key, click **Test connection**. With the OriginShape profile, sign in instead (the header badge) |
| "Your OriginShape sign-in has ended" | Sign in again from the card or the header badge. Signing out elsewhere, or revoking this computer on the website, ends it |
| A waitlist or "no credits left" card | The account is waiting for approval, or has spent its credits: **Check again**, or **Choose a plan** / **Buy credits** on the card (credit packs are for subscribers). Your own provider works meanwhile |
| "This chat is too long for the credits left" | Each request sets credits aside for the whole chat first. Start a new chat, or add credits |
| A local model is slow or stops early | Use a bigger model if your computer allows it, or simplify the request. Check that Ollama or LM Studio is running |
| Oscar asks for network access | It wants to download a reference file or read a web page. Allow it once, or deny it and it continues without |
| Something went wrong in a turn | Click **↩ revert changes** on the turn's card. The full log is `Oscar/logs/agent.log` in your FreeCAD user folder |

## Uninstall

Sign out first (**⚙ → OriginShape account → Sign out**) if you signed in. Then use the Addon Manager's
**Uninstall**, or delete the `Mod/Oscar` folder of a ZIP install (or run `python3 scripts/dev_install.py --uninstall` for a
development link). Your projects in `~/Documents/Oscar/Projects/` and your settings (`oscar_settings.json`) are kept
until you delete them.

## Develop

```bash
scripts/launch.sh             # FreeCAD with this checkout, in a throw-away profile
scripts/test.sh fast          # quick tests against your installed FreeCAD
scripts/test.sh full          # the whole suite, one FreeCAD process per test module
```

[DEVELOPMENT.md](DEVELOPMENT.md) explains how Oscar is built, how to add a tool or an AI provider, the project
folder, the settings, the logs and the tests. [CONTRIBUTING.md](CONTRIBUTING.md) has the rules for pull requests.

## License

Apache-2.0. The CAD engine in `oscar/cad/` is derived from [freecad-mcp](https://github.com/neka-nat/freecad-mcp)
(MIT); see [NOTICE](NOTICE). Oscar runs inside FreeCAD (LGPL-2.1-or-later) and does not include or modify it.
