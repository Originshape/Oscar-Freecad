#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Link this checkout into FreeCAD's user Mod folder, so FreeCAD loads Oscar from your working copy.

    python3 scripts/dev_install.py                 # link as Mod/Oscar
    python3 scripts/dev_install.py --uninstall     # remove the link

FreeCAD itself is asked where its user data folder is (it is versioned since 1.1, e.g.
~/Library/Application Support/FreeCAD/v1-1 on macOS, ~/.local/share/FreeCAD/v1-1 on Linux,
%APPDATA%\\FreeCAD\\v1-1 on Windows), so the link always lands where that FreeCAD looks.
Set FREECAD_USER_HOME to install into an isolated profile instead.
"""
import argparse
import os
import shutil
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANDIDATES = [
    os.environ.get("FREECADCMD", ""),
    "/Applications/FreeCAD.app/Contents/Resources/bin/freecadcmd",
    r"C:\Program Files\FreeCAD 1.1\bin\freecadcmd.exe",
    "freecadcmd", "FreeCADCmd",
]


def find_freecadcmd(explicit: str) -> str:
    for c in [explicit] + CANDIDATES:
        if c and (os.path.isfile(c) or shutil.which(c)):
            return shutil.which(c) or c
    sys.exit("dev_install: freecadcmd not found; pass --freecadcmd or set FREECADCMD")


def user_data_dir(freecadcmd: str) -> str:
    out = subprocess.run([freecadcmd, "-c", "import FreeCAD; print('USERDIR=' + FreeCAD.getUserAppDataDir())"],
                         capture_output=True, text=True, timeout=120)
    for line in (out.stdout or "").splitlines():
        if line.startswith("USERDIR="):
            return line[len("USERDIR="):].strip()
    sys.exit(f"dev_install: could not ask FreeCAD for its user folder:\n{out.stdout}\n{out.stderr}")


def is_link(path: str) -> bool:
    return os.path.islink(path) or bool(getattr(os.path, "isjunction", lambda _p: False)(path))


def unlink(path: str) -> None:
    os.unlink(path) if os.path.islink(path) else os.rmdir(path)  # rmdir removes a junction, never its target


def link(src: str, dst: str, force: bool) -> None:
    if os.path.lexists(dst):
        if is_link(dst):
            unlink(dst)
        elif force:
            backup = dst + ".bak"
            shutil.move(dst, backup)
            print(f"moved the existing {dst} to {backup}")
        else:
            sys.exit(f"dev_install: {dst} exists and is not a link; remove it or pass --force")
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.name == "nt":
        subprocess.run(["cmd", "/c", "mklink", "/J", dst, src], check=True)  # junction: no admin rights needed
    else:
        os.symlink(src, dst)
    print(f"linked {dst} -> {src}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--freecadcmd", default="")
    ap.add_argument("--name", default="Oscar")
    ap.add_argument("--uninstall", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    mod = os.path.join(user_data_dir(find_freecadcmd(a.freecadcmd)), "Mod")
    targets = [(ROOT, os.path.join(mod, a.name))]
    for src, dst in targets:
        if a.uninstall:
            if is_link(dst):
                unlink(dst)
                print(f"removed {dst}")
        else:
            link(src, dst, a.force)
    if not a.uninstall:
        print("Restart FreeCAD, then pick the Oscar workbench (or press Ctrl+Shift+O).")


if __name__ == "__main__":
    main()
