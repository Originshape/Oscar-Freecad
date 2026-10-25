#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Launch FreeCAD with this checkout of Oscar in a throw-away profile (your normal FreeCAD profile,
# preferences and addons are untouched).
#
#   scripts/launch.sh                            # a fresh profile in .devhome/
#   OSCAR_DEV_HOME=~/oscar-dev scripts/launch.sh  # keep the profile somewhere else between runs
#
# FREECAD points at the GUI binary; the defaults cover the official macOS, Linux AppImage and Windows builds.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ -z "${FREECAD:-}" ]; then
  for c in /Applications/FreeCAD.app/Contents/MacOS/FreeCAD "$(command -v freecad || true)" "$(command -v FreeCAD || true)"; do
    if [ -n "$c" ] && [ -x "$c" ]; then FREECAD="$c"; break; fi
  done
fi
[ -n "${FREECAD:-}" ] || { echo "launch.sh: set FREECAD to the FreeCAD GUI binary" >&2; exit 2; }
HOME_DIR="${OSCAR_DEV_HOME:-$ROOT/.devhome}"
mkdir -p "$HOME_DIR/Mod"
ln -sfn "$ROOT" "$HOME_DIR/Mod/Oscar"
export FREECAD_USER_HOME="$HOME_DIR"
exec "$FREECAD" "$@"
