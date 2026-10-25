#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Run the Oscar test suite against an official FreeCAD build, in a throw-away FreeCAD user home.
#
#   scripts/test.sh fast                    # the fast set, one process
#   scripts/test.sh full [test_module ...]  # every (or the named) test module, ONE PROCESS PER MODULE
#   scripts/test.sh gui                     # GUI smoke test (needs a display; Xvfb on Linux)
#
# One process per module because a long single freecadcmd run can hit order-dependent native crashes
# in the CAD kernel; isolated modules do not. FREECADCMD / FREECAD_GUI point at the console and GUI
# binaries (defaults: /Applications/FreeCAD.app on macOS, freecadcmd / freecad on PATH elsewhere).
# Exit code: 0 when every module passed.
set -uo pipefail
MODE="${1:-}"; shift || true
case "$MODE" in fast | full | gui) ;; *) sed -n '2,12p' "$0"; exit 2 ;; esac
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
APP="${FREECAD_APP:-/Applications/FreeCAD.app}"
if [ -x "$APP/Contents/Resources/bin/freecadcmd" ]; then
  FREECADCMD="${FREECADCMD:-$APP/Contents/Resources/bin/freecadcmd}"
  FREECAD_GUI="${FREECAD_GUI:-$APP/Contents/Resources/bin/freecad}"
else
  FREECADCMD="${FREECADCMD:-$(command -v freecadcmd || command -v FreeCADCmd || true)}"
  FREECAD_GUI="${FREECAD_GUI:-$(command -v freecad || command -v FreeCAD || true)}"
fi
TESTS="$ROOT/tests"
export PYTHONUTF8=1 PYTHONDONTWRITEBYTECODE=1 LC_ALL="${OSCAR_TEST_LOCALE:-en_US.UTF-8}"

sandbox() {   # a fresh FreeCAD user home with the addon linked in; prints its path
  local scr; scr="$(mktemp -d "${TMPDIR:-/tmp}/oscar-test.XXXXXX")"
  mkdir -p "$scr/fchome/Mod" "$scr/projects" "$scr/logs" "$scr/cache" "$scr/downloads"
  ln -sfn "$ROOT" "$scr/fchome/Mod/Oscar"
  echo "$scr"
}
use() {
  export FREECAD_USER_HOME="$1/fchome" OSCAR_AI_SETTINGS="$1/settings.json" OSCAR_PROJECTS_DIR="$1/projects"
  export OSCAR_LOG_DIR="$1/logs" OSCAR_CACHE_DIR="$1/cache" OSCAR_DOWNLOAD_DIR="$1/downloads"
  # model-written scripts are off by default and asked about per call; the suite (and the GUI smoke test) runs them unasked
  export OSCAR_ALLOW_CODE=1
  # no daily "is there a newer Oscar?" request to the OriginShape server from a test run
  export OSCAR_UPDATE_CHECK=0
}
run_py() {    # run tests/run.py under freecadcmd with OSCAR_TEST_MODULES already set
  "$FREECADCMD" -c "import sys; sys.argv=['run.py']; __file__='$TESTS/run.py'; exec(open(__file__, encoding='utf-8').read())"
}

if [ "$MODE" = gui ]; then
  [ -x "$FREECAD_GUI" ] || { echo "test.sh: GUI binary not found (set FREECAD_GUI)" >&2; exit 2; }
  SCR="$(sandbox)"; use "$SCR"
  export OSCAR_SMOKE_LOG="$SCR/gui_smoke.log" OSCAR_BRIDGE_PORT="${OSCAR_BRIDGE_PORT:-19876}"
  export OSCAR_SMOKE_MOCK_PORT="${OSCAR_SMOKE_MOCK_PORT:-18765}"
  mkdir -p "$SCR/gui" && cd "$SCR/gui"
  # a portable time limit: stock macOS has no GNU `timeout`
  "$FREECAD_GUI" "$ROOT/tests/gui_smoke.py" & PID=$!
  LIMIT="${OSCAR_GUI_TIMEOUT:-300}"; WAITED=0
  while kill -0 "$PID" 2>/dev/null; do
    if [ "$WAITED" -ge "$LIMIT" ]; then
      echo "test.sh: GUI smoke test exceeded ${LIMIT}s; killing it" >&2
      kill "$PID" 2>/dev/null; sleep 2; kill -9 "$PID" 2>/dev/null; break
    fi
    sleep 1; WAITED=$((WAITED + 1))
  done
  wait "$PID"; CODE=$?
  [ -f "$OSCAR_SMOKE_LOG" ] && tail -n 40 "$OSCAR_SMOKE_LOG"
  grep -q OSCAR_SMOKE_OK "$OSCAR_SMOKE_LOG" 2>/dev/null || CODE=1
  exit "$CODE"
fi

[ -x "$FREECADCMD" ] || { echo "test.sh: freecadcmd not found (set FREECADCMD)" >&2; exit 2; }
cd "$ROOT"
if [ "$MODE" = fast ]; then
  SCR="$(sandbox)"; use "$SCR"
  export OSCAR_TEST_MODULES="fast${*:+,$(IFS=,; echo "$*")}"
  run_py; exit $?
fi

if [ $# -gt 0 ]; then MODULES=("$@"); else
  MODULES=(); for f in "$TESTS"/test_*.py; do MODULES+=("$(basename "$f" .py)"); done
fi
FAILED=(); TOTAL=0
for m in "${MODULES[@]}"; do
  SCR="$(sandbox)"; use "$SCR"
  export OSCAR_TEST_MODULES="$m"
  OUT="$(run_py 2>&1)"; CODE=$?
  # not anchored at the line start: FreeCAD's progress output (stdout, no newline) can precede it on the same line
  LINE="$(printf '%s\n' "$OUT" | grep -oE 'OSCAR_TESTS: ran [0-9]+.*' | tail -n 1)"
  N="$(printf '%s' "$LINE" | sed -E 's/.*ran ([0-9]+).*/\1/')"; TOTAL=$((TOTAL + ${N:-0}))
  printf '%-40s %s\n' "$m" "${LINE#OSCAR_TESTS: }"
  if [ "$CODE" -ne 0 ]; then FAILED+=("$m"); printf '%s\n' "$OUT" | grep -E '^(FAIL|ERROR):' | head -n 20; fi
  [ "${OSCAR_TEST_KEEP:-0}" = 1 ] || rm -rf "$SCR"
done
echo "OSCAR_TESTS_TOTAL: $TOTAL tests in ${#MODULES[@]} modules; failed modules: ${FAILED[*]:-none}"
[ ${#FAILED[@]} -eq 0 ]
