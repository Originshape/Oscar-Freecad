# Init.py — runs in BOTH console (freecadcmd) and GUI sessions.
#
# Deliberately empty: the tool runtime imports FreeCADGui and is only
# meaningful in the GUI. Keeping console startup import-free lets headless
# tests and freecadcmd scripts load this Mod without side effects.
