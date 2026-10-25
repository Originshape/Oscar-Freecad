"""oscar.project_paths and its use in the chat: a path inside an Oscar project is said as "Oscar project[/rest]",
never with the full folder. Covered: redaction for the workspace (project paths, Windows slashes, other paths under
home become ``~``, nothing else touched) and the panel's markdown, where the same paths become a link to the real
folder, inside backticks too, while other paths keep their own link.
Pure python: no Qt and no FreeCAD.
"""
import os
import sys
import unittest
import unittest.mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))   # oscar
sys.path.insert(0, os.path.dirname(__file__))                        # tests/support
from support import isolation  # noqa: E402

isolation.isolate_process()   # temp settings / projects / logs / cache / downloads BEFORE importing oscar (HOME is NOT touched)

from oscar import panel_render, project_paths  # noqa: E402

ROOT = "/Users/ada/Documents/Oscar/Projects"
REPLY = ("I exported the shell and lid to `/Users/ada/Documents/Oscar/Projects/workspace-hello-3aad94/exports` "
         "(`picase-shell-2`).\nproject: /Users/ada/Documents/Oscar/Projects/workspace-hello-3aad94.")


class RedactTests(unittest.TestCase):
    def test_project_paths_become_oscar_project(self):
        text = project_paths.redact(REPLY, roots=[ROOT], home="/Users/ada")
        self.assertEqual(text, "I exported the shell and lid to `Oscar project/exports` (`picase-shell-2`).\n"
                               "project: Oscar project.")

    def test_windows_paths_and_other_home_paths(self):
        text = project_paths.redact(
            r"C:\Users\ada\Documents\Oscar\Projects\bracket-1\exports\a.stl and /Users/ada/Desktop/ref.step "
            "and /opt/data/x.step",
            roots=[ROOT, r"C:\Users\ada\Documents\Oscar\Projects"], home="/Users/ada")
        self.assertEqual(text, "Oscar project/exports/a.stl and ~/Desktop/ref.step and /opt/data/x.step")

    def test_a_folder_that_only_starts_like_home_is_left_alone(self):
        self.assertEqual(project_paths.redact("/Users/adam/x.step", roots=[ROOT], home="/Users/ada"),
                         "/Users/adam/x.step")
        self.assertEqual(project_paths.redact("", roots=[ROOT]), "")


class PanelLinkTests(unittest.TestCase):
    def setUp(self):
        panel_render.set_project_paths(project_paths.project_path_re([ROOT]))
        self.addCleanup(panel_render.set_project_paths, None)

    def test_a_project_path_is_a_link_named_oscar_project(self):
        html = panel_render.md_to_html(REPLY)
        self.assertNotIn("workspace-hello-3aad94</a>", html)
        self.assertIn(">Oscar project/exports</a>", html)
        self.assertIn(f"href='file://{ROOT}/workspace-hello-3aad94/exports'", html)
        self.assertIn(f"href='file://{ROOT}/workspace-hello-3aad94'>Oscar project</a>.", html)   # the period stays out
        self.assertIn("<code", html)   # other code spans are still code

    def test_install_uses_the_configured_roots(self):
        with unittest.mock.patch.object(project_paths, "project_roots", return_value=["/srv/oscar"]):
            project_paths.install()
        self.assertIn(">Oscar project/a.step</a>", panel_render.md_to_html("see /srv/oscar/chat-1/a.step"))

    def test_other_paths_keep_their_own_link(self):
        html = panel_render.md_to_html("See /opt/data/ref.step")
        self.assertIn(">/opt/data/ref.step</a>", html)


if __name__ == "__main__":
    unittest.main()
