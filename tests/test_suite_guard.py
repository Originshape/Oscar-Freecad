"""Suite guard: a test that is written but never collected is worse than no test.

Pure source analysis (``ast`` and file names), so it runs in the fast CI job without FreeCAD. It fails when a file
in ``tests/`` would silently run nothing:

* ``NO_TESTS``            a ``test_*.py`` file with no collectable test method;
* ``HIDDEN_TEST``         a ``test_*`` method in a class that is not a ``TestCase`` (never run);
* ``MODULE_LEVEL_TEST``   a ``def test_*`` at module level (unittest does not collect functions);
* ``DUPLICATE_NAME``      two methods of one class with the same name (the first is silently shadowed);
* ``NOT_DISCOVERED``      test classes in a file that discovery (``test_*.py``, package dirs only) will not pick up;
* ``NO_ISOLATION``        a test module that imports ``oscar`` before calling ``isolation.isolate_process()``
                          (the four legacy files below are grandfathered);
and, with FreeCAD importable, that real discovery collects at least as many tests as the source declares.
A ``test_*.py`` file that is a helper module on purpose (shared fixtures) says so in its docstring ("no test case
on purpose") and is exempt from ``NO_TESTS``. A second guard is a pyflakes-lite check for names that are used but
defined nowhere (``oscar/``, ``tests/``), built on ``symtable``: it catches the NameError that only a rarely taken
branch would raise, without a linter dependency. The detectors are tested against synthetic trees too, so the guard
cannot rot into a test that never fails.
"""

import ast
import builtins
import os
import re
import symtable
import sys
import tempfile
import textwrap
import unittest

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
MOD_DIR = os.path.dirname(TESTS_DIR)
REPO_ROOT = os.path.abspath(os.path.join(TESTS_DIR, *[".."] * 6))

# Written before the isolation header existed; they run isolated through the OSCAR_* variables of
# scripts/test.sh and the CI environment. NEW files must have the header (their owners may migrate these).
LEGACY_UNISOLATED = frozenset({"test_attachments.py", "test_oscar_regressions.py",
                               "test_tools_and_params.py"})

_TC_BASES = {"TestCase", "IsolatedAsyncioTestCase"}
_SKIP_DIRS = {"__pycache__", "fixtures"}


def _base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def analyse(path):
    """``{"classes": {name: {"tc": bool, "tests": [names], "bases": [names]}}, "module_tests": [...], "dups": [...]}``
    for one source file (a syntax error raises ``SyntaxError`` with the file name)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=path)
    classes = {}
    dups = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            names = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            tests = [n for n in names if n.startswith("test")]
            for n in {x for x in tests if tests.count(x) > 1}:
                dups.append(f"{node.name}.{n}")
            classes[node.name] = {"bases": [_base_name(b) for b in node.bases], "tests": sorted(set(tests)), "tc": False}
    changed = True
    while changed:                                  # TestCase-ness flows down through in-file inheritance
        changed = False
        for name, info in classes.items():
            if info["tc"]:
                continue
            for base in info["bases"]:
                imported_case = base not in classes and re.search(r"(Case|Test|Tests)$", base or "") is not None
                if base in _TC_BASES or imported_case or (base in classes and classes[base]["tc"]):
                    info["tc"] = True
                    changed = True
                    break
    module_tests = [n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and n.name.startswith("test")]
    helper = "no test case on purpose" in (ast.get_docstring(tree) or "").lower()
    first_import, isolate_call = None, None
    for node in tree.body:
        if first_import is None and isinstance(node, ast.Import) and any(a.name.split(".")[0] == "oscar" for a in node.names):
            first_import = node.lineno
        if first_import is None and isinstance(node, ast.ImportFrom) and (node.module or "").split(".")[0] == "oscar" \
                and node.level == 0:
            first_import = node.lineno
        if isolate_call is None and isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) \
                and _base_name(node.value.func) == "isolate_process":
            isolate_call = node.lineno
    return {"classes": classes, "module_tests": module_tests, "dups": dups, "helper": helper,
            "oscar_import": first_import, "isolate_call": isolate_call}


def expected_test_count(info):
    """Tests the file declares in TestCase classes (inherited ones from imported bases are not counted)."""
    return sum(len(c["tests"]) for c in info["classes"].values() if c["tc"])


def scan_tests_dir(tests_dir, *, legacy=LEGACY_UNISOLATED):
    """Problems found in ``tests_dir`` as ``[(code, relative_path, detail)]`` (empty = the suite is fully collectable)."""
    problems = []
    for root, dirs, files in os.walk(tests_dir):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        rel_root = os.path.relpath(root, tests_dir)
        is_package_path = all(os.path.isfile(os.path.join(tests_dir, *p, "__init__.py"))
                              for p in _prefixes(rel_root))
        for name in sorted(files):
            if not name.endswith(".py"):
                continue
            path = os.path.join(root, name)
            rel = os.path.normpath(os.path.join(rel_root, name)) if rel_root != "." else name
            try:
                info = analyse(path)
            except SyntaxError as e:
                problems.append(("SYNTAX_ERROR", rel, str(e)))
                continue
            has_tc = any(c["tc"] for c in info["classes"].values())
            looks_like_test = name.startswith("test_") and name != "test_.py"
            collected = looks_like_test and is_package_path
            used_as_base = {b for c in info["classes"].values() if c["tc"] for b in c["bases"]}
            if has_tc and not collected:
                problems.append(("NOT_DISCOVERED", rel, "contains TestCase classes but discovery (test_*.py in the tests "
                                 "directory or a package below it) will not pick this file up"))
            if not looks_like_test:
                continue
            if collected and not info["helper"] and expected_test_count(info) == 0 and not any(
                    c["tc"] and not c["tests"] for c in info["classes"].values()):
                problems.append(("NO_TESTS", rel, "no TestCase class with a test method"))
            for cname, cinfo in info["classes"].items():
                if cinfo["tests"] and not cinfo["tc"] and cname not in used_as_base:
                    problems.append(("HIDDEN_TEST", rel, f"{cname} has test methods ({', '.join(cinfo['tests'][:3])}) "
                                     "but is not a TestCase, so they never run"))
            for fn in info["module_tests"]:
                problems.append(("MODULE_LEVEL_TEST", rel, f"{fn}() is a module-level function; unittest does not run it"))
            for dup in info["dups"]:
                problems.append(("DUPLICATE_NAME", rel, f"{dup} is defined twice; the first definition is shadowed"))
            if info["oscar_import"] and name not in legacy and (
                    not info["isolate_call"] or info["isolate_call"] > info["oscar_import"]):
                problems.append(("NO_ISOLATION", rel, f"imports oscar at line {info['oscar_import']} without calling "
                                 "isolation.isolate_process() first (the isolation header)"))
    return problems


# -- undefined names (pyflakes-lite) ----------------------------------------------------------------------

_MODULE_DUNDERS = {"__name__", "__file__", "__doc__", "__package__", "__spec__", "__loader__", "__builtins__", "__path__",
                   "__cached__", "__annotations__", "__dict__", "__class__", "__module__", "__qualname__"}
_KNOWN_GLOBALS = set(dir(builtins)) | _MODULE_DUNDERS | {"WindowsError"}
# names the host injects into a module that is exec'd by FreeCAD
_INJECTED = {"InitGui.py": {"Workbench"}}


def undefined_names(path, injected=()):
    """``[(name, line)]`` of names ``path`` uses that no scope defines (module-level assignments, imports, defs and
    classes, ``global`` assignments inside functions, builtins). Skips files with ``from x import *``."""
    with open(path, encoding="utf-8") as f:
        src = f.read()
    tree = ast.parse(src, path)
    if any(isinstance(n, ast.ImportFrom) and any(a.name == "*" for a in n.names) for n in ast.walk(tree)):
        return []
    top = symtable.symtable(src, path, "exec")
    defined = {s.get_name() for s in top.get_symbols() if s.is_assigned() or s.is_imported() or s.is_namespace()}
    defined |= set(injected)

    def collect(table):
        for sym in table.get_symbols():
            if sym.is_declared_global() and sym.is_assigned():
                defined.add(sym.get_name())
        for child in table.get_children():
            collect(child)

    collect(top)
    bad = []

    def check(table):
        for sym in table.get_symbols():
            name = sym.get_name()
            if name in defined or name in _KNOWN_GLOBALS:
                continue
            if table.get_type() == "module":
                if sym.is_referenced() and not (sym.is_assigned() or sym.is_imported() or sym.is_namespace()):
                    bad.append((name, table.get_lineno()))
            elif sym.is_global() and sym.is_referenced():
                bad.append((name, table.get_lineno()))
        for child in table.get_children():
            check(child)

    check(top)
    return sorted(set(bad))


def scan_undefined(root):
    """``[(relative path, line, name)]`` over every ``*.py`` below ``root`` (``fixtures`` and caches skipped)."""
    out = []
    for dirpath, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS)
        for name in sorted(files):
            if name.endswith(".py"):
                path = os.path.join(dirpath, name)
                try:
                    hits = undefined_names(path, _INJECTED.get(name, ()))
                except SyntaxError as e:
                    out.append((os.path.relpath(path, root), getattr(e, "lineno", 0) or 0, "SyntaxError"))
                    continue
                out += [(os.path.relpath(path, root), line, n) for n, line in hits]
    return out


def _prefixes(rel_root):
    """``a/b`` -> ``[("a",), ("a", "b")]`` (the package chain a subdirectory needs); ``.`` -> ``[]``."""
    if rel_root in (".", ""):
        return []
    parts = rel_root.split(os.sep)
    return [tuple(parts[:i]) for i in range(1, len(parts) + 1)]


def _format(problems):
    return "\n".join(f"  {code:<18} {rel}: {detail}" for code, rel, detail in problems)


class RealTreeTest(unittest.TestCase):
    """The guard applied to the actual ``tests/`` directory."""

    def test_nothing_in_the_suite_is_silently_uncollected(self):
        problems = scan_tests_dir(TESTS_DIR)
        self.assertEqual(problems, [], "uncollectable or unisolated test code:\n" + _format(problems))

    def test_legacy_allowlist_has_no_stale_entries(self):
        for name in LEGACY_UNISOLATED:
            path = os.path.join(TESTS_DIR, name)
            self.assertTrue(os.path.isfile(path), f"{name} no longer exists: remove it from LEGACY_UNISOLATED")
            info = analyse(path)
            self.assertFalse(info["isolate_call"] and (not info["oscar_import"] or info["isolate_call"] < info["oscar_import"]),
                             f"{name} now isolates itself: remove it from LEGACY_UNISOLATED")

    def test_the_guard_and_the_runner_are_present(self):
        self.assertTrue(os.path.isfile(os.path.join(TESTS_DIR, "run.py")))
        self.assertTrue(os.path.isfile(os.path.join(TESTS_DIR, "support", "__init__.py")))

    def test_there_are_a_meaningful_number_of_tests(self):
        total = sum(expected_test_count(analyse(os.path.join(TESTS_DIR, f)))
                    for f in os.listdir(TESTS_DIR) if f.startswith("test_") and f.endswith(".py"))
        self.assertGreater(total, 200, "the declared test count collapsed: a file lost its tests?")


class DetectorTest(unittest.TestCase):
    """Every problem code fires on a synthetic tree (and the clean tree stays clean)."""

    def tree(self, files, *, packages=()):
        root = tempfile.mkdtemp(prefix="guard_")
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        for rel, source in files.items():
            path = os.path.join(root, *rel.split("/"))
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(textwrap.dedent(source))
        for pkg in packages:
            os.makedirs(os.path.join(root, pkg), exist_ok=True)
            open(os.path.join(root, pkg, "__init__.py"), "w").close()
        return root

    def codes(self, files, **kw):
        return sorted({(code, rel) for code, rel, _ in scan_tests_dir(self.tree(files, **kw))})

    GOOD = """
        import unittest
        from support import isolation
        isolation.isolate_process()
        from oscar import flow

        class A(unittest.TestCase):
            def test_one(self):
                pass
        """

    def test_a_good_file_is_clean(self):
        self.assertEqual(self.codes({"test_ok.py": self.GOOD}), [])

    def test_no_tests(self):
        self.assertEqual(self.codes({"test_empty.py": "import unittest\nclass Helper:\n    pass\n"}),
                         [("NO_TESTS", "test_empty.py")])

    def test_a_class_without_the_testcase_base(self):
        src = "import unittest\nclass Forgot:\n    def test_x(self):\n        pass\nclass Real(unittest.TestCase):\n    def test_y(self):\n        pass\n"
        self.assertEqual(self.codes({"test_hidden.py": src}), [("HIDDEN_TEST", "test_hidden.py")])

    def test_a_mixin_used_as_a_base_is_not_hidden(self):
        src = ("import unittest\nclass Mixin:\n    def test_shared(self):\n        pass\n"
               "class Real(Mixin, unittest.TestCase):\n    pass\n")
        self.assertEqual(self.codes({"test_mixin.py": src}), [])

    def test_module_level_test_function(self):
        src = "import unittest\ndef test_fn():\n    pass\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
        self.assertEqual(self.codes({"test_fn.py": src}), [("MODULE_LEVEL_TEST", "test_fn.py")])

    def test_duplicate_method_names(self):
        src = ("import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
               "    def test_a(self):\n        pass\n")
        self.assertEqual(self.codes({"test_dup.py": src}), [("DUPLICATE_NAME", "test_dup.py")])

    def test_the_same_name_in_two_classes_is_fine(self):
        src = ("import unittest\nclass A(unittest.TestCase):\n    def test_a(self):\n        pass\n"
               "class B(unittest.TestCase):\n    def test_a(self):\n        pass\n")
        self.assertEqual(self.codes({"test_two.py": src}), [])

    def test_misnamed_file_is_not_discovered(self):
        src = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
        self.assertEqual(self.codes({"flow_test.py": src, "tests_flow.py": src}),
                         [("NOT_DISCOVERED", "flow_test.py"), ("NOT_DISCOVERED", "tests_flow.py")])

    def test_helper_files_without_testcases_may_have_any_name(self):
        self.assertEqual(self.codes({"helper.py": "def test_data():\n    return 1\n", "gui_smoke.py": "x = 1\n"}), [])

    def test_a_test_file_in_a_plain_subdirectory_is_not_discovered(self):
        src = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
        self.assertEqual(self.codes({"sub/test_deep.py": src}), [("NOT_DISCOVERED", os.path.join("sub", "test_deep.py"))])
        self.assertEqual(self.codes({"sub/test_deep.py": src}, packages=("sub",)), [])

    def test_fixtures_directory_is_ignored(self):
        src = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
        self.assertEqual(self.codes({"fixtures/test_data.py": src}), [])

    def test_oscar_import_before_isolation(self):
        src = ("import unittest\nfrom oscar import flow\nfrom support import isolation\nisolation.isolate_process()\n"
               "class T(unittest.TestCase):\n    def test_a(self):\n        pass\n")
        self.assertEqual(self.codes({"test_late.py": src}), [("NO_ISOLATION", "test_late.py")])

    def test_oscar_import_without_isolation_and_the_legacy_exemption(self):
        src = "import unittest\nimport oscar.params\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
        self.assertEqual(self.codes({"test_bare.py": src}), [("NO_ISOLATION", "test_bare.py")])
        root = self.tree({"test_legacy.py": src})
        self.assertEqual(scan_tests_dir(root, legacy=frozenset({"test_legacy.py"})), [])

    def test_a_helper_module_named_test_is_exempt_from_no_tests_when_it_says_so(self):
        src = '"""Shared fixtures. Contains no test case on purpose."""\nX = 1\n'
        self.assertEqual(self.codes({"test_support_helper.py": src}), [])
        self.assertEqual(self.codes({"test_support_helper.py": "X = 1\n"}), [("NO_TESTS", "test_support_helper.py")])

    def test_files_that_do_not_import_oscar_need_no_isolation(self):
        src = "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
        self.assertEqual(self.codes({"test_pure.py": src}), [])

    def test_a_bare_isolate_process_call_also_counts(self):
        src = ("import unittest\nfrom support.isolation import isolate_process\nisolate_process()\nimport oscar\n"
               "class T(unittest.TestCase):\n    def test_a(self):\n        pass\n")
        self.assertEqual(self.codes({"test_bare_call.py": src}), [])

    def test_inherited_testcase_through_a_local_base_and_an_imported_case(self):
        src = ("import unittest\nfrom test_other import SharedCase\nclass Base(unittest.TestCase):\n    pass\n"
               "class Child(Base):\n    def test_a(self):\n        pass\nclass Other(SharedCase):\n    def test_b(self):\n        pass\n")
        self.assertEqual(self.codes({"test_inherit.py": src}), [])

    def test_a_syntax_error_is_reported_not_raised(self):
        self.assertEqual(self.codes({"test_broken.py": "def (:\n"}), [("SYNTAX_ERROR", "test_broken.py")])

    def test_expected_test_count_counts_testcase_methods_only(self):
        root = self.tree({"test_count.py": "import unittest\nclass T(unittest.TestCase):\n    def test_a(self):\n        pass\n"
                          "    def test_b(self):\n        pass\n    def helper(self):\n        pass\n"
                          "class Plain:\n    def test_c(self):\n        pass\n"})
        self.assertEqual(expected_test_count(analyse(os.path.join(root, "test_count.py"))), 2)


class UndefinedNamesTest(unittest.TestCase):
    def test_no_undefined_names_in_the_workbench_or_its_tests(self):
        hits = scan_undefined(MOD_DIR)
        self.assertEqual(hits, [], "names used but never defined (a NameError waiting for its branch):\n" +
                         "\n".join(f"  {rel}:{line}: {name}" for rel, line, name in hits))

    def lint(self, source, injected=()):
        root = tempfile.mkdtemp(prefix="lint_")
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        path = os.path.join(root, "m.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(source))
        return [name for name, _ in undefined_names(path, injected)]

    def test_a_typo_in_a_function_is_found(self):
        self.assertEqual(self.lint("def f(x):\n    return x + undefined_thing\n"), ["undefined_thing"])

    def test_a_typo_in_a_rarely_taken_branch_is_found(self):
        self.assertEqual(self.lint("def f(x):\n    if x > 100:\n        return helper_that_was_renamed(x)\n    return 0\n"),
                         ["helper_that_was_renamed"])

    def test_module_level_use_before_nothing(self):
        self.assertEqual(self.lint("print(missing)\n"), ["missing"])

    def test_defined_things_are_not_reported(self):
        src = """
            import os
            from os import path as p
            import xml.dom as dom
            try:
                import optional_lib
            except ImportError:
                optional_lib = None
            CONST = 1
            def helper(): return CONST
            class K:
                attr = 1
                def m(self): return self.attr + K.attr + helper() + len([i for i in range(3) if i])
            def uses(): return os.sep, p, dom, optional_lib, K, __name__, __file__
            lam = lambda y: y + CONST
            for loop_var in range(2): pass
            with open(__file__) as handle: pass
            def g():
                global LATE
                LATE = 5
            def reads_late(): return LATE
            """
        self.assertEqual(self.lint(src), [])

    def test_names_injected_by_the_host_are_allowed_when_declared(self):
        self.assertEqual(self.lint("class W(Workbench):\n    pass\n"), ["Workbench"])
        self.assertEqual(self.lint("class W(Workbench):\n    pass\n", injected={"Workbench"}), [])

    def test_star_imports_are_skipped_not_guessed(self):
        self.assertEqual(self.lint("from os.path import *\nprint(anything)\n"), [])

    def test_comprehension_and_closure_scopes(self):
        self.assertEqual(self.lint("def f(items):\n    k = 2\n    return [i * k for i in items if i > k], (lambda: k)\n"), [])

    def test_a_syntax_error_is_reported_by_the_tree_scan(self):
        root = tempfile.mkdtemp(prefix="lint_")
        self.addCleanup(lambda: __import__("shutil").rmtree(root, ignore_errors=True))
        with open(os.path.join(root, "bad.py"), "w") as f:
            f.write("def (:\n")
        self.assertEqual([(rel, name) for rel, _, name in scan_undefined(root)], [("bad.py", "SyntaxError")])


def _freecad_importable():
    try:
        import FreeCAD  # noqa: F401
        return True
    except Exception:  # noqa: BLE001
        return False


@unittest.skipUnless(_freecad_importable(), "needs FreeCAD to import the test modules (the fast job has none)")
class DiscoveryMatchesSourceTest(unittest.TestCase):
    """Real discovery collects at least as many tests per file as the source declares, and imports them all."""

    def test_discovery_agrees_with_the_source(self):
        if TESTS_DIR not in sys.path:
            sys.path.insert(0, TESTS_DIR)
        if MOD_DIR not in sys.path:
            sys.path.insert(0, MOD_DIR)
        suite = unittest.TestLoader().discover(TESTS_DIR, pattern="test_*.py", top_level_dir=TESTS_DIR)
        found, failed = {}, {}
        stack = [suite]
        while stack:
            for item in stack.pop():
                if isinstance(item, unittest.TestSuite):
                    stack.append(item)
                    continue
                module = type(item).__module__ if type(item).__name__ != "_FailedTest" else item.id().split(".")[-1]
                if type(item).__name__ == "_FailedTest":
                    failed[item.id()] = item
                else:
                    found[module] = found.get(module, 0) + 1
        self.assertEqual(sorted(failed), [], "test modules that fail to import:\n" + "\n".join(
            f"  {k}: ...{str(getattr(v, '_exception', ''))[-400:]}" for k, v in sorted(failed.items())[:3]))
        problems = []
        for name in sorted(os.listdir(TESTS_DIR)):
            if name.startswith("test_") and name.endswith(".py"):
                want = expected_test_count(analyse(os.path.join(TESTS_DIR, name)))
                got = found.get(name[:-3], 0)
                if got < want:
                    problems.append(f"{name}: source declares {want} tests, discovery collected {got}")
        self.assertEqual(problems, [], "\n".join(problems))


class WorkflowAndScriptTest(unittest.TestCase):
    """CI uses discovery, not hand-written module lists; scripts/test.sh offers the three modes."""

    def read(self, *parts):
        path = os.path.join(REPO_ROOT, *parts)
        if not os.path.isfile(path):
            self.skipTest(f"{os.path.join(*parts)} is not in this checkout (sparse checkout)")
        with open(path, encoding="utf-8") as f:
            return f.read()

    def test_workflows_do_not_hand_list_test_modules(self):
        for name in ("ci.yml", "_build-app.yml", "release.yml"):
            text = self.read(".github", "workflows", name)
            self.assertIsNone(re.search(r"unittest\s+(?:-\S+\s+)*test_\w+", text),
                              f"{name} lists test modules by hand: use discovery (tests/run.py)")

    def test_the_build_workflow_runs_the_whole_suite_through_the_runner_on_both_oses(self):
        text = self.read(".github", "workflows", "_build-app.yml")
        self.assertGreaterEqual(len(re.findall(r"run\.py", text)), 4)       # fast + full on macOS and on Windows
        self.assertIn("OSCAR_TEST_ALLOW_REAL_HOME", text)
        self.assertIn("--fast", text)

    def test_ci_has_a_fast_job_before_the_long_build(self):
        text = self.read(".github", "workflows", "ci.yml")
        self.assertRegex(text, r"needs:\s*\[?\s*(quick|fast|static)")
        self.assertIn("compileall", text)
        self.assertRegex(text, r"ruff|pyflakes")
        self.assertIn("test_suite_guard", text)

    def test_test_script_is_valid_bash_and_rejects_bad_usage_before_touching_anything(self):
        import shutil
        import subprocess
        bash = shutil.which("bash")
        path = os.path.join(REPO_ROOT, "scripts", "test.sh")
        if not bash or not os.path.isfile(path):
            self.skipTest("needs bash and scripts/test.sh")
        self.assertEqual(subprocess.run([bash, "-n", path], capture_output=True).returncode, 0)
        for args in ([], ["bogus"]):
            proc = subprocess.run([bash, path, *args], capture_output=True, timeout=30)
            self.assertEqual(proc.returncode, 2, proc.stderr.decode("utf-8", "replace"))
            self.assertIn(b"scripts/test.sh", proc.stdout + proc.stderr)

    def test_test_script_has_the_three_modes_and_isolation(self):
        text = self.read("scripts", "test.sh")
        for token in ("fast", "full", "gui", "FREECAD_USER_HOME", "OSCAR_LOG_DIR", "OSCAR_CACHE_DIR",
                      "OSCAR_PROJECTS_DIR", "OSCAR_AI_SETTINGS", "OSCAR_DOWNLOAD_DIR", "OSCAR_TEST_MODULES"):
            self.assertIn(token, text)
        self.assertNotRegex(text, r"(^|\s)HOME=", "scripts/test.sh must never override HOME (the geometry tests need it)")


if __name__ == "__main__":
    unittest.main()
