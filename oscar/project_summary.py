"""Generation half of :mod:`oscar.project` (private mixin): NOTES.md, rebuild.py, the Build Summary and the
artifact check. ``Project`` inherits :class:`SummaryMixin`; every method here reads the project's files (lock-free)
and writes only the generated files (under ``Project._lock``, never across a GUI dispatch).
"""

import os

from . import paths
from . import project_docs as _docs
from . import project_gui as gui
from . import project_validation as vrec
from .project_util import (AUTO_EXPORT_INTENTS, BuildSummary, StepRecord, _NOT_REPLAYED, _as_dict,  # noqa: F401
                           _atomic_write_text, _json_safe)
from .toolctx import current_context
from .turnlog import log_event


class SummaryMixin:
    """NOTES.md / rebuild.py / Build Summary / ``verify_artifacts`` of a Project (see ``Project``)."""

    def _artifact_names(self) -> list[str]:
        """Relative names of the files that exist (for NOTES "Artifacts" and the Summary "Files" line)."""
        out = []
        for name in ("model.FCStd", "rebuild.py", "NOTES.md", "params.json", "provenance.json", "spec.json",
                     "validation.json"):
            if os.path.isfile(self.path(name)):
                out.append(name)
        out += [f"docs/{n}" for n in self._listdir("docs")]
        out += [f"exports/{n}" for n in self._listdir("exports")]
        out += [f"refs/{n}" for n in self._listdir("refs")]
        return out

    def _stale_exports(self) -> str:
        """An open-item line when ``exports/`` was written for an older (or rewound) model, else ``""``."""
        exported = self.manifest.get("exported_after_step")
        if exported is None or not self._listdir("exports"):
            return ""
        try:
            after = int(exported)
        except (TypeError, ValueError):
            return ""
        last_geo = self.last_geometry_step()
        if after == last_geo:
            return ""
        return (f"exports/ was written after step {after} but the model is at step {last_geo}; "
                "export again (export_model) so the files match the model")

    def _unresolved_extra(self, steps: list[StepRecord]) -> list[str]:
        """NOTES "Unresolved" lines that come from the step records (what rebuild.py cannot replay)."""
        extra = [line for line in (self._stale_exports(),) if line]
        live = [s for s in steps if not s.superseded]
        gui_only = [s for s in live if s.replay.get("kind") == "script" and s.replay.get("gui_only")]
        for s in gui_only:
            extra.append(f"step {s.index} ({s.title}): not headless-replayable (uses ViewObject/Gui); rebuild.py "
                         "cannot replay it")
        failed = [s.index for s in live if not s.ok]
        if failed:
            extra.append(f"rebuild.py skips {len(failed)} failed call(s): steps {', '.join(map(str, failed[:20]))}")
        by_reason: dict[str, list[int]] = {}
        for s in live:
            if s.ok and s.replay.get("kind") == "none" and s.tool in ("execute_code_async", "source_reference_part",
                                                                       "run_fem_analysis"):
                by_reason.setdefault(s.tool, []).append(s.index)
        for tool_name, indexes in by_reason.items():
            extra.append(f"rebuild.py does not replay {tool_name} (steps {', '.join(map(str, indexes[:20]))}): "
                         f"{_NOT_REPLAYED[tool_name]}")
        return extra

    def write_notes_md(self, ledger: dict | None = None) -> str:
        """Regenerate NOTES.md from the project's files (deterministic: the same inputs give the same text);
        returns the relative path. ``ledger`` defaults to ``ledger.json``. Called after every recorded step."""
        self.ensure_dir()
        ledger_dict = _as_dict(ledger) if ledger is not None else self.read_ledger()
        steps = self.read_steps(include_superseded=True)
        _, starts = self._journal()
        request = (ledger_dict or {}).get("request") or (starts[0].get("request") if starts else "")
        validation = self.read_validation()
        spec = self.read_spec()
        stale = vrec.stale_reason(validation, steps)
        unresolved = self._unresolved_extra(steps) + ([stale] if stale else [])
        text = _docs.render_notes(
            title=self.manifest.get("title") or self.manifest.get("id", ""), ts=str(self.manifest.get("updated", "")),
            request=str(request or ""), ledger=ledger_dict, steps=steps, params=self.read_params()["params"],
            provenance=self.read_provenance(), notes=self.read_notes(), validation=validation, spec=spec,
            reference_status=self.reference_status, rebuild_ok=self._rebuild_ok(), files=self._artifact_names(),
            unresolved_extra=unresolved, stale=stale)
        with self._lock:
            _atomic_write_text(self.path("NOTES.md"), text)
        return "NOTES.md"

    def _step_literal(self, step: StepRecord) -> dict:
        replay = step.replay
        base = {"index": step.index, "tool": step.tool, "title": step.title, "doc": step.doc}
        if replay.get("kind") == "script":
            base.update(kind="script", file=replay.get("file", ""), headless=bool(replay.get("headless")),
                        gui_only=bool(replay.get("gui_only")))
        else:
            base.update(kind="tool", args=replay.get("args") if replay.get("args") is not None else step.args)
        return base

    def rebuild_script_text(self) -> str:
        """The text of rebuild.py: the NON-superseded successful steps whose ``replay.kind`` is not
        ``none``, in index order, as a Python ``STEPS`` literal, with ``DOC_NAME`` / ``PARAMS`` /
        ``PROJECT_DIR`` / ``OSCAR_MOD_DIR`` as ``repr`` literals."""
        steps = [s for s in self.read_steps() if s.ok and s.replay.get("kind") in ("tool", "script")]
        params = {}
        for row in self.read_params()["params"]:
            value = row.get("value")
            shown = f"{value:g}" if isinstance(value, float) else str(value)
            params[str(row.get("name"))] = f"{shown} {row.get('unit', '')}".strip()
        doc_name = self.manifest.get("active_doc") or (self.manifest.get("docs") or [""])[0]
        return _docs.render_rebuild(
            title=self.manifest.get("title") or self.manifest.get("id", ""), script_path=self.path("rebuild.py"),
            mod_dir=paths.mod_dir(), project_dir=self.root, doc_name=doc_name, params=params,
            steps=[self._step_literal(s) for s in steps])

    def write_rebuild_py(self) -> str:
        """Regenerate rebuild.py atomically; returns the relative path ``rebuild.py``."""
        self.ensure_dir()
        text = self.rebuild_script_text()
        with self._lock:
            _atomic_write_text(self.path("rebuild.py"), text)
        return "rebuild.py"

    def _regenerate_docs(self, *, rebuild: bool) -> None:
        """NOTES.md (+ rebuild.py) after a change; never raises (a failed regeneration is logged)."""
        try:
            self.write_notes_md()
            if rebuild:
                self.write_rebuild_py()
        except Exception as exc:  # noqa: BLE001
            log_event("project_error", where="regenerate_docs", error=f"{type(exc).__name__}: {exc}"[:200])

    def _model_tree(self, validation: dict | None) -> list[dict]:
        doc = self.manifest.get("active_doc") or ""
        rows: list[dict] = []
        if doc:
            ctx = current_context()

            def task() -> list[dict]:
                found = gui.resolve_doc(doc)
                return gui.model_tree(found) if found is not None else []

            result, error = gui.gui_call(task, 30.0, "project_model_tree", ctx.should_stop)
            rows = [] if error else list(result or [])
        if not rows and validation:
            rows = vrec.tree_from_health(validation)
        return rows

    def _open_items(self, ledger: dict | None, validation: dict | None, steps: list[StepRecord],
                    stale: str) -> list[str]:
        items: list[str] = []
        for s in (ledger or {}).get("steps") or []:
            if s.get("status") in ("blocked", "skipped", "changed") and s.get("note"):
                items.append(f"{s.get('id')} {s.get('status')}: {s.get('note')}")
            elif s.get("status") in ("pending", "running") and s.get("required", True):
                items.append(f"{s.get('id')} {s.get('status')}: {s.get('title')}")
        if validation:
            gate_keys = {(e.get("code"), e.get("object")) for e in validation.get("gate_errors") or []}
            for e in validation.get("gate_errors") or []:
                items.append(f"health error {e.get('code', '')} {e.get('object', '')}: {e.get('message', '')}".replace("  ", " "))
            for f in (validation.get("health") or {}).get("findings") or []:
                if f.get("severity") == "error" and (f.get("code"), f.get("object")) not in gate_keys:
                    items.append(f"pre-existing: {f.get('code', '')} {f.get('object', '')}: {f.get('message', '')}".replace("  ", " "))
            for row in validation.get("fit_rows") or []:
                if str(row.get("status", "")).upper() == "FAIL":
                    items.append(f"fit FAIL {row.get('id')}: {row.get('detail') or row.get('metric') or ''}".strip())
            for row in validation.get("dfm") or []:
                if str(row.get("status", "")).upper() == "FAIL":
                    items.append(f"printability FAIL {row.get('id')}: {row.get('detail') or ''}".strip())
        if stale:
            items.append(stale)
        stale_exports = self._stale_exports()
        if stale_exports:
            items.append(stale_exports)
        for n in self.read_notes():
            if n.get("kind") == "unresolved":
                items.append(f"{n.get('id')}: {n.get('text')}")
        for s in steps:
            if not s.superseded and s.replay.get("kind") == "script" and s.replay.get("gui_only"):
                items.append(f"step {s.index} ({s.title}) is not headless-replayable (uses ViewObject/Gui)")
        return items

    def build_summary(self, ledger: dict | None, outcome: str, validation: dict | None) -> BuildSummary:
        """The ``## Build Summary`` block plus the same content as JSON.

        Everything the app claims comes from code and files: the model tree from the live document (GUI thread,
        the validation's health rows as fallback), requirements from the ledger (+ the kit's coverage detail),
        parameters from params.json, the reference status from provenance, and the validated / NOT validated
        lists from ``project_docs.classify_validation``. Without a ``validation`` argument the stored
        ``validation.json`` is used (what a validation tool recorded earlier is still a code-made fact). A
        validation that does not describe the model as it is now (older than the newest geometry step, or
        recorded before a rewind) is STALE: nothing it ran counts as validated. At most 70 lines / 7000 characters."""
        ledger_dict = _as_dict(ledger)
        validation_dict = _as_dict(validation) or self.read_validation()
        steps = self.read_steps(include_superseded=True)
        stale = vrec.stale_reason(validation_dict, steps)
        spec = self.read_spec() or {}
        validated, not_validated = _docs.classify_validation(
            validation_dict, reference_status=self.reference_status, material=str(spec.get("material") or ""),
            rebuild_ok=self._rebuild_ok(), stale=stale)
        tree = self._model_tree(validation_dict)
        params = self.read_params()["params"]
        provenance = self.read_provenance()
        open_items = self._open_items(ledger_dict, validation_dict, steps, stale)
        files = []
        for name in ("NOTES.md", "rebuild.py", "model.FCStd", "params.json", "provenance.json"):
            if os.path.isfile(self.path(name)):
                files.append(name)
        if self._listdir("exports"):
            files.append("exports/")
        requirements = (ledger_dict or {}).get("requirements") or []
        coverage = (validation_dict or {}).get("coverage") if not stale else None
        markdown = _docs.render_summary(
            outcome=outcome, reference_status=self.reference_status, project_dir=self.root, tree=tree,
            requirements=requirements, steps=(ledger_dict or {}).get("steps") or [], coverage=coverage,
            params=params, provenance=provenance, validated=validated, not_validated=not_validated,
            open_items=open_items, files=files)
        data = {"outcome": outcome, "reference_status": self.reference_status, "project_dir": self.root,
                "tree": tree, "requirements": requirements, "parameters": params, "provenance": provenance,
                "validated": validated, "not_validated": not_validated, "open_items": open_items, "files": files}
        return BuildSummary(markdown, _json_safe(data))

    def verify_artifacts(self) -> list[str]:
        """Relative paths that SHOULD exist but do not (empty = the dossier is complete).

        Required: ``project.json``, ``steps.jsonl``, ``ledger.json``, ``NOTES.md`` always; ``model.FCStd`` when a
        geometry step exists; ``rebuild.py`` when any step has a replay entry; ``params.json`` when a step
        used a parameter tool or the active document has a Params / Parameters sheet; ``provenance.json`` when
        the ledger ``needs_reference``; and at least one ``exports/*`` file for the intents that auto-export
        (reported as ``exports/*``)."""
        missing = [n for n in ("project.json", "steps.jsonl", "ledger.json", "NOTES.md")
                   if not os.path.isfile(self.path(n))]
        steps = [s for s in self.read_steps() if s.ok]
        if any(s.geometry for s in steps) and not os.path.isfile(self.path("model.FCStd")):
            missing.append("model.FCStd")
        if any(s.replay.get("kind") in ("tool", "script") for s in steps) and not os.path.isfile(self.path("rebuild.py")):
            missing.append("rebuild.py")
        needs_params = any(s.tool in ("set_parameters", "set_parameter", "bind_expression", "build_enclosure")
                           for s in steps)
        doc = self.manifest.get("active_doc") or ""
        if not needs_params and doc:
            def task() -> bool:
                found = gui.resolve_doc(doc)
                return bool(found is not None and gui.param_table(found).get("sheet"))

            result, error = gui.gui_call(task, 10.0, "project_verify_params", current_context().should_stop)
            needs_params = bool(result) and not error
        if needs_params and not os.path.isfile(self.path("params.json")):
            missing.append("params.json")
        ledger = self.read_ledger() or {}
        if ledger.get("needs_reference") and not os.path.isfile(self.path("provenance.json")):
            missing.append("provenance.json")
        if ledger.get("intent") in AUTO_EXPORT_INTENTS and not self._listdir("exports"):
            missing.append("exports/*")
        return missing
