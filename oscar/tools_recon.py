"""``reconstruct_mesh`` — the one-tool mesh workflow surface for the reconstruction engine
(M2-M6): ``analyse`` (diagnostics + recognition survey), ``repair`` (safe copy-based repair,
gated operations on explicit authorization), ``solid`` (the honest NON-PARAMETRIC faceted
solid via the existing converter) and ``parametric`` (the full route: diagnose → recognize →
FeaturePlan → the existing builder → deviation rows → classified outcome). Every mode reports
an explicit output classification; the original mesh is never modified.
"""

import json
import hashlib
import os
from dataclasses import dataclass

from .tools import ToolError, current_context, tool  # noqa: F401  (registry decorator + ctx)
from .tools_ref import _gui_call
from .tools_verify_text import fit_block  # noqa: F401

_DOC = {"type": "string", "description": "Name of the Oscar document"}
_MODES = ("parametric", "analyse", "repair", "solid", "plan")
_TIMEOUT_S = 600.0


@dataclass
class _RepairApproval:
    """One-use, in-memory human approval bound to a single repair call."""

    doc_name: str
    mesh_name: str
    mesh_digest: str
    operations: tuple[str, ...]
    specs: tuple[tuple[str, tuple[tuple[str, object], ...]], ...]
    request_digest: str
    turn_id: str
    call_id: str
    binding: str
    used: bool = False


def _request_digest(request: str) -> str:
    return hashlib.sha256(str(request).encode("utf-8")).hexdigest()


def _repair_binding(mesh_digest: str, specs: dict, request: str,
                    turn_id: str, call_id: str) -> str:
    payload = {"mesh_digest": mesh_digest, "operations": specs,
               "request": str(request), "turn_id": str(turn_id), "call_id": str(call_id)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                    default=str).encode("utf-8")).hexdigest()


def _repair_proposal(mesh):
    """The exact gated operation names and effective arguments used by meshops.repair."""
    from .recon.meshops import GATED_OPERATIONS, _predicted_spec
    from .recon.quality import estimate_sigma

    try:
        mean_edge = estimate_sigma(mesh).get("mean_edge_mm")
    except Exception:  # noqa: BLE001 - a failed estimate uses the repair engine's fallback
        mean_edge = None
    return {op: _predicted_spec(op, mean_edge) for op in GATED_OPERATIONS}


def _repair_preflight(doc_name: str, wanted: str):
    """Read the mesh identity and gated operation proposal on the GUI thread."""
    import FreeCAD as App

    from .recon import provenance

    doc = App.getDocument(doc_name) if doc_name in App.listDocuments() else None
    if doc is None:
        return {"error": f"document '{doc_name}' is not open"}
    mesh_obj, problem = _find_mesh(doc, wanted)
    if mesh_obj is None:
        return {"error": problem}
    return {"mesh_name": mesh_obj.Name,
            "mesh_digest": provenance.mesh_digest(mesh_obj.Mesh),
            "specs": _repair_proposal(mesh_obj.Mesh)}


def _consume_repair_approval(approval, doc_name, mesh_obj, request, turn_id, call_id):
    """Validate and consume an approval immediately before gated repair execution."""
    if approval is None:
        return ()
    from .recon import provenance

    if not isinstance(approval, _RepairApproval) or approval.used:
        raise ValueError("repair approval is missing, invalid, or already used")
    digest = provenance.mesh_digest(mesh_obj.Mesh)
    specs = _repair_proposal(mesh_obj.Mesh)
    operations = tuple(specs)
    packed_specs = tuple((op, tuple(sorted(spec.items()))) for op, spec in specs.items())
    if (approval.doc_name != doc_name or approval.mesh_name != mesh_obj.Name
            or approval.mesh_digest != digest or approval.operations != operations
            or approval.specs != packed_specs
            or approval.request_digest != _request_digest(request)
            or approval.turn_id != str(turn_id) or approval.call_id != str(call_id)):
        raise ValueError("repair approval is stale or does not match this mesh, request, or tool call")
    expected = _repair_binding(digest, specs, request, turn_id, call_id)
    if approval.binding != expected:
        raise ValueError("repair approval binding is invalid")
    approval.used = True
    return approval.operations


def _find_mesh(doc, wanted):
    if wanted:
        obj = doc.getObject(wanted)
        if obj is None or obj.TypeId != "Mesh::Feature":
            return None, f"'{wanted}' is not a Mesh::Feature of document '{doc.Name}'"
        return obj, ""
    for obj in doc.Objects:
        if obj.TypeId == "Mesh::Feature" and int(obj.Mesh.CountFacets) > 0:
            return obj, ""
    return None, f"document '{doc.Name}' holds no mesh; import it first (import_file)"


def _analysis_payload(mode, mesh_obj, doc_name, diag, survey, provenance, verification,
                      extra=None):
    """K3: the structured analysis-evidence record for one mesh (diagnostics digest +
    quality/tolerances + survey digest + the four-category provenance ledger). The mesh
    digest lets any later reader detect STALE evidence: if the mesh's digest differs from
    the file's, the file describes a different geometric state. G5 adds the scan-quality
    digest (readiness verdict + loop/spike/doubled-surface evidence) when the caller
    passes one in ``extra['scan_quality']``."""
    from .recon import provenance as _prov

    regions = (survey or {}).get("regions") or []
    extra = dict(extra or {})
    scan_quality = extra.pop("scan_quality", None)
    return {
        "schema": 1, "mode": mode, "mesh": mesh_obj.Name, "document": doc_name,
        "source_sha256": getattr(mesh_obj, "OscarImportSha256", None),
        "mesh_digest": _prov.mesh_digest(mesh_obj.Mesh),
        "facets": int(mesh_obj.Mesh.CountFacets), "points": int(mesh_obj.Mesh.CountPoints),
        "diagnostics": {k: diag.get(k) for k in (
            "closed", "non_manifold", "self_intersections", "inconsistent_normals",
            "components", "boundary_edges", "boundary_loops", "open_chains",
            "self_intersection_pairs_count", "size_mm", "area_mm2", "warnings")}
        if diag else None,
        "quality": (survey or {}).get("quality"),
        "classification": (survey or {}).get("classification"),
        "coverage": (survey or {}).get("coverage"),
        "holes": (survey or {}).get("holes", []),
        "region_digest": {
            "count": len(regions),
            "by_type": {t: sum(1 for r in regions if r["type"] == t)
                        for t in {r["type"] for r in regions}},
            "evidence": {e: sum(1 for r in regions if r.get("evidence") == e)
                         for e in {r.get("evidence") for r in regions if r.get("evidence")}},
            "sample": [{k: r[k] for k in ("type", "area_mm2", "rms_mm", "evidence")
                        if k in r} for r in regions[:50]]},
        "provenance": provenance,
        "provenance_verification": verification,
        "scan_quality": _scan_quality_digest(scan_quality),
        "extra": extra,
    }


def _plan_digest(plan):
    """V-4: the per-feature/per-parameter digest of a reconstruction plan for the
    accuracy report (values with their evidence wording; snaps live in the log)."""
    if not plan:
        return None
    params = {}
    for name, spec in (plan.get("parameters") or {}).items():
        value = spec.get("value") if isinstance(spec, dict) else None
        source = spec.get("source") if isinstance(spec, dict) else None
        params[name] = {"value": value,
                        "source": str((source or {}).get("ref", ""))[:140]}
    return {"id": plan.get("id"),
            "features": [{"id": f.get("id"), "op": f.get("op")}
                         for f in (plan.get("features") or [])[:60]],
            "parameters": params}


def _scan_quality_digest(sq):
    """G5: the persisted scan-quality evidence (readiness + classified loops capped at 20
    with the full count + spikes/doubled surfaces + plausibility)."""
    if not sq:
        return None
    comp, integ = sq.get("completeness") or {}, sq.get("integrity") or {}
    loops = comp.get("loops") or []
    return {
        "readiness": sq.get("readiness"),
        "boundary_loops": comp.get("boundary_loops"),
        "open_chains": comp.get("open_chains"),
        "loops": [{k: l.get(k) for k in ("kind", "detail", "n_edges", "spanned_area_mm2",
                                         "area_share", "radius_mm", "rim_alignment",
                                         "through")}
                  for l in loops[:20]] + ([f"+{len(loops) - 20} more (count complete)"]
                                         if len(loops) > 20 else []),
        "spikes": integ.get("spikes"),
        "doubled_surfaces": integ.get("doubled_surfaces"),
        "plausibility": sq.get("plausibility"),
    }


def _units_lines(report, user_unit=None):
    """K4: plain-language unit status; a suspected mm/inch issue asks the user to confirm."""
    if not report:
        return []
    out = [f"UNITS: {report['status']} · source unit "
           f"{report.get('source_unit', 'unknown')} · dims "
           f"{report['dims_source']} source = {report.get('dims_mm')} mm = "
           f"{report.get('dims_inch')} in"]
    if report.get("source_unit_note"):
        out.append(f"  source note: {report['source_unit_note']}")
    if user_unit:
        out.append(f"  user confirmed: {user_unit}")
    if report["status"] in ("POSSIBLE_MM_INCH_MISMATCH", "SCALE_AMBIGUOUS") and not user_unit:
        out.append("  The dimensions may indicate a unit-interpretation issue. I cannot "
                   "confirm the original units from this file alone — please confirm the "
                   "intended units (confirm_units='mm' or 'inch') before relying on any "
                   "reconstruction of it.")
    for w in report.get("warnings", [])[:2]:
        out.append(f"  note: {w}")
    return out


def _mesh_flow(mode, mesh_obj, result, authorized: "tuple | list" = ()) -> dict:
    """W-1: the structured facts of one reconstruct_mesh call for the flow Ledger
    (``ctx.state["extra"]["mesh_flow"]`` → ``Ledger._on_mesh_recon``). Facts only — a step
    completes on what is RECORDED here, never on the call having happened.

    Geometry identity (W-1 final fix): ``input_digest`` = the working mesh this call
    operated on; ``output_digest`` = the post-repair/post-conversion working copy when the
    call produced one (the K3 ``provenance.mesh_digest`` — reused, not reinvented). M5-M7
    evidence is bound to these, so a changed working mesh can never read as the geometry
    the recorded repair facts were about."""
    from .recon import provenance as _prov
    from .recon.meshops import GATED_OPERATIONS

    units = result.get("units") or {}
    rep = result.get("repair") or {}
    verdicts: dict = {}
    for v in rep.get("validation") or []:
        verdicts[v.get("verdict")] = verdicts.get(v.get("verdict"), 0) + 1
    # G6: an authorized operation the fill policy SKIPPED did not execute — it must not
    # read as "authorized and ran" in the Ledger evidence
    ops = {a.get("operation") for a in rep.get("actions") or []
           if a.get("verdict") != "SKIPPED"}
    gated_ran = sorted(ops & set(GATED_OPERATIONS))
    build = result.get("build") or {}
    source_digest = _prov.mesh_digest(mesh_obj.Mesh)
    conv = result.get("unit_conversion")
    base_digest = result.get("base_digest")   # the mm mesh repair actually operated on
    input_digest = base_digest or source_digest
    input_unit = "mm" if base_digest else ("mm" if conv else "unknown")
    input_mm = input_digest if input_unit == "mm" else None
    output_digest = ((rep.get("provenance") or {}).get("mesh_digest")
                     or (result.get("provenance") or {}).get("mesh_digest"))
    if not output_digest and mode in ("parametric", "modify") and result.get("supported"):
        output_digest = input_digest       # no repair/conversion: the work copy is the input
    # the working pipeline (repair/conversion) is always in mm; repair MODE on a
    # unit-unknown source keeps the source representation for its output
    output_unit = input_unit if mode == "repair" else "mm"
    output_mm = output_digest if output_unit == "mm" else None
    executed = sorted(({"op": a.get("operation"), "params": dict(a.get("spec") or {})}
                       for a in rep.get("actions") or [] if a.get("operation") in set(gated_ran)),
                      key=lambda s: s["op"])
    payload = {
        "mode": mode,
        "sha256": getattr(mesh_obj, "OscarImportSha256", None),
        "facets": int(mesh_obj.Mesh.CountFacets),
        "source_digest": source_digest,
        "input_digest": input_digest,
        "input_unit": input_unit,
        "input_mm": input_mm,
        "converted": bool(conv),
        "output_digest": output_digest,
        "output_unit": output_unit,
        "output_mm": output_mm,
        "authorized_ops": executed,
        "diag": bool(result.get("diag") or result.get("diagnostics") or rep.get("after")),
        "units": units.get("status"),
        "units_unconfirmed": units.get("status") in ("POSSIBLE_MM_INCH_MISMATCH",
                                                     "SCALE_AMBIGUOUS")
                             and not result.get("user_unit"),
        "quality": bool((result.get("survey") or {}).get("quality")),
        "repair_verdicts": verdicts or None,
        "gated_needed": bool(rep.get("refused")) or bool(gated_ran),
        "authorized": bool(gated_ran),
        "solid_created": mode == "solid" and bool(result.get("created")),
        "survey": bool(result.get("survey")),
        "plan": bool(result.get("plan")),
        "deviation": bool(result.get("deviation")),
        "outcome": result.get("outcome"),
    }
    if result.get("supported") and build:
        payload["build_success"] = bool(build.get("success"))
        payload["build_error"] = str(build.get("error") or "")[:160]
    return payload


def tools_ask_surface_live() -> bool:
    """A live decision-card surface AND a bound conversation (the panel mounts mid-session)."""
    from . import tools_ask
    from .tools import current_context

    ctx = current_context()
    return (tools_ask.surface is not None
            and bool(str(getattr(ctx, "conversation_id", "") or "")))


def _askuser_store(root, ctx):
    from .recon.approve import AnswerStore

    return AnswerStore(os.path.join(root, "analysis"),
                       str(getattr(getattr(ctx, "project", None), "name", "") or "oscar"))


def _classification_probe(doc_name, wanted):
    """(read-only, GUI thread) the survey's classification — warms reconstruct's survey cache so
    the later _task run does NOT repeat the analysis (cache hit on the same mesh digest)."""
    import FreeCAD as App

    from .recon import recognize

    doc = App.getDocument(doc_name) if doc_name in App.listDocuments() else None
    if doc is None:
        return None
    mesh_obj, _problem = _find_mesh(doc, wanted)
    if mesh_obj is None:
        return None
    return recognize.survey(mesh_obj.Mesh).get("classification")


def _units_gate_report(doc_name, wanted):
    """(read-only, GUI thread) the units report when it is gate-ambiguous AND no durable
    user decision exists, else None — the §16 units ask is needed only then."""
    import FreeCAD as App

    from .recon import units as _units

    doc = App.getDocument(doc_name) if doc_name in App.listDocuments() else None
    if doc is None:
        return None
    mesh_obj, _problem = _find_mesh(doc, wanted)
    if mesh_obj is None:
        return None
    report = _units.analyse_units(mesh_obj.Mesh,
                                  source_path=getattr(mesh_obj, "OscarImportPath", None),
                                  user_unit=getattr(mesh_obj, "OscarSourceUnit", None) or None)
    return report if report.get("status") in _units.GATE_STATUSES else None


def _task(doc_name, mode, wanted, warn_mm, confirm_units=None,
          plan_override=None, repair_decision="", repair_approval=None,
          request="", turn_id="", call_id="", prefer_class=None):
    import FreeCAD as App

    from .recon import meshops, reconstruct, units

    doc = App.getDocument(doc_name) if doc_name in App.listDocuments() else None
    if doc is None:
        return {"error": f"document '{doc_name}' is not open"}
    mesh_obj, problem = _find_mesh(doc, wanted)
    if mesh_obj is None:
        return {"error": problem}
    authorized = ()
    if mode == "repair":
        if repair_decision not in ("approved", "safe_only"):
            return {"error": "repair requires a fresh decision from the user approval dialog"}
        if repair_decision == "approved":
            try:
                authorized = _consume_repair_approval(
                    repair_approval, doc_name, mesh_obj, request, turn_id, call_id)
            except ValueError as exc:
                return {"error": str(exc)}
        elif repair_approval is not None:
            return {"error": "safe-only repair cannot carry gated-operation approval"}
    if confirm_units in ("mm", "inch"):
        # durable user decision, travels with the document (K4-E)
        if "OscarSourceUnit" not in mesh_obj.PropertiesList:
            mesh_obj.addProperty("App::PropertyString", "OscarSourceUnit", "Oscar",
                                 "user-confirmed source unit of this mesh")
        mesh_obj.OscarSourceUnit = confirm_units
    user_unit = confirm_units or getattr(mesh_obj, "OscarSourceUnit", None) or None
    mesh = mesh_obj.Mesh
    doc.openTransaction("Oscar: reconstruct_mesh")
    try:
        if mode == "analyse":
            diag = meshops.diagnose(mesh)
            survey = reconstruct.recognize.survey(mesh, diag)
            units_report = units.analyse_units(
                mesh, source_path=getattr(mesh_obj, "OscarImportPath", None),
                user_unit=user_unit)
            from .recon import datums as _datums
            from .recon import quality as _quality

            try:
                scan_quality = _quality.scan_quality(
                    mesh, units_status=units_report.get("status"))
            except Exception:  # noqa: BLE001 - the report must never fail the analysis
                scan_quality = None
            datum_proposal = None
            try:
                datum_proposal = _datums.propose_datums(mesh, survey)
            except Exception:  # noqa: BLE001 - a proposal must never fail the analysis
                datum_proposal = None
            doc.abortTransaction()          # analysis changes nothing
            return {"mode": mode, "diag": diag, "survey": survey, "mesh": mesh_obj.Name,
                    "mesh_flow": _mesh_flow(mode, mesh_obj,
                                            {"diag": diag, "survey": survey,
                                             "units": units_report,
                                             "user_unit": user_unit}),
                    "units": units_report, "user_unit": user_unit,
                    "scan_quality": scan_quality, "datum_proposal": datum_proposal,
                    "analysis_payload": _analysis_payload(
                        mode, mesh_obj, doc_name, diag, survey, None, None,
                        extra={"units": units_report, "user_unit": user_unit,
                               "scan_quality": scan_quality})}
        if mode == "plan":
            # P2.3: the plan WITHOUT the build — the agent reads it, amends it,
            # and passes it back as plan_override (or lets the engine build)
            result = reconstruct.reconstruct(doc, mesh, warn_mm=warn_mm,
                                             user_unit=user_unit,
                                             authorized=authorized, preview=True)
            doc.abortTransaction()              # a preview changes nothing
            result["mode"] = mode
            result["mesh"] = mesh_obj.Name
            result["user_unit"] = user_unit
            result["mesh_flow"] = _mesh_flow(mode, mesh_obj, result, authorized)
            result["analysis_payload"] = _analysis_payload(
                mode, mesh_obj, doc_name, result.get("diagnostics"),
                result.get("survey"), None, None,
                extra={"preview": True, "plan": _plan_digest(result.get("plan")),
                       "units": result.get("units"), "user_unit": user_unit,
                       "scan_quality": result.get("scan_quality")})
            return result
        if mode == "repair":
            result = meshops.repair(mesh, authorized=authorized)
            fixed = doc.addObject("Mesh::Feature", f"{mesh_obj.Name}_repaired")
            fixed.Mesh = result["mesh"]
            fixed.Label = f"{mesh_obj.Name} repaired"
            if result.get("synthetic"):
                # durable provenance: mesh segments do NOT survive save/reopen (verified),
                # so the synthetic-region record travels as an object property
                if "OscarSyntheticRegions" not in fixed.PropertiesList:
                    fixed.addProperty("App::PropertyString", "OscarSyntheticRegions", "Oscar",
                                      "repair-created geometry (NOT measured scan evidence)")
                fixed.OscarSyntheticRegions = json.dumps(
                    [{"operation": s["operation"], "facet_count": s["count"]} for s in
                     result["synthetic"]], separators=(",", ":"))
            from .recon import provenance as _prov

            if "OscarProvenance" not in fixed.PropertiesList:
                fixed.addProperty("App::PropertyString", "OscarProvenance", "Oscar",
                                  "four-category provenance digest (K3)")
            fixed.OscarProvenance = json.dumps(
                _prov.digest_for_property(result.get("provenance") or {},
                                          result.get("provenance_verification") or {}),
                separators=(",", ":"), default=str)
            doc.recompute()
            doc.commitTransaction()
            return {"mode": mode, "repair": result, "mesh": mesh_obj.Name,
                    "created": fixed.Name,
                    "mesh_flow": _mesh_flow(mode, mesh_obj, result, authorized),
                    "analysis_payload": _analysis_payload(
                        mode, mesh_obj, doc_name, result["after"], None,
                        result.get("provenance"), result.get("provenance_verification"),
                        extra={"created": fixed.Name,
                               "validation": [{k: v[k] for k in ("operation", "verdict")}
                                              for v in result.get("validation", [])],
                               "missing_regions": [
                                   {k: l.get(k) for k in ("kind", "detail", "n_edges",
                                                          "perimeter_mm",
                                                          "spanned_area_mm2",
                                                          "centroid_mm")}
                                   for l in (result.get("missing_regions") or [])[:10]]})}
        if mode == "solid":
            from .cad.imported import solid_from_mesh

            shape, why = solid_from_mesh(mesh)
            if shape is None:
                doc.abortTransaction()
                return {"error": f"the mesh cannot become a solid: {why}"}
            obj = doc.addObject("Part::Feature", f"{mesh_obj.Name}_solid")
            obj.Shape = shape
            obj.Label = f"{mesh_obj.Name} solid (non-parametric)"
            doc.recompute()
            doc.commitTransaction()
            return {"mode": mode, "mesh": mesh_obj.Name, "created": obj.Name,
                    "facets": int(mesh.CountFacets),
                    "mesh_flow": _mesh_flow(mode, mesh_obj,
                                            {"created": obj.Name,
                                             "diag": meshops.diagnose(mesh),
                                             "survey": None})}
        result = reconstruct.reconstruct(doc, mesh, warn_mm=warn_mm, user_unit=user_unit,
                                         prefer_class=prefer_class,
                                         authorized=authorized,
                                         plan_override=plan_override)
        if result.get("build", {}).get("success") or not result.get("supported"):
            doc.commitTransaction()
        else:
            doc.commitTransaction()     # the builder already rolled its own features back
        result["mesh"] = mesh_obj.Name
        result["user_unit"] = user_unit
        result["mesh_flow"] = _mesh_flow(mode, mesh_obj, result, authorized)
        result["analysis_payload"] = _analysis_payload(
            mode, mesh_obj, doc_name, result.get("diagnostics"),
            result.get("survey"),
            result.get("provenance") or (result.get("repair") or {}).get("provenance"),
            result.get("provenance_verification")
            or (result.get("repair") or {}).get("provenance_verification"),
            extra={"outcome": result.get("outcome"), "supported": result.get("supported"),
                   "blocked": result.get("blocked"),
                   "units": result.get("units"), "user_unit": user_unit,
                   "unit_conversion": result.get("unit_conversion"),
                   "scan_quality": result.get("scan_quality"),
                   "measured_mask": result.get("measured_mask"),
                   "nominal_snaps": result.get("nominal_snaps") or [],
                   "residual_clusters": (result.get("residual_analysis") or {}).get("clusters") or [],
                   "plan": _plan_digest(result.get("plan")),
                   "missing_regions": len(result.get("missing_regions") or
                                          ((result.get("repair") or {}).get(
                                              "missing_regions") or [])),
                   "synthetic_dimension_flags": result.get("meta", {}).get(
                       "synthetic_dimension_flags")})
        return result
    except Exception:
        doc.abortTransaction()
        raise


def _fmt_list(items, limit=8):
    return ", ".join(str(i) for i in items[:limit]) + ("…" if len(items) > limit else "")


def _provenance_line(ledger, verification, facet_count, survey=None):
    """H6: plain-language provenance summary. Counts are COMPLETE (ledger.counts); the scope
    wording never implies full verification when only a subset was checked."""
    if not ledger or not (ledger.get("operations")):
        return "PROVENANCE: all original — no repair was applied to this mesh"
    counts = ledger.get("counts") or {}
    synthetic = counts.get("synthetic", len(ledger.get("synthetic") or []))
    modified = counts.get("modified", len(ledger.get("modified") or []))
    removed = counts.get("removed", len(ledger.get("removed") or []))
    uncertain = counts.get("uncertain", 0)
    original = max(0, (facet_count or 0) - synthetic - modified)
    verdict = (verification or {}).get("verdict", "UNKNOWN")
    scope = (verification or {}).get("scope") or ""
    line = (f"PROVENANCE: {verdict} · {original} original · {modified} modified · "
            f"{synthetic} synthetic · {removed} removed facet(s)")
    if scope == "partial":
        line += " · PARTIAL check (key lists were truncated)"
    if uncertain:
        line += f" · {uncertain} low-confidence region match(es)"
    reasons = (verification or {}).get("reasons") or []
    if verdict != "VERIFIED":
        line += " — some repaired regions could not be matched confidently to the scan"
        if reasons:
            line += " (" + "; ".join(reasons[:2]) + ")"
    return line


def _scan_quality_lines(sq) -> list:
    """G5: the human-readable SCAN QUALITY block (verdict first, evidence after)."""
    if not sq:
        return []
    r = sq.get("readiness") or {}
    comp, integ = sq.get("completeness") or {}, sq.get("integrity") or {}
    pl = sq.get("plausibility") or {}
    parts = []
    loops = comp.get("loops") or []
    if loops:
        kinds = {}
        for l in loops:
            kinds[l.get("kind", "?")] = kinds.get(l.get("kind", "?"), 0) + 1
        parts.append("%d boundary loop(s): %s" % (comp.get("boundary_loops", len(loops)),
                                                  ", ".join(f"{v}×{k}"
                                                            for k, v in sorted(kinds.items()))))
    paired = sum(1 for l in loops if l.get("through"))
    if paired:
        parts.append(f"{paired // 2} through-paired opening(s)")
    if integ.get("spikes", {}).get("count"):
        parts.append(f"{integ['spikes']['count']} spike(s)")
    ds = integ.get("doubled_surfaces") or []
    if ds:
        parts.append(f"{len(ds)} doubled-surface region(s) (first offset "
                     f"{ds[0]['offset_mm']} mm)")
    if pl.get("scale_status"):
        parts.append(f"scale {pl['scale_status']}")
    if pl.get("min_detectable_hole_radius_mm") is not None:
        parts.append(f"holes below r={pl['min_detectable_hole_radius_mm']} mm are below "
                     "scan resolution")
    out = ["SCAN QUALITY: %s%s" % (str(r.get("verdict", "unknown")).upper(),
                                   " · " + " · ".join(parts) if parts else "")]
    for reason in (r.get("reasons") or [])[:3]:
        out.append(f"  DEGRADED: {reason}")
    for note in (r.get("notes") or [])[:3]:
        out.append(f"  note: {note}")
    return out


def _render(result) -> str:
    mode = result.get("mode", "parametric")
    lines = [f"RECONSTRUCT_MESH {mode} on '{result.get('mesh', '')}'"]
    if result.get("analysis_file"):
        lines.append(f"analysis evidence: {result['analysis_file']} "
                     "(diagnostics, tolerances, survey digest, provenance ledger)")
        if result.get("macro_file"):
            lines.append(f"plan macro: {result['macro_file']} "
                         "(rerunnable .FCMacro of the same FeaturePlan)")
    if result.get("error"):
        lines.append(f"error: {result['error']}")
        return "\n".join(lines)
    if result.get("preview"):
        plan = result.get("plan") or {}
        meta = result.get("meta") or {}
        val = result.get("validation") or {}
        lines.append(f"route {meta.get('route')} · {len(plan.get('features') or [])} "
                     f"feature(s) · {len(plan.get('parameters') or {})} parameter(s)")
        for f in (plan.get("features") or [])[:24]:
            dim = f.get("length") or f.get("depth") or f.get("size")                 or f.get("diameter") or f.get("spacing") or ""
            lines.append(f"  {str(f.get('op', '')):>14} {f.get('id', '')} {dim}")
        for k in ("pads", "pockets", "holes", "patterns", "dressings",
                  "skipped_bands", "datums", "fillets", "chamfers"):
            if k in meta:
                lines.append(f"  {k}: {meta[k]}")
        if (result.get("nominal_snaps") or []):
            applied = sum(1 for s in result["nominal_snaps"] if s.get("applied"))
            lines.append(f"  nominal snapping: {applied} applied, "
                         f"{len(result['nominal_snaps']) - applied} suggested")
        if val.get("errors"):
            lines.append("VALIDATION ERRORS (fix before building):")
            lines.extend(f"  - {e}" for e in val["errors"][:8])
        elif val.get("warnings"):
            lines.extend(f"warning: {w}" for w in val["warnings"][:4])
        lines.append("RESULT CLASS: PLAN PREVIEW (nothing was created) — amend and pass "
                     "back as plan_override, or call mode='parametric' to build "
                     "Oscar's own plan")
        return "\n".join(lines)
    if mode == "analyse":
        d, s = result["diag"], result["survey"]
        lines.append(f"closed={d['closed']} non_manifold={d['non_manifold']} "
                     f"self_intersections={d['self_intersections']} "
                     f"inconsistent_normals={d['inconsistent_normals']} "
                     f"duplicates(pts/facets)={d['duplicate_points']}/{d['duplicate_facets']} "
                     f"degenerate={d['degenerate_facets']} facets={d['facets']} "
                     f"components={d['components']} size={d['size_mm']} mm")
        lines.append(f"class={s['classification']['class']} evidence={s['classification']['evidence']}")
        q = s.get("quality") or {}
        if q.get("sigma_mm") is not None:
            lines.append(f"noise sigma={q['sigma_mm']} mm (mean edge {q.get('mean_edge_mm')} mm) "
                         f"→ gates: plane {q.get('plane_rms_mm')}, cylinder {q.get('cylinder_rms_mm')} mm")
        lines.append(_provenance_line(None, None, d.get("facets")))
        lines.extend(_units_lines(result.get("units"), result.get("user_unit")))
        lines.extend(_scan_quality_lines(result.get("scan_quality")))
        dp = result.get("datum_proposal") or {}
        if dp.get("primary_plane"):
            lines.append("DATUMS (proposal): primary plane %.1f mm2 · secondary %s"
                         " · tertiary %s%s" % (
                             dp["primary_plane"]["area_mm2"],
                             dp["secondary_axis"]["basis"],
                             dp["tertiary_point"]["basis"],
                             " · AMBIGUOUS" if dp.get("ambiguities") else ""))
        lines.append(f"coverage={s['coverage']} holes={len(s['holes'])} "
                     f"parallel_pairs={len(s['parallel_planes'])}")
        lines.append("RESULT CLASS: ANALYSIS ONLY (nothing was created or changed)")
        return "\n".join(lines)
    if mode == "repair":
        r = result["repair"]
        lines.append(f"created '{result['created']}' (a repaired COPY; '{result['mesh']}' is untouched)")
        lines.append(f"actions: {_fmt_list([a['operation'] for a in r['actions']])}")
        verdicts = [v.get("verdict") for v in r.get("validation", [])]
        if verdicts:
            counts = {v: verdicts.count(v) for v in ("ACCEPTED", "NEEDS_REVIEW",
                                                     "REJECTED", "FAILED") if v in verdicts}
            lines.append("validation: " + ", ".join(f"{k}×{n}" for k, n in counts.items()))
            for v in r["validation"]:
                if v["verdict"] in ("REJECTED", "NEEDS_REVIEW", "FAILED"):
                    lines.append(f"  {v['operation']}: {v['verdict']} — "
                                 f"{_fmt_list(v.get('reasons', []))}")
        if r.get("synthetic"):
            total = sum(s["count"] for s in r["synthetic"])
            lines.append(f"SYNTHETIC: {total} filled facet(s) recorded on the copy — repair "
                         "geometry, NOT measured scan evidence (exclude from deviation "
                         "coverage)")
        for a in r.get("actions") or []:
            if a.get("verdict") == "SKIPPED":
                lines.append(f"SKIPPED {a['operation']}: {a.get('skipped', 'policy')}")
        missing = r.get("missing_regions") or []
        if missing:
            biggest = max(missing, key=lambda l: l["spanned_area_mm2"])
            lines.append(f"MISSING REGIONS: {len(missing)} unfilled opening(s) recorded "
                         f"(largest {biggest['spanned_area_mm2']} mm² at "
                         f"{biggest['centroid_mm']}) — never auto-filled (G6 policy), "
                         "excluded from deviation coverage as unmeasured")
        lines.append(_provenance_line(r.get("provenance"),
                                      r.get("provenance_verification"),
                                      r["after"].get("facets")))
        if r["refused"]:
            specs = r.get("refused_specs") or {}
            shown = [f"{op}({', '.join(f'{k}={v}' for k, v in sorted(specs[op].items()))})"
                     if specs.get(op) else op for op in r["refused"]]
            lines.append(f"gated operations not run without human approval: {_fmt_list(shown)} — "
                         "the approval dialog shows these effective parameters before execution")
        lines.append(f"before: closed={r['before']['closed']} facets={r['before']['facets']} · "
                     f"after: closed={r['after']['closed']} facets={r['after']['facets']}")
        if r["unresolved"]:
            lines.append(f"UNRESOLVED: {r['unresolved']}")
        lines.append("RESULT CLASS: REPAIRED MESH (a mesh, not CAD)")
        return "\n".join(lines)
    if mode == "solid":
        lines.append(f"created '{result['created']}' from {result['facets']:,} triangles — a "
                     "sewn faceted B-rep solid")
        lines.append("RESULT CLASS: NON-PARAMETRIC SOLID (no sketches, no features, no driving "
                     "dimensions — never report it as parametric CAD)")
        return "\n".join(lines)
    # parametric
    if not result.get("supported"):
        if result.get("blocked") == "mesh_not_closed":
            lines.append("BLOCKED: " + result["note"])
            if result.get("missing_regions"):
                lines.append(f"  {len(result['missing_regions'])} missing region(s) recorded "
                             "with their rings (repair.missing_regions) — mask/deviation "
                             "data for when those regions are measured")
            lines.append("RESULT CLASS: RECONSTRUCTION UNVERIFIED (nothing was built)")
            return "\n".join(lines)
        lines.append(f"route not implemented: {result.get('note', '')}")
        lines.append(f"classification: {result.get('classification')}")
        lines.append("RESULT CLASS: RECONSTRUCTION UNVERIFIED (no approximation was labelled a "
                     "reconstruction)")
        return "\n".join(lines)
    build = result.get("build", {})
    if not build.get("success"):
        lines.append(f"the plan build failed: {build.get('error', '')}")
        lines.append("RESULT CLASS: RECONSTRUCTION FAILED")
        return "\n".join(lines)
    meta = result.get("meta", {})
    lines.append(f"route={meta.get('route')} holes={meta.get('holes')} "
                 f"class={result['classification']['class']}")
    lines.extend(_units_lines(result.get("units"), result.get("user_unit")))
    if result.get("unit_conversion"):
        cv = result["unit_conversion"]
        lines.append(f"  applied conversion ×{cv['factor']} on a working copy (validated: "
                     f"{cv.get('validation', {}).get('valid')}) — the original mesh is "
                     "unchanged")
    if meta.get("units_unconfirmed"):
        lines.append("LIMITATION: the unit interpretation is UNCONFIRMED "
                     f"({meta['units_unconfirmed']}) — the outcome is capped at PARTIAL "
                     "until the user confirms the source units")
    if meta.get("scan_degraded"):
        for i, reason in enumerate(meta["scan_degraded"][:3]):
            lines.append(("LIMITATION: scan quality DEGRADED — the outcome is capped at "
                          "PARTIAL: " if i == 0 else "  ") + reason)
    elif (result.get("scan_quality") or {}).get("readiness", {}).get("verdict") == "caution":
        for note in result["scan_quality"]["readiness"].get("notes", [])[:2]:
            lines.append(f"scan note: {note}")
    lines.append(_provenance_line(
        result.get("provenance") or (result.get("repair") or {}).get("provenance"),
        result.get("provenance_verification")
        or (result.get("repair") or {}).get("provenance_verification"),
        (result.get("diagnostics") or {}).get("facets")))
    lines.append(f"body='{build.get('body')}' sheet='{build.get('spreadsheet')}' "
                 f"features={build.get('feature_count')} params={build.get('parameter_count')}")
    cv = meta.get("coverage_vector")
    if cv:
        cov = (f"coverage: native {cv.get('recognized_native')} · "
               f"unrecognized {cv.get('unrecognized')}")
        if cv.get("skipped_level_bands"):
            cov += f" · {cv['skipped_level_bands']} skipped level band(s)"
        lines.append(cov)
    rc = meta.get("reconciliation")
    if rc:
        rline = (f"reconciliation: holes {rc.get('holes_native')}/{rc.get('holes_recognized')} "
                 f"native · dressings {rc.get('dress_evidence_strips')} strip(s) → "
                 f"{rc.get('dress_ops')} op(s)")
        if rc.get("patterns"):
            rline += f" · {rc['patterns']} pattern(s)"
        lines.append(rline)
        for u in rc.get("unexplained") or []:
            lines.append(f"  UNEXPLAINED: {u}")
    if result.get("deviation"):
        lines.append(fit_block(result["deviation"]))
    iou = result.get("exact_iou") or {}
    if iou.get("iou") is not None:
        lines.append(f"EXACT IOU (opt-in evidence; volume overlap only, not used by the result "
                     f"class): {iou['iou']} · {iou.get('elapsed_s', 0.0)} s")
    elif os.environ.get("OSCAR_RECON_IOU", "") == "1":
        lines.append(f"EXACT IOU (opt-in): {iou.get('note') or 'not computed'}")
    from .recon.localize import localization_block
    lines.extend(localization_block(result.get("localized_deviations")))
    mm = result.get("measured_mask")
    if mm and (mm.get("scan_excluded") or mm.get("model_excluded")):
        lines.append(f"MEASURED-ONLY DEVIATION: {mm['scan_excluded']} synthetic scan "
                     f"vertex/vertices and {mm['model_excluded']} model sample(s) in "
                     f"{mm['regions']} missing region(s) excluded — the rows above cover "
                     "measured regions only")
    reg = result.get("registration")
    if reg is not None:
        aligned = result.get("deviation_aligned")
        if reg.get("valid") and reg.get("ambiguous"):
            state = ("AMBIGUOUS — several alignments fit equally well; the aligned numbers "
                     "below are unreliable and never affect the outcome")
        elif reg.get("valid"):
            state = "valid and converged (a lower-bound diagnostic; it never affects the outcome)"
        else:
            state = "INVALID — aligned deviation withheld, raw numbers above remain authoritative"
        lines.append("REGISTRATION (ICP, %s): %s" % (reg.get("method"), state))
        if aligned:
            raw_h = next((r["value"] for r in result["deviation"] if r["id"] == "hausdorff"), None)
            alg_h = next((r["value"] for r in aligned if r["id"] == "hausdorff"), None)
            raw_r = next((r["value"] for r in result["deviation"] if r["id"] == "scan_to_model.rms"), None)
            alg_r = next((r["value"] for r in aligned if r["id"] == "scan_to_model.rms"), None)
            lines.append("  before → after registration: hausdorff %s → %s mm, scan_to_model.rms %s → %s mm "
                         "(aligned is a LOWER BOUND; the raw numbers decide the outcome)" %
                         (raw_h, alg_h, raw_r, alg_r))
        lines.append("  transform (scan→model): rotation %.4f deg, translation %.4f mm, %d iterations, "
                     "%d correspondences, residual RMSE %s mm" % (
                         reg.get("rotation_deg", 0.0), reg.get("translation_mm", 0.0),
                         reg.get("iterations", 0), reg.get("correspondences", 0),
                         reg.get("residual_rmse_mm")))
        if reg.get("weak_translation_axes"):
            lines.append("  WEAKLY OBSERVED translation along %s (e.g. planar faces cannot see "
                         "in-plane sliding)" % ", ".join(reg["weak_translation_axes"]))
        for warning in reg.get("warnings", [])[:4]:
            lines.append("  note: %s" % warning)
    if meta.get("extra_geometry"):
        lines.append("LIMITATION: " + meta["extra_geometry"])
    snaps = result.get("nominal_snaps") or []
    applied = [s for s in snaps if s.get("applied")]
    if applied:
        lines.append("NOMINALS (G8): " + "; ".join(
            f"{s['param']} {s['measured']:g}→{s['nominal']:g} "
            f"(Δ{s['delta']:.3g}, {s['sigma_ratio']:.2g}σ)" for s in applied[:4]) +
            ("…" if len(applied) > 4 else ""))
    suggested = [s for s in snaps if not s.get("applied")]
    if suggested:
        lines.append("  drill candidates (suggestion only, not forced): " + "; ".join(
            f"{s['param']} Ø{s['measured']:g}→{s['nominal']:g}" for s in suggested[:4]))
    outcome = result.get("outcome", "reconstruction_unverified")
    label = {"verified_parametric": "PARAMETRIC CAD MODEL (verified against the scan)",
             "partially_verified_parametric": "PARTIALLY PARAMETRIC MODEL",
             }.get(outcome, "RECONSTRUCTION UNVERIFIED")
    lines.append(f"RESULT CLASS: {label}")
    return "\n".join(lines)


@tool(
    "reconstruct_mesh",
    "Mesh workflows for an imported scan: parametric (default: native editable CAD), analyse, "
    "repair (a copy; gated ops require explicit user approval), solid (non-parametric), plan (PREVIEW the "
    "FeaturePlan without building; amend it and pass it back as plan_override). Always states "
    "its RESULT CLASS.",
    {"type": "object", "properties": {
        "doc_name": _DOC,
        "object": {"type": "string", "description": "The mesh object; default = the document's mesh."},
        "mode": {"type": "string", "enum": list(_MODES), "default": "parametric"},
        "warn_mm": {"type": "number", "default": 0.3, "minimum": 0.01, "maximum": 5,
                    "description": "WARN threshold for the deviation rows, mm."},
        "confirm_units": {"type": "string", "enum": ["mm", "inch"],
                          "description": "The user-confirmed source unit of the mesh. "
                                         "'inch' converts a working copy ×25.4 before "
                                         "reconstruction (the original is never modified); "
                                         "recorded on the mesh as OscarSourceUnit."},
        "plan_override": {"type": "object",
                          "description": "An AMENDED FeaturePlan (the object returned by "
                                         "mode='plan') to build AS GIVEN instead of the "
                                         "engine's own synthesis — same builder, same "
                                         "honesty caps, same deviation rows."}},
     "required": ["doc_name"]},
    category="build", mutating=True, geometry=True,
    title=lambda a: "Mesh: %s" % (a.get("mode") or "parametric"),
)
def _reconstruct_mesh(doc_name: str, object: str | None = None, mode: str = "parametric",  # noqa: A002
                      warn_mm: float = 0.3, confirm_units: "str | None" = None,
                      plan_override: "dict | None" = None):
    ctx = current_context()
    if mode not in _MODES:
        from .tools import ToolError

        raise ToolError(f"reconstruct_mesh: mode must be one of {', '.join(_MODES)}")
    if confirm_units is not None and confirm_units not in ("mm", "inch"):
        from .tools import ToolError

        raise ToolError("reconstruct_mesh: confirm_units must be 'mm' or 'inch' "
                        "(omit it to leave the source unit unknown)")

    if plan_override is not None and not isinstance(plan_override, dict):
        from .tools import ToolError

        raise ToolError("reconstruct_mesh: plan_override must be the plan OBJECT "
                        "(as returned by mode='plan'), not a string")

    request = str(getattr(ctx, "request", "") or
                  getattr(getattr(ctx, "ledger", None), "request", "") or "")
    repair_decision = ""
    repair_approval = None
    if mode == "repair":
        if not request.strip():
            from .tools import ToolError

            raise ToolError("reconstruct_mesh repair: the original user request is unavailable; "
                            "gated repair cannot be approved safely")
        if not str(getattr(ctx, "turn_id", "") or "") or not str(getattr(ctx, "call_id", "") or ""):
            from .tools import ToolError

            raise ToolError("reconstruct_mesh repair: missing turn or tool-call identity; "
                            "gated repair cannot be approved safely")
        proposal, failure = _gui_call(
            lambda: _repair_preflight(doc_name, str(object or "")),
            "reconstruct_mesh repair approval", _TIMEOUT_S)
        if failure or not isinstance(proposal, dict):
            from .tools import ToolError

            raise ToolError(f"reconstruct_mesh repair: approval preflight failed: {failure or 'no result'}")
        if proposal.get("error"):
            from .tools import ToolError

            raise ToolError(f"reconstruct_mesh repair: {proposal['error']}")
        specs = proposal["specs"]
        digest = str(proposal["mesh_digest"])
        mesh_name = str(proposal["mesh_name"])
        detail = "\n".join(f"- {op}: {json.dumps(spec, sort_keys=True) or '(no parameters)'}"
                            for op, spec in specs.items())
        from .netguard import Choice, ask_user

        answer = ask_user(
            "Approve mesh repair operations?",
            f"The repair will work on a copy of mesh '{mesh_name}'. Review the gated operations below.",
            informative=(f"Original request: {request[:1000]}\n"
                         f"Mesh digest: {digest}\n"
                         "Approval is for this mesh and this repair call only."),
            detailed=detail,
            choices=(Choice("approve", "Approve listed operations"),
                     Choice("safe_only", "Safe repairs only", role="NoRole"),
                     Choice("cancel", "Cancel repair", role="RejectRole")),
            default="cancel", object_name="oscarRepairApproval", op="reconstruct_mesh_repair")
        if answer == "approve":
            operations = tuple(specs)
            packed_specs = tuple((op, tuple(sorted(spec.items()))) for op, spec in specs.items())
            repair_approval = _RepairApproval(
                doc_name=doc_name, mesh_name=mesh_name, mesh_digest=digest,
                operations=operations, specs=packed_specs,
                request_digest=_request_digest(request), turn_id=str(ctx.turn_id),
                call_id=str(ctx.call_id),
                binding=_repair_binding(digest, specs, request, ctx.turn_id, ctx.call_id))
            repair_decision = "approved"
        elif answer == "safe_only":
            repair_decision = "safe_only"
        else:
            from .tools import ToolError

            raise ToolError(f"reconstruct_mesh repair: user approval was not granted ({answer})")

    units_decision = None
    if mode in ("parametric", "plan") and not confirm_units:
        # §16 units ask: only when the evidence is gate-ambiguous, a live decision-card surface
        # exists and this chat can answer it. Headless behaviour is unchanged (honest gate).
        try:
            from . import tools_ask

            if tools_ask.surface is not None and str(getattr(ctx, "conversation_id", "") or ""):
                report, gate_failure = _gui_call(
                    lambda: _units_gate_report(doc_name, str(object or "")),
                    "reconstruct_mesh units check", _TIMEOUT_S)
                if not gate_failure and isinstance(report, dict):
                    from .recon import askwire
                    from .recon.approve import AnswerStore

                    root = str(getattr(getattr(ctx, "project", None), "root", "") or "")
                    store = AnswerStore(os.path.join(root, "analysis"),
                                        str(getattr(getattr(ctx, "project", None), "name", "")
                                            or "oscar")) if root else None
                    unit, units_decision = askwire.resolve_units(
                        report, ask=lambda card: tools_ask.ask_user(**card), store=store)
                    if unit in ("mm", "inch"):
                        confirm_units = unit
        except Exception:   # noqa: BLE001 - the ask is an enhancement, never a blocker
            units_decision = None
    prefer_class = None
    if mode in ("parametric", "plan") and tools_ask_surface_live():
        # §16 ambiguous interpretation: approve's own gate fires only for a low-confidence
        # classification WITH a real alternative — a clear winner never asks. The user's
        # confirmed alternative steers THIS run; headless keeps the measured classification.
        try:
            from . import tools_ask
            from .recon import askwire

            classification, probe_failure = _gui_call(
                lambda: _classification_probe(doc_name, str(object or "")),
                "reconstruct_mesh classification probe", _TIMEOUT_S)
            if (not probe_failure and isinstance(classification, dict)
                    and classification.get("alternatives")):
                root = str(getattr(getattr(ctx, "project", None), "root", "") or "")
                store = _askuser_store(root, ctx) if root else None
                prefer_class, _decision = askwire.resolve_interpretation(
                    classification, ask=lambda card: tools_ask.ask_user(**card), store=store)
        except Exception:   # noqa: BLE001 - the ask is an enhancement, never a blocker
            prefer_class = None

    def task():
        return _task(doc_name, mode, str(object or ""), float(warn_mm),
                     confirm_units, plan_override, repair_decision, repair_approval,
                     request, str(getattr(ctx, "turn_id", "") or ""),
                     str(getattr(ctx, "call_id", "") or ""), prefer_class)

    result, failure = _gui_call(task, "reconstruct_mesh", _TIMEOUT_S)
    if failure or not isinstance(result, dict):
        from .tools import ToolError

        raise ToolError(f"reconstruct_mesh: {failure or 'the task returned nothing'}")
    if units_decision is not None:
        result["units_decision"] = units_decision
    mesh_flow = result.pop("mesh_flow", None)
    if mesh_flow is not None:
        # the flow Ledger's evidence channel (same pattern as reference_result / param_names)
        ctx.state.setdefault("extra", {})["mesh_flow"] = mesh_flow
    payload = result.pop("analysis_payload", None)
    if payload is not None and getattr(ctx, "project", None) is not None:
        try:                                    # evidence file must never fail the tool
            result["analysis_file"] = ctx.project.write_analysis(
                payload.get("mesh") or doc_name, payload)
        except Exception:  # noqa: BLE001
            result["analysis_file"] = None
        plan = result.get("plan")
        if plan is not None:
            try:                                # S8: the plan as a rerunnable .FCMacro
                from .plan.builder import macro_text

                macro_dir = ctx.project.path("analysis")
                os.makedirs(macro_dir, exist_ok=True)
                macro_path = os.path.join(
                    macro_dir, f"{payload.get('mesh') or doc_name}_recon.FCMacro")
                with open(macro_path, "w", encoding="utf-8") as fh:
                    fh.write(macro_text(plan))
                result["macro_file"] = ctx.project.rel(macro_path) or macro_path
            except Exception:  # noqa: BLE001 - the macro is a bonus, never a blocker
                result["macro_file"] = None
    text = _render(result)
    rows = result.get("deviation") or []
    if rows and getattr(ctx, "project", None) is not None:
        try:
            from .tools_verify import _after_step, _persist, _ran_ids

            _persist(ctx, {"doc": doc_name, "fit_rows": rows, "fit_ran": True, "fail_rows": [],
                           "ran": _ran_ids(False, rows, True, [], {}),
                           "after_step": _after_step(ctx)}, doc_name)
        except Exception:  # noqa: BLE001 - persistence must never fail the reconstruction
            pass
    if result.get("error"):
        from .tools import ToolError

        raise ToolError(text)
    return text, None
