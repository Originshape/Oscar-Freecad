"""P5.4: the local PARAMETRIC-QUALITY grader (Parametric CAD Bench V3
inspired, clean-room): separate GEOMETRY rows and PARAMETRIC rows for a
reconstruction-built body — never one blended score that could hide a serious
individual failure.

Geometry rows compare the built body with the scan's own solid (sewn from the
mesh when bounded): volume ratio, bbox agreement per axis. Parametric rows
check what makes the model genuinely editable: native-feature share (no loose
primitives), sketch solver status and full-constraint share, spreadsheet
expression-binding count, recompute stability after a forced touch-all, a
±10 % edit probe on the most-bound numeric parameter (geometry must follow and
stay valid), and that nothing is a faceted fallback. PASS/WARN only — the
caller decides what gates.

Pure FreeCAD/Part + the recon deviation helpers; call on the GUI thread in
GUI sessions (console runs inline)."""

KIND = "recon_grader"

_EDIT_DELTA = 0.10


def _row(rid, metric, value, limit, status, detail):
    return {"id": rid, "kind": KIND, "a": "body", "b": "sheet", "metric": metric,
            "value": round(float(value), 4) if isinstance(value, (int, float)) else value,
            "limit": limit, "status": status, "detail": detail[:220]}


def grade(doc, body_name, mesh=None, sheet_name=None, edit_probe=True) -> list:
    """Grade one reconstruction body. ``mesh`` (a Mesh.Mesh) enables the
    geometry rows; ``sheet_name`` defaults to the first spreadsheet."""
    body = doc.getObject(body_name)
    if body is None:
        return [_row("grader.body", "present", 0, 1, "WARN", f"no object '{body_name}'")]
    rows = []
    shape = body.Shape

    # ---- parametric rows (always available) ----
    feats = [o for o in body.Group
             if o.isDerivedFrom("PartDesign::Feature")
             and not o.isDerivedFrom("PartDesign::Body")]
    sketches = [o for o in body.Group if o.isDerivedFrom("Sketcher::SketchObject")]
    loose = [o for o in doc.Objects
             if o.isDerivedFrom("Part::Feature")
             and not o.isDerivedFrom("PartDesign::Feature")
             and not o.isDerivedFrom("PartDesign::Body")
             and not getattr(o, "NoTouch", False)
             and o.Visibility]
    native = len(feats) - len([o for o in feats if o.TypeId == "PartDesign::Feature"])
    rows.append(_row("grader.parametric.native_features", "count", max(native, 0),
                     1, "PASS" if native >= 1 else "WARN",
                     f"{len(feats)} PartDesign feature(s), {len(sketches)} sketch(es)"))
    rows.append(_row("grader.parametric.loose_solids", "count", len(loose), 0,
                     "PASS" if not loose else "WARN",
                     "loose Part primitives visible in the document: "
                     + ", ".join(o.Name for o in loose[:5])))
    bad_sk = [s.Name for s in sketches if "Invalid" in s.State or s.FullyConstrained is False]
    fc = sum(1 for s in sketches if s.FullyConstrained)
    rows.append(_row("grader.parametric.sketches_fully_constrained", "share",
                     (fc / len(sketches)) if sketches else 1.0, 0.95,
                     "PASS" if not bad_sk else "WARN",
                     f"{fc}/{len(sketches)} fully constrained"
                     + (f"; problems: {', '.join(bad_sk[:4])}" if bad_sk else "")))
    bindings = sum(len(getattr(o, "ExpressionEngine", []) or []) for o in feats)
    rows.append(_row("grader.parametric.expression_bindings", "count", bindings, 1,
                     "PASS" if bindings >= max(1, len(feats) // 2) else "WARN",
                     f"{bindings} expression binding(s) across {len(feats)} feature(s)"))

    # recompute stability: touch everything, one forced recompute, still valid
    import FreeCAD as App  # noqa: F401  (document operations below)

    for o in list(body.Group):
        o.touch()
    doc.recompute()
    stable = shape.isValid() and "Invalid" not in body.State
    rows.append(_row("grader.parametric.recompute_stable", "bool", 1 if stable else 0, 1,
                     "PASS" if stable else "WARN",
                     "touch-all + forced recompute keeps the body valid"
                     if stable else f"recompute broke the body: {body.getStatusString()}"))

    sheet = doc.getObject(sheet_name) if sheet_name else next(
        (o for o in doc.Objects if o.isDerivedFrom("Spreadsheet::Sheet")), None)
    if sheet is not None and edit_probe and bindings:
        # the ±10% probe on the most-referenced numeric alias
        from collections import Counter

        refs = Counter()
        for o in feats:
            for prop, expr in (getattr(o, "ExpressionEngine", []) or []):
                for token in str(expr).replace("(", " ").replace(")", " ").split():
                    if token.startswith(f"{sheet.Name}."):
                        refs[token.split(".", 1)[1]] += 1
        if refs:
            alias = refs.most_common(1)[0][0]
            try:
                cell = sheet.getCellFromAlias(alias)
                base = sheet.getContents(cell)
                got = sheet.get(alias)
                val = float(got.Value) if hasattr(got, "Value") else float(got)
                unit = ""
                text = str(got)
                if " " in text:
                    unit = text.split(None, 1)[1].strip()
                ok_edits = 0
                v0 = body.Shape.Volume
                for sign in (1, -1):
                    new = f"={val * (1 + sign * _EDIT_DELTA)}"
                    if unit:
                        new += f" {unit}"
                    sheet.set(cell, new)
                    doc.recompute()
                    if (abs(body.Shape.Volume - v0) > 1e-6 and body.Shape.isValid()
                            and "Invalid" not in body.State):
                        ok_edits += 1
                sheet.set(cell, base)
                doc.recompute()
                rows.append(_row("grader.parametric.edit_probe", "passed_edits", ok_edits,
                                 2, "PASS" if ok_edits == 2 else "WARN",
                                 f"±10% on '{alias}' ({refs[alias]} binding(s)): geometry "
                                 "followed and stayed valid both ways"
                                 if ok_edits == 2 else
                                 f"±10% on '{alias}': only {ok_edits}/2 edits followed "
                                 "cleanly — the tree is not robustly editable"))
            except Exception as exc:  # noqa: BLE001
                rows.append(_row("grader.parametric.edit_probe", "error", 0, 2, "WARN",
                                 f"probe on '{alias}' failed: {exc}"))

    # ---- geometry rows (need the scan) ----
    if mesh is not None:
        try:
            from ..cad.imported import solid_from_mesh

            scan_solid, _why = solid_from_mesh(mesh)
            if scan_solid is not None and scan_solid.Volume > 0:
                inter = shape.common(scan_solid).Volume
                union = shape.Volume + scan_solid.Volume - inter
                iou = inter / union if union > 0 else 0.0
                rows.append(_row("grader.geometry.iou", "share", iou, 0.98,
                                 "PASS" if iou >= 0.98 else "WARN",
                                 f"boolean IoU body vs sewn scan: {iou:.4f}"))
                bbs, bbb = shape.BoundBox, scan_solid.BoundBox
                for axis, a, b in (("dx", bbs.XLength, bbb.XLength),
                                   ("dy", bbs.YLength, bbb.YLength),
                                   ("dz", bbs.ZLength, bbb.ZLength)):
                    rel = abs(a - b) / max(b, 1e-9)
                    rows.append(_row(f"grader.geometry.bbox_{axis}", "rel_err", rel, 0.01,
                                     "PASS" if rel <= 0.01 else "WARN",
                                     f"{a:.3f} vs {b:.3f} mm"))
        except Exception as exc:  # noqa: BLE001
            rows.append(_row("grader.geometry.iou", "error", 0, 0.98, "WARN",
                             f"geometry grading unavailable: {exc}"))
    return rows
