"""The mesh workflow wiring of the flow (private module of ``oscar.flow``, W-1): the mesh
intent resolver, the M0-M14 mesh Ledger template, evidence-driven step completion and the
controlled retarget. Kept in its own module so ``flow.py`` stays under its size invariant;
``flow.py`` imports this at the BOTTOM of the file (the same pattern as ``flow_run``), so
``LedgerStep``/``NOT_REQUESTED`` are already defined when this loads.

 PURE PYTHON (no FreeCAD, no Qt): everything here is a function of typed text and recorded
tool facts. A step completes ONLY on recorded facts; a failed/blocked result blocks its step
with the reason; optional stages seed as skipped and are WOKEN by real evidence.
"""

from .flow_intent import mesh_ambiguous, mesh_operation

MESH_INTENTS = ("analyse", "repair", "solid", "brep", "parametric", "modify", "ambiguous")

_MESH_STEP_DEPS = {  # stated in each step's acceptance text; phases order them implicitly
    "M2": "M1", "M3": "M1", "M4": "M2", "M5": "M2", "M6": "M5", "M7": "M6",
    "M8": "M4", "M9": "M4", "M10": "M9", "M11": "M10", "M12": "M11", "M13": "M12", "M14": "M13",
}


def resolve_mesh_intent(text: str) -> "str | None":
    """The one entry point ``_begin_turn`` uses — ``"ambiguous"`` when the user offered
    several operations without choosing, ``mesh_operation`` otherwise (explicit wins; a bare
    mesh import defaults to parametric; non-mesh and deliver phrasing return None)."""
    return "ambiguous" if mesh_ambiguous(text) else mesh_operation(text)


def _mesh_steps(mesh_intent: str) -> list:
    """The M0-M14 mesh template pruned to the requested workflow. ``analyse`` stops at the
    evidence stages; ``repair`` adds the repair stages; ``solid``/``brep`` add conversion
    (BREP is the same sewn-triangle path, labelled faceted and NEVER parametric);
    ``parametric``/``modify`` run the full route (modify = edits through the parametric
    model); ``ambiguous`` seeds only M0 — no executable build workflow before the user
    chooses."""
    from .flow import LedgerStep, _step

    def dep(sid: str, base: str) -> str:
        d = _MESH_STEP_DEPS.get(sid, "")
        return f"{base} (after {d})" if d else base

    core = [
        _step("M1", "SOURCE", "Import and source identity",
              dep("M1", "mesh imported; sha256 + source path recorded"),
              hint=["import_file", "reconstruct_mesh"]),
        _step("M2", "PLAN", "Mesh diagnosis (read-only)",
              dep("M2", "diagnostics recorded; the source mesh untouched"), owner="app",
              hint=["reconstruct_mesh"]),
        _step("M3", "PLAN", "Unit and scale verification",
              dep("M3", "unit report recorded; a mismatch/ambiguity is user-confirmed"),
              owner="app", hint=["reconstruct_mesh"]),
        _step("M4", "PARAMS", "Quality assessment",
              dep("M4", "noise sigma and tolerances recorded"), owner="app",
              hint=["reconstruct_mesh"]),
    ]
    repair = [
        _step("M5", "BUILD", "Repair recommendation",
              dep("M5", "per-operation verdicts/recommendations recorded"), owner="app",
              required=False, hint=["reconstruct_mesh"]),
        _step("M6", "BUILD", "User repair approval (gated operations only)",
              dep("M6", "gated operations authorized, or none were needed"), owner="app",
              required=False, hint=["reconstruct_mesh"]),
        _step("M7", "BUILD", "Repair execution and validation (on a copy)",
              dep("M7", "repair ran with ACCEPTED verdicts; REJECTED/FAILED blocks this step"),
              required=False, hint=["reconstruct_mesh"]),
    ]
    solid_title = "Faceted BREP conversion (sewn triangles — non-parametric)" \
        if mesh_intent == "brep" else "Solid conversion (non-parametric)"
    conversion = [
        _step("M8", "BUILD", solid_title,
              dep("M8", "solid/BREP object exists and is labelled non-parametric"),
              hint=["reconstruct_mesh"]),
    ]
    parametric = repair + [
        _step("M9", "BUILD", "Geometry survey and recognition",
              dep("M9", "survey with regions/holes/coverage recorded"), owner="app",
              hint=["reconstruct_mesh"]),
        _step("M10", "BUILD", "Reconstruction planning",
              dep("M10", "FeaturePlan recorded with evidence sources"), owner="app",
              hint=["reconstruct_mesh"]),
        _step("M11", "BUILD", "Parametric construction"
              + (" (apply the requested change to the editable model)" if mesh_intent == "modify" else ""),
              dep("M11", "builder success; a failed build blocks this step"),
              hint=["reconstruct_mesh", "apply_feature_plan"]),
        _step("M12", "VERIFY", "Accuracy validation against the scan",
              dep("M12", "deviation rows ran after the last geometry change"), owner="app",
              hint=["reconstruct_mesh", "compare_to_reference"]),
        _step("M13", "VERIFY", "User review",
              dep("M13", "flags (synthetic dims, unconfirmed units, uncertain provenance) "
                         "escalated; unconfirmed never reads verified"), owner="app",
              required=False),
        _step("M14", "DELIVER", "Final delivery",
              dep("M14", "exports + analysis evidence file present"), owner="app",
              hint=["export_model", "save_document"]),
    ]
    for step in parametric:
        if step.id == "M13":
            # W-1 hardening: user review can NEVER be auto-completed or auto-skipped — it
            # stays pending until an explicit user-acknowledgement mechanism exists
            step.status = "pending"
            step.note = ("awaiting explicit user acknowledgement; automatic validation "
                         "must never complete or skip this step")
    if mesh_intent == "ambiguous":
        return [
            _step("M0", "INTAKE", "Clarify the mesh operation with the user",
                  "the user chose one operation (analyse / repair / solid / brep / parametric / modify)",
                  owner="app"),
        ]
    if mesh_intent == "analyse":
        return core
    if mesh_intent == "repair":
        return core + repair + [
            _step("M14", "DELIVER", "Deliver the repaired copy and its evidence",
                  dep("M14", "repaired copy + repair evidence present"), owner="app",
                  hint=["export_model", "save_document"]),
        ]
    if mesh_intent in ("solid", "brep"):
        return core + conversion + [
            _step("M12", "VERIFY", "Validate the conversion honestly",
                  dep("M12", "labelled non-parametric; never reported as parametric CAD"),
                  owner="app", hint=["reconstruct_mesh"]),
            _step("M14", "DELIVER", "Final delivery", dep("M14", "exports present"), owner="app",
                  hint=["export_model", "save_document"]),
        ]
    return core + parametric


def _newest_identity(ledger, sid: str) -> "dict | None":
    """The newest GEOMETRY IDENTITY (``evidence["mesh"]``) recorded on a step, or None."""
    step = next((s for s in ledger.steps if s.id == sid), None)
    if step is None:
        return None
    for e in reversed(step.evidence):
        if isinstance(e, dict) and isinstance(e.get("mesh"), dict):
            return e["mesh"]
    return None


def _same_geometry(a: dict, b: dict) -> bool:
    """Do two identities denote the SAME input geometry in a comparable representation?
    Exact digest equality; or both carry canonical mm digests that match (each canonical
    digest was COMPUTED from actually-converted coordinates — verified equivalence, never
    assumed); or one side was produced by a recorded unit conversion of the very source
    object the other side operated on directly (``converted`` + matching ``source``).
    Anything else — different geometry, or representations with no established
    equivalence — is NOT the same geometry (conservative)."""
    if a.get("input") == b.get("input") and a.get("input"):
        return True
    if a.get("input_mm") and b.get("input_mm") and a["input_mm"] == b["input_mm"]:
        return True
    for x, y in ((a, b), (b, a)):
        if x.get("converted") and x.get("source") and x["source"] == y.get("source"):
            return True
    return False


def _spec_list(ops) -> "list | None":
    """The op specs as a comparable list, or None when they are not parameter-specific
    (legacy name-only approvals are NEVER treated as parameter-specific)."""
    if isinstance(ops, list) and ops and all(isinstance(o, dict) and o.get("op") for o in ops):
        return sorted(({ "op": o["op"], "params": dict(o.get("params") or {})} for o in ops),
                      key=lambda s: s["op"])
    return None


def _repair_chain_current(ledger) -> bool:
    """Are M5-M7 CURRENT completed evidence? Every link must have recorded the same input
    geometry in a comparable representation, the approved specification (M6) must EXACTLY
    equal the executed specification (M7) — operation names AND effective parameters, as
    full spec dicts (name-only/legacy never passes) — and M7's output must be the live
    working mesh (``ledger.mesh_live``). M1 completion and step indices prove nothing."""
    m5 = _newest_identity(ledger, "M5")
    m6 = _newest_identity(ledger, "M6")
    m7 = _newest_identity(ledger, "M7")
    live = getattr(ledger, "mesh_live", None)
    if not (m5 and m6 and m7 and isinstance(live, dict)):
        return False
    approved, executed = _spec_list(m6.get("ops")), _spec_list(m7.get("ops"))
    if not (approved and executed and approved == executed):
        return False
    if not (_same_geometry(m5, m7) and _same_geometry(m6, m7)):
        return False
    if not (m7.get("output") and live.get("input")):
        return False
    return (m7.get("output") == live.get("input") or m7.get("output") == live.get("output")
            or (m7.get("output_mm") and m7.get("output_mm") == live.get("input_mm")))


def mesh_evidence(ledger, outcome) -> bool:
    """Evidence-driven completion of the M-steps from one ``reconstruct_mesh`` call (the
    ``Ledger._on_mesh_recon`` delegation). The tool pushes a ``mesh_flow`` dict through
    ``ctx.state["extra"]``; a step completes ONLY on the recorded facts inside it, and a
    failed/blocked result BLOCKS its step with the reason instead of completing it."""
    def _g(obj, key, default=None):
        return getattr(obj, key, default) if not isinstance(obj, dict) else obj.get(key, default)

    extra = _g(outcome, "extra", None) or {}
    flow = extra.get("mesh_flow") if isinstance(extra, dict) else None
    if not isinstance(flow, dict):
        return False
    changed = False
    ev = ledger._evidence(outcome, kind="mesh_flow")
    # the geometry identity this outcome's repair facts are bound to (W-1 final fix):
    # input = digest of the mesh the call operated on, output = digest of the post-repair
    # working copy (K3 provenance.mesh_digest, reused — no competing algorithm)
    identity = {"source": flow.get("source_digest"),
                "input": flow.get("input_digest"), "input_unit": flow.get("input_unit"),
                "input_mm": flow.get("input_mm"), "converted": bool(flow.get("converted")),
                "output": flow.get("output_digest"), "output_unit": flow.get("output_unit"),
                "output_mm": flow.get("output_mm"),
                "ops": list(flow.get("authorized_ops") or [])}
    chain = _newest_identity(ledger, "M7") or _newest_identity(ledger, "M5")
    live_in = identity["input"]
    same_input = bool(chain and live_in) and _same_geometry(identity, chain)
    consumes_output = bool(chain and chain.get("output") and live_in) and (
        live_in == chain.get("output")
        or (identity.get("input_mm") and identity["input_mm"] == chain.get("output_mm")))
    if chain and live_in and chain.get("input") and not (same_input or consumes_output):
        # the working mesh changed under the workflow (neither the chain's input nor its
        # output): current repair evidence is STALE — demote to pending, keep the history
        for sid in ("M5", "M6", "M7"):
            st = next((s for s in ledger.steps if s.id == sid), None)
            if st is not None and st.status == "done" and not st.claimed:
                st.status = "pending"
                st.note = ("working mesh changed since this evidence was recorded (digest "
                           "mismatch); historical evidence retained — re-run to re-verify")
                changed = True

    def mark(sid: str, done: bool, note: str = "", mesh: "dict | None" = None) -> None:
        nonlocal changed
        from .flow import NOT_REQUESTED

        step = next((s for s in ledger.steps if s.id == sid), None)
        if step is None:
            return

        def newer_than_recorded() -> bool:
            """This outcome's evidence is at least as fresh as the step's newest record."""
            idx = ev.get("step_index", -1)
            last = max((e.get("step_index", -1) for e in step.evidence if isinstance(e, dict)),
                       default=-1)
            return not isinstance(idx, int) or idx >= last

        if step.status in ("done", "changed"):
            if not newer_than_recorded():
                return                 # a stale fact can neither confirm nor demote
            if done and mesh is None:
                return                 # already satisfied; nothing new to record
        if step.status == "blocked" and not done:
            return                     # keep the FIRST failure reason on the record
        entry = dict(ev)
        if mesh is not None:
            entry["mesh"] = dict(mesh)   # identity travels WITH the evidence entry
        ledger._add_evidence(step, entry)
        if step.status in ("done", "changed") and done:
            changed = True              # re-confirmation recorded (newest identity wins)
            return
        if done:
            # real evidence wakes a step that seeded as optional/skipped ("not requested")
            step.status, step.claimed = "done", False
            step.note = note or (NOT_REQUESTED if step.note == NOT_REQUESTED else step.note)
        else:
            step.status = "blocked"
            step.note = note
        changed = True

    if flow.get("sha256") or flow.get("facets"):
        mark("M1", bool(flow.get("sha256")),
             "" if flow.get("sha256") else
             "mesh identified but no source hash is on record (imported before K4?)")
    if flow.get("diag"):
        mark("M2", True, "diagnostics recorded (read-only)")
    units = str(flow.get("units") or "")
    if units:
        unconfirmed = bool(flow.get("units_unconfirmed"))
        mark("M3", units in ("UNIT_KNOWN", "SCALE_PLAUSIBLE") and not unconfirmed,
             "" if not unconfirmed else
             f"unit status {units} is unconfirmed — ask the user before trusting dimensions")
    if flow.get("quality"):
        mark("M4", True, "noise sigma + tolerances recorded")
    verdicts = flow.get("repair_verdicts") or {}
    if verdicts:
        rec_mesh = {"source": identity["source"], "input": identity["input"],
                    "input_unit": identity["input_unit"],
                    "input_mm": identity["input_mm"], "converted": identity["converted"],
                    "ops": identity["ops"], "output": None}
        mark("M5", True, f"repair verdicts: "
             + ", ".join(f"{k}x{v}" for k, v in sorted(verdicts.items())), mesh=rec_mesh)
        gated_needed = bool(flow.get("gated_needed"))
        mark("M6", not (gated_needed and not flow.get("authorized")),
             "gated operations authorized" if flow.get("authorized")
             else "no gated operations were needed" if not gated_needed
             else "gated repair needs explicit user authorization",
             mesh=rec_mesh)
        failed = [k for k in ("REJECTED", "FAILED") if verdicts.get(k)]
        mark("M7", not failed,
             "" if not failed else
             f"repair did not cleanly complete: {', '.join(f'{k}x{verdicts[k]}' for k in failed)}",
             mesh=identity)
    if flow.get("solid_created"):
        mark("M8", True, "non-parametric conversion created (labelled, never parametric CAD)")
    if flow.get("survey"):
        mark("M9", True, "survey recorded")
    if flow.get("plan"):
        mark("M10", True, "FeaturePlan recorded")
    if "build_success" in flow:
        ok = bool(flow.get("build_success"))
        mark("M11", ok, "" if ok else f"builder failed: {flow.get('build_error', '')}")
    if flow.get("deviation"):
        mark("M12", True, "deviation rows on record (RAW authoritative)")
    elif flow.get("outcome") in ("reconstruction_failed", "reconstruction_unverified") \
            and "build_success" in flow:
        mark("M12", False, "no deviation evidence — the reconstruction is unverified")
    # the LIVE working-mesh digest as of this outcome: the repair/conversion post-state
    # when there is one, else (first call, analysis-only) the mesh the call targeted
    if identity["output"]:
        ledger.mesh_digest = identity["output"]
        ledger.mesh_live = {"source": identity["source"], "input": identity["output"],
                            "input_unit": identity["output_unit"],
                            "input_mm": identity["output_mm"], "converted": False}
    elif not chain and identity["input"]:
        ledger.mesh_digest = identity["input"]
        ledger.mesh_live = {"source": identity["source"], "input": identity["input"],
                            "input_unit": identity["input_unit"],
                            "input_mm": identity["input_mm"],
                            "converted": identity["converted"]}
    return changed


def absorb_explicit_change(ledger, text: str) -> bool:
    """``_begin_turn`` delegation on a carried continuation: retarget when the new turn
    names a DIFFERENT explicit mesh operation. Returns True when retargeted."""
    new_op = resolve_mesh_intent(text)
    if ledger.mesh_intent and new_op not in (None, ledger.mesh_intent):
        return retarget_mesh_intent(ledger, new_op)
    return False


_REPAIR_FAMILY = ("repair", "parametric", "modify")   # routes whose M5-M7 mean the same thing


def retarget_mesh_intent(ledger, new_intent: str, *, note: str = "") -> bool:
    """A CONTROLLED intent change on a live mesh workflow: keeps completed evidence whose
    facts still hold, replaces the pending route steps with the new template, and journals
    the transition — nothing flips silently.

    Evidence rules (geometry-bound): the core (M1-M4) survives when done. Repair evidence
    (M5-M7) survives AS CURRENT only on repair-family routes AND when
    ``_repair_chain_current`` holds — the recorded chain's input/output digests are
    internally consistent and its output IS the live working mesh (``ledger.mesh_digest``),
    with M6's approved operations exactly matching the executed repair. M1 completion and
    step indices prove nothing. When the chain is stale or its identity is unavailable,
    the historical evidence is RETAINED on the steps but they are demoted to pending with a
    visible note — never represented as currently verified, never deleted."""
    if new_intent not in MESH_INTENTS or new_intent in (ledger.mesh_intent, "ambiguous"):
        return False
    old = ledger.mesh_intent
    core, repair_ids = ("M1", "M2", "M3", "M4"), ("M5", "M6", "M7")
    done_now = {s.id: s for s in ledger.steps if s.status == "done" and not s.claimed}
    kept_ids = set(core) & set(done_now)
    carry_ids: set[str] = set()
    if old in _REPAIR_FAMILY and new_intent in _REPAIR_FAMILY:
        recorded = [s for s in ledger.steps if s.id in repair_ids and s.evidence]
        if _repair_chain_current(ledger):
            kept_ids |= {s.id for s in recorded if s.status == "done" and not s.claimed}
            for st in recorded:                # blocked/failed steps keep their reason too
                if st.id not in kept_ids:
                    carry_ids.add(st.id)
        else:
            for st in recorded:                # stale/unavailable identity: keep history,
                if st.status == "done" and not st.claimed:
                    st.status = "pending"      # visibly unverified, awaiting a live check
                st.note = (st.note if st.status == "blocked" else
                           "historical repair evidence retained; the live working-mesh "
                           "identity does not match (or is unavailable) — re-run the "
                           "repair stage to re-verify")
                carry_ids.add(st.id)
    kept = [done_now[sid] for sid in (*core, *repair_ids) if sid in kept_ids] \
        + [next(s for s in ledger.steps if s.id == sid) for sid in sorted(carry_ids)]
    fresh = _mesh_steps(new_intent)
    ledger.steps = kept + [s for s in fresh
                           if s.id not in kept_ids and s.id not in carry_ids]
    ledger.mesh_intent = new_intent
    ledger.waivers.append({"what": "mesh_intent", "status": "changed",
                           "reason": note or f"user retargeted the mesh workflow "
                                            f"{old!r} -> {new_intent!r}"})
    ledger._bump()
    return True
