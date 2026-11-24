"""Four-category geometry provenance for the reconstruction pipeline (K3).

Every repair/preparation operation is diffed GEOMETRICALLY (coordinate-keyed, renumbering-
safe — triangle indices are NOT stable across topology changes and mesh segments do NOT
survive save/reopen, both verified 2026-11-14): facets are identified by their three
rounded vertex coordinates, vertices by their rounded coordinates. Coordinates round-trip
through FCStd save/reopen bit-exactly on FreeCAD 1.1.4 (verified), so keys persist; when
later edits move or remove geometry, recorded keys legitimately stop matching —
``verify_against_mesh`` then reports a reduced mapping confidence and the verdict turns
UNCERTAIN instead of silently claiming provenance survived.

Categories (never mixed up; synthetic/modified are NEVER "original measured scan evidence"):
  original  — untouched source geometry (the default; what no ledger entry claims)
  modified  — original geometry an applied operation changed (moved vertices / retouched
              facets); dimensions from here are weakened evidence, flagged
  synthetic — geometry an operation CREATED (fills, patches); it may not independently
              establish the dimensions of a reconstructed feature
  removed   — original geometry an applied operation deleted

The ledger records, per applied operation: operation name, verdict, before/after counts,
the three key lists (capped; counts always complete), and mapping notes. ``verify_against_mesh``
re-matches every recorded key against a mesh and returns a confidence in [0, 1] plus
VERIFIED / UNCERTAIN. Pure numpy + the K1 neighbour kernel; read-only on meshes.
"""

import numpy as np

from .quality import _knn

KEY_DECIMALS = 5          # 1e-5 mm quantization; save/reopen round-trips bit-exact (verified)
MATCH_TOL_MM = 0.05       # a moved vertex = an after-vertex within this radius of a vanished
                          # before one. Missing a real nudge would misclassify it as
                          # removed+added (synthetic!) — worse than a tolerant match; 0.05 mm
                          # is repair-nudge scale, far below feature sizes
AMBIGUOUS_RADIUS_MM = 0.5  # unmatched vanished→appeared pairs within this radius are LOW-
                           # CONFIDENCE moves (still modified, never synthetic); beyond it a
                           # vanished vertex really is removed and an appeared one created
MAX_RECORDED_KEYS = 200   # per category per operation in the ledger; counts are always complete
VERIFY_CONFIDENCE = 0.9   # recorded-key survival below this → UNCERTAIN


def mesh_digest(mesh) -> str:
    """A digest over the COMPLETE sorted multiset of facet keys — bit-level identity of the
    mesh's coordinate geometry. Two meshes with the same facets in any order share it; any
    vertex move, facet add/remove/duplicate changes it. Used as the complete-verification
    tier: when it matches, every ledger claim holds for the whole mesh, capped key lists
    included."""
    import hashlib

    keys = sorted(facet_keys(mesh))
    return hashlib.sha256("\n".join(map(repr, keys)).encode("utf-8")).hexdigest()


def identity_health(mesh) -> dict:
    """Where coordinate identity is DEGRADED on this mesh: duplicate vertex keys (nearly
    coincident vertices inside one 1e-5 quantum cell) and duplicate facet keys (repeated
    facets, or distinct facets collapsed by quantization). Duplicate facet keys mean two
    different triangles share one identity — verification then cannot tell them apart and
    must say UNCERTAIN instead of claiming to have matched the right one."""
    pts, fac = _mesh_arrays(mesh)
    vk = list(map(tuple, np.round(pts, KEY_DECIMALS)))
    fk = facet_keys(mesh)
    return {"vertices": len(vk), "duplicate_vertex_keys": len(vk) - len(set(vk)),
            "facets": len(fk), "duplicate_facet_keys": len(fk) - len(set(fk))}


def _vertex_keys(pts):
    return np.round(np.asarray(pts, dtype=float), KEY_DECIMALS)


def _sorted_tris(pts, fac):
    """(m,3,3) rounded vertex triples per facet, each facet's three vertices sorted
    lexicographically — a true canonical form (element-wise column sorting would MIX
    coordinates between vertices and break per-vertex lookups)."""
    tri = np.round(pts[fac], KEY_DECIMALS)
    order = np.lexsort((tri[:, :, 2], tri[:, :, 1], tri[:, :, 0]))
    return np.take_along_axis(tri, order[:, :, None], axis=1)


def facet_keys(mesh) -> list:
    """Stable per-facet identifiers: the sorted, rounded coordinate triple."""
    pts, fac = _mesh_arrays(mesh)
    if len(fac) == 0:
        return []
    tri = _sorted_tris(pts, fac)
    return [tuple(r) for r in tri.reshape(len(fac), -1)]


def _mesh_arrays(mesh):
    from ._arrays import mesh_arrays

    return mesh_arrays(mesh)   # P4.1: the one shared materializer


def diff_mesh(before_mesh, after_mesh) -> dict:
    """Geometric truth of one operation: which facets/vertices were removed, added
    (created) or moved, keyed by coordinates so renumbering cannot hide a change."""
    pb, fb = _mesh_arrays(before_mesh)
    pa, fa = _mesh_arrays(after_mesh)
    if len(fb) == 0 and len(fa) == 0:
        return {"removed_facets": [], "added_facets": [], "modified_facets": [],
                "removed_vertices": [], "added_vertices": [], "moved_vertices": [],
                "uncertain_moves": [], "ambiguous_merges": 0, "unchanged_vertices": 0,
                "counts": {"facets_before": 0, "facets_after": 0,
                           "points_before": 0, "points_after": 0}}
    kb, ka = _vertex_keys(pb), _vertex_keys(pa)
    set_b, set_a = set(map(tuple, kb)), set(map(tuple, ka))
    common = set_b & set_a
    gone_b, new_a = set_b - set_a, set_a - set_b
    # H2 matching: a vanished vertex finds its destination among ALL after-vertices (a merge
    # moves onto an unchanged vertex's key, not a new one). Two tiers: within MATCH_TOL_MM a
    # confident move; within AMBIGUOUS_RADIUS_MM a LOW-CONFIDENCE move (still a move — never
    # synthetic); beyond that it truly vanished. Two vanished vertices claiming the same
    # destination (a merge) are flagged ambiguous.
    moved, uncertain_moves = [], []
    ambiguous_merges = 0
    claimed: dict = {}
    if gone_b:
        query = np.array(sorted(gone_b), dtype=float)
        dest_all = np.array(sorted(set_a), dtype=float)
        d, i, _ = _knn(query, dest_all, 1, exclude_self=False)
        d = np.asarray(d).reshape(-1)
        i = np.asarray(i).reshape(-1)
        for row in range(len(query)):
            v, dist = tuple(query[row]), float(d[row])
            dest = tuple(dest_all[i[row]])
            if dist <= MATCH_TOL_MM:
                if dest in claimed:
                    ambiguous_merges += 1        # two origins, one destination
                else:
                    claimed[dest] = v
                moved.append((v, dest))
            elif dist <= AMBIGUOUS_RADIUS_MM:
                uncertain_moves.append((v, dest, round(dist, 5)))
                moved.append((v, dest))
    moved_to = {b: a for b, a in moved}
    moved_set = {a for _, a in moved}
    added_vertices = [k for k in new_a if k not in moved_set]

    def keys(pts, fac):
        tri = _sorted_tris(pts, fac)
        return tri, [tuple(r) for r in tri.reshape(len(fac), -1)]

    def touches(key, vertex_keys):
        return any(tuple(key[i:i + 3]) in vertex_keys for i in (0, 3, 6))

    moved_from_keys = {b for b, _ in moved}
    _, keys_b = keys(pb, fb)
    _, keys_a = keys(pa, fa)
    set_kb, set_ka = set(keys_b), set(keys_a)
    # facets whose key changed ONLY because a vertex moved are MODIFIED, not removed+created
    removed_facets = {k for k in set_kb - set_ka if not touches(k, moved_from_keys)}
    added_facets = {k for k in set_ka - set_kb if not touches(k, moved_set)}
    modified_facets = {k for k in set_ka
                       if any(v in moved_set for v in (k[0:3], k[3:6], k[6:9]))}
    return {
        "removed_facets": sorted(removed_facets),
        "added_facets": sorted(added_facets),
        "modified_facets": sorted(modified_facets),
        "removed_vertices": sorted(gone_b - {b for b, _ in moved}),
        "added_vertices": sorted(added_vertices),
        "moved_vertices": moved,
        "uncertain_moves": uncertain_moves,      # low-confidence matches (radius, not tol)
        "ambiguous_merges": ambiguous_merges,    # several origins onto one destination
        "unchanged_vertices": len(common),
        "counts": {"facets_before": len(fb), "facets_after": len(fa),
                   "points_before": len(pb), "points_after": len(pa)},
    }


def new_ledger(source_hash: "str | None" = None, mesh_name: "str | None" = None) -> dict:
    return {"schema": 1, "source_hash": source_hash, "mesh": mesh_name,
            "key_decimals": KEY_DECIMALS, "operations": [],
            "modified": [], "synthetic": [], "removed": [],
            "note": "categories are coordinate-keyed facet/vertex lists; counts complete, "
                    "listed keys capped at %d per operation" % MAX_RECORDED_KEYS}


def record_operation(ledger: dict, op: str, verdict: str, diff: dict) -> None:
    """Accumulate one APPLIED operation's diff into the ledger categories. Category COUNTS
    are always complete; the stored key LISTS are capped at MAX_RECORDED_KEYS per operation —
    the truncation flag lets verification distinguish complete from partial checks."""
    entry = {"operation": op, "verdict": verdict,
             "counts": diff["counts"],
             "removed_facets": len(diff["removed_facets"]),
             "added_facets": len(diff["added_facets"]),
             "modified_facets": len(diff["modified_facets"]),
             "moved_vertices": len(diff["moved_vertices"]),
             "uncertain_moves": len(diff.get("uncertain_moves") or []),
             "ambiguous_merges": int(diff.get("ambiguous_merges") or 0),
             "removed_vertices": len(diff["removed_vertices"]),
             "added_vertices": len(diff["added_vertices"]),
             "keys": {"removed_facets": diff["removed_facets"][:MAX_RECORDED_KEYS],
                      "added_facets": diff["added_facets"][:MAX_RECORDED_KEYS],
                      "modified_facets": diff["modified_facets"][:MAX_RECORDED_KEYS]}}
    truncated = any(entry["keys"][k] and len(entry["keys"][k]) < entry[
        {"removed_facets": "removed_facets", "added_facets": "added_facets",
         "modified_facets": "modified_facets"}[k]] for k in entry["keys"])
    entry["truncated"] = truncated
    ledger["operations"].append(entry)
    ledger.setdefault("counts", {"synthetic": 0, "modified": 0, "removed": 0, "uncertain": 0})
    ledger["counts"]["synthetic"] += entry["added_facets"]
    ledger["counts"]["removed"] += entry["removed_facets"]
    ledger["counts"]["modified"] += entry["modified_facets"]
    ledger["counts"]["uncertain"] += entry["uncertain_moves"]
    ledger["truncated"] = bool(ledger.get("truncated") or truncated)
    ledger["removed"].extend(diff["removed_facets"][:MAX_RECORDED_KEYS])
    ledger["synthetic"].extend(diff["added_facets"][:MAX_RECORDED_KEYS])
    ledger["modified"].extend(diff["modified_facets"][:MAX_RECORDED_KEYS])


def evidence_class(sample_keys, ledger: "dict | None") -> str:
    """"original" (no ledger claim), "synthetic" (majority created), "mixed", or
    "uncertain" (keys that no longer exist in the mesh the sample came from)."""
    if not ledger:
        return "original"
    synthetic = set(ledger.get("synthetic") or [])
    if not sample_keys:
        return "original"
    hits = sum(1 for k in sample_keys if k in synthetic)
    if hits == 0:
        return "original"
    return "synthetic" if 2 * hits > len(sample_keys) else "mixed"


def verify_against_mesh(ledger: "dict | None", mesh) -> dict:
    """Do the ledger's recorded claims still hold for ``mesh``? Two tiers:

    1. COMPLETE (mesh digest): when the ledger carries the post-repair ``mesh_digest`` and
       the mesh still matches it, EVERY claim holds for the whole mesh — capped key lists
       included — scope "complete", verdict VERIFIED.
    2. PER-KEY (fallback): every recorded key is re-matched. synthetic/modified keys must
       still EXIST; removed keys must still be ABSENT (a removed facet that reappeared is a
       mapping failure). Scope is "partial" whenever key lists were truncated — a partial
       check is never silently reported as VERIFIED (UNCERTAIN with the reason instead).

    Identity degradation (duplicate facet coordinate keys — nearly coincident vertices or
    repeated facets collapsing distinct triangles onto one identity) also forces UNCERTAIN:
    a recorded key may then "match" the wrong twin."""
    if not ledger:
        return {"confidence": None, "verdict": "NO_LEDGER", "matched": 0, "recorded": 0,
                "scope": None, "reasons": [], "method": None}
    reasons: list[str] = []
    dig = ledger.get("mesh_digest")
    if dig:
        if mesh_digest(mesh) == dig:
            return {"confidence": 1.0, "verdict": "VERIFIED", "matched": None,
                    "recorded": None, "scope": "complete", "method": "mesh-digest",
                    "reasons": [],
                    "note": "the mesh is identical (by coordinate-key digest) to the state "
                            "the provenance ledger was written for"}
        reasons.append("the mesh changed after the ledger was written (digest mismatch)")
    health = identity_health(mesh)
    if health["duplicate_facet_keys"] > 0:
        reasons.append(f"{health['duplicate_facet_keys']} duplicate facet key(s) — distinct "
                       "triangles share one coordinate identity; a key match may hit the "
                       "wrong twin")
    live = set(facet_keys(mesh))
    recorded = matched = missing = reappeared = 0
    for cat in ("synthetic", "modified", "removed"):
        for k in ledger.get(cat) or []:
            recorded += 1
            if cat == "removed":
                if k in live:
                    reappeared += 1
                else:
                    matched += 1
            elif k in live:
                matched += 1
            else:
                missing += 1
    if missing:
        reasons.append(f"{missing} recorded region(s) no longer present in the mesh")
    if reappeared:
        reasons.append(f"{reappeared} removed facet(s) unexpectedly reappeared")
    if ledger.get("truncated"):
        reasons.append("key lists were truncated — only a subset of the recorded regions "
                       "could be checked")
    scope = "partial" if ledger.get("truncated") else "complete"
    if recorded == 0 and not reasons:
        return {"confidence": 1.0, "verdict": "VERIFIED", "matched": 0, "recorded": 0,
                "scope": "complete", "method": "per-key", "reasons": [],
                "note": "no repair operations were applied — the mesh is all original"}
    confidence = (matched / recorded) if recorded else 1.0
    verdict = "VERIFIED" if (matched == recorded and not reasons) else "UNCERTAIN"
    return {"confidence": round(confidence, 4), "recorded": recorded, "matched": matched,
            "verdict": verdict, "scope": scope, "method": "per-key", "reasons": reasons}


def digest_for_property(ledger: dict, verification: dict) -> dict:
    """The compact, property-persistable form (key lists stay in the analysis JSON file)."""
    return {"schema": ledger.get("schema"),
            "source_hash": ledger.get("source_hash"),
            "mesh": ledger.get("mesh"),
            "operations": [{k: v for k, v in op.items() if k != "keys"}
                           for op in ledger.get("operations", [])],
            "counts": {"synthetic": len(ledger.get("synthetic") or []),
                       "modified": len(ledger.get("modified") or []),
                       "removed": len(ledger.get("removed") or [])},
            "verification": verification,
            "note": "synthetic/modified geometry is NOT measured scan evidence; full key "
                    "lists live in the project's analysis/<mesh>.json"}
