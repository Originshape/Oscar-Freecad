"""The ONE mesh→numpy materializer (P4.1) + the vectorized facet normals
(P4.2). Every recon stage used to rebuild its own points/facets arrays from
``mesh.Topology`` — four separate allocations of the same data per pipeline
run; one shared helper ends that. Pure numpy; no FreeCAD import (works on any
object exposing ``Topology``)."""

import numpy as np


def mesh_arrays(mesh):
    """``(points (n,3) float64, facets (m,3) int64)`` from a mesh's Topology.
    The reshape keeps EMPTY meshes well-formed ((0,3), not (0,)) — the edge
    the four former per-module materializers each had to remember."""
    points, facets = mesh.Topology
    pts = np.array([[p.x, p.y, p.z] for p in points], dtype=float).reshape(-1, 3)
    fac = np.array(facets, dtype=np.int64).reshape(-1, 3)
    return pts, fac


def facet_normals_arrays(pts, fac):
    """(m,3) unit facet normals, VECTORIZED — the per-facet Python loop over
    ``mesh.Facets`` was the prismatic route's CPU hotspot. The normal is the
    winding cross product (the same convention as FreeCAD's ``Facet.Normal``);
    degenerate facets get a zero normal."""
    a, b, c = pts[fac[:, 0]], pts[fac[:, 1]], pts[fac[:, 2]]
    n = np.cross(b - a, c - a)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    ok = ln[:, 0] > 1e-30
    out = np.zeros_like(n)
    out[ok] = n[ok] / ln[ok]
    return out
