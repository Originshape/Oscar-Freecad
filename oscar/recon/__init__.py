"""Oscar's mesh-to-parametric reconstruction engine (master spec v1.2, M2+).

Modules are FreeCAD-side (document/mesh access, GUI-thread for document writes) except where
noted; pure-numpy parts run on any thread. The engine never modifies an original imported mesh:
everything that changes triangles works on ``Mesh.copy()`` copies and reports what it did.
"""
