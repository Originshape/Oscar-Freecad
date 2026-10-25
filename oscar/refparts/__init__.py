"""Reference parts: the vetted catalog, the board library and file identity checks.

Everything the build flow knows about the real part a design wraps (a Raspberry
Pi, an Arduino, a motor...) lives here, with a recorded source and confidence:

* ``catalog``  - ``catalog.json`` (what can be sourced, from where) and
  ``boards/*.json`` (published board dimensions), the board-schema validator and
  the derived wall openings. PURE: no FreeCAD, no network.
* ``identity`` - magic-byte sniffing of downloaded files (STEP, STL, ZIP, PDF,
  HTML, git-LFS pointer) and the bounding-box / outline plausibility check
  against the catalog. PURE.
* ``fetch``    - catalog entry -> at most ONE allowlisted download (or the
  ``OSCAR_REFPART_LOCAL_DIR`` mirror) -> verification -> import-ready file, and
  the per-turn attempt rules. PURE of FreeCAD and Qt (network through netguard).
* ``pdftext``  - best-effort PDF text for datasheet attachments. PURE.
* ``importref``, ``measure``, ``measure_params``, ``measure_mesh`` - the FreeCAD
  side, GUI-thread only: the hidden ``Ref_<ID>`` ``App::Part`` and its frame
  normalization, the exact B-rep measurements (PCB, holes, connector
  candidates), the ``Params`` rows and the STL path.

The sub-modules are imported explicitly (``from oscar.refparts import catalog``);
importing the package itself costs nothing and touches no FreeCAD API.
"""
