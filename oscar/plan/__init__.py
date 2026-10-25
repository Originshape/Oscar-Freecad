# SPDX-License-Identifier: Apache-2.0
"""FeaturePlan: a CAD-neutral JSON description of a parametric part, and its FreeCAD builder.

    schema.py      versioned FeaturePlan schema (v1 and v2) and validation
    builder.py     FeaturePlan -> Spreadsheet + Sketcher + PartDesign in an open document
                   (apply_plan / apply_feature / rebuild_from / sync_parameters / format_build_failure);
                   the translators live in _sketchbuild.py, _featurebuild.py, _dressbuild.py and
                   _partbuild.py, the shared contract in _buildcore.py

Used by the ``apply_feature_plan``, ``reapply_plan_feature`` and ``feature_tree_status`` tools for any
structured part (plates, flanges, brackets, fixtures) that Oscar models from a chat request.
"""
