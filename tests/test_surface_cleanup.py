from __future__ import annotations

from pathlib import Path
import stat

import numpy as np
import pytest
import trimesh

from fure_defur.io import digest, export_obj
from fure_defur.reconstruction import (
    _strict_surface_audit,
    clean_surface,
    close_small_boundary_loops,
)


def _direct_cleanup(source: Path, output: Path, max_hole_diameter_fraction: float) -> None:
    """Expected single-component path without the face-reordering split."""
    model = trimesh.load(source, process=False, force="mesh")
    model.merge_vertices(digits_vertex=5)
    model.update_faces(model.nondegenerate_faces())
    model.update_faces(model.unique_faces())
    model.remove_unreferenced_vertices()
    model.fix_normals()
    close_small_boundary_loops(model, max_hole_diameter_fraction, 128)
    model.fix_normals()
    export_obj(output, np.asarray(model.vertices), np.asarray(model.faces))


def test_clean_surface_single_component_matches_direct_order(tmp_path):
    source = tmp_path / "source.obj"
    expected = tmp_path / "expected.obj"
    output = tmp_path / "clean.obj"
    model = trimesh.creation.icosphere(subdivisions=1)
    export_obj(source, np.asarray(model.vertices), np.asarray(model.faces))
    _direct_cleanup(source, expected, .015)
    clean_surface(source, output, iterations=0, merge_vertex_digits=5)
    assert digest(output) == digest(expected)
    assert stat.S_IMODE(output.stat().st_mode) == 0o644


def test_clean_surface_bounded_triangle_cap_adds_no_centroid(tmp_path):
    model = trimesh.creation.icosphere(subdivisions=2)
    model.update_faces(np.arange(len(model.faces) - 1))
    source = tmp_path / "open.obj"
    output = tmp_path / "clean.obj"
    export_obj(source, np.asarray(model.vertices), np.asarray(model.faces))

    report = clean_surface(
        source,
        output,
        iterations=0,
        max_hole_diameter_fraction=.2,
    )
    cleaned = trimesh.load(output, process=False, force="mesh")

    assert report["boundary_repairs"][0]["boundary_vertices"] == 3
    assert len(cleaned.vertices) == len(model.vertices)
    assert len(cleaned.faces) == len(model.faces) + 1
    assert cleaned.is_watertight


def test_surface_audit_rejects_near_degenerate_faces():
    model = trimesh.Trimesh(
        vertices=np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1e-12, 0.0],
            [0.0, 0.0, 1.0],
        ]),
        faces=np.array([
            [0, 2, 1],
            [0, 1, 3],
            [1, 2, 3],
            [2, 0, 3],
        ]),
        process=False,
    )
    model.fix_normals()

    assert model.is_watertight
    assert np.count_nonzero(~model.nondegenerate_faces())
    with pytest.raises(ValueError, match="degenerate or near-degenerate"):
        _strict_surface_audit(model)
