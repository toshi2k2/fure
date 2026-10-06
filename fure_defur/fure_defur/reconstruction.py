"""Fail-fast GPU reconstruction stages with isolated run directories."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil
import tempfile
import numpy as np
import trimesh
from .io import export_obj, mesh, write_json


def commands(prepared: Path, output: Path, gpu: int, python: str, neus2_python: str, steps: int) -> list[dict]:
    repo = Path(__file__).resolve().parents[1]
    engine = repo / "vendor/frosting"
    scene = prepared / "frosting_scene"
    gs = output / "gs_checkpoint"
    stages = [
        dict(name="3dgs", cwd=str(output), command=[python, str(engine / "gaussian_splatting/train.py"), "-s", str(scene), "-m", str(gs),
            "--iterations", "30000", "--save_iterations", "30000", "--test_iterations", "30000", "--port", "0"]),
        dict(name="neus2", cwd=str(output), command=[neus2_python, str(repo / "scripts/train_neus2.py"),
            "--scene", str(prepared / "neus2/transform.json"), "--output", str(output / "neus2"), "--steps", str(steps)]),
        dict(name="frosting", cwd=str(output), command=[python, str(engine / "train.py"), "-s", str(scene), "-c", str(gs)+"/", "-i", "30000",
            "-r", "density", "--low_poly", "True", "--refinement_time", "long", "--gaussians_in_frosting", "500000",
            "--extract_background_mesh", "False", "--bboxmin=-0.9,-0.9,-0.9", "--bboxmax=0.9,0.9,0.9",
            "--center_bbox", "False", "--export_obj", "True", "--export_ply", "True", "--gpu", "0"]),
    ]
    for stage in stages:
        # Upstream tools see the selected physical device as their local GPU 0.
        # Keep this mapping in dry-runs and commands.json, not only in process state.
        stage["environment"] = {"CUDA_VISIBLE_DEVICES": str(gpu)}
    return stages


def train(prepared: Path, output: Path, gpu: int, python: str, neus2_python: str, steps: int, dry_run: bool) -> list[dict]:
    prepared, output = prepared.resolve(), output.resolve()
    stages = commands(prepared, output, gpu, python, neus2_python, steps)
    if dry_run:
        return stages
    if output.exists():
        raise FileExistsError(output)
    for required in (prepared / "frosting_scene/sparse/0/cameras.txt", prepared / "neus2/transform.json"):
        if not required.is_file():
            raise FileNotFoundError(required)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    cuda_check = (
        "import torch\n"
        "if not torch.cuda.is_available():\n"
        "    raise RuntimeError('CUDA is unavailable')\n"
        "print(torch.cuda.get_device_name(0))"
    )
    subprocess.run([python, "-c", cuda_check], env=env, check=True)
    output.mkdir(parents=True)
    repo = Path(__file__).resolve().parents[1]
    engine = repo / "vendor/frosting"
    # Upstream Frosting imports this relative directory; keep per-run output isolated.
    (output / "gaussian_splatting").symlink_to(os.path.relpath(engine / "gaussian_splatting", output), target_is_directory=True)
    env["PYTHONPATH"] = str(engine) + os.pathsep + str(engine / "gaussian_splatting") + os.pathsep + env.get("PYTHONPATH", "")
    write_json(output / "commands.json", stages)
    for stage in stages:
        with (output / f"{stage['name']}.log").open("w") as log:
            subprocess.run(stage["command"], cwd=stage["cwd"], env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    shell = output / "output/refined_frosting_shell/frosting_scene/shell_stats.npz"
    if not shell.is_file() or not (output / "neus2/outer_raw.obj").is_file():
        raise RuntimeError("An engine exited without the expected shell/mesh output")
    write_json(output / "status.json", dict(reconstruction_complete=True, annotation_required=True,
                shell_stats=str(shell.relative_to(output)), outer_mesh="neus2/outer_raw.obj"))
    return stages


def _select_surface_component(components: list[trimesh.Trimesh]) -> tuple[int, trimesh.Trimesh, str]:
    """Prefer a substantial closed subject over a larger open extraction sheet.

    NeuS marching cubes can emit a high-area sheet along the extraction box in
    addition to a closed foreground animal. Selecting by area alone promotes
    that artifact. A closed component must still be substantial relative to
    the largest component so that a tiny floating bubble cannot win.
    """
    if not components:
        raise ValueError("Mesh contains no surface components")
    largest_faces = max(len(component.faces) for component in components)
    largest_scale = max(float(np.linalg.norm(np.ptp(component.vertices, axis=0))) for component in components)
    significant = [
        (index, component) for index, component in enumerate(components)
        if component.is_watertight
        and len(component.faces) >= max(100, int(np.ceil(.05 * largest_faces)))
        and float(np.linalg.norm(np.ptp(component.vertices, axis=0))) >= .1 * largest_scale
    ]
    if significant:
        index, component = max(significant, key=lambda item: item[1].area)
        return index, component, "largest_significant_watertight"
    index, component = max(enumerate(components), key=lambda item: item[1].area)
    return index, component, "largest_area_no_significant_watertight"


def clean_surface(source: Path, output: Path, iterations: int = 5,
                  max_hole_diameter_fraction: float = .015,
                  max_hole_vertices: int = 128,
                  merge_vertex_digits: int | None = None) -> dict:
    """Clean a surface and record component selection, bounded repairs, and the final audit."""
    if output.exists():
        raise FileExistsError(output)
    if (not isinstance(iterations, (int, np.integer)) or isinstance(iterations, bool)
            or int(iterations) < 0):
        raise ValueError("iterations must be a nonnegative integer")
    if not np.isfinite(max_hole_diameter_fraction) or max_hole_diameter_fraction < 0:
        raise ValueError("max_hole_diameter_fraction must be finite and nonnegative")
    if not isinstance(max_hole_vertices, int) or max_hole_vertices < 3:
        raise ValueError("max_hole_vertices must be an integer of at least 3")
    if (merge_vertex_digits is not None
            and (not isinstance(merge_vertex_digits, (int, np.integer))
                 or isinstance(merge_vertex_digits, bool)
                 or not 0 <= int(merge_vertex_digits) <= 15)):
        raise ValueError("merge_vertex_digits must be an integer from 0 to 15 or None")
    model = trimesh.load(source, process=False, force="mesh")
    original_faces = len(model.faces)
    # Preserve the raw marching-cubes connectivity while identifying the
    # subject. Global vertex merging can join nearly coincident extraction
    # components or turn zero-area seam triangles into artificial holes.
    components = model.split(only_watertight=False, repair=False)
    selected_index, selected, selection_reason = _select_surface_component(components)
    selected_faces = len(selected.faces)
    # Splitting a single connected mesh needlessly reorders its faces. Keep the
    # loaded mesh in that case so a deterministic cleanup preserves face order;
    # multi-component inputs still use the selected subject component.
    if len(components) != 1:
        model = selected
    model.merge_vertices(digits_vertex=merge_vertex_digits)
    model.update_faces(model.nondegenerate_faces())
    model.update_faces(model.unique_faces())
    model.remove_unreferenced_vertices()
    model.fix_normals()
    repair = close_small_boundary_loops(model, max_hole_diameter_fraction, max_hole_vertices)
    if iterations:
        trimesh.smoothing.filter_taubin(model, lamb=.5, nu=.53, iterations=iterations)
    model.fix_normals()
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=output.parent, prefix=f".{output.name}.",
                                     suffix=".obj", delete=False) as temporary:
        temporary_path = Path(temporary.name)
    try:
        export_obj(temporary_path, np.asarray(model.vertices), np.asarray(model.faces))
        temporary_path.chmod(0o644)
        saved = mesh(temporary_path)
        if (not np.array_equal(np.asarray(saved.vertices), np.asarray(model.vertices))
                or not np.array_equal(np.asarray(saved.faces), np.asarray(model.faces))):
            raise ValueError("High-precision OBJ did not preserve cleaned vertex/face order exactly")
        audit = _strict_surface_audit(saved)
        temporary_path.replace(output)
    finally:
        temporary_path.unlink(missing_ok=True)
    report = dict(input_faces=original_faces, output_faces=len(model.faces), components_before=len(components),
                  selected_component_index=selected_index, selected_component_faces=selected_faces,
                  component_selection=selection_reason,
                  **audit, topology_changed=True, annotations_must_match_new_mesh=True,
                  merge_vertex_digits=merge_vertex_digits,
                  iterations=int(iterations),
                  max_hole_diameter_fraction=float(max_hole_diameter_fraction),
                  max_hole_vertices=max_hole_vertices,
                  boundary_repairs=repair)
    write_json(output.with_suffix(".cleanup.json"), report)
    return report


def _strict_surface_audit(model: trimesh.Trimesh) -> dict:
    """Check the exact serialized surface before it can be annotated."""
    components = model.split(only_watertight=False, repair=False)
    component_count = len(components)
    degenerate_faces = int(np.count_nonzero(~model.nondegenerate_faces()))
    _, counts = np.unique(model.edges_sorted, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    if component_count != 1:
        raise ValueError(f"Cleaned surface has {component_count} connected components; expected one")
    if degenerate_faces:
        raise ValueError(f"Cleaned surface has {degenerate_faces} degenerate or near-degenerate faces")
    if boundary_edges or nonmanifold_edges or not model.is_watertight:
        raise ValueError(
            "Mesh still has unsafe holes or edge topology "
            f"(boundary={boundary_edges}, nonmanifold={nonmanifold_edges}). "
            "Refine or re-extract the surface instead"
        )
    volume = float(model.volume)
    if not model.is_winding_consistent or not np.isfinite(volume) or volume <= 0:
        raise ValueError("Cleaned surface must have consistent outward winding and positive volume")
    from .pipeline import intersections
    self_intersections = len(intersections(np.asarray(model.vertices), np.asarray(model.faces)))
    if self_intersections:
        raise ValueError(f"Cleaned surface has {self_intersections} self-intersecting triangle pairs")
    return dict(component_count=component_count, watertight=True, winding_consistent=True,
                positive_volume=True, volume=volume, boundary_edges=boundary_edges,
                nonmanifold_edges=nonmanifold_edges, degenerate_faces=0,
                self_intersection_pairs=0)


def close_small_boundary_loops(model: trimesh.Trimesh, max_diameter_fraction: float = .015,
                               max_vertices: int = 128) -> list[dict]:
    """Cap only simple, small boundary loops; reject large or branched defects."""
    from collections import defaultdict
    edges = model.edges_sorted
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    boundary = unique[counts == 1]
    if not len(boundary):
        return []
    neighbors = defaultdict(list)
    for a, b in boundary:
        neighbors[a].append(b)
        neighbors[b].append(a)
    remaining = set(neighbors)
    vertices, faces = model.vertices.tolist(), model.faces.tolist()
    diag = np.linalg.norm(np.ptp(model.vertices, axis=0))
    report = []
    while remaining:
        first = min(remaining)
        group, pending = set(), [first]
        while pending:
            value = pending.pop()
            if value in group:
                continue
            group.add(value)
            pending.extend(neighbors[value])
        remaining.difference_update(group)
        # A branched/open boundary group is not a hole that can be safely
        # triangulated. Continue so it does not suppress independent loops.
        if any(len(neighbors[value]) != 2 for value in group):
            continue
        loop, previous, current = [first], -1, first
        for _ in range(len(group)+1):
            following = next(x for x in neighbors[current] if x != previous)
            if following == first:
                break
            if following in loop:
                raise ValueError("Non-simple boundary loop")
            loop.append(following)
            previous, current = current, following
        else:
            continue
        points = model.vertices[loop]
        diameter = np.linalg.norm(np.ptp(points, axis=0))
        # This guard prevents closing missing bellies/legs as if they were tiny defects.
        if diameter > max_diameter_fraction*diag or len(loop) > max_vertices:
            continue
        if len(loop) == 3:
            # A bounded triangular hole needs one face, not a new centroid and
            # three near-degenerate faces. The final normal repair below checks
            # and corrects its orientation against the surrounding surface.
            faces.append(list(reversed(loop)))
        else:
            center = points.mean(0)
            idx = len(vertices)
            vertices.append(center.tolist())
            faces.extend([[loop[i], loop[(i+1)%len(loop)], idx] for i in range(len(loop))])
        report.append(dict(boundary_vertices=len(loop), diameter_fraction=float(diameter/diag)))
    model.vertices = np.asarray(vertices)
    model.faces = np.asarray(faces)
    model.fix_normals()
    return report
