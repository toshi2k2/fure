"""Run the constrained estimator, then rebuild all topology-dependent inputs."""
from __future__ import annotations
import logging
from pathlib import Path
import time
import numpy as np
import open3d as o3d
import trimesh
from scipy.spatial import cKDTree
from . import geometry as g
from .io import read_config, local_path, mesh, annotations, face_labels, write_json, digest, export_obj
from .package import ray_scene, write_sdf, render_masks, write_interface
from .tangents import generate
from .qa import render_report

LOG = logging.getLogger(__name__)


def intersections(v: np.ndarray, f: np.ndarray) -> set:
    """Run Open3D triangle-intersection tests after AABB subdivision.

    A triangle crossing a split occurs in both children. Therefore an intersecting
    pair always reaches a common leaf; partitioning does not omit possible pairs.
    This avoids Open3D's quadratic all-triangle broad phase on large surfaces.
    """
    low, high = v[f].min(1), v[f].max(1)
    result = set()
    stack = [(np.arange(len(f)), 0)]
    while stack:
        ids, depth = stack.pop()
        if len(ids) > 512 and depth < 32:
            centers = (low[ids]+high[ids])*.5
            axis = int(np.argmax(np.ptp(centers, axis=0)))
            split = np.median(centers[:, axis])
            left, right = ids[low[ids, axis] <= split], ids[high[ids, axis] >= split]
            if max(len(left), len(right)) < .95*len(ids):
                stack.extend(((left, depth+1), (right, depth+1)))
                continue
        used, remap = np.unique(f[ids], return_inverse=True)
        model = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v[used]),
                                         o3d.utility.Vector3iVector(remap.reshape(-1, 3)))
        pairs = np.asarray(model.get_self_intersecting_triangles(), dtype=int)
        result.update(tuple(sorted(p)) for p in ids[pairs].tolist())
    return result


def collision_guard(v: np.ndarray, f: np.ndarray, n: np.ndarray, d: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, dict]:
    original = intersections(v, f)
    history = []
    for iteration in range(17):
        d, flip_guard = g.safe_displacement(v, f, n, d)
        pairs = intersections(v-d[:, None]*n, f)
        new = pairs-original
        history.append(dict(iteration=iteration, total=len(pairs), new=len(new)))
        if not new:
            return d, dict(source_pairs=len(original), final_pairs=len(pairs), new_pairs=0, history=history, **flip_guard)
        ids = np.unique(f[np.asarray(list(new), dtype=int).ravel()])
        affected = np.zeros(len(v), bool)
        affected[ids] = True
        d[np.unique(edges[affected[edges].any(1)])] *= .8
        d[ids] *= .25
    raise ValueError("Collision guard could not remove all new triangle intersections")


def containment_guard(v: np.ndarray, f: np.ndarray, n: np.ndarray, d: np.ndarray,
                      edges: np.ndarray) -> tuple[np.ndarray, dict]:
    """Back off concave normal displacements that leave the outer mesh."""
    d = d.copy()
    scene = ray_scene(v, f)
    diagonal = float(np.linalg.norm(np.ptp(v, axis=0)))
    tolerance = diagonal * 1e-5
    history = []
    for iteration in range(12):
        root = v - d[:, None] * n
        signed = scene.compute_signed_distance(
            o3d.core.Tensor(root.astype(np.float32)), nsamples=3
        ).numpy()
        if not np.isfinite(signed).all():
            raise ValueError("Outer containment check returned non-finite distances")
        outside = np.flatnonzero(signed > tolerance)
        history.append({
            "iteration": iteration,
            "outside_vertices": int(len(outside)),
            "max_signed_distance": float(signed.max()),
        })
        if not len(outside):
            return d, {
                "tolerance": tolerance,
                "outside_vertices": 0,
                "max_signed_distance": float(signed.max()),
                "history": history,
            }
        affected = np.zeros(len(v), dtype=bool)
        affected[outside] = True
        neighbors = np.unique(edges[affected[edges].any(axis=1)])
        d[neighbors] *= .8
        d[outside] *= .25
        d, _ = g.safe_displacement(v, f, n, d)
    raise ValueError("Containment guard could not move every root vertex inside the outer mesh")


def validate_scale(cfg: dict, eyes: Path | None) -> dict:
    """Check an optional eye-distance scale against the eye landmarks."""
    declared_scale = cfg.get("cm_per_scene_unit")
    if (not isinstance(declared_scale, (int, float)) or isinstance(declared_scale, bool)
            or not np.isfinite(declared_scale) or declared_scale <= 0):
        raise ValueError("cm_per_scene_unit must be finite and positive")
    cm_per_scene_unit = float(declared_scale)
    assumed_cm = cfg.get("eye_distance_cm")
    if (eyes is None) != (assumed_cm is None):
        raise ValueError("eyes and eye_distance_cm must be set together")
    if eyes is None:
        return {"method": "declared_scale", "cm_per_scene_unit": cm_per_scene_unit}
    if (not isinstance(assumed_cm, (int, float)) or isinstance(assumed_cm, bool)
            or not np.isfinite(assumed_cm) or assumed_cm <= 0):
        raise ValueError("eye_distance_cm must be finite and positive")
    assumed_cm = float(assumed_cm)
    landmarks = np.asarray(trimesh.load(eyes, process=False).vertices)
    if landmarks.shape != (2, 3) or not np.isfinite(landmarks).all():
        raise ValueError("eyes.ply must contain exactly two finite eye-center points")
    distance_scene = float(np.linalg.norm(landmarks[0] - landmarks[1]))
    if distance_scene <= 0:
        raise ValueError("eyes.ply must contain two distinct eye-center points")
    implied = assumed_cm / distance_scene
    if not np.isclose(cm_per_scene_unit, implied, rtol=2e-6, atol=1e-9):
        raise ValueError(
            "cm_per_scene_unit disagrees with eye_distance_cm and the eye landmarks: "
            f"expected {implied:.12g}, got {cm_per_scene_unit:.12g}"
        )
    return {
        "method": "inter_eye_distance",
        "assumed_distance_cm": assumed_cm,
        "landmark_distance_scene": distance_scene,
        "cm_per_scene_unit": cm_per_scene_unit,
    }


def run(config_path: Path, output: Path, variant: str, geometry_only: bool = False) -> dict:
    config_path = config_path.resolve()
    output = output.resolve()
    if output.exists():
        raise FileExistsError(f"Output exists; choose a new folder: {output}")
    cfg = read_config(config_path)
    paths = {key: local_path(config_path, cfg[key]) for key in ("outer_mesh", "annotation_mesh", "annotations", "shell_stats", "views")}
    if cfg.get("eyes"):
        paths["eyes"] = local_path(config_path, cfg["eyes"])
    if cfg.get("orientations"):
        paths["orientations"] = local_path(config_path, cfg["orientations"])
    for key, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"{key}: {path}")
    output.mkdir(parents=True)
    try:
        result = _run(cfg, paths, output, variant, geometry_only, config_path)
        return result
    except Exception as error:
        write_json(output / "status.json", dict(ready_for_strand_data=False, state="failed", error=str(error)))
        raise


def _run(cfg: dict, paths: dict, output: Path, variant: str, geometry_only: bool, config_path: Path) -> dict:
    started = time.perf_counter()
    scale_basis = cfg.get("scale_basis")
    if (not isinstance(scale_basis, str) or not scale_basis.strip()
            or scale_basis.strip().lower() == "unspecified"):
        raise ValueError("scale_basis must describe the measurement or assumption used")
    part_labels_approved = cfg.get("part_labels_approved", False)
    if not isinstance(part_labels_approved, bool):
        raise ValueError("part_labels_approved must be true or false")
    model = mesh(paths["outer_mesh"])
    v, f = np.asarray(model.vertices), np.asarray(model.faces)
    if not np.isfinite(v).all():
        raise ValueError("Outer mesh vertices must be finite")
    component_count = len(model.split(only_watertight=False, repair=False))
    degenerate_faces = int(np.count_nonzero(~model.nondegenerate_faces()))
    if component_count != 1:
        raise ValueError(f"Outer mesh has {component_count} connected components; expected one")
    if degenerate_faces:
        raise ValueError(f"Outer mesh has {degenerate_faces} degenerate or near-degenerate faces")
    if (not model.is_winding_consistent or not np.isfinite(model.volume)
            or model.volume <= 0):
        raise ValueError("Outer mesh needs consistent outward winding; run clean-surface before annotation")
    if not model.is_watertight:
        raise ValueError("Defurring requires a watertight outer mesh; repair it before annotation")
    source_intersections = intersections(v, f)
    if source_intersections:
        raise ValueError(
            f"Outer mesh has {len(source_intersections)} self-intersecting triangle pairs; "
            "repair it before annotation"
        )
    n, edges = g.normals_and_edges(v, f)
    annotation_mesh = mesh(paths["annotation_mesh"])
    names, original_labels = annotations(paths["annotations"], len(annotation_mesh.vertices))
    annotation_names = set(names)
    if np.array_equal(annotation_mesh.vertices, v) and np.array_equal(annotation_mesh.faces, f):
        labels, distances = original_labels, np.zeros(len(v))
    else:
        distances, ids = cKDTree(annotation_mesh.vertices).query(v, workers=4)
        labels = original_labels[ids]
    present = np.unique(labels)
    empty_parts = [name for i, name in enumerate(names) if i not in present]
    names = [names[i] for i in present]
    labels = np.searchsorted(present, labels)
    diag = float(np.linalg.norm(np.ptp(v, axis=0)))
    transfer_p95 = float(np.percentile(distances, 95)/diag)
    if transfer_p95 > cfg.get("max_annotation_transfer_fraction", .08):
        raise ValueError(
            "Annotation transfer failed: the P95 nearest-vertex distance is "
            f"{transfer_p95:.3f} of the mesh bounding-box diagonal"
        )
    scale_qa = validate_scale(cfg, paths.get("eyes"))
    cm = scale_qa["cm_per_scene_unit"]
    with np.load(paths["shell_stats"], allow_pickle=False) as loaded:
        shell = dict(loaded)
    shell_qa = g.validate_shell(shell)
    if np.asarray(shell["reliability_mask_init"]).size != len(shell["verts"]):
        raise ValueError("Shell reliability mask has wrong topology")
    write_json(output / "shell_validation.json", shell_qa)
    shell, normal_frame = g.align_shell_normal_frame(v, n, shell, .03*diag)
    transfer = g.transfer_widths(v, n, labels, shell, .03*diag)
    if np.percentile(transfer["nearest_distance"], 95) > .12*diag:
        raise ValueError("Shell and mesh appear misaligned; check their coordinate transforms")
    configured_nonfur = cfg.get("nonfur_parts", [])
    if (not isinstance(configured_nonfur, list)
            or any(not isinstance(name, str) or not name for name in configured_nonfur)
            or len(configured_nonfur) != len(set(configured_nonfur))):
        raise ValueError("nonfur_parts must be a list of unique part names")
    unknown_nonfur = sorted(set(configured_nonfur) - annotation_names)
    if unknown_nonfur:
        raise ValueError(f"nonfur_parts contains unknown parts: {', '.join(unknown_nonfur)}")
    nonfur = set(g.ZERO_PARTS) | set(configured_nonfur) | {"unlabeled"}
    pinned = np.isin(np.asarray(names)[labels], sorted(nonfur))
    semantic = cfg.get("semantic_prior", {}) if variant == "shape_vlm" else {}
    if variant == "shape_vlm":
        if not isinstance(semantic.get("length_cm"), dict) or not semantic["length_cm"]:
            raise ValueError("shape_vlm requires semantic_prior.length_cm")
        if not isinstance(semantic.get("thickness_cm"), dict) or not semantic["thickness_cm"]:
            raise ValueError("shape_vlm requires semantic_prior.thickness_cm")
        if not semantic.get("basis"):
            raise ValueError("shape_vlm requires a basis for its part-level values")
        for key in ("length_cm", "thickness_cm"):
            missing = sorted(set(names) - set(semantic[key]))
            if missing:
                raise ValueError(f"semantic_prior.{key} is missing parts: {', '.join(missing)}")
    prior_cm = semantic.get("thickness_cm", {})
    if any(not np.isfinite(value) or value < 0 for value in prior_cm.values()):
        raise ValueError("semantic_prior.thickness_cm values must be finite, nonnegative centimeters")
    clipped, prior, ratio = g.part_calibrated_targets(transfer["nearest_width"], labels, names,
                                                    {p: float(x)/cm for p, x in prior_cm.items()})
    scene = ray_scene(v, f)
    hits = scene.cast_rays(o3d.core.Tensor(np.column_stack([v-diag*1e-5*n, -n]).astype(np.float32)))["t_hit"].numpy()
    upper = np.minimum(.45*np.where(np.isfinite(hits), hits, .1*diag), .05*diag)
    target = np.clip(transfer["width"]*ratio, 0, upper)
    safe_prior = np.minimum(prior, upper)
    displacement, solver = g.solve_displacement(target, safe_prior, transfer["confidence"], upper, pinned,
                                                g.laplacian(v, edges, labels), cfg.get("smoothness", 8.0))
    displacement, containment = containment_guard(v, f, n, displacement, edges)
    displacement, guard = collision_guard(v, f, n, displacement, edges)
    root = v-displacement[:, None]*n
    if guard["final_pairs"] and not geometry_only:
        raise ValueError(f"Root retains {guard['final_pairs']} self-intersections")
    properties = g.calibrate_lengths(transfer["nearest_width"], labels, names, cm, semantic.get("length_cm", {}))
    for name in nonfur & properties.keys():
        for key in ("effective_thickness_cm", "multiplier", "v1_length_cm", "v2_length_cm", "v1_width_cm", "v2_width_cm"):
            properties[name][key] = 0.0
    topology = {name: np.flatnonzero(labels == i).tolist() for i, name in enumerate(names)}
    triangle_labels = face_labels(f, labels, pinned)
    for filename in ("annotations.json", "annotations_furless_reshaped2.json"):
        write_json(output / filename, topology)
    write_json(output / "annotations_faces.json", {name: np.flatnonzero(triangle_labels == i).tolist() for i, name in enumerate(names)})
    export_obj(output / "neus.obj", v, f)
    export_obj(output / "furless.obj", root, f)
    # All downstream arrays bind to the exact serialized topology. Check the
    # disk representation instead of assuming an OBJ float round-trip cannot
    # create a near-contact intersection.
    disk_root = mesh(output / "furless.obj")
    if not np.array_equal(np.asarray(disk_root.faces), f) or not np.array_equal(np.asarray(disk_root.vertices), root):
        raise ValueError("High-precision OBJ did not preserve root vertex/face order exactly")
    serialized_pairs = intersections(np.asarray(disk_root.vertices), np.asarray(disk_root.faces))
    guard["serialized_pairs"] = len(serialized_pairs)
    if serialized_pairs and not geometry_only:
        raise ValueError(f"Serialized root has {len(serialized_pairs)} self-intersections")
    root = np.asarray(disk_root.vertices)
    np.savez_compressed(output / "defurring_fields.npz", displacement=displacement, labels=labels, confidence=transfer["confidence"],
                        transferred_width=transfer["width"], nearest_width=transfer["nearest_width"], upper=upper)
    prop = dict(variant=variant, per_part=properties, cm_per_scene_unit=cm,
                scale_basis=scale_basis,
                eye_distance_cm=cfg.get("eye_distance_cm"),
                geometry_prior="part-pooled shell only" if variant == "shape" else "65% shell / 35% part thickness",
                part_prior_basis=semantic.get("basis", "not used"))
    write_json(output / "strand_properties.json", prop)
    metrics = dict(vertices=len(v), faces=len(f), source_watertight=bool(model.is_watertight),
                   annotation_transfer_p95_diagonal=transfer_p95, support_fraction=float(transfer["supported"].mean()),
                   shrink_median_cm=float(np.median(displacement)*cm), shrink_p95_cm=float(np.percentile(displacement, 95)*cm),
                   anchor_drift_max=float(displacement[pinned].max()) if pinned.any() else 0.,
                   empty_parts_removed=empty_parts,
                   normal_frame=normal_frame, solver=solver, collision_guard=guard,
                   containment_guard=containment,
                   scale_check=scale_qa,
                   scale_basis=scale_basis)
    if model.is_watertight:
        signed = scene.compute_signed_distance(o3d.core.Tensor(root.astype(np.float32)), nsamples=3).numpy()
        metrics["inside_outer_fraction"] = float(np.mean(signed <= diag*1e-5))
        if metrics["inside_outer_fraction"] < 1.0 and not geometry_only:
            raise ValueError("Root vertices remain outside the outer-mesh tolerance")
    tangent = None
    if not geometry_only:
        LOG.info("Generating Directional tangents on the new root topology")
        repo = Path(__file__).resolve().parents[1]
        metrics["tangents"] = generate(output / "furless.obj", output, repo)
        tangent = np.load(output / "tan_furless.npy")
        metrics["sdf"] = write_sdf(output, root, f, cfg.get("sdf_resolution", 64))
        LOG.info("Rendering camera-aligned root and part masks")
        metrics["masks"] = render_masks(
            output, paths["views"], root, f, triangle_labels, names, nonfur,
            paths.get("orientations"), cfg.get("orientation_workers", 1),
        )
        write_interface(output, prop, cfg, paths.get("eyes"), nonfur)
        from .validation import validate
        metrics["package_validation"] = validate(output)
    render_report(output / "output_screenshots", v, root, f, labels, names, transfer["nearest_width"], transfer["confidence"], tangent)
    metrics["seconds"] = time.perf_counter()-started
    write_json(output / "metrics.json", metrics)
    write_json(output / "run.json", dict(
        config=cfg,
        inputs={key: {"file": p.name, "sha256": digest(p)}
                for key, p in paths.items() if p.is_file()},
        algorithm="bounded normal-displacement estimator"))
    geometric_transfer = not np.array_equal(distances, np.zeros_like(distances))
    package_complete = not geometry_only
    status = dict(state="geometry_only" if geometry_only else "prepared", data_generated=not geometry_only,
                  ready_for_strand_data=package_complete,
                  technical_validation_passed=package_complete,
                  readiness_blocker=None if package_complete else "package_not_generated",
                  annotation_origin=cfg.get("annotation_origin", "unspecified"),
                  annotation_transfer_method="nearest_vertex" if geometric_transfer else "same_topology",
                  part_labels_approved=part_labels_approved,
                  downstream_decoder_training_tested=False, downstream_training_scope="external",
                  gpu_reconstruction_tested_here=False)
    write_json(output / "status.json", status)
    return status
