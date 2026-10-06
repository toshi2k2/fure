"""Check an animal 3D model numerically and against calibrated views."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import open3d as o3d
from PIL import Image
import trimesh

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fure_defur.cameras import decompose, validate_views
from fure_defur.io import write_json
from fure_defur.package import ray_scene
from fure_defur.pipeline import intersections


def edge_counts(model: trimesh.Trimesh) -> tuple[int, int]:
    edges = np.sort(np.concatenate([
        model.faces[:, [0, 1]], model.faces[:, [1, 2]], model.faces[:, [2, 0]]
    ]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    return int(np.sum(counts == 1)), int(np.sum(counts > 2))


def topology(model: trimesh.Trimesh, check_intersections: bool) -> dict:
    if not np.isfinite(model.vertices).all():
        raise ValueError("Mesh vertices are not finite")
    boundary, nonmanifold = edge_counts(model)
    components = model.split(only_watertight=False)
    pairs = intersections(np.asarray(model.vertices), np.asarray(model.faces)) if check_intersections else set()
    return dict(vertices=len(model.vertices), faces=len(model.faces), components=len(components),
                largest_component_face_fraction=max(len(part.faces) for part in components) / len(model.faces),
                watertight=bool(model.is_watertight), winding_consistent=bool(model.is_winding_consistent),
                boundary_edges=boundary, nonmanifold_edges=nonmanifold, euler_number=int(model.euler_number),
                volume=float(model.volume), self_intersection_pairs=len(pairs),
                self_intersections_checked=check_intersections,
                bounds=np.asarray(model.bounds).tolist())


def surface_distance(model: trimesh.Trimesh, reference: trimesh.Trimesh, count: int) -> dict:
    np.random.seed(0)
    model_points, _ = trimesh.sample.sample_surface(model, count)
    reference_points, _ = trimesh.sample.sample_surface(reference, count)
    model_scene, reference_scene = ray_scene(reference.vertices, reference.faces), ray_scene(model.vertices, model.faces)
    forward = model_scene.compute_distance(o3d.core.Tensor(model_points.astype(np.float32))).numpy()
    reverse = reference_scene.compute_distance(o3d.core.Tensor(reference_points.astype(np.float32))).numpy()
    diagonal = float(np.linalg.norm(np.ptp(reference.vertices, axis=0)))
    both = np.concatenate([forward, reverse]) / diagonal
    return dict(samples_per_direction=count, normalized_by_reference_bbox_diagonal=diagonal,
                symmetric_mean_fraction=float(np.mean(both)), symmetric_p95_fraction=float(np.percentile(both, 95)),
                symmetric_max_fraction=float(np.max(both)), model_to_reference_p95_fraction=float(np.percentile(forward, 95) / diagonal),
                reference_to_model_p95_fraction=float(np.percentile(reverse, 95) / diagonal))


def silhouettes(model: trimesh.Trimesh, views: Path, output: Path) -> dict:
    images, projections = validate_views(views)
    scene = ray_scene(model.vertices, model.faces)
    overlaps = []
    previews = set(np.linspace(0, len(images)-1, min(4, len(images)), dtype=int))
    overlay = output / "silhouette_overlays"
    overlay.mkdir(parents=True)
    for index, (path, projection) in enumerate(zip(images, projections)):
        width, height = Image.open(path).size
        k, extr = decompose(projection)
        rays = scene.create_rays_pinhole(o3d.core.Tensor(k), o3d.core.Tensor(extr), width, height)
        ids = scene.cast_rays(rays)["primitive_ids"].numpy()
        predicted = ids != np.iinfo(np.uint32).max
        observed = np.asarray(Image.open(views / "foreground" / path.name).convert("L")) > 127
        overlaps.append(float(np.sum(predicted & observed) / max(np.sum(predicted | observed), 1)))
        if index in previews:
            rgb = np.asarray(Image.open(path).convert("RGB")).copy()
            rgb[predicted & observed] = np.round(.55*rgb[predicted & observed] + .45*np.array([40, 220, 80])).astype(np.uint8)
            rgb[predicted & ~observed] = [255, 70, 40]
            rgb[~predicted & observed] = [40, 100, 255]
            Image.fromarray(rgb).save(overlay / path.name)
    return dict(views=len(images), iou_mean=float(np.mean(overlaps)), iou_min=float(np.min(overlaps)),
                iou_max=float(np.max(overlaps)),
                overlay_legend="green overlap, red mesh-only, blue observed-foreground-only")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    parser.add_argument("--views", type=Path)
    parser.add_argument("--distance-samples", type=int, default=20000)
    parser.add_argument("--skip-self-intersections", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    model = trimesh.load(args.mesh, process=False, force="mesh")
    report = {"mesh": args.mesh.as_posix(), "topology": topology(model, not args.skip_self_intersections)}
    if args.reference:
        reference = trimesh.load(args.reference, process=False, force="mesh")
        report["reference"] = args.reference.as_posix()
        report["surface_distance"] = surface_distance(model, reference, args.distance_samples)
    if args.views:
        report["silhouettes"] = silhouettes(model, args.views, args.output)
    write_json(args.output / "audit.json", report)
    print((args.output / "audit.json").read_text())


if __name__ == "__main__":
    main()
