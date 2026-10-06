"""Prepare matching images, masks, and cameras for reconstruction."""
from __future__ import annotations
import json
from pathlib import Path
import shutil
import subprocess
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
import trimesh
from .cameras import decompose
from .colmap import (
    qvec_to_rotation,
    read_cameras_binary,
    read_cameras_text,
    read_views_binary,
    read_views_text,
)
from .io import write_json


def _check_skew(k: np.ndarray) -> None:
    """Reject meaningful pinhole skew while tolerating decomposition roundoff."""
    # K is measured in pixels, so a fixed near-zero threshold is not scale
    # invariant.  The 1e-6 relative guard is sub-pixel even for long lenses.
    tolerance = 1e-6 * max(abs(float(k[0, 0])), abs(float(k[1, 1])), 1.0)
    if abs(float(k[0, 1])) > tolerance:
        raise ValueError("Nonzero camera skew is unsupported")


def extract_video(video: Path, output: Path, fps: float = 2) -> None:
    if output.exists():
        raise FileExistsError(output)
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not shutil.which("ffmpeg"):
        raise RuntimeError("FFmpeg is required to extract video frames")
    output.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "warning", "-i", str(video),
                    "-vf", f"fps={fps}", "-start_number", "0", str(output / "%04d.png")], check=True)


def visual_hull(projections: np.ndarray, masks: list[np.ndarray], resolution: int = 56) -> np.ndarray:
    grid = np.linspace(-.8, .8, resolution)
    points = np.stack(np.meshgrid(grid, grid, grid, indexing="ij"), -1).reshape(-1, 3)
    keep = np.ones(len(points), bool)
    for p, mask in zip(projections, masks):
        uvz = np.column_stack([points, np.ones(len(points))]) @ p[:3].T
        uv = np.rint(uvz[:, :2]/np.maximum(uvz[:, 2:3], 1e-12)).astype(int)
        visible = (uvz[:, 2] > 0) & (uv[:, 0] >= 0) & (uv[:, 0] < mask.shape[1]) & (uv[:, 1] >= 0) & (uv[:, 1] < mask.shape[0])
        sample = np.zeros(len(points), bool)
        sample[visible] = mask[uv[visible, 1], uv[visible, 0]] > 127
        keep &= sample
    if keep.sum() < 100:
        raise ValueError(
            "Visual hull retained fewer than 100 grid samples; check the camera convention, "
            "masks, static pose, and bounds"
        )
    return points[keep]


def prepare(images: Path, masks: Path, projections_path: Path, output: Path,
            bounds: list[float] | None = None, points_path: Path | None = None) -> None:
    """Rectify a capture and apply one optional normalization to cameras and seed points."""
    if output.exists():
        raise FileExistsError(output)
    images_list = sorted(p for p in images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    projection = np.load(projections_path, allow_pickle=False).astype(np.float64)
    if len(images_list) < 3 or projection.shape not in ((len(images_list), 4, 4), (len(images_list), 3, 4)):
        raise ValueError("Provide at least three calibrated views, with projections in sorted image order")
    transform = np.eye(4)
    if bounds:
        low, high = np.asarray(bounds).reshape(2, 3)
        if not np.all(high > low):
            raise ValueError("Bounds must be xmin ymin zmin xmax ymax zmax")
        transform[:3, :3] *= np.max(high-low)/1.6
        transform[:3, 3] = (high+low)/2
    p = np.repeat(np.eye(4)[None], len(projection), axis=0)
    p[:, :3] = projection[:, :3, :4] @ transform
    # Upstream 3DGS assumes centered principal points, and parts of Frosting use
    # shared intrinsics. Rectify pixels and masks and update K accordingly.
    import cv2
    decomposed = [decompose(value) for value in p]
    sizes = [Image.open(image).size for image in images_list]
    fx = min(k[0, 0] for k, _ in decomposed)
    fy = min(k[1, 1] for k, _ in decomposed)
    half_w = max(max(abs(k[0, 2]), abs(w-k[0, 2]))*fx/k[0, 0] for (k, _), (w, h) in zip(decomposed, sizes))
    half_h = max(max(abs(k[1, 2]), abs(h-k[1, 2]))*fy/k[1, 1] for (k, _), (w, h) in zip(decomposed, sizes))
    w, h = 2*int(np.ceil(half_w)), 2*int(np.ceil(half_h))
    if w*h > 4*max(width*height for width, height in sizes):
        raise ValueError("Rectified canvas exceeds four times the largest input image area; inspect camera intrinsics")
    common_k = np.array([[fx, 0, w/2], [0, fy, h/2], [0, 0, 1.]])
    output.mkdir(parents=True)
    for folder in ("views/images_2", "views/foreground", "frosting_scene/images", "frosting_scene/sparse/0", "neus2/images"):
        (output / folder).mkdir(parents=True)
    frames, mask_arrays, intrinsics, extrinsics, mapping = [], [], [], [], []
    cameras_txt, images_txt = [], []
    for index, image in enumerate(images_list):
        rgb = Image.open(image).convert("RGB")
        mask_path = masks / (image.stem + ".png")
        alpha = Image.open(mask_path).convert("L")
        if alpha.size != rgb.size or not np.array(alpha).any():
            raise ValueError(f"Invalid mask: {mask_path}")
        original_k, extr = decomposed[index]
        _check_skew(original_k)
        pixel_transform = common_k @ np.linalg.inv(original_k)
        rgb = Image.fromarray(cv2.warpPerspective(np.asarray(rgb), pixel_transform, (w, h), flags=cv2.INTER_LINEAR))
        alpha = Image.fromarray(cv2.warpPerspective(np.asarray(alpha), pixel_transform, (w, h), flags=cv2.INTER_LINEAR))
        p[index, :3] = common_k @ extr[:3]
        name = f"{index:04d}.png"
        rgb.save(output / "views/images_2" / name)
        alpha.save(output / "views/foreground" / name)
        rgba = rgb.copy()
        rgba.putalpha(alpha)
        rgba.save(output / "frosting_scene/images" / name)
        shutil.copy2(output / "frosting_scene/images" / name, output / "neus2/images" / name)
        k, extr = decompose(p[index])
        _check_skew(k)
        intrinsic = np.eye(4)
        intrinsic[:3, :3] = k
        intrinsics.append(intrinsic)
        extrinsics.append(extr)
        frames.append(dict(file_path=f"images/{name}", transform_matrix=np.linalg.inv(extr).tolist(), intrinsic_matrix=intrinsic.tolist()))
        q = Rotation.from_matrix(extr[:3, :3]).as_quat()[[3, 0, 1, 2]]
        w, h = rgb.size
        cameras_txt.append(f"{index+1} PINHOLE {w} {h} {k[0,0]:.14g} {k[1,1]:.14g} {k[0,2]:.14g} {k[1,2]:.14g}")
        images_txt.extend([f"{index+1} " + " ".join(f"{x:.17g}" for x in np.r_[q, extr[:3, 3]]) + f" {index+1} {name}", ""])
        mask_arrays.append(np.array(alpha))
        mapping.append(dict(index=index, source=image.name, output=name, pixel_homography=pixel_transform.tolist()))
    np.save(output / "views/projection.npy", p.astype(np.float32))
    np.save(output / "views/cameras_intr.npy", np.asarray(intrinsics, np.float32))
    np.save(output / "views/cameras_extr.npy", np.asarray(extrinsics, np.float32))
    write_json(output / "neus2/transform.json", dict(from_na=True, w=w, h=h, scale=.5, offset=[.5]*3, aabb_scale=1., frames=frames))
    sparse = output / "frosting_scene/sparse/0"
    (sparse / "cameras.txt").write_text("\n".join(cameras_txt)+"\n")
    (sparse / "images.txt").write_text("\n".join(images_txt)+"\n")
    (sparse / "points3D.txt").write_text("# Seed point cloud provided in points3D.ply\n")
    if points_path:
        seed = np.asarray(trimesh.load(points_path, process=False).vertices)
        seed = (seed-transform[:3, 3])/transform[0, 0]
    else:
        seed = visual_hull(p, mask_arrays)
    # These points initialize reconstruction; they do not encode coat thickness.
    from plyfile import PlyData, PlyElement
    values = np.zeros(len(seed), dtype=[(k, "f4") for k in ("x", "y", "z", "nx", "ny", "nz")] + [(k, "u1") for k in ("red", "green", "blue")])
    for axis, key in enumerate(("x", "y", "z")):
        values[key] = seed[:, axis]
    for key in ("red", "green", "blue"):
        values[key] = 128
    PlyData([PlyElement.describe(values, "vertex")]).write(sparse / "points3D.ply")
    write_json(output / "normalization.json", dict(scene_to_original=transform.tolist(), original_to_scene=np.linalg.inv(transform).tolist(),
               projection_rule="P_prepared is projectively equivalent to H_pixel @ P_original @ scene_to_original",
               common_intrinsics=common_k.tolist(), seed_points=len(seed)))
    write_json(output / "image_manifest.json", mapping)


def calibrate(images: Path, output: Path, masks: Path | None = None) -> None:
    """Run COLMAP structure from motion and image undistortion for a static capture."""
    if not shutil.which("colmap"):
        raise RuntimeError("COLMAP is required for camera calibration")
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    database = output / "database.db"
    features = ["colmap", "feature_extractor", "--database_path", str(database), "--image_path", str(images),
                "--ImageReader.camera_model", "OPENCV", "--SiftExtraction.use_gpu", "0"]
    if masks:
        features += ["--ImageReader.mask_path", str(masks)]
    subprocess.run(features, check=True)
    subprocess.run(["colmap", "exhaustive_matcher", "--database_path", str(database), "--SiftMatching.use_gpu", "0"], check=True)
    sparse = output / "sparse"
    sparse.mkdir()
    subprocess.run(["colmap", "mapper", "--database_path", str(database), "--image_path", str(images), "--output_path", str(sparse)], check=True)
    models = sorted(p for p in sparse.iterdir() if p.is_dir())
    if len(models) != 1:
        raise ValueError("COLMAP did not produce one connected model; inspect overlap/pose rather than joining unaligned reconstructions")
    subprocess.run(["colmap", "image_undistorter", "--image_path", str(images), "--input_path", str(models[0]),
                    "--output_path", str(output / "undistorted"), "--output_type", "COLMAP"], check=True)
    # Generate masks for the undistorted views so pixels and cameras stay aligned.
    write_json(output / "next_step.json", dict(required="Generate masks for undistorted/images, then use import-colmap to obtain matching projection.npy"))


def import_colmap(model: Path, images: Path, output: Path) -> None:
    """Convert an undistorted COLMAP model to projections in sorted image order."""
    if output.exists():
        raise FileExistsError(output)
    if (model / "cameras.bin").is_file():
        cameras = read_cameras_binary(model / "cameras.bin")
        views = read_views_binary(model / "images.bin")
    else:
        cameras = read_cameras_text(model / "cameras.txt")
        views = read_views_text(model / "images.txt")
    by_name = {v.name: v for v in views.values()}
    paths = sorted(p for p in images.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    if set(p.name for p in paths) != set(by_name):
        raise ValueError("Image set and registered COLMAP image set differ; remove unregistered views explicitly")
    projections = []
    for path in paths:
        v = by_name[path.name]
        camera = cameras[v.camera_id]
        if camera.model == "PINHOLE":
            fx, fy, cx, cy = camera.params
        elif camera.model == "SIMPLE_PINHOLE":
            fx, cx, cy = camera.params
            fy = fx
        else:
            raise ValueError("Use COLMAP image_undistorter first; distorted cameras are not accepted")
        k = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        extr = np.column_stack([qvec_to_rotation(v.qvec), v.tvec])
        p = np.eye(4)
        p[:3] = k @ extr
        projections.append(p)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.save(output, np.asarray(projections, np.float32))
