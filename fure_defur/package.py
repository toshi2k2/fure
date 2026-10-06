"""Root-dependent SDFs, visible part masks and a portable training interface."""
from __future__ import annotations
from pathlib import Path
import json
import shutil
import pickle
import numpy as np
import open3d as o3d
from PIL import Image
import yaml
from .cameras import decompose, validate_views
from .io import write_json, digest


def identity_scale_bytes() -> bytes:
    """Return the exact compatibility transform written with each package."""
    return pickle.dumps(dict(scale=1.0, translation=np.zeros(3)), protocol=4)


def ray_scene(vertices: np.ndarray, faces: np.ndarray) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.t.geometry.TriangleMesh(o3d.core.Tensor(vertices.astype(np.float32)),
                                                  o3d.core.Tensor(faces.astype(np.int32))))
    return scene


def write_sdf(output: Path, vertices: np.ndarray, faces: np.ndarray, size: int = 64) -> dict:
    pad = np.linalg.norm(np.ptp(vertices, axis=0))*.02
    low, high = vertices.min(0)-pad, vertices.max(0)+pad
    axes = [np.linspace(low[i], high[i], size, dtype=np.float32) for i in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    scene = ray_scene(vertices, faces)
    values = np.concatenate([scene.compute_signed_distance(o3d.core.Tensor(p), nsamples=3).numpy()
                             for p in np.array_split(points, max(1, len(points)//65536))])
    if not np.isfinite(values).all() or values.min() >= 0 or values.max() <= 0:
        raise ValueError("SDF must have finite inside-negative and outside-positive samples")
    np.save(output / "sdf_grid.npy", values.reshape(size, size, size).astype(np.float32))
    np.save(output / "min_bound.npy", low.astype(np.float32))
    np.save(output / "max_bound.npy", high.astype(np.float32))
    return dict(shape=[size]*3, order="xyz", sign="negative inside", minimum=float(values.min()), maximum=float(values.max()))


def render_masks(output: Path, views: Path, vertices: np.ndarray, faces: np.ndarray,
                 face_parts: np.ndarray, names: list[str], nonfur: set[str],
                 orientations: Path | None = None, orientation_workers: int = 1) -> dict:
    """Z-buffered raycasting; foreground and observed coat silhouette are distinct."""
    images, projections = validate_views(views)
    scene = ray_scene(vertices, faces)
    for name in ("body", "hair", "bald", "foreground"):
        (output / "masks_2" / name).mkdir(parents=True, exist_ok=True)
    (output / "part_masks").mkdir(exist_ok=True)
    shutil.copytree(views / "images_2", output / "images_2")
    orientation_source = orientations
    if orientation_source is None and (views / "orientations_2").is_dir():
        orientation_source = views / "orientations_2"
    if orientation_source is not None:
        shutil.copytree(orientation_source, output / "orientations_2")
    if not (output / "orientations_2/angles").is_dir():
        from .orientations import generate
        generate(output / "images_2", views / "foreground", output / "orientations_2",
                 orientation_workers)
        orientation_method = "generated"
    else:
        orientation_method = "precomputed"
    np.save(output / "projection.npy", projections.astype(np.float32))
    overlaps = []
    previews = set(np.linspace(0, len(images)-1, min(4, len(images)), dtype=int))
    nonfur_ids = np.array([name in nonfur for name in names])
    for index, (image, projection) in enumerate(zip(images, projections)):
        width, height = Image.open(image).size
        k, extr = decompose(projection)
        # create_rays_pinhole uses the same OpenCV world-to-camera convention.
        rays = scene.create_rays_pinhole(o3d.core.Tensor(k), o3d.core.Tensor(extr), width, height)
        hits = scene.cast_rays(rays)
        ids = hits["primitive_ids"].numpy()
        valid = ids != np.iinfo(np.uint32).max
        foreground = np.asarray(Image.open(views / "foreground" / image.name).convert("L")) > 127
        overlap = float(np.sum(valid & foreground)/max(np.sum(valid | foreground), 1))
        overlaps.append(overlap)
        part = np.zeros((height, width), np.uint16)
        visible = valid & foreground
        part[visible] = face_parts[ids[visible]] + 1
        bald = np.zeros_like(visible)
        bald[visible] = nonfur_ids[face_parts[ids[visible]]]
        for name, mask in (("body", visible), ("hair", foreground & ~bald), ("bald", bald), ("foreground", foreground)):
            Image.fromarray(mask.astype(np.uint8)*255).save(output / "masks_2" / name / image.name)
        Image.fromarray(part).save(output / "part_masks" / image.name)
        if index in previews:
            import cv2
            import matplotlib.pyplot as plt
            colors = plt.get_cmap("tab20")(np.linspace(0, 1, len(names)))[:, :3]*255
            preview = np.asarray(Image.open(image).convert("RGB")).astype(float)
            preview[visible] = .45*preview[visible]+.55*colors[part[visible]-1]
            contour = cv2.Canny(valid.astype(np.uint8)*255, 50, 150) > 0
            preview[contour] = [0, 230, 255]
            folder = output / "output_screenshots/camera_overlays"
            folder.mkdir(parents=True, exist_ok=True)
            Image.fromarray(preview.clip(0, 255).astype(np.uint8)).save(folder / image.name)
    write_json(output / "part_masks/legend.json", {str(i+1): name for i, name in enumerate(names)} | {"0": "background"})
    return dict(views=len(images), root_vs_foreground_iou_mean=float(np.mean(overlaps)),
                root_vs_foreground_iou_min=float(np.min(overlaps)),
                hair_mask="observed foreground excluding visible non-fur parts; overlaps body intentionally",
                part_ids="0 background, i+1 indexes part_names",
                orientations=orientation_method)


def write_interface(output: Path, properties: dict, config: dict, eyes: Path | None,
                    nonfur: set[str]) -> None:
    cm = float(config["cm_per_scene_unit"])
    scale_basis = config.get("scale_basis")
    if (not isinstance(scale_basis, str) or not scale_basis.strip()
            or scale_basis.strip().lower() == "unspecified"):
        raise ValueError("scale_basis must describe the measurement or assumption used")
    parts = properties["per_part"]
    variant = properties["variant"]
    length_key = "v1_length_cm" if variant == "shape" else "v2_length_cm"
    width_key = "v1_width_cm" if variant == "shape" else "v2_width_cm"
    length_cm = {p: r[length_key] for p, r in parts.items()}
    length_scene = {p: x/cm for p, x in length_cm.items()}
    present_nonfur = sorted(set(parts) & nonfur)
    interface = dict(schema_version=1, data_root=".", variant=variant,
                     mesh="furless.obj", outer_mesh="neus.obj",
                     tangent="tan_furless.npy", vertex_parts="annotations_furless_reshaped2.json",
                     face_parts="annotations_faces.json", projection="projection.npy", images="images_2",
                     masks="masks_2", orientations="orientations_2", sdf="sdf_grid.npy",
                     nonfur_parts=present_nonfur,
                     cm_per_scene_unit=cm,
                     scale_basis=scale_basis,
                     length_cm=length_cm, length_scene=length_scene,
                     width_cm={p: r[width_key] for p, r in parts.items()},
                     thickness_cm={p: r["effective_thickness_cm"] for p, r in parts.items()},
                     width_interpretation="heuristic initialization parameter, not measured fiber diameter",
                     gravity_loss_enabled=False, topology_sha256=digest(output / "furless.obj"))
    # Meshes and cameras already share a coordinate frame, so the adapter scale is identity.
    (output / "scale.pickle").write_bytes(identity_scale_bytes())
    textured = dict(path_to_scale="", path_to_mesh="furless.obj", path_to_classes="annotations_furless_reshaped2.json",
                    precomputed_tangent="tan_furless.npy", mapping_length=length_scene,
                    use_metrical_space=False, normalize_decoder_output=False, metric_length=1,
                    mapping_gravity={p: [0., 0., 0.] for p in parts}, fix_normals=False,
                    decoder_checkpoint_path="", num_strands=15000, max_num_strands=500000,
                    texture_size=256, geometry_descriptor_size=64, appearance_descriptor_size=65,
                    texture_type="mlp", num_freqs=10, path_to_uvmap="")
    if eyes is not None:
        import trimesh
        landmarks = np.asarray(trimesh.load(eyes, process=False).vertices)
        if (landmarks.shape != (2, 3) or not np.isfinite(landmarks).all()
                or np.linalg.norm(landmarks[0]-landmarks[1]) <= 0):
            raise ValueError("eyes.ply must contain exactly two finite, distinct eye-center points in mesh coordinates")
        shutil.copy2(eyes, output / "eyes.ply")
        implied_eye_distance_cm = float(np.linalg.norm(landmarks[0]-landmarks[1])*cm)
        eye_distance_cm = float(config.get("eye_distance_cm", implied_eye_distance_cm))
        interface["eye_distance_cm"] = eye_distance_cm
        textured.update(use_metrical_space=True, normalize_decoder_output=True, mapping_length=length_cm,
                        eye_dists_geom="eyes.ply", eye_dists_VQA=eye_distance_cm)
    write_json(output / "strand_package.json", interface)
    adapter = dict(sdf_penetration=dict(sdf_path="sdf_grid.npy", bounds_min="min_bound.npy", bounds_max="max_bound.npy"),
                   sdf_chamfer=dict(path_to_mesh="neus.obj", path_to_scale="", num_points=10000),
                   textured_strands=textured, general=dict(lr=.001, starting_rendering_iter=1, gamma=.5, milestones=[40000,60000,80000]),
                   extra_args=dict(num_guiding_strands=0))
    (output / "hair_config.yaml").write_text(yaml.safe_dump(adapter, sort_keys=False))


def materialize(package: Path, output: Path, decoder: Path) -> None:
    """Resolve package-relative adapter paths for a training environment."""
    if output.exists():
        raise FileExistsError(output)
    if not decoder.is_file():
        raise FileNotFoundError(decoder)
    data = yaml.safe_load((package / "hair_config.yaml").read_text())
    keys = {"sdf_path", "bounds_min", "bounds_max", "path_to_mesh", "path_to_classes", "precomputed_tangent", "eye_dists_geom"}
    for section in data.values():
        for key, value in list(section.items()):
            if key in keys and value:
                resolved = package / value
                if not resolved.is_file():
                    raise FileNotFoundError(resolved)
                section[key] = str(resolved.resolve())
    data["textured_strands"]["decoder_checkpoint_path"] = str(decoder.resolve())
    output.write_text(yaml.safe_dump(data, sort_keys=False))
