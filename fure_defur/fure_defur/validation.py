"""Validate a strand package against this preprocessing schema."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
from PIL import Image
import yaml
from .io import mesh, digest, face_labels as derive_face_labels
from .cameras import decompose
from .tangents import orient


def _declared_path(root: Path, metadata: dict, key: str, directory: bool = False) -> Path:
    value = metadata.get(key)
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"Metadata field {key!r} must be a portable relative path")
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Metadata path escapes the package: {key}={value!r}")
    package = root.resolve()
    path = (package / relative).resolve()
    try:
        path.relative_to(package)
    except ValueError as error:
        raise ValueError(f"Metadata path escapes the package: {key}={value!r}") from error
    valid = path.is_dir() if directory else path.is_file()
    if not valid:
        kind = "directory" if directory else "file"
        raise ValueError(f"Metadata {kind} is missing: {key}={value!r}")
    return path


def _indexed_annotations(path: Path, count: int, kind: str) -> tuple[list[str], np.ndarray]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError(f"{kind} annotations must map part names to index lists")
    names = list(data)
    if any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"{kind} annotation names must be nonempty strings")
    labels = np.full(count, -1, dtype=np.int32)
    for label, name in enumerate(names):
        raw = data[name]
        if not isinstance(raw, list):
            raise ValueError(f"{kind} indices for {name} must be a JSON list")
        if not raw:
            continue
        ids = np.asarray(raw)
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= count:
            raise ValueError(f"Invalid {kind} indices for {name}")
        if len(np.unique(ids)) != len(ids) or (labels[ids] != -1).any():
            raise ValueError(f"Duplicate or overlapping {kind} indices: {name}")
        labels[ids] = label
    missing = int(np.count_nonzero(labels < 0))
    if missing:
        raise ValueError(f"{kind} annotations leave {missing} elements unassigned")
    return names, labels


def _surface_audit(model, label: str) -> dict:
    component_count = len(model.split(only_watertight=False, repair=False))
    degenerate_faces = int(np.count_nonzero(~model.nondegenerate_faces()))
    _, counts = np.unique(model.edges_sorted, axis=0, return_counts=True)
    boundary_edges = int(np.count_nonzero(counts == 1))
    nonmanifold_edges = int(np.count_nonzero(counts > 2))
    volume = float(model.volume)
    if component_count != 1:
        raise ValueError(f"{label} mesh has {component_count} connected components; expected one")
    if degenerate_faces:
        raise ValueError(f"{label} mesh has {degenerate_faces} degenerate or near-degenerate faces")
    if boundary_edges or nonmanifold_edges or not model.is_watertight:
        raise ValueError(
            f"{label} mesh has unsafe edge topology "
            f"(boundary={boundary_edges}, nonmanifold={nonmanifold_edges})"
        )
    if not model.is_winding_consistent or not np.isfinite(volume) or volume <= 0:
        raise ValueError(f"{label} mesh must have consistent outward winding and positive volume")
    from .pipeline import intersections
    self_intersections = len(intersections(np.asarray(model.vertices), np.asarray(model.faces)))
    if self_intersections:
        raise ValueError(f"{label} mesh has {self_intersections} self-intersecting triangle pairs")
    return dict(components=component_count, watertight=True, winding_consistent=True,
                volume=volume, boundary_edges=0, nonmanifold_edges=0,
                degenerate_faces=0, self_intersection_pairs=0)


def _adapter_paths(root: Path, expected: dict[str, Path]) -> tuple[Path, Path, dict]:
    path = root / "hair_config.yaml"
    if not path.is_file():
        raise ValueError("Missing hair_config.yaml")
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("hair_config.yaml must contain a mapping")
    for section in ("sdf_chamfer", "textured_strands"):
        block = data.get(section)
        if not isinstance(block, dict) or block.get("path_to_scale") != "":
            raise ValueError(
                f"hair_config.yaml {section}.path_to_scale must be empty because package coordinates are aligned"
            )

    def get(section: str, key: str) -> Path:
        block = data.get(section)
        if not isinstance(block, dict):
            raise ValueError(f"hair_config.yaml is missing section {section!r}")
        return _declared_path(root, block, key)

    sdf = get("sdf_penetration", "sdf_path")
    low = get("sdf_penetration", "bounds_min")
    high = get("sdf_penetration", "bounds_max")
    declared = {
        "sdf": sdf,
        "outer_mesh": get("sdf_chamfer", "path_to_mesh"),
        "mesh": get("textured_strands", "path_to_mesh"),
        "vertex_parts": get("textured_strands", "path_to_classes"),
        "tangent": get("textured_strands", "precomputed_tangent"),
    }
    for key, declared_path in declared.items():
        if declared_path != expected[key]:
            raise ValueError(f"hair_config.yaml {key} path disagrees with strand_package.json")
    eye = data.get("textured_strands", {}).get("eye_dists_geom", "")
    if eye:
        _declared_path(root, {"eye_dists_geom": eye}, "eye_dists_geom")
    return low, high, data


def _part_tables(metadata: dict, names: list[str]) -> set[str]:
    expected = set(names)
    for key in ("length_cm", "length_scene", "width_cm", "thickness_cm"):
        table = metadata.get(key)
        if not isinstance(table, dict) or set(table) != expected:
            raise ValueError(f"{key} keys differ from annotation keys")
        if any(not isinstance(value, (int, float)) or isinstance(value, bool)
               for value in table.values()):
            raise ValueError(f"{key} must contain finite nonnegative numbers")
        values = np.asarray(list(table.values()), dtype=float)
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{key} must contain finite nonnegative values")
    scale = metadata.get("cm_per_scene_unit")
    if (not isinstance(scale, (int, float)) or isinstance(scale, bool)
            or not np.isfinite(scale) or scale <= 0):
        raise ValueError("cm_per_scene_unit must be finite and positive")
    for name in names:
        expected_scene = metadata["length_cm"][name] / scale
        if not np.isclose(metadata["length_scene"][name], expected_scene, rtol=1e-8, atol=1e-12):
            raise ValueError(f"length_scene and length_cm disagree for {name}")
    raw_nonfur = metadata.get("nonfur_parts")
    if (not isinstance(raw_nonfur, list)
            or any(not isinstance(name, str) or not name for name in raw_nonfur)
            or len(raw_nonfur) != len(set(raw_nonfur))):
        raise ValueError("nonfur_parts must be a list of unique part names")
    nonfur = set(raw_nonfur)
    unknown = sorted(nonfur - expected)
    if unknown:
        raise ValueError(f"nonfur_parts contains unknown parts: {', '.join(unknown)}")
    from .geometry import ZERO_PARTS
    missing_defaults = sorted((ZERO_PARTS & expected) - nonfur)
    if missing_defaults:
        raise ValueError(f"nonfur_parts omits default non-fur parts: {', '.join(missing_defaults)}")
    for name in nonfur:
        for key in ("length_cm", "length_scene", "width_cm", "thickness_cm"):
            if abs(float(metadata[key][name])) > 1e-12:
                raise ValueError(f"Non-fur part {name!r} must have zero {key}")
    return nonfur


def _raw_face_field(path: Path, face_count: int) -> np.ndarray:
    if not path.is_file():
        raise ValueError("Missing rawFaceField.dmat")
    with path.open() as handle:
        try:
            columns = int(handle.readline())
            rows = int(handle.readline())
        except (TypeError, ValueError) as error:
            raise ValueError("rawFaceField.dmat has an invalid header") from error
        if columns != face_count or rows != 6:
            raise ValueError("rawFaceField.dmat must contain six values per face")
        expected = rows * columns
        try:
            values = np.fromfile(handle, dtype=np.float64, sep=" ", count=expected + 1)
        except (TypeError, ValueError) as error:
            raise ValueError("rawFaceField.dmat contains an invalid value") from error
    if values.size != expected:
        raise ValueError("rawFaceField.dmat dimensions do not match its contents")
    field = values.reshape(rows, columns).T
    if field.shape != (face_count, 6) or not np.isfinite(field).all():
        raise ValueError("rawFaceField.dmat must contain six finite values per face")
    return field


def validate(root: Path) -> dict:
    root = root.resolve()
    metadata_path = root / "strand_package.json"
    if not metadata_path.is_file():
        raise ValueError("Missing strand_package.json")
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(metadata, dict) or metadata.get("schema_version") != 1:
        raise ValueError("strand_package.json must use schema_version 1")
    if metadata.get("data_root") != ".":
        raise ValueError("strand_package.json data_root must be '.'")
    if metadata.get("variant") not in {"shape", "shape_vlm"}:
        raise ValueError("strand_package.json must name the shape or shape_vlm variant")

    paths = {
        "mesh": _declared_path(root, metadata, "mesh"),
        "outer_mesh": _declared_path(root, metadata, "outer_mesh"),
        "tangent": _declared_path(root, metadata, "tangent"),
        "vertex_parts": _declared_path(root, metadata, "vertex_parts"),
        "face_parts": _declared_path(root, metadata, "face_parts"),
        "projection": _declared_path(root, metadata, "projection"),
        "images": _declared_path(root, metadata, "images", directory=True),
        "masks": _declared_path(root, metadata, "masks", directory=True),
        "orientations": _declared_path(root, metadata, "orientations", directory=True),
        "sdf": _declared_path(root, metadata, "sdf"),
    }
    low_path, high_path, adapter = _adapter_paths(root, paths)

    model = mesh(paths["mesh"])
    outer = mesh(paths["outer_mesh"])
    root_audit = _surface_audit(model, "Root")
    outer_audit = _surface_audit(outer, "Outer")
    if (len(outer.vertices) != len(model.vertices)
            or not np.array_equal(np.asarray(outer.faces), np.asarray(model.faces))):
        raise ValueError("Outer and root meshes must use the same vertex and face topology")
    if metadata.get("topology_sha256") != digest(paths["mesh"]):
        raise ValueError("Mesh has changed since tangent/SDF generation")

    names, labels = _indexed_annotations(paths["vertex_parts"], len(model.vertices), "Vertex")
    face_names, saved_face_labels = _indexed_annotations(paths["face_parts"], len(model.faces), "Face")
    if face_names != names:
        raise ValueError("Vertex and face annotation part order differs")
    nonfur = _part_tables(metadata, names)
    scale_basis = metadata.get("scale_basis")
    if (not isinstance(scale_basis, str) or not scale_basis.strip()
            or scale_basis.strip().lower() == "unspecified"):
        raise ValueError("strand_package.json must record scale_basis")
    from .package import identity_scale_bytes
    scale_path = root / "scale.pickle"
    if not scale_path.is_file() or scale_path.is_symlink():
        raise ValueError("Missing regular scale.pickle compatibility transform")
    if scale_path.read_bytes() != identity_scale_bytes():
        raise ValueError("scale.pickle must contain the generated identity compatibility transform")
    textured = adapter.get("textured_strands", {})
    metadata_has_eye_scale = "eye_distance_cm" in metadata
    adapter_has_eye_path = bool(textured.get("eye_dists_geom"))
    adapter_has_eye_scale = "eye_dists_VQA" in textured
    if not (metadata_has_eye_scale == adapter_has_eye_path == adapter_has_eye_scale):
        raise ValueError(
            "Eye-distance scale must be present together in landmarks, metadata, and hair_config.yaml"
        )
    use_metrical_space = textured.get("use_metrical_space")
    normalize_decoder_output = textured.get("normalize_decoder_output")
    if (use_metrical_space is not metadata_has_eye_scale
            or normalize_decoder_output is not metadata_has_eye_scale):
        raise ValueError(
            "hair_config.yaml metric-mode flags disagree with the package eye-scale mode"
        )
    expected_lengths = metadata["length_cm"] if metadata_has_eye_scale else metadata["length_scene"]
    mapping_length = textured.get("mapping_length")
    if not isinstance(mapping_length, dict) or set(mapping_length) != set(names):
        raise ValueError("hair_config.yaml mapping_length keys differ from annotation keys")
    for name in names:
        value = mapping_length[name]
        if (not isinstance(value, (int, float)) or isinstance(value, bool)
                or not np.isfinite(value) or value < 0
                or not np.isclose(value, expected_lengths[name], rtol=1e-8, atol=1e-12)):
            raise ValueError(f"hair_config.yaml mapping_length disagrees for {name}")
    if metadata_has_eye_scale:
        eye_distance_cm = metadata["eye_distance_cm"]
        if (not isinstance(eye_distance_cm, (int, float)) or isinstance(eye_distance_cm, bool)
                or not np.isfinite(eye_distance_cm)
                or eye_distance_cm <= 0):
            raise ValueError("eye_distance_cm must be finite and positive")
        eye_path = _declared_path(root, textured, "eye_dists_geom")
        landmarks = np.asarray(trimesh.load(eye_path, process=False).vertices)
        if landmarks.shape != (2, 3) or not np.isfinite(landmarks).all():
            raise ValueError("eyes.ply must contain exactly two finite eye-center points")
        measured_cm = float(np.linalg.norm(landmarks[0] - landmarks[1]) * metadata["cm_per_scene_unit"])
        adapter_cm = textured.get("eye_dists_VQA")
        if (not isinstance(adapter_cm, (int, float)) or isinstance(adapter_cm, bool)
                or not np.isfinite(adapter_cm)
                or not np.isclose(adapter_cm, eye_distance_cm, rtol=2e-6, atol=1e-9)
                or not np.isclose(measured_cm, eye_distance_cm, rtol=2e-6, atol=1e-9)):
            raise ValueError("Eye-distance scale disagrees across landmarks, metadata, and hair_config.yaml")

    tangent = np.load(paths["tangent"], allow_pickle=False)
    if tangent.shape != (len(model.faces), 3) or not np.isfinite(tangent).all():
        raise ValueError("Tangent dimensions do not match the exact root mesh")
    dot = float(np.max(np.abs(np.sum(tangent * model.face_normals, 1))))
    if dot > 1e-4 or np.max(np.abs(np.linalg.norm(tangent, axis=1)-1)) > 1e-4:
        raise ValueError("Tangents are nonunit or not in their face planes")
    _, flips = orient(model.faces, tangent)
    if flips:
        raise ValueError(f"Tangent sign pass changes {flips} faces")
    _raw_face_field(root / "rawFaceField.dmat", len(model.faces))
    unoriented = np.load(root / "field_furless_reshaped.npy", allow_pickle=False)
    if unoriented.shape != (len(model.faces), 3) or not np.isfinite(unoriented).all():
        raise ValueError("field_furless_reshaped.npy must contain one finite xyz vector per face")

    fields_path = root / "defurring_fields.npz"
    if not fields_path.is_file():
        raise ValueError("Missing defurring_fields.npz")
    required_fields = ("displacement", "labels", "confidence", "transferred_width", "nearest_width", "upper")
    with np.load(fields_path, allow_pickle=False) as fields:
        missing_fields = [key for key in required_fields if key not in fields]
        if missing_fields:
            raise ValueError(f"defurring_fields.npz is missing: {', '.join(missing_fields)}")
        arrays = {key: np.asarray(fields[key]) for key in required_fields}
    for key, values in arrays.items():
        if values.shape != (len(model.vertices),) or not np.isfinite(values).all():
            raise ValueError(f"Defurring field {key} must have one finite value per vertex")
    if arrays["labels"].dtype.kind not in "iu" or not np.array_equal(arrays["labels"], labels):
        raise ValueError("Defurring labels differ from vertex annotations")
    for key in ("displacement", "confidence", "transferred_width", "nearest_width", "upper"):
        if (arrays[key] < 0).any():
            raise ValueError(f"Defurring field {key} contains negative values")
    tolerance = max(1e-12, float(arrays["upper"].max(initial=0)) * 1e-8)
    if (arrays["displacement"] > arrays["upper"] + tolerance).any():
        raise ValueError("Defurring displacement exceeds its recorded upper bound")
    if (arrays["confidence"] > 1 + 1e-8).any():
        raise ValueError("Defurring confidence must be between zero and one")

    part_names = np.asarray(names)
    pinned = np.isin(part_names[labels], sorted(nonfur))
    if pinned.any() and np.max(np.abs(arrays["displacement"][pinned])) > tolerance:
        raise ValueError("Non-fur vertices must have zero displacement")
    expected_face_labels = derive_face_labels(np.asarray(model.faces), labels, pinned)
    if not np.array_equal(saved_face_labels, expected_face_labels):
        raise ValueError("Face annotations disagree with vertex annotations")

    from .geometry import normals_and_edges
    outer_vertices = np.asarray(outer.vertices)
    root_vertices = np.asarray(model.vertices)
    normals, _ = normals_and_edges(outer_vertices, np.asarray(outer.faces))
    reconstructed = outer_vertices - arrays["displacement"][:, None] * normals
    diagonal = float(np.linalg.norm(np.ptp(outer_vertices, axis=0)))
    relation_error = float(np.max(np.linalg.norm(root_vertices - reconstructed, axis=1)))
    relation_tolerance = max(1e-10, 1e-8 * diagonal)
    if relation_error > relation_tolerance:
        raise ValueError("Root vertices disagree with the recorded normal displacement")

    from .package import ray_scene
    signed = ray_scene(outer_vertices, np.asarray(outer.faces)).compute_signed_distance(
        o3d.core.Tensor(root_vertices.astype(np.float32)), nsamples=3
    ).numpy()
    if not np.isfinite(signed).all():
        raise ValueError("Outer containment check returned non-finite distances")
    inside_outer_fraction = float(np.mean(signed <= diagonal * 1e-5))
    if inside_outer_fraction < 1.0:
        raise ValueError("Root vertices remain outside the outer-mesh tolerance")

    images = sorted(paths["images"].glob("*.png"))
    projections = np.load(paths["projection"], allow_pickle=False)
    if (not images or projections.shape not in ((len(images), 3, 4), (len(images), 4, 4))
            or not np.isfinite(projections).all()):
        raise ValueError("projection.npy must have one finite 3x4/4x4 projection per image")
    if [image.stem for image in images] != [f"{i:04d}" for i in range(len(images))]:
        raise ValueError("Package images must use contiguous 0000.png names")

    legend_path = root / "part_masks/legend.json"
    if not legend_path.is_file():
        raise ValueError("Missing part_masks/legend.json")
    legend = json.loads(legend_path.read_text())
    expected_legend = {"0": "background"} | {str(i + 1): name for i, name in enumerate(names)}
    if legend != expected_legend:
        raise ValueError("Part-mask legend differs from annotation part order")
    allowed_part_ids = np.arange(len(names) + 1)
    nonfur_part_ids = np.asarray([index + 1 for index, name in enumerate(names) if name in nonfur])
    orientation_variance_min = float("inf")
    orientation_variance_max = float("-inf")

    for image, projection in zip(images, projections):
        decompose(projection)
        with Image.open(image) as opened:
            size = opened.size
        for folder in (paths["masks"] / "body", paths["masks"] / "hair",
                       paths["masks"] / "bald", paths["masks"] / "foreground",
                       root / "part_masks", paths["orientations"] / "angles"):
            path = folder / image.name
            if not path.is_file():
                raise ValueError(f"Missing view asset: {path}")
            with Image.open(path) as opened:
                if opened.size != size:
                    raise ValueError(f"Wrong-sized view asset: {path}")
        part_path = root / "part_masks" / image.name
        with Image.open(part_path) as opened:
            part_mask = np.asarray(opened)
        if part_mask.ndim != 2 or not np.isin(np.unique(part_mask), allowed_part_ids).all():
            raise ValueError(f"Part mask uses an ID absent from the legend: {part_path}")
        masks = {}
        for name in ("body", "hair", "bald", "foreground"):
            path = paths["masks"] / name / image.name
            with Image.open(path) as opened:
                values = np.asarray(opened)
            if values.ndim != 2 or not np.isin(np.unique(values), [0, 255]).all():
                raise ValueError(f"Mask must be a binary grayscale PNG: {path}")
            masks[name] = values > 127
        if (masks["body"] & ~masks["foreground"]).any():
            raise ValueError(f"Body mask extends outside the foreground: {image.name}")
        if (masks["bald"] & ~masks["body"]).any():
            raise ValueError(f"Non-fur mask extends outside the visible root: {image.name}")
        if not np.array_equal(masks["hair"], masks["foreground"] & ~masks["bald"]):
            raise ValueError(f"Hair mask disagrees with foreground/non-fur masks: {image.name}")
        if not np.array_equal(part_mask > 0, masks["body"]):
            raise ValueError(f"Part mask disagrees with the visible-root mask: {image.name}")
        expected_bald = masks["body"] & np.isin(part_mask, nonfur_part_ids)
        if not np.array_equal(masks["bald"], expected_bald):
            raise ValueError(f"Non-fur mask disagrees with part labels: {image.name}")
        variance = np.load(paths["orientations"] / "vars" / (image.stem + ".npy"), allow_pickle=False)
        if (variance.shape != size[::-1] or not np.isfinite(variance).all()
                or (variance < 0).any() or (variance > np.pi**2 / 4 + 1e-3).any()):
            raise ValueError("Orientation variance is incomplete or invalid")
        orientation_variance_min = min(orientation_variance_min, float(variance.min()))
        orientation_variance_max = max(orientation_variance_max, float(variance.max()))
        with Image.open(paths["orientations"] / "angles" / image.name) as opened:
            angle = np.asarray(opened)
        if angle.ndim != 2 or angle.max(initial=0) >= 180:
            raise ValueError("Orientation PNG must encode degrees modulo 180")

    sdf = np.load(paths["sdf"], allow_pickle=False)
    low = np.load(low_path, allow_pickle=False)
    high = np.load(high_path, allow_pickle=False)
    if (sdf.ndim != 3 or min(sdf.shape) < 2 or not np.isfinite(sdf).all()
            or not sdf.min() < 0 < sdf.max()):
        raise ValueError("Invalid inside-negative SDF")
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite(low).all() or not np.isfinite(high).all():
        raise ValueError("SDF bounds must be finite xyz vectors")
    if not (low < high).all():
        raise ValueError("SDF minimum bounds must be smaller than maximum bounds")
    span = high - low
    bound_tolerance = float(span.max()) * 1e-6
    vertices = np.asarray(model.vertices)
    if (vertices.min(0) < low - bound_tolerance).any() or (vertices.max(0) > high + bound_tolerance).any():
        raise ValueError("SDF bounds do not contain the root mesh")

    sample_ids = np.unique(np.linspace(0, sdf.size - 1, min(4096, sdf.size), dtype=np.int64))
    grid_indices = np.column_stack(np.unravel_index(sample_ids, sdf.shape))
    denominator = np.asarray(sdf.shape, dtype=np.float64) - 1
    sample_points = low + grid_indices / denominator * (high - low)
    sampled_distance = ray_scene(vertices, np.asarray(model.faces)).compute_signed_distance(
        o3d.core.Tensor(sample_points.astype(np.float32)), nsamples=3
    ).numpy()
    stored_distance = sdf.reshape(-1)[sample_ids]
    sdf_error = float(np.max(np.abs(sampled_distance - stored_distance)))
    sdf_tolerance = max(1e-6, diagonal * 5e-5)
    if not np.isfinite(sampled_distance).all() or sdf_error > sdf_tolerance:
        raise ValueError("SDF samples disagree with the root mesh")
    away_from_surface = np.abs(sampled_distance) > sdf_tolerance
    if not np.array_equal(sampled_distance[away_from_surface] < 0,
                          stored_distance[away_from_surface] < 0):
        raise ValueError("SDF sign disagrees with the root mesh")

    return dict(valid=True, vertices=len(model.vertices), faces=len(model.faces), views=len(images),
                tangent_residual_sign_flips=flips, tangent_max_normal_dot=dot,
                watertight=True, components=root_audit["components"], self_intersection_pairs=0,
                outer_watertight=True, outer_components=outer_audit["components"],
                outer_self_intersection_pairs=0,
                root_displacement_max_error=relation_error,
                inside_outer_fraction=inside_outer_fraction,
                sdf_sample_max_error=sdf_error,
                orientation_variance_min=orientation_variance_min,
                orientation_variance_max=orientation_variance_max,
                nonfur_parts=sorted(nonfur),
                downstream_training_tested=False)
