"""Explicit file contracts and mesh-order-preserving I/O."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import trimesh


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b""):
            value.update(block)
    return value.hexdigest()


def mesh(path: Path) -> trimesh.Trimesh:
    value = trimesh.load(path, process=False, maintain_order=True)
    if not isinstance(value, trimesh.Trimesh) or not len(value.faces):
        raise ValueError(f"Expected one triangle mesh: {path}")
    if not np.isfinite(value.vertices).all():
        raise ValueError(f"Nonfinite mesh: {path}")
    return value


def export_obj(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write a topology-ordered OBJ without precision-induced intersections."""
    vertices, faces = np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3 or not len(vertices) or not np.isfinite(vertices).all():
        raise ValueError("OBJ vertices must be a nonempty finite [V,3] array")
    if (faces.ndim != 2 or faces.shape[1] != 3 or not len(faces)
            or faces.min() < 0 or faces.max() >= len(vertices)):
        raise ValueError("OBJ faces must be a nonempty in-range [F,3] array")
    with path.open("w") as handle:
        handle.write("# FurE topology-ordered float64 OBJ\n")
        # Seventeen significant decimal digits round-trip every IEEE float64;
        # trimesh's OBJ writer uses fixed decimals, which loses significance
        # for small coordinates even when `digits=17`.
        np.savetxt(handle, vertices, fmt="v %.17g %.17g %.17g")
        np.savetxt(handle, faces + 1, fmt="f %d %d %d")


def annotations(path: Path, count: int) -> tuple[list[str], np.ndarray]:
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not data:
        raise ValueError("Annotations must map part names to vertex-index lists")
    names = list(data)
    labels = np.full(count, -1, dtype=np.int32)
    for p, name in enumerate(names):
        ids = np.asarray(data[name])
        if ids.size == 0:
            continue
        if ids.ndim != 1 or ids.dtype.kind not in "iu" or ids.min() < 0 or ids.max() >= count:
            raise ValueError(f"Invalid vertex indices for {name}; use the annotated mesh's unchanged ordering")
        if len(np.unique(ids)) != len(ids) or (labels[ids] != -1).any():
            raise ValueError(f"Duplicate/overlapping annotation indices: {name}")
        labels[ids] = p
    if (labels < 0).any():
        if "unlabeled" not in names:
            names.append("unlabeled")
        labels[labels < 0] = names.index("unlabeled")
    return names, labels


def face_labels(faces: np.ndarray, labels: np.ndarray, nonfur: np.ndarray) -> np.ndarray:
    """Majority label; a non-fur corner wins ties/overrides to protect small regions."""
    triangle = labels[faces]
    result = triangle[:, 0].copy()
    result[triangle[:, 1] == triangle[:, 2]] = triangle[triangle[:, 1] == triangle[:, 2], 1]
    for k in range(3):
        mask = nonfur[faces[:, k]]
        result[mask] = triangle[mask, k]
    return result


def local_path(config_path: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (config_path.parent / path).resolve()


def read_config(path: Path) -> dict:
    import yaml
    data = yaml.safe_load(path.read_text())
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Expected schema_version: 1")
    return data
