"""Directional degree-2 field with the downstream sign convention.

The compatibility pass compares adjacent ambient vectors. It is not
Levi-Civita transport or a physical hair-growth direction.
"""
from __future__ import annotations
from collections import deque
from pathlib import Path
import os
import subprocess
import numpy as np
import trimesh
from .io import digest, write_json


def orient(faces: np.ndarray, tangent: np.ndarray) -> tuple[np.ndarray, int]:
    """Match trimesh adjacency order, including all disconnected components."""
    pairs = trimesh.graph.face_adjacency(faces=faces)
    neighbors = [[] for _ in faces]
    for a, b in pairs:
        neighbors[a].append(b)
        neighbors[b].append(a)
    output = np.asarray(tangent, dtype=np.float64).copy()
    visited = np.zeros(len(faces), bool)
    flips = 0
    for seed in range(len(faces)):
        if visited[seed]:
            continue
        visited[seed] = True
        queue = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor in neighbors[current]:
                if visited[neighbor]:
                    continue
                if output[current] @ output[neighbor] < 0:
                    output[neighbor] *= -1
                    flips += 1
                visited[neighbor] = True
                queue.append(neighbor)
    return output.astype(np.float32), flips


def build_helper(repo: Path) -> Path:
    source = repo / "native/directional_raw_face_field.cpp"
    root = repo / "vendor/Directional"
    if not (root / "include/directional/power_field.h").is_file():
        raise FileNotFoundError("Directional missing. Run python scripts/bootstrap.py --directional")
    binary = repo / "build/directional_raw_face_field"
    stamp = binary.with_suffix(".json")
    identity = {"source": digest(source), "compiler": os.environ.get("CXX", "g++")}
    import json
    if binary.is_file() and stamp.is_file() and json.loads(stamp.read_text()) == identity:
        return binary
    binary.parent.mkdir(exist_ok=True)
    subprocess.run([identity["compiler"], "-std=c++20", "-O1", "-g0", "-DNDEBUG",
                    f"-I{root / 'include'}", f"-I{root / 'external/eigen'}", str(source), "-o", str(binary)], check=True)
    write_json(stamp, identity)
    return binary


def generate(mesh_path: Path, output: Path, repo: Path) -> dict:
    model = trimesh.load(mesh_path, process=False, maintain_order=True)
    raw = output / "rawFaceField.dmat"
    subprocess.run([str(build_helper(repo)), str(mesh_path), str(raw), "2"], check=True)
    with raw.open() as handle:
        cols, rows = int(handle.readline()), int(handle.readline())
        field = np.fromstring(handle.read(), sep=" ").reshape(rows, cols).T
    if field.shape != (len(model.faces), 6) or not np.isfinite(field).all():
        raise ValueError("Directional produced an invalid field; repair input topology, not a substitute tangent")
    tangent = field[:, 3:6]
    normals = model.face_normals
    # Remove only floating-point normal components, not field failures.
    normal_error = np.abs(np.sum(tangent * normals, axis=1))
    norm = np.linalg.norm(tangent, axis=1)
    if np.any(norm < 1e-8) or np.max(normal_error / norm) > 1e-3:
        raise ValueError("Directional field is not a valid tangent field")
    tangent -= np.sum(tangent * normals, axis=1)[:, None] * normals
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    oriented, flips = orient(model.faces, tangent)
    second, remaining = orient(model.faces, oriented)
    if remaining or not np.array_equal(oriented, second):
        raise ValueError("Tangent sign pass is not idempotent")
    np.save(output / "field_furless_reshaped.npy", tangent.astype(np.float32))
    np.save(output / "tan_furless.npy", oriented)
    metrics = dict(method="Directional power_field degree=2, consumer-compatible BFS sign pass",
                   faces=len(model.faces), shape=list(oriented.shape), initial_sign_flips=flips,
                   residual_sign_flips=remaining, mesh_sha256=digest(mesh_path),
                   abs_normal_dot_max=float(np.max(np.abs(np.sum(oriented * normals, axis=1)))),
                   norm_max_error=float(np.max(np.abs(np.linalg.norm(oriented, axis=1)-1))))
    write_json(output / "tangent_validation.json", metrics)
    return metrics
