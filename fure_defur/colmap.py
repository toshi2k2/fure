"""Read the camera and image records needed from a COLMAP sparse model."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import struct

import numpy as np


@dataclass(frozen=True)
class Camera:
    camera_id: int
    model: str
    width: int
    height: int
    params: np.ndarray


@dataclass(frozen=True)
class View:
    image_id: int
    qvec: np.ndarray
    tvec: np.ndarray
    camera_id: int
    name: str


# COLMAP stores the model number, rather than its name, in cameras.bin.
_CAMERA_MODELS = {
    0: ("SIMPLE_PINHOLE", 3),
    1: ("PINHOLE", 4),
    2: ("SIMPLE_RADIAL", 4),
    3: ("RADIAL", 5),
    4: ("OPENCV", 8),
    5: ("OPENCV_FISHEYE", 8),
    6: ("FULL_OPENCV", 12),
    7: ("FOV", 5),
    8: ("SIMPLE_RADIAL_FISHEYE", 4),
    9: ("RADIAL_FISHEYE", 5),
    10: ("THIN_PRISM_FISHEYE", 12),
    11: ("RAD_TAN_THIN_PRISM_FISHEYE", 16),
}


def _unpack(handle, fmt: str):
    size = struct.calcsize("<" + fmt)
    value = handle.read(size)
    if len(value) != size:
        raise ValueError("Truncated COLMAP model")
    return struct.unpack("<" + fmt, value)


def qvec_to_rotation(qvec: np.ndarray) -> np.ndarray:
    """Convert a COLMAP (w, x, y, z) quaternion to a rotation matrix."""
    w, x, y, z = np.asarray(qvec, dtype=np.float64)
    norm = np.linalg.norm([w, x, y, z])
    if norm == 0:
        raise ValueError("COLMAP image has a zero quaternion")
    w, x, y, z = np.array([w, x, y, z]) / norm
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z, 2*x*z + 2*w*y],
        [2*x*y + 2*w*z, 1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x*x - 2*y*y],
    ])


def read_cameras_text(path: Path) -> dict[int, Camera]:
    cameras = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split()
        camera_id, model = int(fields[0]), fields[1]
        cameras[camera_id] = Camera(
            camera_id,
            model,
            int(fields[2]),
            int(fields[3]),
            np.asarray(fields[4:], dtype=np.float64),
        )
    return cameras


def read_views_text(path: Path) -> dict[int, View]:
    views = {}
    with path.open() as handle:
        while True:
            line = handle.readline()
            if not line:
                break
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            fields = line.split()
            if len(fields) < 10:
                raise ValueError(f"Invalid COLMAP image record: {line}")
            image_id = int(fields[0])
            views[image_id] = View(
                image_id,
                np.asarray(fields[1:5], dtype=np.float64),
                np.asarray(fields[5:8], dtype=np.float64),
                int(fields[8]),
                " ".join(fields[9:]),
            )
            # Every image record is followed by its POINTS2D line, which may
            # be empty when a model only carries calibrated cameras.
            if not handle.readline() and not handle.closed:
                raise ValueError("COLMAP image record is missing its POINTS2D line")
    return views


def read_cameras_binary(path: Path) -> dict[int, Camera]:
    cameras = {}
    with path.open("rb") as handle:
        count, = _unpack(handle, "Q")
        for _ in range(count):
            camera_id, model_id, width, height = _unpack(handle, "iiQQ")
            if model_id not in _CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model number: {model_id}")
            model, num_params = _CAMERA_MODELS[model_id]
            params = np.asarray(_unpack(handle, "d" * num_params))
            cameras[camera_id] = Camera(camera_id, model, width, height, params)
    return cameras


def read_views_binary(path: Path) -> dict[int, View]:
    views = {}
    file_size = path.stat().st_size
    with path.open("rb") as handle:
        count, = _unpack(handle, "Q")
        for _ in range(count):
            values = _unpack(handle, "idddddddi")
            name_bytes = bytearray()
            while True:
                character = handle.read(1)
                if not character:
                    raise ValueError("Truncated COLMAP image name")
                if character == b"\0":
                    break
                name_bytes.extend(character)
            image_id = values[0]
            views[image_id] = View(
                image_id,
                np.asarray(values[1:5]),
                np.asarray(values[5:8]),
                values[8],
                name_bytes.decode("utf-8"),
            )
            point_count, = _unpack(handle, "Q")
            point_bytes = 24 * point_count
            handle.seek(point_bytes, 1)
            if handle.tell() > file_size:
                raise ValueError("Truncated COLMAP POINTS2D data")
    return views
