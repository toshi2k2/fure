"""Camera utilities for OpenCV intrinsics and world-to-camera matrices."""
from __future__ import annotations
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def decompose(projection: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.asarray(projection, dtype=np.float64)[:3, :4]
    k, r, c = cv2.decomposeProjectionMatrix(p)[:3]
    k /= k[2, 2]
    c = (c[:3]/c[3]).ravel()
    w2c = np.eye(4)
    w2c[:3, :3], w2c[:3, 3] = r, -r @ c
    if not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0 or np.linalg.det(r) < .999:
        raise ValueError("Invalid pinhole projection; undistort/calibrate first")
    rebuilt = k @ w2c[:3]
    if not np.allclose(p/np.linalg.norm(p), rebuilt/np.linalg.norm(rebuilt), atol=1e-5):
        raise ValueError("Projection decomposition did not reproduce the input matrix")
    return k, w2c


def validate_views(root: Path) -> tuple[list[Path], np.ndarray]:
    paths = sorted((root / "images_2").glob("*.png"))
    p = np.load(root / "projection.npy", allow_pickle=False)
    if not paths or p.shape not in ((len(paths), 4, 4), (len(paths), 3, 4)) or not np.isfinite(p).all():
        raise ValueError("projection.npy must have one finite 3x4/4x4 projection per sorted PNG")
    if [x.stem for x in paths] != [f"{i:04d}" for i in range(len(paths))]:
        raise ValueError(
            "Image filenames must be contiguous (0000.png, 0001.png, ...); "
            "run prepare to rename images and reorder cameras together"
        )
    for image, projection in zip(paths, p):
        decompose(projection)
        mask = root / "foreground" / image.name
        if not mask.is_file() or Image.open(mask).size != Image.open(image).size:
            raise ValueError(f"Missing or wrong-size foreground mask: {mask}")
        if not np.asarray(Image.open(mask).convert("L")).any():
            raise ValueError(f"Empty foreground mask: {mask}")
    return paths, p
