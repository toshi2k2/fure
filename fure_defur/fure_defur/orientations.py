"""Compute axial image orientations with the package's Gabor filter bank."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
import numpy as np
import cv2
from PIL import Image
from skimage.filters import difference_of_gaussians, gabor_kernel
from .io import write_json


def estimate(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return axial angles and response-weighted squared angular deviation."""
    gray = np.asarray(rgb, dtype=np.float64) @ np.array([.2989, .5870, .1140])
    filtered = difference_of_gaussians(gray, .4, 10)
    responses = []
    for degree in range(180):
        kernel = np.real(gabor_kernel(.23, theta=np.pi-np.deg2rad(degree), sigma_x=1.8, sigma_y=2.4))
        # filter2D is correlation, like the original torch Conv2d; zero padding.
        responses.append(np.abs(cv2.filter2D(filtered.astype(np.float32), -1,
                                             kernel.astype(np.float32), borderType=cv2.BORDER_CONSTANT)))
    responses = np.stack(responses)
    best = responses.argmax(0).astype(np.uint8)
    total = np.maximum(responses.sum(0), 1e-12)
    variance = np.zeros(best.shape, np.float32)
    for degree, response in enumerate(responses):
        delta = np.abs(np.deg2rad(best.astype(np.float32)-degree))
        delta = np.minimum(delta, np.pi-delta)
        variance += delta**2 * response / total
    return best, variance


def _generate_view(image: Path, masks: Path, output: Path) -> None:
    rgb = np.array(Image.open(image).convert("RGB"))
    foreground = np.array(Image.open(masks / image.name).convert("L")) > 127
    if foreground.shape != rgb.shape[:2]:
        raise ValueError(f"Orientation mask size differs from its image: {image.name}")
    # Bound memory by tiling with a halo larger than DoG+Gabor support.
    angle = np.zeros(foreground.shape, np.uint8)
    variance = np.full(foreground.shape, np.pi**2/12, np.float32)
    for y in range(0, len(rgb), 256):
        for x in range(0, rgb.shape[1], 256):
            y0, x0 = max(0, y-64), max(0, x-64)
            block = rgb[y0:y+320, x0:x+320]
            a, v = estimate(block)
            h, w = angle[y:y+256, x:x+256].shape
            angle[y:y+h, x:x+w] = a[y-y0:y-y0+h, x-x0:x-x0+w]
            variance[y:y+h, x:x+w] = v[y-y0:y-y0+h, x-x0:x-x0+w]
    angle[~foreground] = 0
    variance[~foreground] = np.pi**2/12
    Image.fromarray(angle).save(output / "angles" / image.name)
    np.save(output / "vars" / (image.stem + ".npy"), variance.astype(np.float16))
    hsv = np.stack([angle, foreground.astype(np.uint8)*255, foreground.astype(np.uint8)*255], -1)
    Image.fromarray(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)).save(output / "vis_imgs" / image.name)


def generate(images: Path, masks: Path, output: Path, workers: int = 1) -> None:
    """Generate deterministic axial orientation maps for a numbered view set."""
    if output.exists():
        raise FileExistsError(output)
    if not isinstance(workers, int) or isinstance(workers, bool) or workers < 1:
        raise ValueError("workers must be a positive integer")
    image_paths = sorted(images.glob("*.png"))
    if not image_paths:
        raise ValueError(f"No PNG images found in {images}")
    expected_masks = {path.name for path in image_paths}
    actual_masks = {path.name for path in masks.glob("*.png")}
    if actual_masks != expected_masks:
        raise ValueError("Orientation images and masks must use the same PNG filenames")
    for directory in ("angles", "vars", "vis_imgs"):
        (output / directory).mkdir(parents=True)
    previous_cv_threads = cv2.getNumThreads()
    cv2.setNumThreads(1)
    try:
        task = partial(_generate_view, masks=masks, output=output)
        if workers == 1:
            for image in image_paths:
                task(image)
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(task, image_paths))
    finally:
        cv2.setNumThreads(previous_cv_threads)
    write_json(output / "metadata.json", dict(method="Gabor default bank; CPU tiled correlation",
               degrees=180, frequency=.23, sigma_x=1.8, sigma_y=2.4, dog_low=.4, dog_high=10,
               angle_units="degrees modulo 180; axial, not directed flow",
               variance_units="radians squared", views=len(image_paths)))
