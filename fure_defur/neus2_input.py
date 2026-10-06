"""Isolate NeuS2's input-side logging and dynamic-scene JSON discovery."""
from __future__ import annotations
import json
from pathlib import Path
import shutil


def stage_scene(scene: Path, destination: Path) -> Path:
    """Stage exactly one time frame; upstream writes output/ next to its data."""
    if destination.exists():
        raise FileExistsError(destination)
    data = json.loads(scene.read_text())
    if not data.get("frames"):
        raise ValueError("NeuS2 scene has no frames")
    (destination / "images").mkdir(parents=True)
    for index, frame in enumerate(data["frames"]):
        path = (scene.parent / frame["file_path"]).resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        name = f"{index:06d}{path.suffix}"
        shutil.copy2(path, destination / "images" / name)
        frame["file_path"] = f"images/{name}"
    (destination / "transform.json").write_text(json.dumps(data, indent=2))
    return destination
