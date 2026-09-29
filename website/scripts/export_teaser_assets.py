"""Export website previews from the approved, unmodified all-strand Blender scene.

Studio renders keep every strand. Interactive previews sample 30000 existing
curves without interpolation and simplify only the browser body mesh.
Run with Blender --background <scene.blend> --python this.py -- --output ... .
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

import bpy
from mathutils import Vector
import numpy as np


def material_color(material):
    if material and material.use_nodes:
        for node in material.node_tree.nodes:
            for key in ("Base Color", "Color"):
                if node.type in ("BSDF_PRINCIPLED", "BSDF_HAIR_PRINCIPLED", "BSDF_DIFFUSE") and key in node.inputs:
                    return list(node.inputs[key].default_value)[:3]
    return list(material.diffuse_color)[:3] if material else [.3, .2, .1]


def transformed(points, matrix):
    a = np.array(matrix, dtype=np.float64)
    return points @ a[:3, :3].T + a[:3, 3]


def export_model(name: str, folder: Path, limit: int) -> dict:
    body = bpy.data.objects[name + "_body"]
    fur = bpy.data.objects[name + "_strands"]
    data = fur.data
    positions = np.empty(len(data.points) * 3, dtype=np.float32)
    data.attributes["position"].data.foreach_get("vector", positions)
    positions = positions.reshape(-1, 3)
    offsets = np.empty(len(data.curve_offset_data), dtype=np.int32)
    data.curve_offset_data.foreach_get("value", offsets)
    counts = np.diff(offsets)
    assert counts.min() == counts.max(), "Expected uniform teaser control points"
    curves = positions.reshape(len(counts), int(counts[0]), 3)
    ids = np.sort(np.random.default_rng(73).choice(len(curves), min(limit, len(curves)), replace=False))
    sampled = transformed(curves[ids].reshape(-1, 3), fur.matrix_world)
    duplicate = body.copy()
    duplicate.data = body.data.copy()
    bpy.context.collection.objects.link(duplicate)
    if len(duplicate.data.polygons) > 60000:
        decimate = duplicate.modifiers.new("Website preview only", "DECIMATE")
        decimate.ratio = 60000 / len(duplicate.data.polygons)
    evaluated = duplicate.evaluated_get(bpy.context.evaluated_depsgraph_get())
    mesh = evaluated.to_mesh()
    mesh.calc_loop_triangles()
    verts = np.empty(len(mesh.vertices) * 3, dtype=np.float32)
    mesh.vertices.foreach_get("co", verts)
    verts = transformed(verts.reshape(-1, 3), body.matrix_world)
    faces = np.empty(len(mesh.loop_triangles) * 3, dtype=np.int32)
    mesh.loop_triangles.foreach_get("vertices", faces)
    center = (verts.min(0) + verts.max(0)) / 2
    scale = 2.8 / np.ptp(verts, axis=0).max()
    axis = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    verts = ((verts - center) * scale @ axis).astype("<f4")
    sampled = ((sampled - center) * scale @ axis).astype("<f4")
    primitives = {"bodyPositions": verts.reshape(-1), "bodyIndices": faces.astype("<u4"),
                  "strandPositions": sampled.reshape(-1)}
    offset, binary, layout = 0, [], {}
    for key, values in primitives.items():
        payload = values.tobytes()
        layout[key] = {"offset": offset, "length": len(values), "type": values.dtype.str}
        offset += len(payload)
        binary.append(payload)
    target = folder / f"{name}.bin"
    with target.open("wb") as handle:
        handle.write(b"".join(binary))
    record = dict(name=name, url=f"assets/models/{name}.bin", bytes=offset, layout=layout,
                  pointsPerStrand=int(counts[0]), previewStrands=len(ids), sourceStrands=len(curves),
                  sourceBodyFaces=len(body.data.polygons), previewBodyFaces=len(faces)//3,
                  bodyColor=material_color(body.active_material), hairColor=material_color(fur.active_material),
                  bounds=[np.minimum(verts.min(0), sampled.min(0)).tolist(), np.maximum(verts.max(0), sampled.max(0)).tolist()],
                  sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
                  note="Existing curves sampled without interpolation; body/strands share the same coordinate transform.")
    (folder / f"{name}.json").write_text(json.dumps(record, indent=2))
    evaluated.to_mesh_clear()
    bpy.data.objects.remove(duplicate, do_unlink=True)
    return record


def render_animal(name: str, output: Path, samples: int, direction: Vector):
    parent = bpy.data.objects[name + "_turntable"]
    body = bpy.data.objects[name + "_body"]
    for obj in bpy.data.objects:
        if obj.type in ("MESH", "CURVES"):
            obj.hide_render = obj.parent != parent and obj.name != "infinite_studio_floor"
    coordinates = np.array([tuple(body.matrix_world @ Vector(corner)) for corner in body.bound_box])
    low, high = coordinates.min(0), coordinates.max(0)
    target = Vector((low + high) / 2)
    camera = bpy.context.scene.camera
    camera.location = target + direction * 14
    camera.rotation_euler = (target - camera.location).to_track_quat("-Z", "Y").to_euler()
    camera.data.type = "ORTHO"
    bpy.context.view_layer.update()
    # Horizontal and vertical extents in the actual camera plane, with fur margin.
    view = camera.matrix_world.inverted()
    projected = np.array([tuple(view @ Vector(v)) for v in coordinates])
    camera.data.ortho_scale = max(np.ptp(projected[:, 0])*1.20, np.ptp(projected[:, 1])*1.6*1.2)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = samples
    scene.cycles.use_denoising = True
    scene.render.resolution_x, scene.render.resolution_y = 1600, 1000
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.image_settings.color_depth = "8"
    scene.render.filepath = str(output)
    bpy.ops.render.render(write_still=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--animals", nargs="+", default=["cat", "fox", "tiger", "beagle", "panda", "bison"])
    parser.add_argument("--mode", choices=("models", "renders"), default="models")
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args(sys.argv[sys.argv.index("--")+1:])
    args.output.mkdir(parents=True, exist_ok=True)
    direction = (bpy.context.scene.camera.location - Vector((0, 0, .8))).normalized()
    for name in args.animals:
        if args.mode == "models":
            export_model(name, args.output, 30000)
        else:
            render_animal(name, args.output / f"{name}.png", args.samples, direction)


if __name__ == "__main__":
    main()
