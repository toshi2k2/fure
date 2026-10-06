"""CPU raycast diagnostics and preview rendering."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import open3d as o3d
import trimesh
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from .io import write_json


def display_frame(v: np.ndarray, labels: np.ndarray, names: list[str]) -> np.ndarray:
    """Derive a display-only frame from body/feet and face/tail labels when available."""
    def center(parts: set[str]) -> np.ndarray | None:
        selected = np.isin(labels, [i for i, name in enumerate(names) if name in parts])
        return v[selected].mean(0) if selected.any() else None
    body, feet = center({"body", "belly", "neck"}), center({"paws", "front_paws", "paw_pads"})
    up = body-feet if body is not None and feet is not None else np.array([0., 1., 0.])
    up /= max(np.linalg.norm(up), 1e-12)
    face, tail = center({"face", "nosetip"}), center({"tail", "under_tail"})
    front = face-tail if face is not None and tail is not None else np.array([1., 0., 0.])
    front -= up*np.dot(front, up)
    if np.linalg.norm(front) < 1e-8:
        front = np.cross(up, [0., 0., 1.])
    front /= np.linalg.norm(front)
    side = np.cross(front, up)
    return np.stack([front, side, up], axis=1)


def cast(v: np.ndarray, f: np.ndarray, rgb: np.ndarray, direction: np.ndarray, span: float, size: int = 600) -> tuple:
    from .package import ray_scene
    direction = direction/np.linalg.norm(direction)
    right = np.cross([0., 0., 1.], direction)
    right /= np.linalg.norm(right)
    up = np.cross(direction, right)
    x, y = np.meshgrid(np.linspace(-span/2, span/2, size), np.linspace(span/2, -span/2, size))
    origins = x[..., None]*right + y[..., None]*up + direction*span*2
    rays = np.concatenate([origins, np.broadcast_to(-direction, origins.shape)], axis=-1).astype(np.float32)
    hit = ray_scene(v, f).cast_rays(o3d.core.Tensor(rays))
    ids = hit["primitive_ids"].numpy()
    valid = ids != np.iinfo(np.uint32).max
    uv = hit["primitive_uvs"].numpy()[valid]
    weights = np.column_stack([1-uv.sum(1), uv])
    colors = (rgb[f[ids[valid]]] * weights[..., None]).sum(1)
    normal = hit["primitive_normals"].numpy()[valid]
    light = direction + np.array([-.2, -.2, .8])
    light /= np.linalg.norm(light)
    shade = .58+.42*np.maximum(normal @ light, 0)
    image = np.ones((size, size, 3))
    image[valid] = np.clip(colors*shade[:, None], 0, 1)
    return image, ids, right, up


def render_report(output: Path, vertices: np.ndarray, root: np.ndarray, faces: np.ndarray,
                  labels: np.ndarray, names: list[str], width: np.ndarray, confidence: np.ndarray,
                  tangent: np.ndarray | None = None) -> None:
    output.mkdir(parents=True, exist_ok=True)
    palette = plt.get_cmap("tab20")(np.linspace(0, 1, len(names)))[:, :3]
    p95 = max(float(np.percentile(width, 95)), 1e-12)
    width_rgb = plt.get_cmap("YlOrRd")(np.clip(width/p95, 0, 1))[:, :3]
    for name, colors in (("parts", palette[labels]), ("shell_width", width_rgb),
                          ("support", plt.get_cmap("viridis")(confidence)[:, :3])):
        model = trimesh.Trimesh(vertices, faces, process=False)
        model.visual.vertex_colors = np.round(colors*255).astype(np.uint8)
        model.export(output / f"{name}.ply")
    write_json(output / "legend.json", dict(parts={name: palette[i].tolist() for i, name in enumerate(names)},
               shell_width=dict(min=0, max=p95, units="scene units", clipped_percentile=95),
               support=dict(min=0, max=1, interpretation="geometric correspondence support, not fur density")))
    frame = display_frame(vertices, labels, names)
    center = (vertices.max(0)+vertices.min(0))/2
    v, r = (vertices-center) @ frame, (root-center) @ frame
    span = np.linalg.norm(np.ptp(v, axis=0))*1.12
    directions = [np.array([x, y, .24]) for x, y in ((.65, 1), (-.65, 1), (.65, -1), (-.65, -1))]
    title = dict(
        parts="Part annotations",
        outer_root_overlay="Outer coat (orange) and estimated root surface (blue)",
        thickness="Nearest Frosting shell-width proxy",
        tangents="Directional tangent axis, not measured hair-growth direction",
    )
    for kind in title:
        if kind == "tangents" and tangent is None:
            continue
        fig, axes = plt.subplots(1, 4, figsize=(14, 4.6))
        for ax, direction in zip(axes, directions):
            points = r if kind == "tangents" else v
            rgb = palette[labels] if kind == "parts" else width_rgb if kind == "thickness" else np.tile([.82, .82, .81], (len(v), 1))
            if kind == "outer_root_overlay":
                rgb[:] = [.9, .54, .19]
            picture, ids, right, up = cast(points, faces, rgb, direction, span)
            if kind == "outer_root_overlay":
                inner, inner_ids, _, _ = cast(r, faces, np.tile([.12, .44, .7], (len(v), 1)), direction, span)
                inside = inner_ids != np.iinfo(np.uint32).max
                picture[inside] = .3*picture[inside]+.7*inner[inside]
            ax.imshow(picture, extent=(-span/2, span/2, -span/2, span/2))
            if kind == "tangents":
                visible = np.unique(ids[ids != np.iinfo(np.uint32).max])
                take = visible[np.linspace(0, len(visible)-1, min(850, len(visible)), dtype=int)]
                centers, vectors = r[faces[take]].mean(1), tangent[take] @ frame
                ax.quiver(centers @ right, centers @ up, vectors @ right, vectors @ up,
                          color="#b22123", angles="xy", scale_units="xy", scale=55/span, width=.002)
            ax.set_axis_off()
        fig.suptitle(title[kind], fontsize=13)
        if kind == "parts":
            fig.legend(handles=[Patch(color=palette[i], label=name) for i, name in enumerate(names)],
                       loc="lower center", ncol=5, frameon=False, fontsize=9)
        if kind == "thickness":
            bar = fig.add_axes([.36, .1, .28, .02])
            fig.colorbar(plt.cm.ScalarMappable(norm=Normalize(0, p95), cmap="YlOrRd"), cax=bar, orientation="horizontal", label="scene units (clipped at P95)")
        fig.subplots_adjust(left=.005, right=.995, top=.92, bottom=.25, wspace=.01)
        fig.savefig(output / f"{kind}_quadview.png", dpi=180)
        plt.close(fig)
    write_json(output / "display_frame.json", dict(rotation_columns=frame.tolist(), display_only=True))
    (output / "summary.md").write_text(
        "# Diagnostic views\n\n"
        "Parts: input annotations after any required nearest-vertex transfer to the selected "
        "mesh. Colors are in legend.json.\n\n"
        "Shell width: nearest Frosting shell width in scene units, clipped at this animal's P95. "
        "It is a geometric proxy, not measured coat thickness, strand length, or fur density.\n\n"
        "Outer/root overlay: orange outer coat and blue estimated root surface.\n\n"
        "Tangents: per-face Directional axes on the root surface, not physical hair-growth directions.\n\n"
        "Display orientation uses body/feet and face/tail labels when available; fixed-axis "
        "fallbacks are used otherwise. OBJ/PLY coordinates remain unchanged. "
        "Inspect the face, ears, paws, tail and every part boundary.\n")
