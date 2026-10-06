"""Report installed dependencies and optionally check the real CUDA extensions."""
from __future__ import annotations
import argparse
import importlib
import importlib.metadata
import json
import math
from pathlib import Path
import shutil
import sys


def check_rasterizer(torch) -> dict:
    """Execute a tiny differentiable render, not merely an extension import."""
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer

    device = torch.device("cuda")
    means3d = torch.tensor(
        [[-0.25, -0.15, 0.5], [0.2, -0.1, 0.6], [0.0, 0.25, 0.7]],
        dtype=torch.float32, device=device, requires_grad=True,
    )
    means2d = torch.zeros_like(means3d, requires_grad=True)
    opacity = torch.full((3, 1), 0.8, device=device, requires_grad=True)
    scales = torch.full((3, 3), 0.08, device=device, requires_grad=True)
    rotations = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0]] * 3, device=device, requires_grad=True,
    )
    colors = torch.tensor(
        [[1.0, 0.2, 0.1], [0.1, 1.0, 0.2], [0.2, 0.1, 1.0]],
        device=device, requires_grad=True,
    )
    settings = GaussianRasterizationSettings(
        image_height=32, image_width=32,
        tanfovx=math.tan(math.pi / 4), tanfovy=math.tan(math.pi / 4),
        bg=torch.zeros(3, device=device), scale_modifier=1.0,
        viewmatrix=torch.eye(4, device=device), projmatrix=torch.eye(4, device=device),
        sh_degree=0, campos=torch.zeros(3, device=device), prefiltered=False, debug=False,
    )
    image, radii = GaussianRasterizer(raster_settings=settings)(
        means3D=means3d, means2D=means2d, shs=None, colors_precomp=colors,
        opacities=opacity, scales=scales, rotations=rotations, cov3D_precomp=None,
    )
    loss = image.square().mean()
    loss.backward()
    gradients = (means3d.grad, means2d.grad, opacity.grad, scales.grad, rotations.grad, colors.grad)
    if not torch.isfinite(image).all() or not all(g is not None and torch.isfinite(g).all() for g in gradients):
        raise RuntimeError("Gaussian rasterizer produced non-finite output or gradients")
    visible = int((radii > 0).sum().item())
    if visible != len(means3d) or not float(loss) > 0:
        raise RuntimeError("Gaussian rasterizer smoke scene was not rendered as expected")
    return {"image_shape": list(image.shape), "visible_gaussians": visible,
            "loss": float(loss), "finite_gradients": True}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gpu", action="store_true")
    args = p.parse_args()
    modules = ["numpy", "scipy", "trimesh", "open3d", "PIL", "cv2", "skimage", "yaml"]
    if args.gpu:
        modules += ["torch", "pytorch3d", "diff_gaussian_rasterization", "simple_knn._C"]
    checks = {name: str(getattr(importlib.import_module(name), "__version__", "import OK")) for name in modules}
    checks["python"] = sys.executable
    environment_bin = Path(sys.prefix) / "bin"
    checks["external_tools"] = {
        name: shutil.which(name) or (str(environment_bin / name) if (environment_bin / name).is_file() else None)
        for name in ("ffmpeg", "colmap", "git", "cmake", "ninja")
    }
    checks["compiler"] = shutil.which("g++") or shutil.which("c++")
    if not checks["compiler"]:
        raise RuntimeError("A C++20 compiler is needed for Directional")
    if args.gpu:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        checks["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        from simple_knn._C import distCUDA2
        points = torch.rand(256, 3, device="cuda")
        distances = distCUDA2(points)
        if not torch.isfinite(distances).all():
            raise RuntimeError("The simple-knn extension returned non-finite distances")
        checks["knn_kernel"] = "executed"
        checks["rasterizer_kernel"] = check_rasterizer(torch)
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
