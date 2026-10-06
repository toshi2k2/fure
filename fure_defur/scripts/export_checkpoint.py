"""Re-export a checkpoint from a trusted source without retraining (GPU required)."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fure_defur.neus2_input import stage_scene


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", choices=["neus2", "frosting"])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene", type=Path, help="NeuS2 transform.json with the checkpoint's original normalization")
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    if args.scene is not None:
        args.scene = args.scene.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if args.engine == "neus2":
        if args.scene is None or not args.scene.is_file():
            raise ValueError("NeuS2 export needs its original scene transform.json")
        libraries = list((ROOT / "vendor/NeuS2").glob("build*/**/pyngp*.so"))
        if len(libraries) != 1:
            raise RuntimeError("Build the pinned NeuS2 engine first")
        sys.path.insert(0, str(libraries[0].parent))
        import pyngp as ngp
        args.output.mkdir(parents=True)
        staged = stage_scene(args.scene, args.output / "input")
        os.chdir(ROOT / "vendor/NeuS2")
        model = ngp.Testbed(ngp.TestbedMode.Nerf)
        model.load_training_data(str(staged))
        model.reload_network_from_file(str(ROOT / "vendor/NeuS2/configs/nerf/dtu.json"), str(args.output / "engine"))
        model.load_snapshot(str(args.checkpoint.resolve()))
        model.shall_train = False
        model.compute_and_save_marching_cubes_mesh(str(args.output / "outer_raw.obj"), [args.resolution]*3, thresh=0.0)
    else:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("The upstream Frosting exporter requires CUDA")
        engine = ROOT / "vendor/frosting"
        sys.path[:0] = [str(engine), str(engine / "gaussian_splatting")]
        from frosting_scene.frosting_model import load_frosting_model, convert_frosting_into_gaussians
        from animal_skin.shell_export import export_shell_stats_from_frosting
        model = load_frosting_model(str(args.checkpoint.resolve()), device=torch.device("cuda"))
        args.output.mkdir(parents=True)
        export_shell_stats_from_frosting(model, scene_name="shell", output_root=args.output)
        convert_frosting_into_gaussians(model).save_ply(str(args.output / "frosting_gaussians.ply"))


if __name__ == "__main__":
    main()
