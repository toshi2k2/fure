"""Run headless NeuS2 training with the pinned Python bindings."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fure_defur.neus2_input import stage_scene


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scene", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--steps", type=int, default=60000)
    p.add_argument("--resolution", type=int, default=512)
    args = p.parse_args()
    args.scene, args.output = args.scene.resolve(), args.output.resolve()
    if args.output.exists():
        raise FileExistsError(args.output)
    engine = ROOT / "vendor/NeuS2"
    libraries = list(engine.glob("build*/**/pyngp*.so"))
    if len(libraries) != 1:
        raise RuntimeError("Build exactly one matching pyngp module; see docs/installation.md")
    sys.path.insert(0, str(libraries[0].parent))
    import pyngp as ngp
    config = json.loads((engine / "configs/nerf/dtu.json").read_text())
    if "parent" in config:
        config["parent"] = str((engine / "configs/nerf" / config["parent"]).resolve())
    config["hyperparams"]["mask_loss_weight"] = .05
    config["hyperparams"]["first_frame_max_training_step"] = args.steps
    args.output.mkdir(parents=True)
    staged = stage_scene(args.scene, args.output / "input")
    network = args.output / "network.json"
    network.write_text(json.dumps(config, indent=2))
    # NeuS2 reads utils/mlp_weights.txt relative to cwd.
    os.chdir(engine)
    model = ngp.Testbed(ngp.TestbedMode.Nerf)
    model.load_training_data(str(staged))
    model.reload_network_from_file(str(network), str(args.output / "engine"))
    model.shall_train = True
    start = time.perf_counter()
    while model.training_step < args.steps:
        if not model.frame():
            raise RuntimeError("NeuS2 stopped before reaching the requested step count")
    model.shall_train = False
    model.save_snapshot(str(args.output / "checkpoint.msgpack"), False)
    model.compute_and_save_marching_cubes_mesh(str(args.output / "outer_raw.obj"), [args.resolution]*3, thresh=0.0)
    (args.output / "metrics.json").write_text(json.dumps(dict(steps=int(model.training_step), seconds=time.perf_counter()-start,
                loss=float(model.loss), resolution=args.resolution, mask_loss_weight=.05), indent=2))


if __name__ == "__main__":
    main()
