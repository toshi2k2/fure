"""Command-line interface for capture preparation and strand preprocessing."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import sys


def main() -> None:
    p = argparse.ArgumentParser(
        prog="fure-defur",
        description=(
            "Prepare calibrated captures, reconstruct coat geometry, estimate a "
            "strand-root surface, and validate exported packages."
        ),
    )
    sub = p.add_subparsers(dest="command", required=True, metavar="COMMAND")
    for name in ("run", "geometry"):
        summary = (
            "Estimate a root surface and export a schema-complete preprocessing package"
            if name == "run"
            else "Estimate a root surface and write geometry diagnostics only"
        )
        parser = sub.add_parser(name, help=summary, description=summary)
        parser.add_argument("--config", type=Path, required=True, help="YAML configuration file")
        parser.add_argument("--output", type=Path, required=True, help="New output directory")
        parser.add_argument(
            "--variant",
            choices=("shape", "shape_vlm"),
            default="shape_vlm",
            help="Use shell-based heuristics alone (shape) or add the configured part priors (shape_vlm)",
        )
    parser = sub.add_parser(
        "validate",
        help="Check an exported package",
        description="Check the package's internal structure and geometric consistency.",
    )
    parser.add_argument("package", type=Path, help="Package directory")
    parser = sub.add_parser(
        "extract-video",
        help="Extract numbered PNG frames from a video",
        description="Extract numbered PNG frames with FFmpeg.",
    )
    parser.add_argument("video", type=Path, help="Input video")
    parser.add_argument("--output", type=Path, required=True, help="New frame directory")
    parser.add_argument("--fps", type=float, default=2., help="Frames to extract per second")
    parser = sub.add_parser(
        "calibrate",
        help="Calibrate a static capture with COLMAP",
        description="Run COLMAP structure from motion and image undistortion for a static capture.",
    )
    parser.add_argument("--images", type=Path, required=True, help="Input image directory")
    parser.add_argument("--masks", type=Path, help="Optional COLMAP feature-mask directory")
    parser.add_argument("--output", type=Path, required=True, help="New COLMAP output directory")
    parser = sub.add_parser(
        "prepare",
        help="Stage images, masks, and cameras for reconstruction",
        description="Rectify and stage matching images, masks, and projection matrices.",
    )
    for name in ("images", "masks", "projections", "output"):
        help_text = {
            "images": "Input image directory",
            "masks": "Matching single-channel mask directory",
            "projections": "NumPy file with one projection matrix per sorted image",
            "output": "New prepared-capture directory",
        }[name]
        parser.add_argument("--"+name, type=Path, required=True, help=help_text)
    parser.add_argument(
        "--bounds",
        nargs=6,
        type=float,
        metavar=("XMIN", "YMIN", "ZMIN", "XMAX", "YMAX", "ZMAX"),
        help="Animal bounds in the original camera coordinate system",
    )
    parser.add_argument("--points", type=Path, help="Optional seed point cloud in the original coordinate system")
    parser = sub.add_parser(
        "import-colmap",
        help="Export projection matrices from COLMAP",
        description="Convert an undistorted registered COLMAP model to projection matrices in image order.",
    )
    parser.add_argument("--model", type=Path, required=True, help="COLMAP sparse-model directory")
    parser.add_argument("--images", type=Path, required=True, help="Registered undistorted image directory")
    parser.add_argument("--output", type=Path, required=True, help="Output projection.npy path")
    parser = sub.add_parser(
        "train",
        help="Run 3DGS, NeuS2, and Frosting reconstruction",
        description="Run the three GPU reconstruction stages on a prepared capture.",
    )
    parser.add_argument("--prepared", type=Path, required=True, help="Prepared-capture directory")
    parser.add_argument("--output", type=Path, required=True, help="New reconstruction directory")
    parser.add_argument("--gpu", type=int, default=0, help="Physical CUDA device index")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter for 3DGS and Frosting")
    parser.add_argument("--neus2-python", default=sys.executable, help="Python interpreter for NeuS2")
    parser.add_argument("--steps", type=int, default=60000, help="NeuS2 training steps")
    parser.add_argument("--dry-run", action="store_true", help="Print stage definitions without running them")
    parser = sub.add_parser(
        "clean-surface",
        help="Clean and audit an unlabeled surface mesh",
        description="Select the subject component, repair small defects, optionally smooth, serialize, and audit the mesh.",
    )
    parser.add_argument("mesh", type=Path, help="Input triangle mesh")
    parser.add_argument("--output", type=Path, required=True, help="New OBJ path")
    parser.add_argument("--iterations", type=int, default=5, help="Taubin smoothing iterations")
    parser.add_argument(
        "--max-hole-diameter-fraction",
        type=float,
        default=.015,
        help="Largest repaired boundary-loop diameter as a fraction of the mesh diagonal",
    )
    parser.add_argument(
        "--max-hole-vertices",
        type=int,
        default=128,
        help="Largest repaired boundary loop in vertices",
    )
    parser.add_argument(
        "--merge-vertex-digits",
        type=int,
        help="Merge vertices after rounding coordinates to this many decimal digits",
    )
    parser = sub.add_parser(
        "materialize-config",
        help="Resolve package paths for an external trainer",
        description="Resolve package-relative adapter paths and insert an external decoder checkpoint path.",
    )
    parser.add_argument("package", type=Path, help="Exported package directory")
    parser.add_argument("--output", type=Path, required=True, help="New training YAML path")
    parser.add_argument("--decoder", type=Path, required=True, help="External strand-decoder checkpoint")
    parser = sub.add_parser(
        "orientations",
        help="Generate axial image-orientation maps",
        description="Generate Gabor orientation angles and variances for matching images and masks.",
    )
    parser.add_argument("--images", type=Path, required=True, help="Numbered PNG image directory")
    parser.add_argument("--masks", type=Path, required=True, help="Matching foreground-mask directory")
    parser.add_argument("--output", type=Path, required=True, help="New orientation-map directory")
    parser.add_argument("--workers", type=int, default=1, help="Views to process concurrently")
    args = p.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    result = None
    if args.command in ("run", "geometry"):
        from .pipeline import run
        result = run(args.config, args.output, args.variant, args.command == "geometry")
    elif args.command == "validate":
        from .validation import validate
        result = validate(args.package)
    elif args.command == "extract-video":
        from .prepare import extract_video
        extract_video(args.video, args.output, args.fps)
    elif args.command == "calibrate":
        from .prepare import calibrate
        calibrate(args.images, args.output, args.masks)
    elif args.command == "prepare":
        from .prepare import prepare
        prepare(args.images, args.masks, args.projections, args.output, args.bounds, args.points)
    elif args.command == "import-colmap":
        from .prepare import import_colmap
        import_colmap(args.model, args.images, args.output)
    elif args.command == "train":
        from .reconstruction import train
        result = train(args.prepared, args.output, args.gpu, args.python, args.neus2_python, args.steps, args.dry_run)
    elif args.command == "clean-surface":
        from .reconstruction import clean_surface
        result = clean_surface(args.mesh, args.output, args.iterations,
                               args.max_hole_diameter_fraction, args.max_hole_vertices,
                               args.merge_vertex_digits)
    elif args.command == "materialize-config":
        from .package import materialize
        materialize(args.package, args.output, args.decoder)
    elif args.command == "orientations":
        from .orientations import generate
        generate(args.images, args.masks, args.output, args.workers)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
