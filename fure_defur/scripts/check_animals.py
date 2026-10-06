"""Build and validate example-animal packages, or run geometry-only checks."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fure_defur.pipeline import run
from fure_defur.io import write_json
from fure_defur.validation import validate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--animals", nargs="+", default=["cat", "fox", "whiteTiger", "beagle_dog", "panda", "bison"])
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Write geometry diagnostics only; do not generate or validate strand packages",
    )
    parser.add_argument("--variants", nargs="+", choices=["shape", "shape_vlm"],
                        default=["shape", "shape_vlm"])
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    results = {}
    for animal in args.animals:
        results[animal] = {}
        for variant in args.variants:
            print(f"Checking {animal}/{variant}", flush=True)
            target = args.output / animal / variant
            try:
                result = run(ROOT / "configs" / f"{animal}.yaml", target,
                             variant, args.geometry_only)
                if not args.geometry_only:
                    result["validation"] = validate(target)
            except Exception as error:
                result = dict(state="failed", error=str(error),
                              ready_for_strand_data=False)
            results[animal][variant] = result
            write_json(args.output / "summary.json", results)
            print(f"{animal}/{variant}: {result}", flush=True)
    if any(result.get("state") == "failed"
           for variants in results.values() for result in variants.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
