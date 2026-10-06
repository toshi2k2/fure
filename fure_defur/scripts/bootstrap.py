"""Fetch the pinned reconstruction dependencies."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
FROSTING_PATCH = ROOT / "third_party" / "frosting.patch"
PINS = {
    "frosting": ("https://github.com/Anttwo/Frosting.git", "08b1cb27b476b3c9a77b8405c28de3cd82dcaa70"),
    "Directional": ("https://github.com/avaxman/Directional.git", "5b7368eef67df09a809cbfa2908d226c068124bd"),
    "NeuS2": ("https://github.com/19reborn/NeuS2.git", "081b8191d049158a8296bfcb15c7d9cf065540df"),
}


def _apply_frosting_patch(target: Path) -> None:
    command = ["git", "-C", str(target), "apply", "-p3"]
    already_applied = subprocess.run(
        command + ["--reverse", "--check", str(FROSTING_PATCH)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode == 0
    if not already_applied:
        subprocess.run(command + ["--check", str(FROSTING_PATCH)], check=True)
        subprocess.run(command + [str(FROSTING_PATCH)], check=True)


def fetch(name: str) -> Path:
    url, revision = PINS[name]
    target = ROOT / "vendor" / name
    if target.exists():
        if not (target / ".git").exists():
            raise FileExistsError(f"Existing dependency is not a Git checkout: {target}")
        actual = subprocess.check_output(["git", "-C", str(target), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError(f"Existing {name} checkout is at a different revision")
        if name == "frosting":
            _apply_frosting_patch(target)
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--no-checkout", url, str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", revision], check=True)
    if name in {"NeuS2", "frosting"}:
        subprocess.run(["git", "-C", str(target), "submodule", "update", "--init", "--recursive"], check=True)
    if name == "frosting":
        _apply_frosting_patch(target)
    return target


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--frosting", action="store_true", help="Fetch Frosting and apply the FurE changes")
    p.add_argument("--directional", action="store_true", help="Fetch the pinned Directional source tree")
    p.add_argument("--neus2", action="store_true", help="Fetch the pinned NeuS2 source tree and submodules")
    args = p.parse_args()
    if args.frosting:
        fetch("frosting")
    if args.directional:
        fetch("Directional")
    if args.neus2:
        fetch("NeuS2")


if __name__ == "__main__":
    main()
