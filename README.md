# FurE preprocessing

This repository contains the furless-mesh preprocessing and evaluation code
used in the paper. It cleans a reconstructed 3D model, estimates a furless
mesh, rebuilds topology-dependent fields and masks, and writes the files used
by the strand-learning stage.

The inputs are calibrated multi-view images, foreground masks, reconstruction
geometry, and a human-approved part segmentation. The output is a complete
package for strand decoding.

The strand decoder and strand-training code are separate.

```text
multi-view images + foreground masks + cameras
        |-- NeuS2 ----------------------------> outer 3D model
        `-- 3D Gaussian Splatting + Frosting -> shell-width samples

outer 3D model + shell samples + human-approved 3D part labels
        -> furless mesh
        -> strand-learning data
```

Part labels are read from a segmented triangle mesh. This code does not create
the part segmentation.

## Installation

The CPU environment covers image preparation, mesh cleanup, defurring, data
export, and validation. A C++20 compiler is required for the Directional
helper.

```bash
conda env create -f environment.yml
conda activate fure-defur
python scripts/bootstrap.py --directional
python scripts/doctor.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m pytest -q
```

NeuS2 and Frosting reconstruction require an NVIDIA GPU. See
[docs/installation.md](docs/installation.md).

## Prepare a capture

Use images of one static pose with broad view coverage. Foreground masks and
cameras must match the image resolution and filename order.

```bash
python -m fure_defur extract-video capture.mp4 --fps 2 --output work/frames

python -m fure_defur calibrate \
  --images work/frames \
  --output work/colmap

python -m fure_defur import-colmap \
  --model work/colmap/undistorted/sparse \
  --images work/colmap/undistorted/images \
  --output work/projection.npy

python -m fure_defur prepare \
  --images work/colmap/undistorted/images \
  --masks work/masks \
  --projections work/projection.npy \
  --bounds xmin ymin zmin xmax ymax zmax \
  --output work/prepared
```

The bounds must enclose the animal in the original camera coordinate system.

## Reconstruct and clean the outer 3D model

```bash
python scripts/bootstrap.py --frosting --neus2

python -m fure_defur train \
  --prepared work/prepared \
  --output work/reconstruction \
  --gpu 0 \
  --python /path/to/frosting-env/bin/python \
  --neus2-python /path/to/neus2-env/bin/python

python -m fure_defur clean-surface \
  work/reconstruction/neus2/outer_raw.obj \
  --output work/outer.obj

python scripts/audit_surface.py work/outer.obj \
  --views work/prepared/views \
  --output work/outer_check
```

The cleanup step selects the animal component, removes duplicate and
degenerate elements, repairs small boundary loops, and checks the written mesh.
Large holes and remaining self-intersections stop the run.

## Build the furless mesh and strand-learning data

Copy [configs/example.yaml](configs/example.yaml) and set the paths to the
cleaned outer mesh, segmented mesh, vertex labels, Frosting shell statistics,
and prepared views. The config also records the scene scale and the part-level
length and thickness values used by `shape_vlm`.

```bash
python -m fure_defur run \
  --config configs/my_animal.yaml \
  --variant shape_vlm \
  --output outputs/my_animal

python -m fure_defur validate outputs/my_animal
```

`shape` uses the shell-width estimates and fixed part multipliers. `shape_vlm`
also uses the per-part table in the config. Both modes use the same constrained
surface estimator.

Important output files are:

| Path | Contents |
| --- | --- |
| `furless.obj` | furless mesh used for strand roots |
| `neus.obj` | cleaned outer 3D model |
| `annotations_furless_reshaped2.json` | vertex part labels on the furless mesh |
| `annotations_faces.json` | face part labels |
| `tan_furless.npy`, `rawFaceField.dmat` | per-face tangent field |
| `images_2/`, `projection.npy` | ordered images and projection matrices |
| `masks_2/`, `part_masks/` | foreground, fur, non-fur, and part masks |
| `orientations_2/` | axial image-orientation maps |
| `sdf_grid.npy`, `min_bound.npy`, `max_bound.npy` | inside-negative signed-distance grid |
| `strand_package.json` | paths, units, part values, and topology hash |
| `hair_config.yaml` | paths and values for the strand-learning code |
| `metrics.json`, `status.json`, `run.json` | numerical checks, status, config, and input hashes |

See [docs/data_format.md](docs/data_format.md) for file conventions and
[docs/method.md](docs/method.md) for the estimator.

## Part masks

The repository includes the human-approved part masks for cat, fox, white
tiger, beagle, panda, and bison under `part_masks/`. Cat, fox, white tiger,
beagle, and panda are from Artemis data. The bison was captured at the
Smithsonian National Museum of Natural History in Washington, DC. Mask values
are listed in each animal's `legend.json`.

The FurE code is licensed under Apache-2.0. The part masks use CC BY-NC 4.0.
Third-party components keep their own licenses; see
[THIRD_PARTY.md](THIRD_PARTY.md).
