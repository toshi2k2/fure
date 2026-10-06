# Installation

## CPU preprocessing

The CPU environment supports capture preparation, mesh cleanup, defurring,
package export, and validation. FFmpeg and COLMAP are installed with the
environment.

```bash
conda env create -f environment.yml
conda activate fure-defur
python scripts/bootstrap.py --directional
python scripts/doctor.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
  python -m pytest -q
```

Directional is pinned to the revision in `scripts/bootstrap.py`. Its helper is
compiled on first use and stored under `build/`. A C++20 compiler is required.
On systems where Open3D cannot find `libGL.so.1`, install the distribution's
OpenGL runtime package.

Editable installation in an existing Python 3.10 environment is also supported:

```bash
python -m pip install -e '.[test]'
python scripts/bootstrap.py --directional
```

## GPU reconstruction

The GPU environment pins Python 3.9, PyTorch 2.0.1, CUDA 11.8, and
PyTorch3D 0.7.4.

```bash
conda env create -f environment-gpu.yml
conda activate fure-defur-gpu
export CUDA_HOME="$CONDA_PREFIX"

python scripts/bootstrap.py --frosting --neus2
python -m pip install -e '.[test]'
python -m pip install --no-build-isolation \
  ./vendor/frosting/gaussian_splatting/submodules/simple-knn \
  ./vendor/frosting/gaussian_splatting/submodules/diff-gaussian-rasterization

cmake -S vendor/NeuS2 -B vendor/NeuS2/build \
  -DCMAKE_BUILD_TYPE=Release \
  -DNGP_BUILD_WITH_GUI=OFF \
  -DNGP_BUILD_WITH_VULKAN=OFF
cmake --build vendor/NeuS2/build -j 4

python scripts/doctor.py --gpu
```

`CUDA_HOME` must point to the toolkit used to build the extensions. Set `CC`
and `CXX` if the host compiler is incompatible with CUDA 11.8. NeuS2 may use a
separate environment; pass its interpreter with `train --neus2-python`.

The default reconstruction schedule is 30,000 3DGS iterations, 60,000 NeuS2
steps, long Frosting refinement, and 500,000 Frosting Gaussians. Inspect the
camera registration, foreground masks, and recovered silhouettes for each
capture.

## Multiple GPUs

Each reconstruction process uses one GPU. Run independent animals as separate
jobs:

```bash
python -m fure_defur train \
  --prepared work/cat \
  --output outputs/cat \
  --gpu 0 \
  --python "$CONDA_PREFIX/bin/python" \
  --neus2-python /path/to/neus2-env/bin/python
```

Choose a different output directory and GPU index for each concurrent run.
