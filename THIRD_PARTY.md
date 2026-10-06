# Third-party software

## Gaussian Frosting

- Upstream: <https://github.com/Anttwo/Frosting>
- Revision: `08b1cb27b476b3c9a77b8405c28de3cd82dcaa70`
- Installed by: `python scripts/bootstrap.py --frosting`
- Local changes: `third_party/frosting.patch`

The bootstrap command checks out Frosting and its dependencies, then applies
the local diff. The changes add foreground-alpha handling, shell-statistics
export, bounded mesh rasterization, and the training entry point used by this
pipeline. Frosting, 3D Gaussian Splatting, and their CUDA components retain
their upstream license terms.

## NeuS2

- Upstream: <https://github.com/19reborn/NeuS2>
- Revision: `081b8191d049158a8296bfcb15c7d9cf065540df`
- Installed by: `python scripts/bootstrap.py --neus2`

The bootstrap command checks out NeuS2 with its submodules and retains the
upstream license files.
Its `LICENSE.txt` is the NVIDIA Source Code License used by instant-ngp and
limits the work and its derivatives to noncommercial research or evaluation.

## Directional

- Upstream: <https://github.com/avaxman/Directional>
- Revision: `5b7368eef67df09a809cbfa2908d226c068124bd`
- Installed by: `python scripts/bootstrap.py --directional`

Directional headers carry MPL-2.0 notices. The fetched Directional checkout's
Eigen tree includes its upstream license files. FurE uses a small C++ bridge in
`native/directional_raw_face_field.cpp` to compute the raw 2-RoSy face field.

## Citation

Please cite the corresponding papers when using these components:

- Antoine Guédon and Vincent Lepetit. *Gaussian Frosting: Editable Complex
  Radiance Fields with Real-Time Rendering*. ECCV, 2024.
- Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis.
  *3D Gaussian Splatting for Real-Time Radiance Field Rendering*. ACM TOG, 2023.
- Yiming Wang, Qin Han, Marc Habermann, Kostas Daniilidis, Christian Theobalt,
  and Lingjie Liu. *NeuS2: Fast Learning of Neural Implicit Surfaces for
  Multi-view Reconstruction*. ICCV, 2023.
- Amir Vaxman et al. *Directional Field Synthesis, Design, and Processing*.
  Computer Graphics Forum, 2016.
