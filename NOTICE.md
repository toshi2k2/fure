# Notices

Frosting, 3D Gaussian Splatting, and their CUDA extensions are downloaded at a
pinned revision by `scripts/bootstrap.py`. Their original license files remain
with the downloaded source. These components are restricted to noncommercial
research and evaluation use.

NeuS2 and Directional are also fetched at pinned revisions by
`scripts/bootstrap.py`. Their repositories and licenses are listed in
`THIRD_PARTY.md`. NeuS2's NVIDIA license also limits use to noncommercial
research or evaluation. Directional is MPL-2.0.

The original FurE preprocessing code is licensed under Apache-2.0. The root
license does not relicense `vendor/`, software fetched by
`scripts/bootstrap.py`, or the part masks. The masks use the terms in
`part_masks/LICENSE.txt`.

Model checkpoints, model directories (including `cfg_args`), and pickle files
should be loaded only from trusted sources. Checksums verify file integrity but
do not make serialized or executable configuration safe to load.
