# Data format

## Coordinates and cameras

Meshes, Frosting shell samples, landmarks, and cameras use one scene coordinate
system. `projection.npy` contains one float32/float64 matrix per image, with
shape `[N, 4, 4]` or `[N, 3, 4]`. The first three rows are
`K [R | t]` using the OpenCV world-to-camera convention: x points right, y
points down, and positive z points forward.

Prepared views use this layout:

```text
views/
  images_2/0000.png
  foreground/0000.png
  projection.npy
```

Images are sorted by filename. Projection matrices and foreground masks follow
the same order. Foreground masks are single-channel PNGs; values above 127 are
treated as foreground. Rectification may introduce intermediate values along
mask boundaries.

`prepare` rectifies the views to shared centered intrinsics for reconstruction.
`image_manifest.json` stores the source filename and pixel homography for each
output view.

When bounds are set, `normalization.json` records the similarity transform
between the capture frame and the normalized scene. Apply `original_to_scene`
to later geometry and landmarks expressed in the capture frame. If the uniform
scale in `scene_to_original` is `a`, and the original scale is `c` centimeters
per original unit, the normalized scale is `c*a` centimeters per scene unit.

## Part labels

Part labels are a JSON object mapping each part name to vertex indices on a
triangle mesh:

```json
{
  "body": [0, 1, 2],
  "face": [3, 4],
  "eyes": [5],
  "horns": [6, 7]
}
```

Each vertex belongs to at most one part. Missing vertices are assigned to
`unlabeled` and receive zero strand length. Face-based labels should be resolved
to a vertex partition before export.

Using the cleaned outer mesh as the annotation mesh gives an exact topological
match. Labels from another mesh are transferred by nearest vertex in the same
coordinate frame; the normalized transfer distance is recorded in
`metrics.json`.

The default non-fur names are `eyes`, `nosetip`, `nose`, `paw_pads`, `horns`,
`inner_earcanal`, and `unlabeled`. Additional names can be listed in
`nonfur_parts`.

## Frosting shell statistics

`shell_stats.npz` contains:

| Array | Shape | Meaning |
| --- | --- | --- |
| `verts` | `[S,3]` | shell vertices |
| `normals` | `[S,3]` | unit vertex normals |
| `faces` | `[F,3]` | shell triangles |
| `inner_dist` | `[S]` | inner Frosting offset |
| `outer_dist` | `[S]` | outer Frosting offset |
| `inner_points` | `[S,3]` | `verts + inner_dist[:,None] * normals` |
| `outer_points` | `[S,3]` | `verts + outer_dist[:,None] * normals` |
| `shell_width` | `[S]` | `abs(outer_dist - inner_dist)` |
| `reliability_mask_init` | `[S]` | shell-width outlier gate |

The reliability array is a geometric outlier mask from the Frosting export. It
is not a calibrated confidence score.

## Strand package

The root mesh keeps the outer mesh's face and vertex order. All topology-bound
arrays are generated after defurring.

- `tan_furless.npy`: float32 `[F,3]` unit face tangents.
- `rawFaceField.dmat`: Directional's raw `[F,6]` 2-RoSy field.
- `field_furless_reshaped.npy`: the reshaped raw field.
- `annotations_furless_reshaped2.json`: root vertex labels.
- `annotations_faces.json`: root face labels.
- `defurring_fields.npz`: `displacement`, `labels`, `confidence`,
  `transferred_width`, `nearest_width`, and `upper` displacement bounds.

`orientations_2/angles/*.png` stores axial angles in integer degrees modulo 180.
`orientations_2/vars/*.npy` stores float16 axial variance in radians squared.
The pipeline can copy compatible maps selected in the config or
generate them with `fure-defur orientations`. Generated variance is bounded by
`pi^2/4`; maps from a different uncertainty convention must be regenerated.

`masks_2/body` is the visible root surface inside the observed foreground.
`masks_2/hair` is the observed foreground excluding visible non-fur regions.
These masks overlap by design. `masks_2/bald` stores visible non-fur regions,
and `masks_2/foreground` copies the observed silhouette. `part_masks` uses
uint16 PNGs: 0 is background and positive IDs are defined by `legend.json`.

## Signed-distance grid

`sdf_grid.npy` is float32 `[X,Y,Z]`, sampled over the inclusive bounds in
`min_bound.npy` and `max_bound.npy`. Values are negative inside the root mesh.

For PyTorch 5D `grid_sample`, query coordinates are ordered `[z,y,x]` after
normalization and `align_corners=True` is used.

## Units and trainer adapter

`strand_package.json` stores `cm_per_scene_unit`, its `scale_basis`, and,
when eye landmarks are set, `eye_distance_cm`. Validation checks that the
landmark separation times the scene scale agrees with that distance. The file
also stores per-part `length_cm`, the corresponding `length_scene`, and
initialization widths. The width field is a renderer/training initialization
value rather than a measured fiber diameter.

`hair_config.yaml` uses paths relative to the package root. Its scale paths stay
empty because geometry and cameras are already aligned. The required
`scale.pickle` records the same identity transform for compatibility and must
not be applied a second time.
The external decoder checkpoint is filled when the package is materialized for
training:

```bash
python -m fure_defur materialize-config PACKAGE \
  --decoder /path/to/decoder.pt \
  --output training.yaml
```
