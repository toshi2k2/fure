# What the estimator does

Using the reconstructed coat as the attachment surface places strand roots
outside the underlying skin. The estimator displaces the reconstructed outer
mesh inward, using Frosting shell width as a local geometric proxy. Part labels
select region-specific priors and identify non-fur vertices that remain fixed.

## Shell-width transfer

Let D be the outer mesh bounding-box diagonal. For each outer vertex, query the
12 nearest Frosting shell vertices within 0.03D. Reject a match if its transferred
part label differs, its stored outlier mask is false, or its normal dot product
is at most 0.3.

Shell exports may use the opposite global normal orientation. Before matching,
we find reliable shell vertices within 0.03D of the outer mesh and take the
median dot product with their nearest outer normals. A negative median flips all
shell normals used for correspondence; an absolute median at most 0.3, or fewer
than `min(20,V)` nearby reliable samples for V outer vertices, stops the run.
This changes only the normals used for matching. Stored offsets and shell
geometry remain unchanged.

Each remaining match has weight

```
w_ij = exp(-(distance_ij / (0.03 D))^2)
       * clamp((dot(n_i, n_j) - 0.3) / 0.7, 0, 1)^2
```

Their normalized weighted width is the local width estimate. The largest weight
is a geometric support score c_i; no supported match means c_i=0. This score is
not an estimated probability that the surface contains fur.

## Part priors and defurring

For geometry, nearest-shell widths are clipped at each part's P95, then pooled
at P75. V1 (`shape`) uses this pooled value as its part target. V2 (`shape_vlm`)
uses 0.65 times that value plus 0.35 times the configured part thickness,
capped at the part's P95. The V2 table must cover every part present in the
annotation. The local supported width is scaled by the part-target/P75 ratio,
capped at 1.15.

We solve for inward normal displacement d, rather than smoothing mesh positions:

```
min_d sum_i c_i (d_i - local_target_i)^2
    + 0.25 sum_i (d_i - part_target_i)^2
    + 8 d^T L d
subject to 0 <= d_i <= upper_i, and d_i = 0 on non-fur parts.
```

L is an edge-length-weighted graph Laplacian. Edges crossing part boundaries
have 0.15 times the same-part weight; boundaries remain coupled instead of
becoming seams. The upper bound is the smaller of 0.45 times the inward-ray
distance to the opposite surface and 0.05D. A missing ray hit uses 0.1D before
the 0.45 factor. Targets are clipped to these bounds.

The root surface is `x_root = x_outer - d*n_outer`. Local backtracking rejects
relative face inversions, severe area collapse and new triangle intersections.
The final displacement after collision correction is not claimed to remain the
exact minimizer of the original quadratic. A complete package additionally
requires a closed root mesh, no detected self-intersections, and every root
vertex inside the outer mesh to a tolerance of `1e-5 D`. Any vertex outside that
tolerance stops package generation and fails validation. This is **outer-mesh
containment**, not a proof of containment inside the full Frosting volume.

Vertices without accepted shell matches are controlled by the part-prior and
smoothness terms. The result is an initialized root surface, not an estimate of
ground-truth skin. Errors in masks, camera coverage, or part labels propagate
into the result.

## Length and width initialization

Part length uses the P75 of the **raw nearest-shell width**, before the geometry
correspondence gates. This is intentionally distinct from the supported local
displacement above.
Let T_p be that statistic times cm_per_scene_unit. For fur-bearing parts:

```
V1: length_p = multiplier_p * T_p
V2: length_p = clip(0.6 * V1_p + 0.4 * reference_p,
                    0.35 * reference_p, 1.35 * reference_p)
```

V2 uses V1 when no positive reference is provided. Non-fur parts are zero.
Multipliers are fixed: belly 3.6, body 3.4, neck 3.5, mane 3.8, face/ears 3.0,
front/rear leg 2.6, front paws 2.2, paws 2.15, tail 2.75, under-tail 2.7.
Other fur-bearing labels use 2.75. These fixed multipliers are heuristic length
priors. V2's bounds use its reference table, so agreement with that table is not
independent accuracy evidence.

The width initialization is `clip(0.03 * length_cm, 0.02, 0.12)` cm, and zero
for non-fur parts. It is a renderer/training initialization, not a physical
fiber-diameter estimate. Renderer-specific density and radius remain separate
parameters.

## Tangents and image orientations

Directional computes a 2-RoSy field on the final mesh with one hard tangent
constraint. We extract a face tangent, check that it lies in the face plane,
normalize it, and resolve its sign over every connected component with an
adjacency-order, dot-product BFS. This is not Levi-Civita transport or a
physical hair-growth field.

New images use 180 Gabor angles, frequency 0.23, and sigmas 1.8 and 2.4 after a
0.4/10-sigma difference-of-Gaussians filter. The maximum absolute response gives
the axial angle. Response-weighted squared angular deviation is stored as a
dispersion value, not calibrated uncertainty. A config may select compatible
precomputed maps explicitly; otherwise, maps under the prepared views are used
when present or generated from the package images. Validation checks filenames,
shapes, angle range, finite variance in `[0, pi^2/4]`, and view coverage, but not
correspondence to image content.

## Interpreting reports

`support_fraction` counts vertices with at least one accepted shell match.
Shrink percentiles describe the final normal displacement, not hair arc length.
`inside_outer_fraction` is the fraction of vertices that pass the containment
tolerance and must equal 1. `root_vs_foreground_iou` compares a shrunken surface
against the furry silhouette, so 1 is not necessarily the objective.
`anchor_drift_max` should be zero. Tangent norm and normal-dot errors should be
small and a repeated sign pass should make zero changes.

No ground-truth skin or strand-length accuracy is implied by these checks.
Quantitative strand results from another root surface require fresh training
and evaluation.
