# Part-level values

`shape` estimates displacement and strand length from the Frosting shell and
fixed part multipliers. `shape_vlm` also reads per-part strand length and
radial thickness from the config.

```yaml
semantic_prior:
  length_cm:
    body: 4.0
    eyes: 0.0
  thickness_cm:
    body: 1.0
    eyes: 0.0
  basis: Values used for this capture
```

Lengths and thicknesses are in centimeters. Non-fur parts should be zero. Part
names must match the vertex-label JSON exactly.

These values initialize the estimator; they are not physical measurements.
Keep the scene scale and its basis separate from the part table. Use `shape`
when no suitable table is available.
