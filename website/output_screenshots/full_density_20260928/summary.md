# Full-Density Browser Inspection

All screenshots are actual Chromium/WebGL captures, not Blender renders.
Select Explore 3D, then Full density to reproduce. The named animal PNGs show
all original teaser curves and the lightweight underlying body together.
`bison-body-only.png` hides strands for an alignment check.

| Animal / screenshot | Original strands displayed | Download (MiB) |
|---|---:|---:|
| cat.png | 310,000 | 43.60 |
| fox.png | 265,000 | 37.42 |
| tiger.png | 265,000 | 37.42 |
| beagle.png | 100,000 | 14.76 |
| panda.png | 220,000 | 31.24 |
| bison.png | 104,993 | 15.17 |

Each strand retains the approved scene's 12 control points. No interpolation,
strand filtering or coordinate realignment was introduced. During export,
every sampled preview control point and both body buffers matched the new
full-density export byte-for-byte. The original preview assets were preserved.
Bodies remain simplified to roughly 60k faces (bison retains 43,868).

Validation: 14 Chromium tests passed, including all six full-density loads,
SHA-256/count/finite-position/bounds checks, identical body buffers, loading
cancellation, release on exit, preview fallback and mobile layout. These tests
use software WebGL, not a physical GPU performance benchmark. `timings.json`
records local UI/load/render waits (roughly 1.6-4.1 seconds); these are not
internet download times or interaction FPS, and cat timing starts just after
clicking Full density, whereas others include the animal-tab click.

Visual limitation: opaque one-pixel WebGL lines have no Blender/Cycles hair
lighting, self-shadowing or anisotropic highlights. At this density they can
merge into a flat-looking silhouette, particularly on cat. This is an honest
full-geometry inspection mode, not a reproduction of the teaser's appearance.
Use the Studio view or film for the photorealistic presentation.

These changes are local until committed/pushed and the website deployment
workflow is run. No remote deployment was performed as part of this task.
