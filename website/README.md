# FurE Project Website

A self-contained static research page for the draft in `../paper/`. No backend,
API key, CDN, Python, Blender or research dataset is needed to serve the prepared
website. Source assets and research results are not modified.

## Run Locally

Requires Node.js 20 or newer. From this directory:

```sh
npm ci
npm run build
npm run serve
```

Open `http://127.0.0.1:4173`. Do not open `index.html` via `file://`; the module
loader and geometry requests need an HTTP server. The server supports byte-range
requests for seeking through the 4K film. To access this server from a laptop,
forward port 4173 through your existing SSH connection.

`dist/` is the complete deployable website. All URLs are relative, so it works at
the repository's GitHub Pages path, `/fure/`, or at a custom domain root.

## Content And Controls

- `index.html`: paper-based text, links, figures and accessible HTML fallback.
- `styles.css`: responsive editorial layout and self-hosted typography.
- `config.js`: animal counts/descriptions, URLs, citation and rotation interval.
- `app.js`: gallery, video, clipboard, five-second co-first-author rotation.
- `viewer.js`: click-to-load Three.js body/strand viewer; only renders on changes.
- `assets/`: all public media, fonts/licenses, preview geometry and provenance.

The author names/credits were read from `../paper/main_iclr.tex`. Srinjay Sarkar
and Prakhar Kaushik alternate every five seconds; equal-contribution markers and
Prakhar's project-lead marker move with the corresponding name. Rotation pauses
on author hover/focus, in a hidden tab, with the pause control, or initially when
the visitor requests reduced motion. Other authors do not move. Author markup
is also present in HTML for search engines and no-JavaScript visitors; update it
alongside `config.js` if authorship changes.

The PDF is a local build of the supplied current draft, not an external preprint.
The BibTeX deliberately contains `year = {TODO}` and a placeholder note. Update
both `config.js` and the HTML fallback when publication metadata is available.
The displayed 52-minute / 10.5-hour comparison is attributed to the current
manuscript's strand-training table (36 views, RTX A5000), not a new benchmark.

## Assets And Scientific Scope

The hero and six animal geometries come from the approved six-animal scene:

```
Frosting/1_paper_vis/0_teaser/final_fur_focus_upclose_balanced_v03/
```

| Animal | Strands in the approved scene | Browser preview |
|---|---:|---:|
| Cat | 310,000 | 30,000 |
| Fox | 265,000 | 30,000 |
| Tiger | 265,000 | 30,000 |
| Beagle | 100,000 | 30,000 |
| Panda | 220,000 | 30,000 |
| Bison | 104,993 | 30,000 |

Studio portraits retain every curve in that scene and its body mesh, lighting
and illustrative palette. The scene uses 12 control points per curve. These are
presentation renders, not color-accuracy evaluations or newly learned textures.
Bison retains the approved scene's pose-grounding corrections and eye detail,
recorded in `assets/teaser-provenance.json`.

Browser previews sample existing curves with seed 73, without interpolating
additional strands. Body meshes are decimated to at most about 60k triangles
for interactive use. Both body and fur receive the same center, scale and axis
transform. Source research geometry is untouched. The `.json` files record
counts, layout, bounds, transforms' convention and SHA-256 hashes; `.bin` files
contain little-endian float32 positions and uint32 triangle indices. Simple
WebGL lines are for inspecting geometry, not reproducing Cycles hair shading.

### Full-Density Interactive Mode

Select **Explore 3D**, then **Full density · all strands**. This loads every
original teaser-scene curve (the counts above), not interpolated filler strands.
The same simplified body and 12 control points per strand are retained. The
exporter checks that the body and sampled preview curves are byte-identical
between modes. Full assets live separately in `assets/models/full/`; previews
and research assets are unchanged.

Downloads are roughly 15–44 MB per animal, on demand only. Progress is shown;
switching density, animal, or returning to Studio cancels the current download.
Only one animal's geometry is retained by the viewer at a time, with GPU buffers
disposed on replacement/exit. Density switches preserve the current camera.
Desktop hardware is recommended; Preview remains available for slower devices.
Full density is a geometry setting, not Blender/Cycles hair shading or lighting.

To regenerate full assets (refuses to overwrite existing full exports):

```sh
blender -b /path/to/animal_showcase_scene.blend -t 8 \
  --python scripts/export_teaser_assets.py -- --mode full-models --output assets/models/full
```

The film is the existing **strong-gust, fixed-camera, all-strand fox simulation**:

```
Frosting/1_paper_vis/fox_wind_animation_3840x2160_60fps_allstrands_strongwind_static_v01/
  composite_fast_ffmpeg_0001_0720/fox_wind_stronggusts_static_allstrands_4k60_12sec_q1.mp4
```

The original MPEG-4 Part 2 file is transcoded to H.264 High/yuv420p for browser
compatibility (CRF 19, medium preset, fast-start). This is a lossy web encode,
not a byte-identical master. It preserves **3840 x 2160, 60 fps, 720 frames,
12 seconds** and fits GitHub's individual-file limit. No lower-resolution movie
is substituted. `assets/video/provenance.json` records probe data and hashes.
Some phones may not decode 4K60 smoothly; a direct download is always available.
The motion is downstream simulation, not motion inferred from the photographs.

## Tests And Screenshots

Validated 2026-09-28: production build and all 14 Chromium tests pass. Reviewed
desktop/mobile screenshots are in [output_screenshots/20260928/](output_screenshots/20260928/summary.md).
Full-density screenshots, exact download sizes and validation notes are in
[output_screenshots/full_density_20260928/](output_screenshots/full_density_20260928/summary.md).

```sh
PLAYWRIGHT_BROWSERS_PATH=.cache/browsers npx playwright install chromium
npm run build
npm test
```

Tests cover 320/390/768/1440px layouts, author timing and reduced motion, all
six tabs, geometry hashes/finite positions/indexing, click-to-load behavior,
interactive rendering/fallback, and actual 4K playback/seeking. Screenshots are
saved under `.build/screenshots/` and are deliberately excluded from git along
with browser caches, logs, render intermediates and test traces.

## Regenerate Assets (Optional Development Step)

Only needed when changing the source media. Use the existing `frosting` conda
environment with Pillow, NumPy and `imageio-ffmpeg`, and Blender 4.3.2.

```sh
python scripts/prepare_media.py --source /path/to/Frosting --stage images
python scripts/prepare_media.py --source /path/to/Frosting --stage fonts
python scripts/prepare_media.py --source /path/to/Frosting --stage video
blender -b /path/to/animal_showcase_scene.blend -t 16 \
  --python scripts/export_teaser_assets.py -- --mode models --output assets/models
blender -b /path/to/animal_showcase_scene.blend -t 16 \
  --python scripts/export_teaser_assets.py -- --mode renders --output .build/posters
python scripts/prepare_media.py --source /path/to/Frosting --stage posters
```

The video step refuses to overwrite its output. Archive/remove only the
generated web copy before deliberately regenerating it. Original assets are
never changed. Run isolated animals with `--animals fox` to shorten rendering.
The current portrait exporter uses CPU rendering because this session's NVIDIA
driver was unavailable. The approved hero/film already contain the high-quality
GPU-rendered work; they are reused, not rerendered at a lower quality.

To refresh the paper PDF, compile a copy of `../paper/` in `.build/paper/` with
two passes of `pdflatex -interaction=nonstopmode -halt-on-error main_iclr.tex`,
then place `main_iclr.pdf` at `assets/paper/FurE.pdf`. Nothing edits the source
LaTeX draft automatically.

## Publish

The repository includes `.github/workflows/website.yml`. It runs **only on manual
dispatch**; no page has been published or pushed by this implementation. After
review, commit the website source, lockfile and `assets/` (not `dist/`), select
GitHub Actions as the repository's Pages source, and run **Deploy FurE website**.
Only `website/dist/` is deployed. Paper sources and research data are not served.

For another static host, upload the contents of `dist/`. Keep MIME types correct
and enable HTTP range requests for the film. Fonts are self-hosted with their
OFL licenses; the build includes the Three.js license. No trackers or analytics
are included.

A VP9/WebM alternative is also packaged at the **same 4K60 resolution** for
browsers without H.264 decoding (CRF 28, quality-based encoding). The player selects a supported codec before
loading; it never silently substitutes a smaller video. Regenerate this export
with `python scripts/prepare_media.py --source /path/to/Frosting --stage webm`.
Its probe data and hash are recorded in `assets/video/webm-provenance.json`.
