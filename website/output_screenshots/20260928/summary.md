# FurE Website Visual Review

Website: `/mnt/data0/prakhar/fure/website/`.

- `desktop-hero.png`: paper title, verified authors and approved six-animal teaser.
- `desktop-full.png`: complete desktop page.
- `mobile-hero.png`: 390px phone layout; all six animals remain in frame.
- `mobile-full.png`: complete phone page.
- `interactive-fox.png`: actual Three.js viewer, with a labeled 30,000-strand sample and corresponding body.
- `film-playing.png`: browser-decoded 4K strong-wind film, paused at six seconds after verifying playback and seeking.

Author order is paused in layout screenshots because they use reduced-motion
settings. A separate test verifies the five-second swap and credit preservation.
Studio portraits retain every curve from the approved scene, not the smaller
browser preview. Teaser palettes are illustrative.

## Verification

Production build passed. All **11 Playwright/Chromium tests passed** on
2026-09-28, including automated WCAG A/AA checks, four viewport widths
(320/390/768/1440px), author behavior, keyboard tabs, all six portraits, finite
geometry and hashes, WebGL rendering and fallback, PDF/citation controls, and
actual video playback/range seeking. This is not a claim of manual screen-reader
testing or a full Safari/Firefox/device compatibility matrix.

Both movie codecs preserve 3840 x 2160, 60 fps, 720 frames and 12 seconds.
H.264 is the primary web export; VP9 is used when H.264 is unavailable. These are
high-quality web transcodes, not lossless masters. Native 4K decoding speed
depends on the visitor's hardware. The original movie remains unchanged.

No scientific experiments, metrics, paper sources or research models were
modified. No GitHub push or deployment was performed.
