import { test, expect } from "@playwright/test";
import { mkdir, readFile } from "node:fs/promises";
import { site } from "../config.js";
import AxeBuilder from "@axe-core/playwright";

test("page passes automated WCAG accessibility checks", async ({ page }) => {
  // Audit settled colors, not intermediate opacity during entrance animations.
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  const audit = await new AxeBuilder({ page })
    .withTags(["wcag2a", "wcag2aa", "wcag21aa"])
    .analyze();
  expect(
    audit.violations.map(({ id, nodes }) => ({
      id,
      targets: nodes.map((node) => node.target),
    })),
  ).toEqual([]);
});

test("paper, citation and protected static serving work", async ({
  page,
  context,
  request,
}) => {
  const paper = await request.head("/assets/paper/FurE.pdf");
  expect(paper.status()).toBe(200);
  expect(paper.headers()["content-type"]).toBe("application/pdf");
  expect((await request.get("/../package.json")).status()).toBe(404);
  expect(
    (
      await request.get("/assets/video/fox-strong-wind-4k60.mp4", {
        headers: { Range: "bytes=9999999999-" },
      })
    ).status(),
  ).toBe(416);
  await context.grantPermissions(["clipboard-read", "clipboard-write"]);
  await page.goto("/");
  await page.locator("#copy-bibtex").click();
  await expect(page.locator("#copy-status")).toHaveText(
    "Placeholder BibTeX copied to the clipboard.",
  );
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain(
    "year   = {TODO}",
  );
});

test("responsive layout, accessible labels, real media and no automatic heavy downloads", async ({
  page,
}) => {
  const requests = [],
    errors = [];
  page.on("request", (request) => requests.push(request.url()));
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await expect(page).toHaveTitle(/FurE/);
  await expect(page.getByRole("heading", { level: 1 })).toContainText(
    "without animal-fur datasets",
  );
  await page.evaluate(() => document.fonts.ready);
  for (const width of [320, 390, 768, 1440]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect
      .poll(() =>
        page.evaluate(() => document.documentElement.scrollWidth <= innerWidth),
      )
      .toBe(true);
    const image = page.locator(".hero-figure > img");
    await expect(image).toBeVisible();
    expect(await image.evaluate((el) => el.naturalWidth)).toBeGreaterThan(0);
    const imageBox = await image.boundingBox();
    expect(imageBox.x).toBeGreaterThanOrEqual(0);
    expect(imageBox.x + imageBox.width).toBeLessThanOrEqual(width + 1);
  }
  expect(
    requests.some(
      (url) =>
        url.endsWith(".mp4") ||
        url.endsWith(".webm") ||
        url.endsWith(".bin") ||
        url.endsWith("/viewer.js"),
    ),
  ).toBe(false);
  expect(errors).toEqual([]);
});

test("co-first authors alternate every five seconds with credits and pause intact", async ({
  page,
}) => {
  await page.clock.install();
  await page.goto("/");
  const first = page.locator("#first-authors a").first();
  await expect(first).toContainText("Srinjay Sarkar");
  await page.clock.runFor(5100);
  await expect(first).toContainText("Prakhar Kaushik");
  await expect(first).toHaveAccessibleName(
    "Prakhar Kaushik, equal contribution, project lead",
  );
  await page.locator("#author-pause").click();
  await page.clock.runFor(10000);
  await expect(first).toContainText("Prakhar Kaushik");
  await page.locator("#author-pause").click();
  await page.clock.runFor(5000);
  await expect(first).toContainText("Srinjay Sarkar");
});

test("reduced motion pauses rotation", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await expect(page.locator("#author-pause")).toHaveAttribute(
    "aria-pressed",
    "true",
  );
});

test("all six full-density portraits and keyboard tabs work", async ({
  page,
}) => {
  await page.goto("/");
  for (const animal of site.animals) {
    await page.locator(`#tab-${animal.id}`).click();
    await expect(page.locator("#animal-name")).toHaveText(animal.name);
    await expect(page.locator("#strand-count")).toHaveText(
      animal.strands.toLocaleString("en-US"),
    );
    await expect
      .poll(() =>
        page
          .locator("#animal-image")
          .evaluate((el) => el.complete && el.naturalWidth),
      )
      .toBe(1600);
  }
  await page.locator("#tab-bison").press("ArrowRight");
  await expect(page.locator("#tab-cat")).toBeFocused();
  await expect(page.locator("#tab-cat")).toHaveAttribute(
    "aria-selected",
    "true",
  );
});

test("geometry export is finite, indexed correctly and hash-verified", async () => {
  const { createHash } = await import("node:crypto");
  for (const animal of site.animals) {
    const metadata = JSON.parse(
      await readFile(`assets/models/${animal.id}.json`),
    );
    const bytes = await readFile(`assets/models/${animal.id}.bin`);
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(
      metadata.sha256,
    );
    expect(bytes.byteLength).toBe(metadata.bytes);
    expect(metadata.sourceStrands).toBe(animal.strands);
    expect(metadata.previewStrands).toBe(30000);
    const buffer = bytes.buffer.slice(
      bytes.byteOffset,
      bytes.byteOffset + bytes.byteLength,
    );
    for (const key of ["bodyPositions", "strandPositions"]) {
      const spec = metadata.layout[key];
      const values = new Float32Array(buffer, spec.offset, spec.length);
      expect(values.every(Number.isFinite)).toBe(true);
      expect(spec.length % 3).toBe(0);
    }
    const indices = metadata.layout.bodyIndices;
    expect(
      new Uint32Array(buffer, indices.offset, indices.length).every(
        (index) => index < metadata.layout.bodyPositions.length / 3,
      ),
    ).toBe(true);
    expect(metadata.layout.strandPositions.length).toBe(
      metadata.previewStrands * metadata.pointsPerStrand * 3,
    );
  }
});

test("interactive geometry loads on demand and switches animals", async ({
  page,
}) => {
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  await page.goto("/");
  await page.locator("#interactive-mode").click();
  await expect(page.locator("#viewer-note")).toContainText(
    "30,000 sampled strands",
    { timeout: 30000 },
  );
  await expect(page.locator("#webgl-viewer canvas")).toBeVisible();
  await page.locator("#show-body").click();
  await expect(page.locator("#show-body")).toHaveAttribute(
    "aria-pressed",
    "false",
  );
  await page.locator("#show-body").click();
  await page.locator("#tab-fox").click();
  await expect(page.locator("#viewer-loading")).toBeHidden({ timeout: 30000 });
  await page.locator("#reset-view").click();
  await mkdir(".build/screenshots", { recursive: true });
  await page
    .locator("#collection")
    .screenshot({ path: ".build/screenshots/interactive-fox.png" });
  await page.locator("#studio-mode").click();
  await expect(page.locator("#animal-image")).toBeVisible();
  expect(errors).toEqual([]);
});

test("failed 3D download keeps the full-density studio view usable", async ({
  page,
}) => {
  await page.route("**/assets/models/*.bin", (route) => route.abort());
  await page.goto("/");
  await page.locator("#interactive-mode").click();
  await expect(page.locator("#viewer-note")).toContainText("3D unavailable", {
    timeout: 30000,
  });
  await expect(page.locator("#animal-image")).toBeVisible();
});

test("full-density assets retain every source strand and the identical preview body", async () => {
  const { createHash } = await import("node:crypto");
  for (const animal of site.animals) {
    const metadata = JSON.parse(
      await readFile(`assets/models/full/${animal.id}.json`),
    );
    const preview = JSON.parse(
      await readFile(`assets/models/${animal.id}.json`),
    );
    const bytes = await readFile(metadata.url);
    const previewBytes = await readFile(preview.url);
    expect(bytes.length).toBe(metadata.bytes);
    expect(createHash("sha256").update(bytes).digest("hex")).toBe(
      metadata.sha256,
    );
    expect(metadata.strandCount).toBe(animal.strands);
    expect(metadata.strandCount).toBe(metadata.sourceStrands);
    expect(metadata.layout.strandPositions.length).toBe(
      animal.strands * 12 * 3,
    );
    expect(metadata.previewCorrespondence).toContain("byte-identical");
    for (const key of ["bodyPositions", "bodyIndices"]) {
      const a = metadata.layout[key],
        b = preview.layout[key];
      expect(
        bytes
          .subarray(a.offset, a.offset + a.length * 4)
          .equals(previewBytes.subarray(b.offset, b.offset + b.length * 4)),
      ).toBe(true);
    }
    const spec = metadata.layout.strandPositions;
    const positions = new Float32Array(
      bytes.buffer,
      bytes.byteOffset + spec.offset,
      spec.length,
    );
    expect(positions.every(Number.isFinite)).toBe(true);
    for (let axis = 0; axis < 3; axis++) {
      expect(
        positions.every(
          (v, i) =>
            i % 3 !== axis ||
            (v >= metadata.bounds[0][axis] && v <= metadata.bounds[1][axis]),
        ),
      ).toBe(true);
    }
  }
});

test("all six animals load full density only on request and release geometry on exit", async ({
  page,
}) => {
  test.setTimeout(180000);
  const errors = [],
    requests = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("request", (request) => requests.push(request.url()));
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.locator("#interactive-mode").click();
  const viewer = page.locator("#webgl-viewer");
  await expect(viewer).toHaveAttribute("data-density", "preview", {
    timeout: 30000,
  });
  expect(requests.some((url) => url.includes("/models/full/"))).toBe(false);
  await page.locator("#full-density").click();
  await mkdir(".build/screenshots/full-density", { recursive: true });
  const timings = [];
  for (const animal of site.animals) {
    const start = Date.now();
    if (animal.id !== "cat") await page.locator(`#tab-${animal.id}`).click();
    await expect(viewer).toHaveAttribute("data-loaded-animal", animal.id, {
      timeout: 30000,
    });
    await expect(viewer).toHaveAttribute("data-density", "full");
    await expect(viewer).toHaveAttribute(
      "data-strand-count",
      String(animal.strands),
    );
    await expect(page.locator("#viewer-note")).toContainText(
      `${animal.strands.toLocaleString()} original strands`,
    );
    timings.push({
      animal: animal.id,
      localLoadAndRenderMs: Date.now() - start,
    });
    await page
      .locator("#collection")
      .screenshot({ path: `.build/screenshots/full-density/${animal.id}.png` });
  }
  await page.locator("#show-strands").click();
  await page
    .locator("#collection")
    .screenshot({
      path: ".build/screenshots/full-density/bison-body-only.png",
    });
  await page.locator("#preview-density").click();
  await expect(viewer).toHaveAttribute("data-strand-count", "30000", {
    timeout: 30000,
  });
  await page.setViewportSize({ width: 390, height: 900 });
  await expect(page.locator("#full-density")).toBeVisible();
  expect(
    await page.evaluate(
      () => document.documentElement.scrollWidth <= innerWidth,
    ),
  ).toBe(true);
  await page.locator("#studio-mode").click();
  await expect(viewer).not.toHaveAttribute("data-loaded-animal");
  expect(errors).toEqual([]);
  const { writeFile } = await import("node:fs/promises");
  await writeFile(
    ".build/screenshots/full-density/timings.json",
    JSON.stringify(timings, null, 2),
  );
});

test("a pending full-density request can be cancelled without stale geometry", async ({
  page,
}) => {
  let release;
  const hold = new Promise((resolve) => {
    release = resolve;
  });
  await page.route("**/models/full/cat.bin", async (route) => {
    await hold;
    await route.abort();
  });
  await page.goto("/");
  await page.locator("#interactive-mode").click();
  await expect(page.locator("#webgl-viewer")).toHaveAttribute(
    "data-density",
    "preview",
  );
  const pending = page.waitForRequest("**/models/full/cat.bin");
  await page.locator("#full-density").click();
  await pending;
  await page.locator("#preview-density").click();
  release();
  await expect(page.locator("#webgl-viewer")).toHaveAttribute(
    "data-density",
    "preview",
    { timeout: 30000 },
  );
  await expect(page.locator("#viewer-note")).toContainText(
    "30,000 sampled strands",
  );
});

test("native 4K film plays and local server supports seeking", async ({
  page,
  request,
}) => {
  const range = await request.get("/assets/video/fox-strong-wind-4k60.mp4", {
    headers: { Range: "bytes=0-1023" },
  });
  expect(range.status()).toBe(206);
  expect((await range.body()).length).toBe(1024);
  await page.goto("/");
  await page.locator("#play-film").click();
  await expect
    .poll(() => page.locator("video").evaluate((video) => video.currentTime), {
      timeout: 30000,
    })
    .toBeGreaterThan(0.05);
  expect(
    await page
      .locator("video")
      .evaluate((video) => [
        video.videoWidth,
        video.videoHeight,
        video.duration,
      ]),
  ).toEqual([3840, 2160, 12]);
  await page.locator("video").evaluate((video) => {
    video.pause();
    video.currentTime = 6;
  });
  await expect
    .poll(() => page.locator("video").evaluate((video) => video.currentTime))
    .toBe(6);
  const metadata = JSON.parse(await readFile("assets/video/provenance.json"));
  expect(metadata.streams[0].avg_frame_rate).toBe("60/1");
  expect(metadata.streams[0].nb_frames).toBe("720");
  const fallback = JSON.parse(
    await readFile("assets/video/webm-provenance.json"),
  );
  expect(fallback.streams[0].avg_frame_rate).toBe("60/1");
  expect(fallback.streams[0].nb_read_frames).toBe("720");
  await page
    .locator("#motion")
    .screenshot({ path: ".build/screenshots/film-playing.png" });
});

test("capture desktop and mobile screenshots", async ({ page }) => {
  await page.emulateMedia({ reducedMotion: "reduce" });
  await page.goto("/");
  await page.evaluate(() => document.fonts.ready);
  await mkdir(".build/screenshots", { recursive: true });
  await page.screenshot({ path: ".build/screenshots/desktop-hero.png" });
  await page.locator("footer").scrollIntoViewIfNeeded();
  await page.evaluate(() => scrollTo(0, 0));
  await page.screenshot({
    path: ".build/screenshots/desktop-full.png",
    fullPage: true,
  });
  await page.setViewportSize({ width: 390, height: 844 });
  await page.screenshot({ path: ".build/screenshots/mobile-hero.png" });
  await page.screenshot({
    path: ".build/screenshots/mobile-full.png",
    fullPage: true,
  });
});
