import { build } from "esbuild";
import { cp, mkdir, writeFile, stat } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import path from "node:path";

const root = fileURLToPath(new URL("..", import.meta.url));
const out = path.join(root, "dist");
await mkdir(out, { recursive: true });
const required = [
  "assets/images/teaser-2560.webp",
  "assets/images/teaser-1280.webp",
  "assets/images/method.webp",
  "assets/images/thickness.webp",
  "assets/images/bison-capture.webp",
  "assets/images/fox-wind-poster.webp",
  "assets/fonts/InstrumentSerif-Regular.ttf",
  "assets/fonts/InstrumentSerif-Italic.ttf",
  "assets/fonts/DMSans-Variable.ttf",
  "assets/video/fox-strong-wind-4k60.mp4",
  "assets/video/fox-strong-wind-4k60.webm",
  "assets/paper/FurE.pdf",
];
for (const animal of ["cat", "fox", "tiger", "beagle", "panda", "bison"])
  required.push(
    `assets/images/${animal}.webp`,
    `assets/models/${animal}.bin`,
    `assets/models/${animal}.json`,
    `assets/models/full/${animal}.bin`,
    `assets/models/full/${animal}.json`,
  );
for (const file of required) {
  try {
    const info = await stat(path.join(root, file));
    if (!info.isFile() || info.size === 0) throw new Error("Empty asset");
  } catch {
    throw new Error(
      `Missing website asset: ${file}. See README.md for media preparation.`,
    );
  }
}
for (const name of [
  "index.html",
  "styles.css",
  "favicon.svg",
  "config.js",
  "app.js",
  "assets",
])
  await cp(path.join(root, name), path.join(out, name), { recursive: true });
await build({
  entryPoints: [path.join(root, "viewer.js")],
  bundle: true,
  format: "esm",
  minify: true,
  target: "es2022",
  outfile: path.join(out, "viewer.js"),
  legalComments: "linked",
});
await writeFile(path.join(out, ".nojekyll"), "");
// Keep third-party licenses with the deployable bundle.
await cp(
  path.join(root, "node_modules/three/LICENSE"),
  path.join(out, "THREE-LICENSE.txt"),
);
console.log(`Static site built: ${out}`);
