/** Small local static server with byte ranges so native-4K video can seek. */
import http from "node:http";
import { stat, realpath } from "node:fs/promises";
import { createReadStream } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(fileURLToPath(new URL("../dist/", import.meta.url)));
const port = Number(process.env.PORT || 4173);
const host = process.env.HOST || "127.0.0.1";
const mime = {
  ".html": "text/html; charset=utf-8",
  ".css": "text/css",
  ".js": "text/javascript",
  ".json": "application/json",
  ".webp": "image/webp",
  ".png": "image/png",
  ".svg": "image/svg+xml",
  ".mp4": "video/mp4",
  ".webm": "video/webm",
  ".pdf": "application/pdf",
  ".ttf": "font/ttf",
  ".bin": "application/octet-stream",
  ".txt": "text/plain",
};
const server = http.createServer(async (req, res) => {
  try {
    if (!["GET", "HEAD"].includes(req.method)) {
      res.writeHead(405, { Allow: "GET, HEAD" }).end();
      return;
    }
    let url = decodeURIComponent(new URL(req.url, "http://localhost").pathname);
    if (url.endsWith("/")) url += "index.html";
    const filename = path.resolve(root, "." + url);
    if (!filename.startsWith(root + path.sep)) {
      res.writeHead(403).end();
      return;
    }
    const resolved = await realpath(filename);
    if (!resolved.startsWith(root + path.sep)) {
      res.writeHead(403).end();
      return;
    }
    const info = await stat(resolved);
    if (!info.isFile()) {
      res.writeHead(404).end();
      return;
    }
    const headers = {
      "Content-Type":
        mime[path.extname(resolved)] || "application/octet-stream",
      "Accept-Ranges": "bytes",
      "X-Content-Type-Options": "nosniff",
    };
    let start = 0,
      end = info.size - 1,
      status = 200;
    if (req.headers.range) {
      const match = /^bytes=(\d*)-(\d*)$/.exec(req.headers.range);
      if (!match || (!match[1] && !match[2])) {
        res.writeHead(416, { "Content-Range": `bytes */${info.size}` }).end();
        return;
      }
      if (!match[1]) start = Math.max(0, info.size - Number(match[2]));
      else {
        start = Number(match[1]);
        if (match[2]) end = Math.min(Number(match[2]), end);
      }
      if (start > end || start >= info.size) {
        res.writeHead(416, { "Content-Range": `bytes */${info.size}` }).end();
        return;
      }
      status = 206;
      headers["Content-Range"] = `bytes ${start}-${end}/${info.size}`;
    }
    headers["Content-Length"] = end - start + 1;
    res.writeHead(status, headers);
    if (req.method === "HEAD") res.end();
    else {
      const stream = createReadStream(resolved, { start, end });
      stream.on("error", () => res.destroy());
      stream.pipe(res);
    }
  } catch {
    if (!res.headersSent) res.writeHead(404);
    res.end("Not found");
  }
});
server.listen(port, host, () =>
  console.log(`FurE website: http://${host}:${port}`),
);
