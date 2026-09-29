"""Package existing research media; no generated animals or changed scientific data."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import urllib.request

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True, help="Frosting repository")
    parser.add_argument("--stage", choices=("images", "video", "webm", "posters", "fonts"), required=True)
    args = parser.parse_args()
    site = Path(__file__).resolve().parents[1]
    root = args.source.resolve()
    out = site / "assets"
    if args.stage == "images":
        source = root / "1_paper_vis/0_teaser/final_fur_focus_upclose_balanced_v03/derived/teaser_luxury_web_2560.png"
        original = Image.open(source).convert("RGB")
        for width in (1280, 2560):
            image = original.resize((width, round(original.height*width/original.width)), Image.Resampling.LANCZOS)
            image.save(out / "images" / f"teaser-{width}.webp", quality=94, method=6)
        shutil.copy2(root / "1_paper_vis/0_teaser/final_fur_focus_upclose_balanced_v03/render_config.json", out / "teaser-provenance.json")
        for filename, output in (("FurE.pdf", "method"), ("frosting_thickness_v2.pdf", "thickness"),
                                 ("Bison_results_input_v2.pdf", "bison-capture")):
            pdf = site.parent / "paper/figures" / filename
            stem = site / ".build" / output
            subprocess.run(["pdftoppm", "-f", "1", "-singlefile", "-scale-to", "2200", "-png", str(pdf), str(stem)], check=True)
            Image.open(stem.with_suffix(".png")).convert("RGB").save(out / "images" / f"{output}.webp", quality=94, method=6)
        (out / "media-provenance.json").write_text(json.dumps({"hero": str(source),
            "hero_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "paper_source": str(site.parent / "paper"), "source_strands": 1264993,
            "presentation": "Approved legacy illustration palettes, not recovered appearance or evaluation renders.",
            "model_previews": "30000 existing strands per animal, click-to-load; full source density in studio images and film."}, indent=2))
    elif args.stage in ("video", "webm"):
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        source = root / "1_paper_vis/fox_wind_animation_3840x2160_60fps_allstrands_strongwind_static_v01/composite_fast_ffmpeg_0001_0720/fox_wind_stronggusts_static_allstrands_4k60_12sec_q1.mp4"
        webm = args.stage == "webm"
        target = out / ("video/fox-strong-wind-4k60.webm" if webm else "video/fox-strong-wind-4k60.mp4")
        codec = (["-c:v", "libvpx-vp9", "-crf", "28", "-b:v", "0", "-deadline", "good", "-cpu-used", "4", "-row-mt", "1"]
                 if webm else ["-c:v", "libx264", "-crf", "19", "-preset", "medium", "-movflags", "+faststart"])
        subprocess.run([ffmpeg, "-n", "-i", str(source), *codec,
                        "-threads", "12", "-pix_fmt", "yuv420p", "-an", str(target)], check=True)
        if not (site / ".build/fox-wind-poster.png").exists():
            subprocess.run([ffmpeg, "-n", "-ss", "3", "-i", str(source), "-frames:v", "1", str(site / ".build/fox-wind-poster.png")], check=True)
        Image.open(site / ".build/fox-wind-poster.png").convert("RGB").resize((1920,1080), Image.Resampling.LANCZOS).save(out / "images/fox-wind-poster.webp", quality=95, method=6)
        probe = root / ".conda-envs/frosting/bin/ffprobe"
        metadata = json.loads(subprocess.check_output([str(probe), "-v", "error", "-count_frames", "-show_streams", "-show_format", "-of", "json", str(target)], text=True))
        video = metadata["streams"][0]
        assert video["width"] == 3840 and video["height"] == 2160 and video["avg_frame_rate"] == "60/1"
        assert abs(float(metadata["format"]["duration"])-12) < .05 and int(video["nb_read_frames"]) == 720
        assert target.stat().st_size < 100 * 1024**2, "Video exceeds GitHub's individual-file limit"
        metadata.update(source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                        export_sha256=hashlib.sha256(target.read_bytes()).hexdigest(), source_strands=265000,
                        provenance="Existing root-pinned wind simulation, not motion reconstructed from input views.")
        (out / "video" / ("webm-provenance.json" if webm else "provenance.json")).write_text(json.dumps(metadata, indent=2))
    elif args.stage == "posters":
        for source in (site / ".build/posters").glob("*.png"):
            image = Image.open(source).convert("RGB")
            image.save(out / "images" / f"{source.stem}.webp", quality=94, method=6)
            image.resize((640, 400), Image.Resampling.LANCZOS).save(out / "images" / f"{source.stem}-thumb.webp", quality=88, method=6)
    else:
        for family, files in {"instrumentserif": ["InstrumentSerif-Regular.ttf", "InstrumentSerif-Italic.ttf", "OFL.txt"],
                              "dmsans": ["DMSans[opsz,wght].ttf", "OFL.txt"]}.items():
            for filename in files:
                url = "https://raw.githubusercontent.com/google/fonts/main/ofl/"+family+"/"+urllib.parse.quote(filename)
                target = out / "fonts" / (family+"-OFL.txt" if filename == "OFL.txt" else filename.replace("[opsz,wght]", "-Variable"))
                urllib.request.urlretrieve(url, target)


if __name__ == "__main__":
    main()
