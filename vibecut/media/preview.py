"""구간 미리보기: 원본에서 start~end를 잘라 저해상도 미리보기 영상(음성 포함)을 만든다. 원본은 수정하지 않는다."""

from __future__ import annotations

from pathlib import Path

from vibecut.config import load_config
from vibecut.media.ffmpeg import run_with_progress
from vibecut.media.ingest import META_FILE
from vibecut.storage.cache import cache_dir, read_json

PREVIEWS_DIR = "previews"


def make_preview(video_id: str, start: float, end: float) -> Path:
    out = cache_dir(video_id) / PREVIEWS_DIR / f"{start:.2f}_{end:.2f}.mp4"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = read_json(cache_dir(video_id) / META_FILE)
    height = load_config()["media"]["analysis_height"]
    tmp = out.with_suffix(".tmp.mp4")
    run_with_progress(
        ["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", meta["source_path"],
         "-vf", f"scale=-2:{height}", "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
         "-c:a", "aac", "-b:a", "128k", str(tmp)],
        end - start, "미리보기",
    )
    tmp.replace(out)
    return out
