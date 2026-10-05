"""영상 입력 처리: video_id 계산, 영상 정보(meta.json), 음성 추출, 저해상도 분석용 사본.

원본 영상은 읽기만 하고 절대 수정하지 않는다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from vibecut.config import load_config
from vibecut.media.ffmpeg import probe, run_with_progress
from vibecut.storage.cache import cache_dir, compute_video_id, write_json

AUDIO_FILE = "audio.wav"
PROXY_FILE = "proxy.mp4"
META_FILE = "meta.json"


def _parse_fps(rate: str) -> float:
    num, _, den = rate.partition("/")
    return float(num) / float(den) if den and float(den) != 0 else float(num)


def read_meta(video_path: Path, video_id: str) -> dict[str, Any]:
    info = probe(video_path)
    video = next((s for s in info["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        raise ValueError(f"영상 스트림이 없습니다: {video_path}")
    if not any(s["codec_type"] == "audio" for s in info["streams"]):
        raise ValueError(f"음성 스트림이 없습니다: {video_path}")
    return {
        "video_id": video_id,
        "source_path": str(video_path.resolve()),
        "duration": round(float(info["format"]["duration"]), 3),
        "fps": round(_parse_fps(video["avg_frame_rate"]), 3),
        "width": int(video["width"]),
        "height": int(video["height"]),
    }


def ingest(video_path: Path) -> dict[str, Any]:
    """영상을 등록하고 캐시 폴더에 meta.json, audio.wav, proxy.mp4를 만든다. 이미 있으면 건너뛴다."""
    cfg = load_config()["media"]
    video_path = video_path.resolve()
    print(f"[ingest] {video_path.name}")
    video_id = compute_video_id(video_path)
    out = cache_dir(video_id)
    print(f"  video_id: {video_id}  ->  {out}")

    # ffprobe는 빠르므로 매번 다시 읽는다 (원본 경로가 바뀌었을 수 있음)
    meta = read_meta(video_path, video_id)
    write_json(out / META_FILE, meta)
    print(f"  길이 {meta['duration']:.1f}초, {meta['width']}x{meta['height']}, {meta['fps']}fps")

    audio_path = out / AUDIO_FILE
    if audio_path.exists():
        print(f"  {AUDIO_FILE}: 캐시 사용")
    else:
        tmp = audio_path.with_suffix(".tmp.wav")
        run_with_progress(
            ["-i", str(video_path), "-vn", "-ac", "1", "-ar", str(cfg["audio_sample_rate"]),
             "-c:a", "pcm_s16le", str(tmp)],
            meta["duration"], "음성 추출",
        )
        tmp.replace(audio_path)

    proxy_path = out / PROXY_FILE
    if proxy_path.exists():
        print(f"  {PROXY_FILE}: 캐시 사용")
    else:
        tmp = proxy_path.with_suffix(".tmp.mp4")
        run_with_progress(
            ["-i", str(video_path), "-an", "-vf", f"scale=-2:{cfg['analysis_height']}",
             "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", str(tmp)],
            meta["duration"], "분석용 사본",
        )
        tmp.replace(proxy_path)

    return meta
