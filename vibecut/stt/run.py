"""음성 인식 실행과 캐싱. 같은 영상은 API를 다시 호출하지 않는다."""

from __future__ import annotations

from typing import Any

from vibecut.config import load_config
from vibecut.media.ingest import AUDIO_FILE, META_FILE
from vibecut.storage.cache import cache_dir, read_json, write_json
from vibecut.stt.base import SttProvider

TRANSCRIPT_FILE = "transcript.json"


def get_provider(cfg: dict[str, Any]) -> SttProvider:
    if cfg["provider"] == "openai-whisper":
        from vibecut.stt.openai_whisper import OpenAIWhisper

        return OpenAIWhisper(cfg)
    raise ValueError(f"지원하지 않는 음성 인식 제공사: {cfg['provider']}")


def transcribe(video_id: str, force: bool = False) -> dict[str, Any]:
    out = cache_dir(video_id)
    path = out / TRANSCRIPT_FILE
    if path.exists() and not force:
        print(f"[transcribe] {TRANSCRIPT_FILE}: 캐시 사용 (다시 하려면 --force)")
        return read_json(path)

    cfg = load_config()["stt"]
    meta = read_json(out / META_FILE)
    provider = get_provider(cfg)
    print(f"[transcribe] {video_id} ({meta['duration']:.1f}초) - {provider.name}")
    segments = provider.transcribe(out / AUDIO_FILE, meta["duration"])
    transcript = {
        "video_id": video_id,
        "provider": provider.name,
        "language": cfg["language"],
        "segments": segments,
    }
    write_json(path, transcript)
    return transcript
