"""무음 탐지: ffmpeg silencedetect."""

from __future__ import annotations

import re
from typing import Any

from vibecut.config import load_config
from vibecut.detectors.common import make_event, save_events
from vibecut.media.ffmpeg import run_filter_log
from vibecut.media.ingest import AUDIO_FILE, META_FILE
from vibecut.storage.cache import cache_dir, read_json

NAME = "silence"


def detect(video_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = {**load_config()["detectors"][NAME], **(params or {})}
    out = cache_dir(video_id)
    duration = read_json(out / META_FILE)["duration"]
    print(f"[{NAME}] 기준 {p['noise_db']}dB, 최소 {p['min_duration']}초")
    log = run_filter_log(["-i", str(out / AUDIO_FILE),
                          "-af", f"silencedetect=noise={p['noise_db']}dB:d={p['min_duration']}"])

    starts = [float(x) for x in re.findall(r"silence_start: (-?[\d.]+)", log)]
    ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", log)]
    if len(ends) < len(starts):  # 파일 끝까지 무음이면 silence_end가 없음
        ends.append(duration)

    events = []
    for s, e in zip(starts, ends):
        s = max(0.0, s)
        length = e - s
        events.append(make_event(
            NAME, len(events) + 1, s, e, "silence",
            score=length / (p["min_duration"] * 5),
            reason=f"{length:.1f}초 동안 {p['noise_db']:.0f}dB 미만",
            data={"duration": round(length, 3)},
        ))
    return save_events(video_id, NAME, p, events)
