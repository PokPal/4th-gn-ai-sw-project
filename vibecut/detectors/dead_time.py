"""지루한 구간 탐지: 말이 없고, 소리가 평소보다 크지 않고, 화면 변화가 거의 없는 구간 (로딩, 대기, 메뉴)."""

from __future__ import annotations

from typing import Any

import numpy as np

from vibecut.config import load_config
from vibecut.detectors.common import make_event, mask_to_intervals, save_events
from vibecut.detectors.loudness import loudness_series
from vibecut.detectors.visual import visual_series
from vibecut.storage.cache import cache_dir, read_json
from vibecut.stt.run import TRANSCRIPT_FILE

NAME = "dead_time"


def detect(video_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg = load_config()["detectors"]
    p = {**cfg[NAME], **(params or {})}
    sample_fps = cfg["visual"]["sample_fps"]
    step = 1.0 / sample_fps
    print(f"[{NAME}] 말 없음 + 조용함 + 화면 변화 {p['max_motion']} 미만이 {p['min_duration']}초 이상")

    times, _, motion = visual_series(video_id, sample_fps)
    if len(times) == 0:
        return save_events(video_id, NAME, p, [])

    # 음량을 화면 샘플과 같은 간격으로 계산해 맞춘다
    db, db_times = loudness_series(video_id, step)
    db_at = np.interp(times, db_times, db) if len(db) else np.full(len(times), -100.0)
    quiet_level = float(np.median(db)) + p["quiet_margin_db"] if len(db) else 0.0

    speech = np.zeros(len(times), dtype=bool)
    for seg in read_json(cache_dir(video_id) / TRANSCRIPT_FILE)["segments"]:
        speech |= (times + step > seg["start"]) & (times < seg["end"])

    mask = (~speech) & (db_at <= quiet_level) & (motion < p["max_motion"])
    events = []
    for a, b in mask_to_intervals(mask, times, step, min_duration=p["min_duration"]):
        start, end = float(times[a]), float(times[b - 1]) + step
        mean_motion = float(motion[a:b].mean())
        events.append(make_event(
            NAME, len(events) + 1, start, end, "dead_time",
            score=1.0 - mean_motion / p["max_motion"],
            reason=f"{end - start:.1f}초 동안 말 없음, 음량 평소 이하, 화면 변화 평균 {mean_motion:.3f}",
            data={"mean_motion": round(mean_motion, 4),
                  "mean_db": round(float(db_at[a:b].mean()), 1),
                  "quiet_level_db": round(quiet_level, 1)},
        ))
    return save_events(video_id, NAME, p, events)
