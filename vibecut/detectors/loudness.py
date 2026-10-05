"""음량 급증 탐지: 짧은 창의 음량(dB)이 주변 평균 음량보다 threshold_db 이상 큰 구간."""

from __future__ import annotations

from typing import Any

import librosa
import numpy as np
from scipy.ndimage import median_filter

from vibecut.config import load_config
from vibecut.detectors.common import make_event, mask_to_intervals, save_events
from vibecut.media.ingest import AUDIO_FILE
from vibecut.storage.cache import cache_dir

NAME = "loudness"


def loudness_series(video_id: str, window_seconds: float) -> tuple[np.ndarray, np.ndarray]:
    """창 단위 음량(dB)과 각 창의 시작 시간(초)."""
    y, sr = librosa.load(str(cache_dir(video_id) / AUDIO_FILE), sr=None, mono=True)
    hop = max(1, int(window_seconds * sr))
    rms = librosa.feature.rms(y=y, frame_length=hop, hop_length=hop, center=False)[0]
    db = 20 * np.log10(np.maximum(rms, 1e-10))
    times = np.arange(len(db)) * hop / sr
    return db, times


def detect(video_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = {**load_config()["detectors"][NAME], **(params or {})}
    print(f"[{NAME}] 주변 {p['baseline_seconds']}초 평균보다 {p['threshold_db']}dB 이상 큰 구간")
    step = p["window_seconds"]
    db, times = loudness_series(video_id, step)
    if len(db) == 0:
        return save_events(video_id, NAME, p, [])

    baseline_len = max(1, int(p["baseline_seconds"] / step))
    baseline = median_filter(db, size=baseline_len, mode="nearest")
    diff = db - baseline
    runs = mask_to_intervals(diff >= p["threshold_db"], times, step,
                             merge_gap=p["merge_gap"], min_duration=p["min_duration"])

    events = []
    for a, b in runs:
        peak = a + int(np.argmax(diff[a:b]))
        start, end = float(times[a]), float(times[b - 1] + step)
        events.append(make_event(
            NAME, len(events) + 1, start, end, "loudness_spike",
            score=diff[peak] / (2 * p["threshold_db"]),
            reason=f"주변 평균보다 최대 {diff[peak]:.1f}dB 큰 소리가 {end - start:.1f}초 지속",
            data={"peak_db": round(float(db[peak]), 1),
                  "baseline_db": round(float(baseline[peak]), 1),
                  "max_diff_db": round(float(diff[peak]), 1)},
        ))
    return save_events(video_id, NAME, p, events)
