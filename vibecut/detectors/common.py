"""탐지기 공통: 이벤트(CLAUDE.md 6장 ②) 생성, 구간 병합, 저장·조회."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import numpy as np

from vibecut.storage.cache import cache_dir, read_json, write_json

EVENTS_DIR = "events"
MIN_LEN = 0.001


def events_path(video_id: str, detector: str) -> Path:
    return cache_dir(video_id) / EVENTS_DIR / f"{detector}.json"


def make_event(detector: str, index: int, start: float, end: float, type_: str,
               score: float, reason: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    start = round(start, 3)
    end = round(max(end, start + MIN_LEN), 3)
    return {
        "id": f"{detector}_{index:04d}",
        "start": start,
        "end": end,
        "type": type_,
        "score": round(float(np.clip(score, 0.0, 1.0)), 3),
        "reason": reason,
        "source": detector,
        "data": data or {},
    }


def save_events(video_id: str, detector: str, params: dict[str, Any],
                events: list[dict[str, Any]]) -> dict[str, Any]:
    doc = {"video_id": video_id, "detector": detector, "params": params, "events": events}
    write_json(events_path(video_id, detector), doc)
    print(f"  {detector}: 이벤트 {len(events)}개 -> {EVENTS_DIR}/{detector}.json")
    return doc


def load_events(video_id: str, detector: str) -> dict[str, Any]:
    return read_json(events_path(video_id, detector))


def mask_to_intervals(mask: np.ndarray, times: np.ndarray, step: float,
                      merge_gap: float = 0.0, min_duration: float = 0.0) -> list[tuple[int, int]]:
    """True가 이어진 구간을 (시작 인덱스, 끝 인덱스 exclusive)로 돌려준다.

    merge_gap초 이내로 떨어진 구간은 합치고, min_duration초보다 짧은 구간은 버린다.
    times[i]는 i번째 창의 시작 시간, step은 창 간격(초).
    """
    runs: list[list[int]] = []
    i, n = 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if runs and (times[i] - (times[runs[-1][1] - 1] + step)) <= merge_gap:
                runs[-1][1] = j
            else:
                runs.append([i, j])
            i = j
        else:
            i += 1
    return [(a, b) for a, b in runs if (b - a) * step >= min_duration]


def merge_intervals(items: Iterable[tuple[float, float]], gap: float = 0.0) -> list[list[float]]:
    out: list[list[float]] = []
    for s, e in sorted(items):
        if out and s - out[-1][1] <= gap:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out
