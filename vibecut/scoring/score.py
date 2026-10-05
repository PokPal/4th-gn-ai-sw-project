"""하이라이트 점수화: 탐지기 이벤트를 시간으로 묶어 후보를 만들고 가중치로 점수를 매긴다.

결과는 이벤트 형식(CLAUDE.md 6장 ②)으로 candidates.json에 저장한다 (type: highlight, source: scoring).
"""

from __future__ import annotations

from typing import Any

from vibecut.config import load_config
from vibecut.detectors.common import events_path, make_event, merge_intervals
from vibecut.media.ingest import META_FILE
from vibecut.storage.cache import cache_dir, read_json, write_json

NAME = "scoring"
CANDIDATES_FILE = "candidates.json"
SIGNAL_DETECTORS = ["loudness", "reaction", "visual"]
TYPE_LABELS = {
    "loudness_spike": "음량 급증",
    "reaction": "반응 표현",
    "scene_change": "장면 전환",
    "brightness_change": "밝기 급변",
    "motion": "큰 움직임",
    "desaturation": "흑백 전환(사망 추정)",
}


def _split_long(start: float, end: float, padded: list[tuple[float, float]],
                max_duration: float) -> list[tuple[float, float]]:
    """신호가 이어 붙어 max_duration보다 길어진 후보를, 신호 사이 가장 큰 틈에서 재귀적으로 나눈다."""
    if end - start <= max_duration:
        return [(start, end)]
    inside = sorted((s, e) for s, e in padded if s >= start and e <= end)
    best, cut = 0.0, None
    reach = inside[0][1] if inside else end
    for s, e in inside[1:]:
        if s - reach > best or cut is None:  # 앞 신호들이 끝난 지점과 다음 신호 시작 사이의 틈
            best, cut = s - reach, (reach, s)
        reach = max(reach, e)
    if cut is None:
        return [(start, end)]
    mid = (cut[0] + cut[1]) / 2 if cut[1] > cut[0] else cut[1]
    if mid <= start or mid >= end:
        return [(start, end)]
    return _split_long(start, mid, padded, max_duration) + _split_long(mid, end, padded, max_duration)


def score(video_id: str) -> dict[str, Any]:
    cfg = load_config()["scoring"]
    weights: dict[str, float] = cfg["weights"]
    out = cache_dir(video_id)
    duration = read_json(out / META_FILE)["duration"]

    signals: list[dict[str, Any]] = []
    for det in SIGNAL_DETECTORS:
        path = events_path(video_id, det)
        if not path.exists():
            raise FileNotFoundError(f"{det} 탐지 결과가 없습니다. 먼저 `python cli.py detect {det} {video_id}`")
        signals += [ev for ev in read_json(path)["events"] if ev["type"] in weights]
    print(f"[{NAME}] 신호 {len(signals)}개를 묶어 후보 생성")

    def window(ev: dict[str, Any]) -> tuple[float, float]:
        anchor = cfg.get("anchor_windows", {}).get(ev["type"])
        if anchor:  # 이벤트 시작 시점 기준 (예: 흑백 전환 → 죽기 직전 교전)
            return max(0.0, ev["start"] - anchor["before"]), min(duration, ev["start"] + anchor["after"])
        return max(0.0, ev["start"] - cfg["padding_before"]), min(duration, ev["end"] + cfg["padding_after"])

    padded = [window(ev) for ev in signals]
    groups = []
    for g in merge_intervals(padded, gap=cfg["merge_gap"]):
        groups += _split_long(g[0], g[1], padded, cfg["max_duration"])

    events = []
    for start, end in groups:
        inside = [ev for ev, (ws, we) in zip(signals, padded) if ws < end and we > start]
        best: dict[str, float] = {}
        for ev in inside:
            best[ev["type"]] = max(best.get(ev["type"], 0.0), ev["score"])
        total = sum(weights[t] * s for t, s in best.items())
        if total < cfg["min_score"]:
            continue
        parts = sorted(best.items(), key=lambda kv: -weights[kv[0]] * kv[1])
        events.append(make_event(
            "highlight", len(events) + 1, start, end, "highlight",
            score=total,
            reason=" + ".join(f"{TYPE_LABELS.get(t, t)}({s:.2f})" for t, s in parts),
            data={"signals": {t: round(s, 3) for t, s in best.items()},
                  "event_ids": [ev["id"] for ev in inside]},
        ))
        events[-1]["source"] = NAME

    doc = {"video_id": video_id, "detector": NAME, "params": cfg, "events": events}
    write_json(out / CANDIDATES_FILE, doc)
    print(f"  후보 {len(events)}개 -> {CANDIDATES_FILE}")
    return doc
