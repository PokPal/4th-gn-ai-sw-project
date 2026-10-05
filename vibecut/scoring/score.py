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
}


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

    padded = [(max(0.0, ev["start"] - cfg["padding_before"]),
               min(duration, ev["end"] + cfg["padding_after"])) for ev in signals]
    groups = merge_intervals(padded, gap=cfg["merge_gap"])

    events = []
    for start, end in groups:
        inside = [ev for ev in signals if ev["start"] < end and ev["end"] > start]
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
