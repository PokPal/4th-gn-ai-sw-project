"""감탄·반응 표현 탐지: 자막의 단어에서 반응 키워드를 찾는다."""

from __future__ import annotations

import re
from typing import Any

from vibecut.config import load_config
from vibecut.detectors.common import make_event, merge_intervals, save_events
from vibecut.media.ingest import META_FILE
from vibecut.storage.cache import cache_dir, read_json
from vibecut.stt.run import TRANSCRIPT_FILE

NAME = "reaction"
PUNCT = re.compile(r"[\s.,!?~…\"'()\[\]]+")
ELONGATION = re.compile(r"[아어오우으이야ㅏㅓㅗㅜㅡㅣ~!?ㅋㅎ]*")


def match_keyword(word: str, keywords: list[str]) -> str | None:
    """단어가 키워드와 같거나, 키워드 뒤에 늘어진 소리만 붙어 있으면 그 키워드를 돌려준다."""
    w = PUNCT.sub("", word)
    for kw in keywords:
        if w.startswith(kw) and ELONGATION.fullmatch(w[len(kw):]):
            return kw
    return None


def detect(video_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = {**load_config()["detectors"][NAME], **(params or {})}
    out = cache_dir(video_id)
    duration = read_json(out / META_FILE)["duration"]
    transcript = read_json(out / TRANSCRIPT_FILE)
    print(f"[{NAME}] 반응 키워드 {len(p['keywords'])}개로 자막 검색")

    hits = []  # (start, end, word, segment_text)
    for seg in transcript["segments"]:
        for w in seg["words"]:
            if match_keyword(w["word"], p["keywords"]):
                hits.append((w["start"], w["end"], w["word"], seg["text"]))

    ctx = p["context_seconds"]
    windows = merge_intervals(
        [(max(0.0, s - ctx), min(duration, e + ctx)) for s, e, _, _ in hits])

    events = []
    for start, end in windows:
        inside = [h for h in hits if start <= h[0] < end]
        words = [h[2] for h in inside]
        lines = list(dict.fromkeys(h[3] for h in inside))
        events.append(make_event(
            NAME, len(events) + 1, start, end, "reaction",
            score=p["base_score"] + p["extra_score"] * (len(inside) - 1),
            reason=f"반응 표현 {len(inside)}개: {', '.join(words)}",
            data={"words": words, "lines": lines},
        ))
    return save_events(video_id, NAME, p, events)
