"""탐지기 모음. 모든 탐지기는 detect(video_id, params) -> 이벤트 문서(CLAUDE.md 6장 ②)."""

from __future__ import annotations

from typing import Any, Callable

from vibecut.detectors import dead_time, loudness, reaction, silence, visual

DETECTORS: dict[str, Callable[..., dict[str, Any]]] = {
    "silence": silence.detect,
    "loudness": loudness.detect,
    "reaction": reaction.detect,
    "visual": visual.detect,
    "dead_time": dead_time.detect,
}
