"""음성 인식 공통 인터페이스.

제공사 어댑터는 SttProvider를 구현하고, 결과를 공통 자막 형식(CLAUDE.md 6장 ①)의
segments 목록으로 돌려준다. 나머지 코드는 제공사를 몰라야 한다.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, TypedDict


class Word(TypedDict):
    word: str
    start: float
    end: float


class Segment(TypedDict):
    id: int
    start: float
    end: float
    text: str
    words: list[Word]


class SttProvider(ABC):
    name: str

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg

    @abstractmethod
    def transcribe(self, audio_path: Path, duration: float) -> list[Segment]:
        """audio_path 전체를 인식해 원본 기준 시간(초)의 segments를 돌려준다."""
