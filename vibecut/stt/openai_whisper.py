"""OpenAI Whisper API 어댑터.

파일당 25MB 제한이 있으므로 음성을 chunk_seconds 단위로 잘라 보내고,
각 조각의 시간을 원본 기준으로 되돌려 합친다.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import soundfile as sf
from openai import OpenAI

from vibecut.stt.base import Segment, SttProvider, Word

MIN_LEN = 0.001  # start < end 보장용 최소 길이


def _interval(start: float, end: float) -> tuple[float, float]:
    start = round(start, 3)
    return start, round(max(end, start + MIN_LEN), 3)


class OpenAIWhisper(SttProvider):
    name = "openai-whisper"

    def transcribe(self, audio_path: Path, duration: float) -> list[Segment]:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY가 .env에 없습니다.")
        client = OpenAI()
        audio, sr = sf.read(str(audio_path), dtype="int16")
        chunk_len = int(self.cfg["chunk_seconds"] * sr)
        n_chunks = max(1, -(-len(audio) // chunk_len))

        segments: list[Segment] = []
        with tempfile.TemporaryDirectory() as tmp:
            for i in range(n_chunks):
                offset = i * chunk_len / sr
                chunk_path = Path(tmp) / f"chunk_{i:03d}.wav"
                sf.write(str(chunk_path), audio[i * chunk_len:(i + 1) * chunk_len], sr)
                print(f"  음성 인식: 조각 {i + 1}/{n_chunks} ({offset:.0f}초~) 전송 중...")
                with open(chunk_path, "rb") as f:
                    result = client.audio.transcriptions.create(
                        file=f,
                        model=self.cfg["model"],
                        language=self.cfg["language"],
                        response_format="verbose_json",
                        timestamp_granularities=["segment", "word"],
                    )
                segments.extend(self._convert(result, offset, start_id=len(segments)))
        print(f"  음성 인식: 완료, 문장 {len(segments)}개")
        return segments

    def _convert(self, result, offset: float, start_id: int) -> list[Segment]:
        """API 응답을 공통 형식으로 바꾼다.

        문장 구간끼리 겹칠 수 있으므로, 각 단어는 가장 많이 겹치는 문장 하나에만 배정한다.
        버린 문장(무음 환각)에 속한 단어도 함께 버린다.
        """
        drop_prob = self.cfg["drop_no_speech_prob"]
        segs = list(result.segments or [])
        assigned: list[list[Word]] = [[] for _ in segs]
        for w in result.words or []:
            if not w.word.strip():
                continue
            w_end = max(w.end, w.start + MIN_LEN)  # 길이 0인 단어도 겹침 계산이 되도록
            overlaps = [min(s.end, w_end) - max(s.start, w.start) for s in segs]
            if not overlaps or max(overlaps) <= 0:
                continue
            best = max(range(len(segs)), key=lambda i: overlaps[i])
            ws, we = _interval(w.start + offset, w.end + offset)
            assigned[best].append({"word": w.word.strip(), "start": ws, "end": we})

        out: list[Segment] = []
        for seg, seg_words in zip(segs, assigned):
            text = seg.text.strip()
            if not text or seg.no_speech_prob > drop_prob:
                continue
            start, end = _interval(seg.start + offset, seg.end + offset)
            out.append({"id": start_id + len(out), "start": start, "end": end,
                        "text": text, "words": seg_words})
        return out
