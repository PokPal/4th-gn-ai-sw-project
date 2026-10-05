"""에이전트 도구: 저장된 분석 결과를 조회하고 편집 결정 목록을 만든다 (판단은 에이전트가, 계산은 도구가)."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import librosa
import numpy as np

from vibecut.agent.memory import add_preference
from vibecut.config import load_config
from vibecut.detectors import DETECTORS
from vibecut.detectors.common import events_path
from vibecut.export.resolve import export_edit
from vibecut.media.ingest import AUDIO_FILE, META_FILE
from vibecut.scoring.score import CANDIDATES_FILE, score
from vibecut.storage.cache import cache_dir, read_json
from vibecut.storage.edits import edit_path, latest_edit, save_edit
from vibecut.stt.run import TRANSCRIPT_FILE

MARKER_COLORS = ["Blue", "Cyan", "Green", "Yellow", "Red", "Pink", "Purple"]

TOOLS: list[dict[str, Any]] = [
    {
        "name": "get_analysis_summary",
        "description": "영상 분석 결과 개요: 영상 길이, 자막 문장 수, 탐지기별·종류별 이벤트 개수, 하이라이트 후보 수와 점수 분포, "
                       "지루한 구간 비율. 편집을 시작할 때 가장 먼저 호출한다.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "list_candidates",
        "description": "하이라이트 후보 또는 탐지 구간을 조회한다. 기본은 type=highlight(점수화된 후보)를 점수 높은 순으로 돌려준다. "
                       "각 항목에 그 구간의 대사 일부(speech)가 붙는다. types로 dead_time, reaction, loudness_spike 등도 볼 수 있다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "types": {"type": "array", "items": {"type": "string", "enum": [
                    "highlight", "silence", "loudness_spike", "reaction", "scene_change",
                    "brightness_change", "motion", "dead_time"]},
                    "description": "조회할 종류. 생략하면 [\"highlight\"]"},
                "min_score": {"type": "number", "description": "이 점수(0~1) 이상만"},
                "start": {"type": "number", "description": "이 시간(초) 이후와 겹치는 구간만"},
                "end": {"type": "number", "description": "이 시간(초) 이전과 겹치는 구간만"},
                "sort": {"type": "string", "enum": ["score", "time"], "description": "정렬 기준 (기본 score)"},
                "limit": {"type": "integer", "description": "최대 개수"},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "get_transcript",
        "description": "start~end초 구간의 자막(문장 단위, 원본 시간)을 돌려준다. 장면의 맥락을 확인할 때 쓴다.",
        "input_schema": {
            "type": "object",
            "properties": {"start": {"type": "number"}, "end": {"type": "number"}},
            "required": ["start", "end"], "additionalProperties": False,
        },
    },
    {
        "name": "get_signals",
        "description": "start~end초 구간의 신호 상세: 그 구간과 겹치는 모든 탐지 이벤트, 구간 음량 통계(영상 전체 중앙값 대비), "
                       "지루한 구간과 겹치는 비율.",
        "input_schema": {
            "type": "object",
            "properties": {"start": {"type": "number"}, "end": {"type": "number"}},
            "required": ["start", "end"], "additionalProperties": False,
        },
    },
    {
        "name": "rerun_detection",
        "description": "후보가 너무 적거나 많을 때 탐지 기준을 바꿔 특정 탐지기를 다시 실행하고, 하이라이트 점수화도 다시 한다. "
                       "params에는 바꿀 값만 넣는다. 예: loudness {\"threshold_db\": 6}, visual {\"scene_threshold\": 20, "
                       "\"motion_threshold\": 0.15}, reaction {\"context_seconds\": 3}, dead_time {\"min_duration\": 10}.",
        "input_schema": {
            "type": "object",
            "properties": {
                "detector": {"type": "string", "enum": list(DETECTORS)},
                "params": {"type": "object", "description": "바꿀 기준값"},
            },
            "required": ["detector", "params"], "additionalProperties": False,
        },
    },
    {
        "name": "build_edit",
        "description": "선택한 구간들로 편집 결정 목록을 만들어 저장하고, 총 길이와 목표 대비 차이를 돌려준다. "
                       "clips는 영상에 놓일 순서대로 준다(보통 시간순). 각 클립 경계는 말소리에 맞춰 자동 보정된다 "
                       "(말 시작 1초 전부터, 말 끝 직후까지, 조용한 공백은 제거 — speech_adjustments에 기록). "
                       "보정 후 총 길이가 허용 오차를 벗어나면 구간을 빼거나 더해 다시 호출한다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "goal": {"type": "string", "description": "사용자의 편집 목표 요약"},
                "target_duration": {"type": "number", "description": "목표 길이(초)"},
                "clips": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "candidate_id": {"type": ["string", "null"],
                                             "description": "후보에서 온 구간이면 그 id, 아니면 null"},
                            "reason": {"type": "string", "description": "이 구간을 넣은 이유 (편집자가 읽을 메모)"},
                            "marker_name": {"type": "string", "description": "마커 이름 — 장면을 짧게 설명"},
                            "marker_note": {"type": "string", "description": "마커 메모 (근거 수치 등)"},
                            "marker_color": {"type": "string", "enum": MARKER_COLORS},
                        },
                        "required": ["start", "end", "reason", "marker_name"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["goal", "target_duration", "clips"], "additionalProperties": False,
        },
    },
    {
        "name": "ask_user",
        "description": "넣을지 확신이 애매한 장면만 사용자에게 묻는다. 한 번에 최대 3개. 모든 장면을 묻지 않는다. "
                       "호출하면 사용자가 답할 때까지 기다리며, 답이 도구 결과로 돌아온다.",
        "input_schema": {
            "type": "object",
            "properties": {
                "questions": {
                    "type": "array", "maxItems": 3,
                    "items": {
                        "type": "object",
                        "properties": {
                            "start": {"type": "number"},
                            "end": {"type": "number"},
                            "summary": {"type": "string", "description": "장면 요약"},
                            "why_unsure": {"type": "string", "description": "넣을지 고민되는 이유"},
                        },
                        "required": ["start", "end", "summary", "why_unsure"],
                        "additionalProperties": False,
                    },
                },
            },
            "required": ["questions"], "additionalProperties": False,
        },
    },
    {
        "name": "save_preference",
        "description": "사용자 응답에서 다음 편집에도 쓸 수 있는 일반적인 선호도를 저장한다. "
                       "특정 시간이 아니라 장면의 성격으로 적는다 (예: '조용한 전략 설명 장면은 빼기').",
        "input_schema": {
            "type": "object",
            "properties": {
                "preference": {"type": "string"},
                "decision": {"type": "string", "enum": ["include", "exclude", "style"]},
                "example": {"type": "string", "description": "이번 영상에서 근거가 된 장면 (선택)"},
            },
            "required": ["preference", "decision"], "additionalProperties": False,
        },
    },
    {
        "name": "export_timeline",
        "description": "편집 결정 목록을 다빈치 리졸브용 타임라인(.otio), 자막(.srt), 마커(.edl)로 내보낸다. "
                       "edit_id를 생략하면 가장 최근 build_edit 결과를 쓴다.",
        "input_schema": {
            "type": "object",
            "properties": {"edit_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
]


class ToolError(Exception):
    """도구 입력이 잘못됐을 때. 에이전트에게 is_error 결과로 돌려준다."""


def _r(x: float) -> float:
    return round(float(x), 3)


class ToolBox:
    def __init__(self, video_id: str) -> None:
        self.video_id = video_id
        self.dir = cache_dir(video_id)
        self.meta = read_json(self.dir / META_FILE)
        self.cfg = load_config()

    # ---- 내부 조회 ----
    def _transcript(self) -> list[dict[str, Any]]:
        return read_json(self.dir / TRANSCRIPT_FILE)["segments"]

    def _all_events(self) -> list[dict[str, Any]]:
        evs = []
        for det in DETECTORS:
            p = events_path(self.video_id, det)
            if p.exists():
                evs += read_json(p)["events"]
        return evs

    def _candidates(self) -> list[dict[str, Any]]:
        p = self.dir / CANDIDATES_FILE
        return read_json(p)["events"] if p.exists() else []

    def _speech(self, start: float, end: float, limit: int = 120) -> str:
        text = " ".join(s["text"] for s in self._transcript() if s["start"] < end and s["end"] > start)
        return text[:limit] + ("…" if len(text) > limit else "")

    def _speech_spans(self) -> list[tuple[float, float]]:
        """문장마다 실제 말소리 구간 (단어 타임스탬프가 있으면 첫 단어 시작 ~ 마지막 단어 끝)."""
        return [(s["words"][0]["start"], s["words"][-1]["end"]) if s["words"] else (s["start"], s["end"])
                for s in self._transcript()]

    def _quiet(self, start: float, end: float) -> bool:
        """말도 없고 하이라이트 신호(음량 급증, 반응, 화면 변화)도 없는 구간인지."""
        signal_types = set(self.cfg["scoring"]["weights"])
        return not any(ev["type"] in signal_types and ev["start"] < end and ev["end"] > start
                       for ev in self._all_events())

    def _snap_to_speech(self, start: float, end: float) -> tuple[float, float, list[str]]:
        """클립 경계를 말소리에 맞춘다: 말 시작 pre_roll초 전부터, 말 끝 post_roll초 후까지.

        말이 잘리면 늘리고, 말 앞뒤의 조용한 공백이 여유 시간보다 길면 줄인다.
        신호가 있는 공백(게임 효과음, 화면 변화)은 하이라이트일 수 있으므로 줄이지 않는다.
        """
        pre, post = self.cfg["edit"]["speech_pre_roll"], self.cfg["edit"]["speech_post_roll"]
        duration = self.meta["duration"]
        spans = self._speech_spans()
        inside = [(on, off) for on, off in spans if off > start and on < end]
        if not inside:
            return start, end, []
        first_on = min(on for on, _ in inside)
        last_off = max(off for _, off in inside)
        notes = []

        new_start = start
        if first_on - pre < start:  # 말이 잘렸거나 앞 여유가 부족 → 앞으로 늘림 (이전 문장 끝은 넘지 않음)
            prev_off = max((off for _, off in spans if off <= first_on), default=0.0)
            new_start = max(0.0, prev_off, first_on - pre)
        elif self._quiet(start, first_on - pre):  # 말 앞 공백이 길고 조용함 → 줄임
            new_start = first_on - pre
        if abs(new_start - start) >= 0.05:
            notes.append(f"시작 {start:.2f}→{new_start:.2f}초 (말 시작 {first_on:.2f}초)")

        new_end = end
        if last_off + post > end:  # 말이 잘렸거나 뒤 여유가 부족 → 뒤로 늘림 (다음 문장 시작은 넘지 않음)
            next_on = min((on for on, _ in spans if on >= last_off), default=duration)
            new_end = min(duration, next_on, last_off + post)
        elif self._quiet(last_off + post, end):  # 말 뒤 공백이 길고 조용함 → 줄임
            new_end = last_off + post
        if abs(new_end - end) >= 0.05:
            notes.append(f"끝 {end:.2f}→{new_end:.2f}초 (말 끝 {last_off:.2f}초)")
        return new_start, new_end, notes

    def _check_range(self, start: float, end: float) -> None:
        if not (0 <= start < end <= self.meta["duration"] + 0.001):
            raise ToolError(f"잘못된 구간 {start}~{end}초 (영상 길이 {self.meta['duration']}초, start < end 필요)")

    # ---- 도구 ----
    def get_analysis_summary(self) -> dict[str, Any]:
        duration = self.meta["duration"]
        events = self._all_events()
        by_type: dict[str, int] = {}
        for ev in events:
            by_type[ev["type"]] = by_type.get(ev["type"], 0) + 1
        dead = sum(ev["end"] - ev["start"] for ev in events if ev["type"] == "dead_time")
        cands = self._candidates()
        scores = sorted((c["score"] for c in cands), reverse=True)
        return {
            "video_id": self.video_id,
            "duration_seconds": duration,
            "transcript_segments": len(self._transcript()),
            "events_by_type": by_type,
            "dead_time_ratio": _r(dead / duration) if duration else 0,
            "highlight_candidates": len(cands),
            "candidate_total_seconds": _r(sum(c["end"] - c["start"] for c in cands)),
            "candidate_scores": {"max": scores[0] if scores else None,
                                 "median": scores[len(scores) // 2] if scores else None,
                                 "min": scores[-1] if scores else None},
        }

    def list_candidates(self, types: list[str] | None = None, min_score: float | None = None,
                        start: float | None = None, end: float | None = None,
                        sort: str = "score", limit: int | None = None) -> dict[str, Any]:
        types = types or ["highlight"]
        pool = (self._candidates() if "highlight" in types else []) + \
               [ev for ev in self._all_events() if ev["type"] in types]
        if min_score is not None:
            pool = [ev for ev in pool if ev["score"] >= min_score]
        if start is not None:
            pool = [ev for ev in pool if ev["end"] > start]
        if end is not None:
            pool = [ev for ev in pool if ev["start"] < end]
        pool.sort(key=(lambda e: -e["score"]) if sort == "score" else (lambda e: e["start"]))
        limit = min(limit or self.cfg["agent"]["candidate_limit"], self.cfg["agent"]["candidate_limit"])
        return {
            "total_matched": len(pool),
            "items": [{"id": ev["id"], "start": ev["start"], "end": ev["end"],
                       "duration": _r(ev["end"] - ev["start"]), "type": ev["type"], "score": ev["score"],
                       "reason": ev["reason"], "speech": self._speech(ev["start"], ev["end"])}
                      for ev in pool[:limit]],
        }

    def get_transcript(self, start: float, end: float) -> dict[str, Any]:
        self._check_range(start, end)
        segs = [{"start": s["start"], "end": s["end"], "text": s["text"]}
                for s in self._transcript() if s["start"] < end and s["end"] > start]
        return {"start": start, "end": end, "segments": segs}

    def get_signals(self, start: float, end: float) -> dict[str, Any]:
        self._check_range(start, end)
        events = [{"id": ev["id"], "type": ev["type"], "start": ev["start"], "end": ev["end"],
                   "score": ev["score"], "reason": ev["reason"]}
                  for ev in self._all_events() if ev["start"] < end and ev["end"] > start]
        dead = sum(min(ev["end"], end) - max(ev["start"], start) for ev in self._all_events()
                   if ev["type"] == "dead_time" and ev["start"] < end and ev["end"] > start)

        window = self.cfg["detectors"]["loudness"]["window_seconds"]
        y, sr = librosa.load(str(self.dir / AUDIO_FILE), sr=None)
        hop = max(1, int(window * sr))

        def db_of(samples: np.ndarray) -> np.ndarray:
            if len(samples) < hop:
                return np.array([])
            rms = librosa.feature.rms(y=samples, frame_length=hop, hop_length=hop, center=False)[0]
            return 20 * np.log10(np.maximum(rms, 1e-10))

        whole, part = db_of(y), db_of(y[int(start * sr):int(end * sr)])
        loud = None
        if len(part) and len(whole):
            med = float(np.median(whole))
            loud = {"mean_db": _r(part.mean()), "max_db": _r(part.max()), "video_median_db": _r(med),
                    "max_above_median_db": _r(part.max() - med)}
        return {"start": start, "end": end, "events": events, "loudness": loud,
                "dead_time_overlap_ratio": _r(dead / (end - start))}

    def rerun_detection(self, detector: str, params: dict[str, Any]) -> dict[str, Any]:
        if detector not in DETECTORS:
            raise ToolError(f"알 수 없는 탐지기: {detector}")
        known = set(self.cfg["detectors"][detector])
        unknown = set(params) - known
        if unknown:
            raise ToolError(f"{detector}에 없는 기준값: {sorted(unknown)}. 가능한 값: {sorted(known)}")
        doc = DETECTORS[detector](self.video_id, params)
        cands = score(self.video_id)["events"]
        return {"detector": detector, "params_used": doc["params"], "events": len(doc["events"]),
                "highlight_candidates": len(cands)}

    def build_edit(self, goal: str, target_duration: float, clips: list[dict[str, Any]]) -> dict[str, Any]:
        if not clips:
            raise ToolError("clips가 비어 있습니다.")
        warnings = []
        adjustments = []
        out_clips = []
        for i, c in enumerate(clips, start=1):
            self._check_range(c["start"], c["end"])
            start, end = c["start"], c["end"]
            if self.cfg["edit"]["snap_to_speech"]:
                start, end, notes = self._snap_to_speech(start, end)
                if notes:
                    adjustments.append(f"클립 {i}: " + ", ".join(notes))
            out_clips.append({
                "order": i, "start": _r(start), "end": _r(end),
                "candidate_id": c.get("candidate_id"),
                "reason": c["reason"],
                "marker": {"name": c["marker_name"], "note": c.get("marker_note", ""),
                           "color": c.get("marker_color") or self.cfg["export"]["default_marker_color"]},
            })
        ordered = sorted(out_clips, key=lambda c: c["start"])
        # 보정 때문에 새로 겹친 이웃 클립은 사이 공백을 반으로 나눠 경계를 맞춘다
        originals = {i: (c["start"], c["end"]) for i, c in enumerate(clips, start=1)}
        for a, b in zip(ordered, ordered[1:]):
            if b["start"] < a["end"] and originals[b["order"]][0] >= originals[a["order"]][1]:
                mid = _r((a["end"] + b["start"]) / 2)
                a["end"], b["start"] = mid, mid
                adjustments.append(f"클립 {a['order']}·{b['order']}: 여유 시간이 겹쳐 {mid:.2f}초에서 나눔")
        for a, b in zip(ordered, ordered[1:]):
            if b["start"] < a["end"]:
                warnings.append(f"구간이 겹침: {a['start']}~{a['end']} 과 {b['start']}~{b['end']} (겹친 부분이 두 번 나옴)")
        total = _r(sum(c["end"] - c["start"] for c in out_clips))
        edit = {
            "video_id": self.video_id, "goal": goal, "target_duration": _r(target_duration),
            "total_duration": total, "created_at": datetime.now().isoformat(timespec="seconds"),
            "clips": out_clips,
        }
        path = save_edit(edit)
        tol = self.cfg["agent"]["duration_tolerance"]
        diff = total - target_duration
        ok = abs(diff) <= target_duration * tol
        return {
            "edit_id": path.stem, "clips": len(out_clips), "total_duration": total,
            "target_duration": target_duration, "difference": _r(diff),
            "allowed_difference": _r(target_duration * tol), "within_tolerance": ok,
            "advice": None if ok else ("목표보다 깁니다. 점수가 낮거나 덜 중요한 구간을 빼거나 줄이세요." if diff > 0
                                       else "목표보다 짧습니다. 후보를 더 넣거나 구간을 늘리세요. 후보가 부족하면 rerun_detection."),
            "speech_adjustments": adjustments,
            "warnings": warnings,
        }

    def save_preference(self, preference: str, decision: str, example: str | None = None) -> dict[str, Any]:
        return {"saved": add_preference(preference, decision, example, self.video_id)}

    def export_timeline(self, edit_id: str | None = None) -> dict[str, Any]:
        path = edit_path(self.video_id, edit_id) if edit_id else latest_edit(self.video_id)
        if path is None or not path.exists():
            raise ToolError("내보낼 편집 결정 목록이 없습니다. 먼저 build_edit를 호출하세요.")
        return export_edit(self.video_id, path)

    def execute(self, name: str, args: dict[str, Any]) -> tuple[str, bool]:
        """도구를 실행하고 (결과 JSON 문자열, 오류 여부)를 돌려준다."""
        fn = getattr(self, name, None)
        if fn is None or name == "ask_user" or name.startswith("_"):
            return f"알 수 없는 도구: {name}", True
        try:
            result = fn(**args)
            return json.dumps(result, ensure_ascii=False), False
        except ToolError as e:
            return str(e), True
        except TypeError as e:
            return f"입력 형식 오류: {e}", True
