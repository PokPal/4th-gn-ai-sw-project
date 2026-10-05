"""에이전트 판단 로그: 입력 → 판단 → 도구 실행 → 결과를 logs/에 JSONL로 남기고, 화면·콘솔에 보여준다."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Callable

from vibecut.config import load_config, project_path

KIND_LABELS = {
    "user": "사용자 입력",
    "thinking": "판단",
    "assistant": "에이전트 응답",
    "tool_call": "도구 호출",
    "tool_result": "도구 결과",
    "question": "사용자에게 질문",
    "error": "오류",
    "usage": "비용",
}


def _summary(text: str, limit: int = 300) -> str:
    text = text.replace("\n", " ")
    return text[:limit] + ("…" if len(text) > limit else "")


class AgentLogger:
    def __init__(self, video_id: str, on_entry: Callable[[dict[str, Any]], None] | None = None,
                 echo: bool = True) -> None:
        log_dir = project_path(load_config()["paths"]["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"{video_id}_{stamp}.jsonl"
        self.entries: list[dict[str, Any]] = []
        self.on_entry = on_entry
        self.echo = echo

    def log(self, kind: str, content: Any, **extra: Any) -> dict[str, Any]:
        text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        entry = {"time": datetime.now().isoformat(timespec="seconds"), "kind": kind,
                 "label": KIND_LABELS.get(kind, kind), "summary": _summary(text), **extra}
        self.entries.append(entry)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**entry, "content": content}, ensure_ascii=False) + "\n")
        if self.echo:
            name = f" {extra['tool']}" if "tool" in extra else ""
            print(f"  [{entry['label']}{name}] {entry['summary']}")
        if self.on_entry:
            self.on_entry(entry)
        return entry
