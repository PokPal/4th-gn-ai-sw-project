"""장기 메모리: 사용자의 넣어줘/빼줘 응답에서 얻은 선호도를 memory/preferences.json에 저장한다."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from vibecut.config import load_config, project_path


def _path():
    return project_path(load_config()["paths"]["memory_file"])


def load_preferences() -> list[dict[str, Any]]:
    path = _path()
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("preferences", [])


def add_preference(preference: str, decision: str, example: str | None, video_id: str) -> dict[str, Any]:
    prefs = load_preferences()
    item = {"preference": preference, "decision": decision, "example": example,
            "video_id": video_id, "saved_at": datetime.now().isoformat(timespec="seconds")}
    prefs.append(item)
    path = _path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"preferences": prefs}, f, ensure_ascii=False, indent=2)
    return item


def preferences_prompt() -> str:
    prefs = load_preferences()
    if not prefs:
        return "아직 저장된 선호도가 없습니다."
    return "\n".join(f"- [{p['decision']}] {p['preference']}" + (f" (예: {p['example']})" if p.get("example") else "")
                     for p in prefs)
