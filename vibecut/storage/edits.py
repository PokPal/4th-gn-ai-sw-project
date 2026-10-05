"""편집 결정 목록(CLAUDE.md 6장 ③) 저장·조회."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from vibecut.storage.cache import cache_dir, read_json, write_json

EDITS_DIR = "edits"


def save_edit(edit: dict[str, Any]) -> Path:
    stamp = datetime.fromisoformat(edit["created_at"]).strftime("%Y%m%d_%H%M%S")
    path = cache_dir(edit["video_id"]) / EDITS_DIR / f"{stamp}.json"
    write_json(path, edit)
    return path


def edit_path(video_id: str, edit_id: str) -> Path:
    return cache_dir(video_id) / EDITS_DIR / f"{edit_id}.json"


def latest_edit(video_id: str) -> Path | None:
    files = sorted((cache_dir(video_id) / EDITS_DIR).glob("*.json"))
    return files[-1] if files else None


def load_edit(video_id: str, edit_id: str) -> dict[str, Any]:
    return read_json(edit_path(video_id, edit_id))
