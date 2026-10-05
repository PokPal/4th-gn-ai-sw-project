"""video_id 계산과 분석 결과 캐시 폴더 관리."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from vibecut.config import load_config, project_path


def compute_video_id(video_path: Path) -> str:
    """파일 크기 + 앞부분 바이트의 해시. 파일 이름이 달라도 같은 영상이면 같은 id."""
    head_bytes = load_config()["video_id"]["hash_head_bytes"]
    h = hashlib.sha256()
    h.update(str(video_path.stat().st_size).encode())
    with open(video_path, "rb") as f:
        h.update(f.read(head_bytes))
    return h.hexdigest()[:12]


def cache_dir(video_id: str) -> Path:
    return project_path(load_config()["paths"]["cache_dir"]) / video_id


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
