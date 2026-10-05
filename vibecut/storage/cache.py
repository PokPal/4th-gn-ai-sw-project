"""video_id 계산과 분석 결과 캐시 폴더 관리."""

from __future__ import annotations

import hashlib
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

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


LOCK_FILE = ".analyze.lock"
LOCK_STALE_SECONDS = 3 * 3600  # 이보다 오래된 잠금은 비정상 종료로 보고 무시


class AlreadyRunning(RuntimeError):
    pass


@contextmanager
def analysis_lock(video_id: str) -> Iterator[None]:
    """같은 영상을 동시에 분석하지 못하게 한다 (음성 인식 API 중복 호출 방지). CLI와 UI 프로세스 간에도 동작."""
    path = cache_dir(video_id) / LOCK_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and time.time() - path.stat().st_mtime > LOCK_STALE_SECONDS:
        path.unlink(missing_ok=True)
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise AlreadyRunning(f"이 영상({video_id})은 이미 다른 곳에서 분석 중입니다. 끝난 뒤 다시 시도하세요.") from None
    os.close(fd)
    try:
        yield
    finally:
        path.unlink(missing_ok=True)


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
