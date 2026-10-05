"""config.yaml과 .env를 읽는 공통 모듈."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "config.yaml"

load_dotenv(PROJECT_ROOT / ".env")


def load_config(path: Path = CONFIG_PATH) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def project_path(relative: str) -> Path:
    """config.yaml의 상대 경로를 프로젝트 루트 기준 절대 경로로 바꾼다."""
    return PROJECT_ROOT / relative
