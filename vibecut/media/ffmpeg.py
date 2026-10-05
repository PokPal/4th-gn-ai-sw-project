"""ffmpeg/ffprobe 실행 도우미. 긴 처리는 진행률을 출력한다."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _require(tool: str) -> str:
    path = shutil.which(tool)
    if path is None:
        raise RuntimeError(f"{tool}를 찾을 수 없습니다. FFmpeg를 설치하고 PATH에 추가하세요.")
    return path


def probe(video_path: Path) -> dict[str, Any]:
    result = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", str(video_path)],
        capture_output=True, text=True, encoding="utf-8", check=True,
    )
    return json.loads(result.stdout)


def run_filter_log(args: list[str]) -> str:
    """출력 파일 없이 필터만 실행하고 ffmpeg 로그(stderr)를 돌려준다. silencedetect 등에 사용."""
    result = subprocess.run(
        [_require("ffmpeg"), "-hide_banner", "-nostats", *args, "-f", "null", "-"],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패: {result.stderr.strip()[-500:]}")
    return result.stderr


def run_with_progress(args: list[str], duration: float, label: str) -> None:
    """ffmpeg를 실행하며 원본 길이 대비 진행률을 한 줄로 갱신해 출력한다."""
    cmd = [_require("ffmpeg"), "-hide_banner", "-loglevel", "error", "-y",
           "-progress", "pipe:1", "-nostats", *args]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            text=True, encoding="utf-8")
    assert proc.stdout is not None
    for line in proc.stdout:
        if line.startswith("out_time_us=") and duration > 0:
            value = line.split("=", 1)[1].strip()
            if value.isdigit():
                pct = min(100.0, int(value) / 1e6 / duration * 100)
                sys.stdout.write(f"\r  {label}: {pct:5.1f}%")
                sys.stdout.flush()
    _, stderr = proc.communicate()
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg 실패 ({label}): {stderr.strip()}")
    sys.stdout.write(f"\r  {label}: 100.0%\n")
