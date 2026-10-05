"""화면 급변 탐지: 장면 전환(PySceneDetect), 밝기 급변·프레임 변화량(OpenCV).

분석은 저해상도 사본(proxy.mp4)으로 한다.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from typing import Any

import cv2
import numpy as np
from scenedetect import ContentDetector
from scenedetect import detect as scene_detect

from vibecut.config import load_config
from vibecut.detectors.common import make_event, mask_to_intervals, save_events
from vibecut.media.ingest import META_FILE, PROXY_FILE
from vibecut.storage.cache import cache_dir, read_json

NAME = "visual"


@lru_cache(maxsize=4)
def visual_series(video_id: str, sample_fps: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """초당 sample_fps장씩 뽑아 (시간, 평균 밝기 0~255, 직전 샘플 대비 변화량 0~1)을 돌려준다."""
    path = cache_dir(video_id) / PROXY_FILE
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / sample_fps))

    times, bright, motion = [], [], []
    prev = None
    idx = 0
    while True:
        if idx % step == 0:
            ok, frame = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            times.append(idx / fps)
            bright.append(float(gray.mean()))
            motion.append(0.0 if prev is None else float(cv2.absdiff(gray, prev).mean()) / 255.0)
            prev = gray
            if len(times) % 200 == 0 and total:
                sys.stdout.write(f"\r  화면 분석: {min(100.0, idx / total * 100):5.1f}%")
                sys.stdout.flush()
        elif not cap.grab():
            break
        idx += 1
    cap.release()
    sys.stdout.write("\r  화면 분석: 100.0%\n")
    return np.array(times), np.array(bright), np.array(motion)


def detect(video_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = {**load_config()["detectors"][NAME], **(params or {})}
    out = cache_dir(video_id)
    duration = read_json(out / META_FILE)["duration"]
    half = p["point_window"]
    print(f"[{NAME}] 장면 전환 민감도 {p['scene_threshold']}, 밝기 급변 {p['brightness_jump']}, "
          f"변화량 {p['motion_threshold']}")

    times, bright, motion = visual_series(video_id, p["sample_fps"])
    step = 1.0 / p["sample_fps"]
    events: list[dict[str, Any]] = []

    def window(t: float) -> tuple[float, float]:
        return max(0.0, t - half), min(duration, t + half)

    # 장면 전환: 각 장면의 시작 시점(첫 장면 제외)
    print("  장면 전환 탐지 중...")
    scenes = scene_detect(str(out / PROXY_FILE), ContentDetector(threshold=p["scene_threshold"]))
    for start_tc, _ in scenes[1:]:
        t = float(start_tc.seconds)
        i = int(np.argmin(np.abs(times - t))) if len(times) else 0
        m = float(motion[min(i + 1, len(motion) - 1)]) if len(motion) else 0.0
        s, e = window(t)
        events.append(make_event(
            NAME, len(events) + 1, s, e, "scene_change",
            score=0.5 + m / (2 * p["motion_threshold"]),
            reason=f"{t:.1f}초에 장면 전환 (프레임 변화량 {m:.2f})",
            data={"time": round(t, 3), "motion": round(m, 3)},
        ))

    # 밝기 급변: 직전 샘플 대비 밝기 차이
    jumps = np.abs(np.diff(bright, prepend=bright[:1])) if len(bright) else np.array([])
    for a, b in mask_to_intervals(jumps >= p["brightness_jump"], times, step, merge_gap=p["merge_gap"]):
        peak = a + int(np.argmax(jumps[a:b]))
        s, e = window(float(times[a]))[0], window(float(times[b - 1]))[1]
        events.append(make_event(
            NAME, len(events) + 1, s, e, "brightness_change",
            score=jumps[peak] / (2 * p["brightness_jump"]),
            reason=f"밝기가 최대 {jumps[peak]:.0f} 변함 (0~255 기준)",
            data={"max_jump": round(float(jumps[peak]), 1),
                  "before": round(float(bright[max(peak - 1, 0)]), 1),
                  "after": round(float(bright[peak]), 1)},
        ))

    # 프레임 변화량 급증: 움직임이 큰 구간
    for a, b in mask_to_intervals(motion >= p["motion_threshold"], times, step, merge_gap=p["merge_gap"]):
        peak = a + int(np.argmax(motion[a:b]))
        s, e = float(times[a]), min(duration, float(times[b - 1]) + step)
        events.append(make_event(
            NAME, len(events) + 1, s, e, "motion",
            score=motion[peak] / (2 * p["motion_threshold"]),
            reason=f"화면 변화량이 최대 {motion[peak]:.2f}로 큰 움직임이 {e - s:.1f}초 지속",
            data={"max_motion": round(float(motion[peak]), 3),
                  "mean_motion": round(float(motion[a:b].mean()), 3)},
        ))

    events.sort(key=lambda ev: ev["start"])
    return save_events(video_id, NAME, p, events)
