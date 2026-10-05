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


def series_for(video_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """config 기준으로 visual_series를 부른다 (탐지기끼리 같은 인자로 불러야 캐시가 공유됨)."""
    p = load_config()["detectors"][NAME]
    return visual_series(video_id, p["sample_fps"], tuple(p["game_region"]), p["color_saturation"])


@lru_cache(maxsize=4)
def visual_series(video_id: str, sample_fps: float, game_region: tuple[float, ...] = (0.0, 0.0, 1.0, 1.0),
                  color_saturation: float = 80.0) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """초당 sample_fps장씩 뽑아 (시간, 평균 밝기 0~255, 직전 샘플 대비 변화량 0~1, 색 있는 픽셀 비율 0~1)을 돌려준다.

    색 있는 픽셀 비율은 game_region(화면 비율 좌표 x0, y0, x1, y1) 안에서 채도가 color_saturation 이상인 픽셀의 비율.
    방송 오버레이(캐릭터·채팅)를 영역에서 빼면 게임 화면만의 흑백 전환을 잴 수 있다.
    """
    path = cache_dir(video_id) / PROXY_FILE
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"영상을 열 수 없습니다: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, round(fps / sample_fps))

    times, bright, motion, color = [], [], [], []
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
            h, w = frame.shape[:2]
            x0, y0, x1, y1 = game_region
            region = frame[int(h * y0):int(h * y1), int(w * x0):int(w * x1)]
            sat_ch = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)[:, :, 1]
            color.append(float((sat_ch >= color_saturation).mean()))
            prev = gray
            if len(times) % 200 == 0 and total:
                sys.stdout.write(f"\r  화면 분석: {min(100.0, idx / total * 100):5.1f}%")
                sys.stdout.flush()
        elif not cap.grab():
            break
        idx += 1
    cap.release()
    sys.stdout.write("\r  화면 분석: 100.0%\n")
    return np.array(times), np.array(bright), np.array(motion), np.array(color)


def detect(video_id: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
    p = {**load_config()["detectors"][NAME], **(params or {})}
    out = cache_dir(video_id)
    duration = read_json(out / META_FILE)["duration"]
    half = p["point_window"]
    print(f"[{NAME}] 장면 전환 민감도 {p['scene_threshold']}, 밝기 급변 {p['brightness_jump']}, "
          f"변화량 {p['motion_threshold']}, 흑백 전환 색 비율 {p['desaturation_ratio']} 미만")

    times, bright, motion, color = series_for(video_id)
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

    # 채도 급감(흑백 전환): 게임 영역의 색 있는 픽셀 비율이 기준 아래로 떨어져 유지됨 (예: 사망 화면).
    # 너무 어두운 화면(장면 전환 중 검은 화면)은 제외한다.
    # 너무 밝은 화면(파스텔 톤 방송 대기 화면 등)도 제외한다. 사망 화면은 어두운 회색이다.
    low = ((color < p["desaturation_ratio"]) & (bright >= p["desaturation_min_brightness"])
           & (bright <= p["desaturation_max_brightness"]))
    for a, b in mask_to_intervals(low, times, step, merge_gap=p["merge_gap"],
                                  min_duration=p["desaturation_min_duration"]):
        s, e = float(times[a]), min(duration, float(times[b - 1]) + step)
        # "색이 많던 화면이 갑자기 흑백으로 바뀐" 경우만: 원래 색이 적던 화면(대기 화면, 어두운 맵)은 제외
        prev = color[max(0, a - int(p["desaturation_lookback"] / step)):a]
        if len(prev) == 0:
            continue
        before = float(np.median(prev))
        if before < p["desaturation_before_min"]:
            continue
        mean_c = float(color[a:b].mean())
        events.append(make_event(
            NAME, len(events) + 1, s, e, "desaturation",
            score=1.0 - mean_c / p["desaturation_ratio"],
            reason=f"{s:.1f}초부터 {e - s:.1f}초 동안 화면이 흑백으로 바뀜 (색 있는 픽셀 {before:.0%} → {mean_c:.0%})",
            data={"time": round(s, 3), "color_ratio": round(mean_c, 3), "color_ratio_before": round(before, 3)},
        ))

    events.sort(key=lambda ev: ev["start"])
    return save_events(video_id, NAME, p, events)
