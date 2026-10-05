"""다빈치 리졸브 내보내기: 편집 결정 목록(CLAUDE.md 6장 ③) 하나를 입력으로 받아

* timeline.otio : 컷이 적용된 타임라인 (영상·음성 트랙, 클립마다 마커와 선택 이유 메모)
* subtitles.srt : 편집된 타임라인 기준 자막
* markers.edl   : 타임라인 마커 (리졸브의 "Import Timeline Markers from EDL"용)

을 만든다. 시간은 이 단계에서만 프레임으로 바꾼다.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import opentimelineio as otio

from vibecut.config import load_config
from vibecut.media.ingest import META_FILE
from vibecut.storage.cache import cache_dir, read_json
from vibecut.stt.run import TRANSCRIPT_FILE

EXPORTS_DIR = "exports"
OTIO_COLORS = {c.lower(): getattr(otio.schema.MarkerColor, c)
               for c in dir(otio.schema.MarkerColor) if c.isupper()}


def _frames(seconds: float, fps: float) -> int:
    return int(round(seconds * fps))


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _timecode(frames: int, fps: float) -> str:
    rate = int(round(fps))
    f = frames % rate
    total_s = frames // rate
    return f"{total_s // 3600:02d}:{total_s // 60 % 60:02d}:{total_s % 60:02d}:{f:02d}"


def _timeline_layout(edit: dict[str, Any], fps: float) -> list[tuple[dict[str, Any], int, int, int]]:
    """클립마다 (클립, 원본 시작 프레임, 길이 프레임, 타임라인 시작 프레임). 프레임 단위로 이어 붙인다."""
    out, pos = [], 0
    for clip in sorted(edit["clips"], key=lambda c: c["order"]):
        src = _frames(clip["start"], fps)
        length = _frames(clip["end"], fps) - src
        if length <= 0:
            continue
        out.append((clip, src, length, pos))
        pos += length
    return out


def write_otio(edit: dict[str, Any], meta: dict[str, Any], path: Path, fps: float, start_s: float) -> None:
    default_color = load_config()["export"]["default_marker_color"]
    source = Path(meta["source_path"])
    available = otio.opentime.TimeRange(
        otio.opentime.RationalTime(0, fps),
        otio.opentime.RationalTime(_frames(meta["duration"], fps), fps))

    timeline = otio.schema.Timeline(name=f"VibeCut {edit['video_id']}")
    timeline.global_start_time = otio.opentime.RationalTime(_frames(start_s, fps), fps)
    v_track = otio.schema.Track(name="V1", kind=otio.schema.TrackKind.Video)
    a_track = otio.schema.Track(name="A1", kind=otio.schema.TrackKind.Audio)

    for clip, src, length, _ in _timeline_layout(edit, fps):
        rng = otio.opentime.TimeRange(otio.opentime.RationalTime(src, fps),
                                      otio.opentime.RationalTime(length, fps))
        for track in (v_track, a_track):
            ref = otio.schema.ExternalReference(target_url=source.as_uri(), available_range=available)
            item = otio.schema.Clip(name=source.name, media_reference=ref, source_range=rng)
            if track is v_track:
                m = clip.get("marker") or {}
                color = OTIO_COLORS.get(str(m.get("color", default_color)).lower(), otio.schema.MarkerColor.BLUE)
                marker = otio.schema.Marker(
                    name=m.get("name") or f"Clip {clip['order']}",
                    marked_range=otio.opentime.TimeRange(otio.opentime.RationalTime(src, fps),
                                                         otio.opentime.RationalTime(1, fps)),
                    color=color,
                )
                marker.comment = " / ".join(x for x in [m.get("note"), clip.get("reason")] if x)
                item.markers.append(marker)
            track.append(item)

    timeline.tracks.append(v_track)
    timeline.tracks.append(a_track)
    otio.adapters.write_to_file(timeline, str(path))


def write_srt(edit: dict[str, Any], transcript: dict[str, Any], path: Path, fps: float) -> int:
    """원본 자막을 잘라 편집된 타임라인 시간으로 옮긴다."""
    lines, n, prev_end = [], 0, 0.0
    for clip, src, length, pos in _timeline_layout(edit, fps):
        c_start, c_end = src / fps, (src + length) / fps
        for seg in transcript["segments"]:
            s, e = max(seg["start"], c_start), min(seg["end"], c_end)
            if e - s <= 0.05:
                continue
            t0 = max(prev_end, pos / fps + (s - c_start))  # 자막끼리 겹치지 않게
            t1 = pos / fps + (e - c_start)
            if t1 - t0 <= 0.05:
                continue
            n += 1
            prev_end = t1
            lines.append(f"{n}\n{_srt_time(t0)} --> {_srt_time(t1)}\n{seg['text']}\n")
    path.write_text("\n".join(lines), encoding="utf-8")
    return n


def write_marker_edl(edit: dict[str, Any], path: Path, fps: float, start_s: float) -> None:
    default_color = load_config()["export"]["default_marker_color"]
    start_f = _frames(start_s, fps)
    rows = [f"TITLE: VibeCut {edit['video_id']}", "FCM: NON-DROP FRAME", ""]
    for i, (clip, _, _, pos) in enumerate(_timeline_layout(edit, fps), start=1):
        m = clip.get("marker") or {}
        tc_in, tc_out = _timecode(start_f + pos, fps), _timecode(start_f + pos + 1, fps)
        note = " / ".join(x for x in [m.get("note"), clip.get("reason")] if x).replace("|", "/")
        name = (m.get("name") or f"Clip {clip['order']}").replace("|", "/")
        rows.append(f"{i:03d}  001      V     C        {tc_in} {tc_out} {tc_in} {tc_out}  ")
        rows.append(f" |C:ResolveColor{str(m.get('color', default_color)).capitalize()} |M:{name} |D:1")
        if note:
            rows.append(f"* {note}")
        rows.append("")
    path.write_text("\n".join(rows), encoding="utf-8")


def export_edit(video_id: str, edit_path: Path) -> dict[str, Any]:
    cfg = load_config()["export"]
    out = cache_dir(video_id)
    meta = read_json(out / META_FILE)
    edit = read_json(edit_path)
    fps = float(meta.get("fps") or cfg["default_fps"])
    start_s = float(cfg["timeline_start"])

    dest = out / EXPORTS_DIR / edit_path.stem
    dest.mkdir(parents=True, exist_ok=True)
    write_otio(edit, meta, dest / "timeline.otio", fps, start_s)
    n_subs = write_srt(edit, read_json(out / TRANSCRIPT_FILE), dest / "subtitles.srt", fps)
    write_marker_edl(edit, dest / "markers.edl", fps, start_s)

    layout = _timeline_layout(edit, fps)
    total = sum(length for _, _, length, _ in layout) / fps
    print(f"[export] 클립 {len(layout)}개, 총 {total:.1f}초, 자막 {n_subs}줄 -> {dest}")
    return {"dir": str(dest), "files": ["timeline.otio", "subtitles.srt", "markers.edl"],
            "clips": len(layout), "total_duration": round(total, 3), "subtitles": n_subs}
