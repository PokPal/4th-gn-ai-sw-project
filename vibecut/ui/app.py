"""바이브컷 화면 (Gradio): 영상 선택·분석 + 에이전트 채팅 + 판단 로그 + 결과·내보내기."""

from __future__ import annotations

import queue
import shutil
import threading
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from vibecut.agent.log import AgentLogger
from vibecut.agent.loop import AgentReply, EditAgent
from vibecut.config import PROJECT_ROOT, load_config, project_path
from vibecut.detectors import DETECTORS
from vibecut.media.ingest import META_FILE, ingest
from vibecut.media.preview import make_preview
from vibecut.scoring.score import score
from vibecut.storage.cache import cache_dir, read_json
from vibecut.storage.edits import latest_edit
from vibecut.stt.run import transcribe

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm"}
LOG_HEADERS = ["시간", "구분", "도구", "내용"]
CLIP_HEADERS = ["#", "시작", "끝", "길이(초)", "마커", "선택 이유"]


def _t(sec: float) -> str:
    return f"{int(sec // 60)}:{sec % 60:05.2f}"


def _data_dir() -> Path:
    return PROJECT_ROOT / "data"


def list_videos() -> list[str]:
    d = _data_dir()
    return sorted(str(p.relative_to(PROJECT_ROOT)) for p in d.iterdir() if p.suffix.lower() in VIDEO_EXTS) \
        if d.exists() else []


def save_upload(file_path: str | None) -> Any:
    """업로드한 영상을 data/로 복사한다 (Gradio 임시 파일은 사라지므로 원본 경로로 쓰지 않음)."""
    if not file_path:
        return gr.update()
    src = Path(file_path)
    dest = _data_dir() / src.name
    if not dest.exists():
        shutil.copy2(src, dest)
    return gr.update(choices=list_videos(), value=str(dest.relative_to(PROJECT_ROOT)))


def analyze(video: str | None, state: dict[str, Any]) -> Iterator[tuple[str, dict[str, Any]]]:
    if not video:
        yield "영상을 먼저 선택하세요.", state
        return
    lines: list[str] = []

    def step(msg: str) -> str:
        lines.append(msg)
        return "\n".join(lines)

    yield step(f"영상 등록 중: {video}"), state
    meta = ingest(PROJECT_ROOT / video)
    vid = meta["video_id"]
    yield step(f"✔ 등록 완료 — video_id {vid}, 길이 {_t(meta['duration'])}"), state
    yield step("음성 인식 중 (영상당 한 번, 이후 캐시 사용)..."), state
    t = transcribe(vid)
    yield step(f"✔ 음성 인식 — 문장 {len(t['segments'])}개"), state
    for name, detect in DETECTORS.items():
        yield step(f"{name} 탐지 중..."), state
        doc = detect(vid)
        yield step(f"✔ {name} — 이벤트 {len(doc['events'])}개"), state
    cands = score(vid)["events"]
    yield step(f"✔ 하이라이트 후보 {len(cands)}개\n\n분석 완료. 오른쪽 채팅에 편집 방향을 말해 주세요."), \
        {**state, "video_id": vid, "agent": None}


def _new_agent(video_id: str, q: queue.Queue) -> EditAgent:
    logger = AgentLogger(video_id, on_entry=q.put, echo=False)
    return EditAgent(video_id, logger=logger)


def _log_rows(entries: list[dict[str, Any]]) -> list[list[str]]:
    return [[e["time"][11:], e["label"], e.get("tool", ""), e["summary"]] for e in entries]


def _edit_view(video_id: str) -> tuple[list[list[Any]], list[str], list[str]]:
    """최신 편집 결정 목록 표, 내보낸 파일 목록, 미리보기 선택지."""
    path = latest_edit(video_id)
    if path is None:
        return [], [], []
    edit = read_json(path)
    rows = [[c["order"], _t(c["start"]), _t(c["end"]), round(c["end"] - c["start"], 1),
             c["marker"]["name"], c["reason"]] for c in edit["clips"]]
    exp = cache_dir(video_id) / "exports" / path.stem
    files = [str(p) for p in sorted(exp.glob("*"))] if exp.exists() else []
    choices = [f"클립 {c['order']} | {c['start']:.2f}-{c['end']:.2f}" for c in edit["clips"]]
    return rows, files, choices


def chat(message: str, history: list[dict[str, Any]], state: dict[str, Any]):
    vid = state.get("video_id")
    if not message.strip():
        yield history, gr.update(), gr.update(), gr.update(), gr.update(), state, ""
        return
    if not vid:
        history = history + [{"role": "user", "content": message},
                             {"role": "assistant", "content": "먼저 왼쪽에서 영상을 선택하고 [분석]을 눌러 주세요."}]
        yield history, gr.update(), gr.update(), gr.update(), gr.update(), state, ""
        return

    q: queue.Queue = queue.Queue()
    agent: EditAgent | None = state.get("agent")
    if agent is None or agent.video_id != vid:
        agent = _new_agent(vid, q)
        state = {**state, "agent": agent, "log": []}
    else:
        agent.log.on_entry = q.put
    log: list[dict[str, Any]] = list(state.get("log", []))

    history = history + [{"role": "user", "content": message}]
    result: dict[str, AgentReply] = {}
    worker = threading.Thread(target=lambda: result.setdefault("reply", agent.send(message)), daemon=True)
    worker.start()

    # 에이전트가 일하는 동안 판단 로그를 실시간으로 갱신
    while worker.is_alive() or not q.empty():
        try:
            log.append(q.get(timeout=0.3))
            yield (history + [{"role": "assistant", "content": "⏳ 작업 중..."}], _log_rows(log),
                   gr.update(), gr.update(), gr.update(), state, "")
        except queue.Empty:
            continue

    reply = result.get("reply") or AgentReply("응답을 받지 못했습니다.")
    text = reply.text or ""
    preview_choices: list[str] = []
    if reply.questions:
        qs = "\n".join(f"**{i}. [{_t(x['start'])} ~ {_t(x['end'])}]** {x['summary']}\n   - 고민되는 이유: {x['why_unsure']}"
                       for i, x in enumerate(reply.questions, start=1))
        text = (text + "\n\n" if text else "") + f"**질문**\n{qs}\n\n왼쪽에서 구간을 미리보고 답해 주세요."
        preview_choices = [f"질문 {i} | {x['start']:.2f}-{x['end']:.2f}" for i, x in enumerate(reply.questions, 1)]
    history = history + [{"role": "assistant", "content": text or "(응답 없음)"}]

    rows, files, clip_choices = _edit_view(vid)
    choices = preview_choices + clip_choices
    state = {**state, "log": log}
    yield (history, _log_rows(log), rows, files or None,
           gr.update(choices=choices, value=choices[0] if preview_choices else None), state, "")


def preview(choice: str | None, state: dict[str, Any]) -> Any:
    vid = state.get("video_id")
    if not choice or not vid:
        return None
    start, end = (float(x) for x in choice.split("|")[1].strip().split("-"))
    return str(make_preview(vid, start, end))


def reset(state: dict[str, Any]):
    return [], [], [], None, gr.update(choices=[], value=None), {**state, "agent": None, "log": []}


def analyzed_videos() -> list[str]:
    root = project_path(load_config()["paths"]["cache_dir"])
    out = []
    for meta_path in sorted(root.glob(f"*/{META_FILE}")) if root.exists() else []:
        m = read_json(meta_path)
        out.append(f"{Path(m['source_path']).name} | {m['video_id']}")
    return out


def select_analyzed(choice: str | None, state: dict[str, Any]):
    if not choice:
        return gr.update(), state
    vid = choice.split("|")[1].strip()
    return f"분석된 영상 선택: {choice}\n오른쪽 채팅에 편집 방향을 말해 주세요.", {**state, "video_id": vid, "agent": None}


def build_app() -> gr.Blocks:
    with gr.Blocks(title="바이브컷 VibeCut", fill_height=True) as app:
        state = gr.State({})
        gr.Markdown("## 🎬 바이브컷 — 말로 하는 게임 하이라이트 편집\n"
                    "영상을 분석하고, 원하는 편집 방향을 말하면 에이전트가 컷 구성 초안을 만들어 다빈치 리졸브로 내보냅니다.")
        with gr.Row(equal_height=False):
            with gr.Column(scale=3):
                gr.Markdown("### 1. 영상")
                with gr.Tab("새 영상 분석"):
                    video_dd = gr.Dropdown(choices=list_videos(), label="data/ 폴더의 영상", interactive=True)
                    upload = gr.File(label="또는 영상 업로드 (data/로 복사됨)", file_types=["video"], type="filepath")
                    analyze_btn = gr.Button("분석", variant="primary")
                with gr.Tab("분석된 영상"):
                    analyzed_dd = gr.Dropdown(choices=analyzed_videos(), label="이미 분석한 영상", interactive=True)
                status = gr.Textbox(label="진행 상황", lines=8, interactive=False)
                gr.Markdown("### 구간 미리보기")
                preview_dd = gr.Dropdown(choices=[], label="질문 구간 또는 편집 클립", interactive=True)
                video = gr.Video(label="미리보기", interactive=False, height=260)
            with gr.Column(scale=4):
                gr.Markdown("### 2. 편집 에이전트")
                chatbot = gr.Chatbot(label="대화", height=520)
                msg = gr.Textbox(placeholder="예: 10분짜리로, 반응 큰 장면 위주로 만들어줘 / 질문에 대한 답", label="메시지",
                                 lines=2)
                with gr.Row():
                    send_btn = gr.Button("보내기", variant="primary")
                    reset_btn = gr.Button("새 대화")
            with gr.Column(scale=4):
                gr.Markdown("### 3. 판단 로그")
                log_df = gr.Dataframe(headers=LOG_HEADERS, wrap=True, interactive=False,
                                      column_widths=["12%", "16%", "18%", "54%"], max_height=380)
                gr.Markdown("### 4. 편집 결과")
                clips_df = gr.Dataframe(headers=CLIP_HEADERS, wrap=True, interactive=False, max_height=260)
                files = gr.File(label="다빈치 리졸브용 파일 (timeline.otio / subtitles.srt / markers.edl)",
                                file_count="multiple", interactive=False)

        upload.upload(save_upload, upload, video_dd)
        analyze_btn.click(analyze, [video_dd, state], [status, state]).then(
            lambda: gr.update(choices=analyzed_videos()), None, analyzed_dd)
        analyzed_dd.change(select_analyzed, [analyzed_dd, state], [status, state])
        outputs = [chatbot, log_df, clips_df, files, preview_dd, state, msg]
        send_btn.click(chat, [msg, chatbot, state], outputs)
        msg.submit(chat, [msg, chatbot, state], outputs)
        reset_btn.click(reset, state, [chatbot, log_df, clips_df, files, preview_dd, state])
        preview_dd.change(preview, [preview_dd, state], video)
    return app


def launch(port: int = 7860) -> None:
    build_app().launch(server_port=port, allowed_paths=[str(PROJECT_ROOT / "data")], theme=gr.themes.Soft())
