"""바이브컷 단계별 실행 명령어.

사용법: python cli.py <명령> [인자]
각 모듈을 만들 때마다 여기에 명령을 추가한다.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def cmd_ingest(args: argparse.Namespace) -> None:
    from vibecut.media.ingest import ingest

    ingest(Path(args.video))


def cmd_transcribe(args: argparse.Namespace) -> None:
    from vibecut.stt.run import transcribe

    t = transcribe(args.video_id, force=args.force)
    for seg in t["segments"][: args.show]:
        print(f"  [{seg['start']:8.3f} ~ {seg['end']:8.3f}] {seg['text']}  (단어 {len(seg['words'])}개)")


def cmd_detect(args: argparse.Namespace) -> None:
    import yaml

    from vibecut.detectors import DETECTORS

    params = {k: yaml.safe_load(v) for k, v in (s.split("=", 1) for s in args.set)}
    names = list(DETECTORS) if args.detector == "all" else [args.detector]
    for name in names:
        doc = DETECTORS[name](args.video_id, params if args.detector != "all" else None)
        for ev in doc["events"][: args.show]:
            print(f"    [{ev['start']:8.3f} ~ {ev['end']:8.3f}] {ev['type']:<17} "
                  f"{ev['score']:.2f}  {ev['reason']}")


def cmd_score(args: argparse.Namespace) -> None:
    from vibecut.scoring.score import score

    doc = score(args.video_id)
    for ev in sorted(doc["events"], key=lambda e: -e["score"])[: args.show]:
        print(f"    [{ev['start']:8.3f} ~ {ev['end']:8.3f}] {ev['id']}  {ev['score']:.2f}  {ev['reason']}")


def cmd_analyze(args: argparse.Namespace) -> None:
    """영상 입력부터 점수화까지 한 번에 (각 단계는 캐시를 사용)."""
    from vibecut.detectors import DETECTORS
    from vibecut.media.ingest import ingest
    from vibecut.scoring.score import score
    from vibecut.stt.run import transcribe

    video_id = ingest(Path(args.video))["video_id"]
    transcribe(video_id)
    for detect in DETECTORS.values():
        detect(video_id)
    score(video_id)
    print(f"\n분석 완료. 편집 시작: python cli.py edit {video_id}")


def cmd_edit(args: argparse.Namespace) -> None:
    from vibecut.agent.loop import EditAgent, format_questions

    agent = EditAgent(args.video_id)
    print(f"판단 로그: {agent.log.path}")
    message = args.goal or input("\n편집 방향을 말해 주세요 > ").strip()
    while message:
        reply = agent.send(message)
        if reply.text:
            print(f"\n[바이브컷] {reply.text}")
        if reply.questions:
            print("\n[질문]\n" + format_questions(reply.questions))
        try:
            message = input("\n> ").strip()
        except EOFError:
            break


def cmd_export(args: argparse.Namespace) -> None:
    from vibecut.export.resolve import export_edit
    from vibecut.storage.edits import edit_path, latest_edit

    path = edit_path(args.video_id, args.edit_id) if args.edit_id else latest_edit(args.video_id)
    if path is None or not path.exists():
        raise SystemExit("편집 결정 목록이 없습니다. 먼저 edit를 실행하세요.")
    export_edit(args.video_id, path)


def cmd_ui(args: argparse.Namespace) -> None:
    from vibecut.ui.app import launch

    launch(port=args.port)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="바이브컷 단계별 실행")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="영상 등록: video_id, meta.json, 음성 추출, 분석용 사본")
    p.add_argument("video", help="원본 영상 경로")
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("transcribe", help="음성 인식 → transcript.json (영상당 1회, 캐시)")
    p.add_argument("video_id")
    p.add_argument("--force", action="store_true", help="캐시를 무시하고 다시 인식")
    p.add_argument("--show", type=int, default=10, help="출력할 문장 수")
    p.set_defaults(func=cmd_transcribe)

    p = sub.add_parser("detect", help="신호 탐지 → events/<탐지기>.json")
    p.add_argument("detector", choices=["all", "silence", "loudness", "reaction", "visual", "dead_time"])
    p.add_argument("video_id")
    p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                   help="config.yaml 값을 이번 실행에만 바꿈 (예: --set threshold_db=6)")
    p.add_argument("--show", type=int, default=10, help="탐지기별 출력할 이벤트 수")
    p.set_defaults(func=cmd_detect)

    p = sub.add_parser("score", help="하이라이트 점수화 → candidates.json")
    p.add_argument("video_id")
    p.add_argument("--show", type=int, default=10, help="점수 높은 순으로 출력할 후보 수")
    p.set_defaults(func=cmd_score)

    p = sub.add_parser("analyze", help="ingest → transcribe → detect all → score 를 한 번에")
    p.add_argument("video", help="원본 영상 경로")
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("edit", help="편집 에이전트와 대화 (빈 줄 입력 시 종료)")
    p.add_argument("video_id")
    p.add_argument("--goal", help="첫 요청 (생략하면 입력받음)")
    p.set_defaults(func=cmd_edit)

    p = sub.add_parser("export", help="편집 결정 목록 → timeline.otio, subtitles.srt, markers.edl")
    p.add_argument("video_id")
    p.add_argument("--edit-id", help="edits/<edit_id>.json (생략하면 최신)")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("ui", help="화면 실행 (브라우저에서 http://127.0.0.1:7860)")
    p.add_argument("--port", type=int, default=7860)
    p.set_defaults(func=cmd_ui)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
