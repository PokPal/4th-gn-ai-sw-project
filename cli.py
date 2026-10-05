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

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
