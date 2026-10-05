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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="바이브컷 단계별 실행")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("ingest", help="영상 등록: video_id, meta.json, 음성 추출, 분석용 사본")
    p.add_argument("video", help="원본 영상 경로")
    p.set_defaults(func=cmd_ingest)

    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
