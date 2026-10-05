"""바이브컷 단계별 실행 명령어.

사용법: python cli.py <명령> [인자]
각 모듈을 만들 때마다 여기에 명령을 추가한다.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cli.py", description="바이브컷 단계별 실행")
    parser.add_subparsers(dest="command", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
