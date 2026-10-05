"""편집 에이전트 루프 (Anthropic SDK tool calling, 프레임워크 없이 직접 구현).

send(사용자 입력)를 부르면 에이전트가 도구를 쓰며 진행하다가
* 최종 응답을 하거나 (questions=None)
* ask_user로 질문하면 (questions=[...]) 멈추고 돌아온다.
질문에 대한 답도 send()로 넘기면 이어서 진행한다.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import anthropic

from vibecut.agent.log import AgentLogger
from vibecut.agent.prompts import build_system_prompt
from vibecut.agent.tools import TOOLS, ToolBox
from vibecut.config import load_config

FALLBACK_BETA = "server-side-fallback-2026-07-01"
# 도구 호출 사이의 진행 설명을 텍스트로 받는다 (내부 추론은 숨겨진 채로 유지됨)
UPDATES_BETA = "thinking-display-updates-2026-08-18"


@dataclass
class AgentReply:
    text: str
    questions: list[dict[str, Any]] | None = None


class EditAgent:
    def __init__(self, video_id: str, logger: AgentLogger | None = None) -> None:
        self.cfg = load_config()["agent"]
        self.video_id = video_id
        self.tools = ToolBox(video_id)
        self.log = logger or AgentLogger(video_id)
        self.client = anthropic.Anthropic()
        self.system = build_system_prompt(self.cfg["max_questions"])
        self.messages: list[dict[str, Any]] = []
        # ask_user로 멈췄을 때: (같은 턴의 다른 도구 결과들, ask_user의 tool_use id)
        self._pending: tuple[list[dict[str, Any]], str] | None = None
        self.total_cost = 0.0  # 이 대화의 누적 예상 비용(USD)

    @property
    def waiting_for_answer(self) -> bool:
        return self._pending is not None

    def send(self, user_text: str) -> AgentReply:
        self.log.log("user", user_text)
        if self._pending:
            results, ask_id = self._pending
            self._pending = None
            results.append({"type": "tool_result", "tool_use_id": ask_id,
                            "content": f"사용자 답변: {user_text}"})
            self.messages.append({"role": "user", "content": results})
        else:
            self.messages.append({"role": "user", "content": user_text})
        return self._run()

    def _create(self) -> Any:
        return self.client.beta.messages.create(
            model=self.cfg["model"],
            max_tokens=self.cfg["max_tokens"],
            system=self.system,
            tools=TOOLS,
            messages=self.messages,
            thinking={"type": "adaptive", "display": "updates"},
            output_config={"effort": self.cfg["effort"]},
            cache_control={"type": "ephemeral"},
            betas=[FALLBACK_BETA, UPDATES_BETA],
            fallbacks="default",
        )

    def _record_usage(self, usage: Any) -> None:
        """호출마다 토큰 사용량과 예상 비용(config 단가 기준)을 판단 로그에 남긴다."""
        price = self.cfg["price_per_mtok"]
        tokens = {
            "input": usage.input_tokens or 0,
            "output": usage.output_tokens or 0,
            "cache_write": getattr(usage, "cache_creation_input_tokens", 0) or 0,
            "cache_read": getattr(usage, "cache_read_input_tokens", 0) or 0,
        }
        cost = sum(tokens[k] * price[k] for k in tokens) / 1_000_000
        self.total_cost += cost
        self.log.log("usage", f"입력 {tokens['input']:,} / 캐시쓰기 {tokens['cache_write']:,} / "
                              f"캐시읽기 {tokens['cache_read']:,} / 출력 {tokens['output']:,} 토큰 → "
                              f"${cost:.4f} (누적 ${self.total_cost:.4f})",
                     tokens=tokens, cost=round(cost, 6), total_cost=round(self.total_cost, 6))

    def _run(self) -> AgentReply:
        for _ in range(self.cfg["max_turns"]):
            try:
                resp = self._create()
            except anthropic.APIError as e:
                self.log.log("error", f"Claude API 오류: {e}")
                return AgentReply(f"Claude API 호출에 실패했습니다: {e}")

            self._record_usage(resp.usage)
            # 응답 블록은 그대로 대화에 붙인다 (thinking 블록 보존)
            self.messages.append({"role": "assistant", "content": resp.content})
            text_parts = []
            # 도구 호출과 함께 온 텍스트는 진행 설명이므로 판단 로그로 분류한다
            has_tools = any(b.type == "tool_use" for b in resp.content)
            for block in resp.content:
                if block.type == "thinking" and block.thinking:
                    self.log.log("thinking", block.thinking)
                elif block.type == "text" and block.text.strip():
                    text_parts.append(block.text)
                    self.log.log("thinking" if has_tools else "assistant", block.text)
            text = "\n".join(text_parts)

            if resp.stop_reason == "refusal":
                d = resp.stop_details
                self.log.log("error", f"요청이 거부되었습니다. (분류: {getattr(d, 'category', None)}, "
                                      f"설명: {getattr(d, 'explanation', None)})")
                return AgentReply(text or "요청이 거부되어 진행할 수 없습니다.")

            tool_uses = [b for b in resp.content if b.type == "tool_use"]
            if not tool_uses:
                return AgentReply(text)

            results: list[dict[str, Any]] = []
            ask = None
            for tu in tool_uses:
                self.log.log("tool_call", tu.input, tool=tu.name)
                if tu.name == "ask_user":
                    ask = tu
                    continue
                content, is_error = self.tools.execute(tu.name, dict(tu.input))
                self.log.log("error" if is_error else "tool_result", content, tool=tu.name)
                results.append({"type": "tool_result", "tool_use_id": tu.id,
                                "content": content, "is_error": is_error})

            if ask is not None:
                questions = list(ask.input.get("questions", []))
                if not questions or len(questions) > self.cfg["max_questions"]:
                    results.append({"type": "tool_result", "tool_use_id": ask.id, "is_error": True,
                                    "content": f"질문은 1~{self.cfg['max_questions']}개여야 합니다."})
                else:
                    self.log.log("question", questions, tool="ask_user")
                    self._pending = (results, ask.id)
                    return AgentReply(text, questions=questions)

            self.messages.append({"role": "user", "content": results})

        self.log.log("error", f"최대 반복 횟수({self.cfg['max_turns']})에 도달해 멈췄습니다.")
        return AgentReply("작업 단계가 너무 길어져 멈췄습니다. 요청을 조금 더 구체적으로 말씀해 주세요.")


def format_questions(questions: list[dict[str, Any]]) -> str:
    def t(s: float) -> str:
        return f"{int(s // 60)}:{s % 60:05.2f}"
    return "\n".join(f"{i}. [{t(q['start'])} ~ {t(q['end'])}] {q['summary']}\n   고민되는 이유: {q['why_unsure']}"
                     for i, q in enumerate(questions, start=1))

