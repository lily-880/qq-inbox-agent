"""M5 agent: choose the next action, optionally call tools, then stop for a human.

Sending is a PendingAction the user would approve. This module never sends mail.
"""

from __future__ import annotations

import json
from typing import Any

from .jsonutil import extract_json_object
from .llm_types import LLM, LLMResponse
from .mailstore import SampleMailStore
from .schema import AgentDecision, AgentRun, AgentStep, EmailOutcome
from .tools import ToolRegistry, build_registry
from .workflow import apply_action

AGENT_SYSTEM = """你是 QQ 收件箱助手，负责处理一批新邮件。你可以用工具查看邮件、检索、查看模拟日历和模拟的求职进度。

每封邮件最后只能做这些动作之一：
- ignore：营销或验证码。给出不含敏感数字的一行摘要后丢弃。
- summarize_only：需要留档但不需要回复（收据、宣讲会、普通通知）。
- draft_followup：需要回复。你只决定动作；起草和审稿由后续步骤完成。不要自己输出可直接发送的终稿。
- ask_user：是否回复并不清楚，先问人。

规则：
- 校园宣讲会、招聘会、就业中心活动通知不是个人求职进展，动作用 summarize_only。
- 验证码不要出现在摘要里。
- 没有发送邮件的工具。不要尝试发送。处理完必须停下来等人类确认。
- 先调用工具把邮件看完并分类，再给出结论。

最后一轮不要再调用工具，只输出 JSON：
{"decisions":[{"email_id":"...","action":"ignore|summarize_only|draft_followup|ask_user","needs_reply":false,"gist":"一行中文摘要","reason":"一句中文理由"}]}"""


def parse_agent_decisions(text: str) -> list[AgentDecision]:
    if not text or not text.strip():
        raise ValueError("Agent 没有给出最终决定")
    data = extract_json_object(text)
    rows = data.get("decisions", data.get("emails"))
    if not isinstance(rows, list):
        raise ValueError("Agent 最终输出缺少 decisions 数组")
    decisions: list[AgentDecision] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("decisions 里有非对象元素")
        item = dict(row)
        if isinstance(item.get("action"), str):
            item["action"] = item["action"].strip()
        if "needs_reply" not in item:
            item["needs_reply"] = item.get("action") in {"draft_followup", "ask_user"}
        if not str(item.get("gist") or "").strip():
            item["gist"] = "（没有摘要）"
        if not str(item.get("reason") or "").strip():
            item["reason"] = "模型没有给理由。"
        decisions.append(AgentDecision.model_validate(item))
    return decisions


def _clip(text: str, limit: int = 420) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[: limit - 1] + "…"


def _assistant_message(response: LLMResponse) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": response.content,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments, ensure_ascii=False),
                },
            }
            for call in response.tool_calls
        ],
    }


def _push(steps: list[AgentStep], kind: str, detail: str, name: str | None = None) -> None:
    steps.append(
        AgentStep(index=len(steps) + 1, kind=kind, name=name, detail=detail)  # type: ignore[arg-type]
    )


def run_agent(
    llm: LLM,
    store: SampleMailStore,
    *,
    max_steps: int = 40,
) -> AgentRun:
    """Tool loop, then draft/reflect for draft_followup, then the confirmation gate."""
    registry = build_registry(store, llm)
    return run_agent_with_registry(llm, store, registry, max_steps=max_steps)


def run_agent_with_registry(
    llm: LLM,
    store: SampleMailStore,
    registry: ToolRegistry,
    *,
    max_steps: int = 40,
) -> AgentRun:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": AGENT_SYSTEM},
        {
            "role": "user",
            "content": "处理收件箱里的新邮件。先用工具把邮件看完并分类，再给出每封邮件的动作。不要发送。",
        },
    ]
    steps: list[AgentStep] = []
    final_text: str | None = None
    for _ in range(max_steps):
        response = llm.complete(messages, tools=registry.openai_tools(), temperature=0)
        if response.tool_calls:
            messages.append(_assistant_message(response))
            for call in response.tool_calls:
                result = registry.call(call.name, call.arguments)
                _push(
                    steps,
                    "tool_call",
                    json.dumps(call.arguments, ensure_ascii=False),
                    call.name,
                )
                _push(steps, "tool_result", _clip(result), call.name)
                messages.append(
                    {"role": "tool", "tool_call_id": call.id, "content": result}
                )
            continue
        final_text = response.content or ""
        break
    else:
        raise RuntimeError(
            f"超过最大步数 {max_steps}，停在人工确认之前。没有发送邮件。"
        )

    decisions = parse_agent_decisions(final_text or "")
    outcomes: list[EmailOutcome] = []
    for decision in decisions:
        try:
            email = store.get(decision.email_id)
        except KeyError:
            outcomes.append(
                EmailOutcome(
                    email_id=decision.email_id,
                    action=decision.action,
                    needs_reply=decision.needs_reply,
                    gist=decision.gist,
                    reason=decision.reason,
                    blocked_reason="邮箱里没有这封邮件。",
                )
            )
            continue
        outcome = apply_action(
            email,
            decision,
            llm,
            gist=decision.gist,
            reason=decision.reason,
        )
        if outcome.draft:
            _push(steps, "draft", _clip(outcome.draft.body), outcome.email_id)
        if outcome.reflection:
            flags = (
                f"tone_ok={outcome.reflection.tone_ok} "
                f"faithful={outcome.reflection.faithful} "
                f"should_send={outcome.reflection.should_send}"
            )
            _push(steps, "reflection", flags, outcome.email_id)
        outcomes.append(outcome)

    pending = [outcome.pending for outcome in outcomes if outcome.pending is not None]
    _push(
        steps,
        "gate",
        f"待确认 {len(pending)} 封。停在人工确认门，没有发送。",
        "human_confirmation",
    )
    return AgentRun(steps=steps, outcomes=outcomes, pending=pending)


def seal_workflow(outcome: EmailOutcome) -> AgentRun:
    """Wrap one fixed-pipeline result in the same gate the agent uses."""
    steps: list[AgentStep] = []
    if outcome.draft:
        _push(steps, "draft", _clip(outcome.draft.body), outcome.email_id)
    if outcome.reflection:
        _push(
            steps,
            "reflection",
            (
                f"tone_ok={outcome.reflection.tone_ok} "
                f"faithful={outcome.reflection.faithful} "
                f"should_send={outcome.reflection.should_send}"
            ),
            outcome.email_id,
        )
    pending = [outcome.pending] if outcome.pending is not None else []
    _push(
        steps,
        "gate",
        "固定流水线结束。停在人工确认门，没有发送。",
        "human_confirmation",
    )
    return AgentRun(steps=steps, outcomes=[outcome], pending=pending)


def tool_names_called(run: AgentRun) -> list[str]:
    return [step.name or "" for step in run.steps if step.kind == "tool_call"]


# Re-exported so callers can build a registry that refuses unknown tools,
# including a model that asks for send_email.
__all__ = [
    "AGENT_SYSTEM",
    "parse_agent_decisions",
    "run_agent",
    "run_agent_with_registry",
    "seal_workflow",
    "tool_names_called",
]
