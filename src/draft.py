"""起草一封回复。这里只产生草稿，不发送。"""

from __future__ import annotations

import json
from typing import Any

from .jsonutil import extract_json_object
from .llm_types import LLM
from .schema import DraftReply

DRAFT_SYSTEM = """你是邮件回复起草助手。根据原邮件和给出的空闲时间写一封简短、礼貌的中文回复。
规则：
- 只使用原邮件和空闲时间里出现的事实，不要编造时间、金额、链接、薪资或承诺。
- 不要复述验证码或密码。
- 不要声称这封回复已经发出。
- 称呼用「您」，落款用「傅公羽」。
只输出 JSON：{"subject": "...", "body": "..."}"""


def build_draft_messages(
    email: dict[str, Any],
    *,
    slots: dict[str, Any] | None,
) -> list[dict[str, str]]:
    slot_lines = []
    proposed = None
    if slots:
        slot_lines = list(slots.get("slot_lines") or [])
        proposed = slots.get("proposed")
    payload = {
        "from": email.get("from", ""),
        "subject": email.get("subject", ""),
        "body": email.get("body", ""),
        "free_slots": slot_lines,
        "proposed": proposed,
    }
    return [
        {"role": "system", "content": DRAFT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_draft(model_text: str) -> DraftReply:
    return DraftReply.model_validate(extract_json_object(model_text))


def request_draft(
    llm: LLM,
    email: dict[str, Any],
    *,
    slots: dict[str, Any] | None,
) -> DraftReply:
    response = llm.complete(
        build_draft_messages(email, slots=slots),
        temperature=0.0,
        json_mode=True,
    )
    if not response.content:
        raise RuntimeError("起草模型没有返回内容")
    return parse_draft(response.content)
