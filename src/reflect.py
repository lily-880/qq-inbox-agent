"""M2 反思：草稿之后检查语气、是否忠于原信、到底该不该发，然后只修订一次。"""

from __future__ import annotations

import json
import re
from typing import Any

from .jsonutil import extract_json_object
from .llm_types import LLM
from .safety import is_otp_email
from .schema import DraftReply, ReflectionResult

REFLECT_SYSTEM = """你是回复审稿人。对照原邮件检查草稿，并且只修订一次。
检查三项：
1. tone_ok：语气是否礼貌，适合对方。用「您」，不要过于随意，不要用表情。
2. faithful：时间、金额、地点、链接和承诺是否都能在原邮件或给出的空闲时间里找到。不能编造。
3. should_send：这封邮件到底该不该回复。验证码、营销、账单、宣讲会通知一律 false。没有明确回复请求也是 false。
tone_ok 和 faithful 针对送审的原稿。revised_body 是修订一次后的全文；原稿没问题就原样返回。
should_send 表示修订稿能不能交给人确认，不表示已经发送。
只输出 JSON：
{"tone_ok": true, "faithful": true, "should_send": true, "issues": [], "revised_body": "..."}"""

_FACT = re.compile(
    r"\d{4}-\d{2}-\d{2} \d{1,2}:\d{2}-\d{1,2}:\d{2}"
    r"|\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|\d{1,2}:\d{2}"
    r"|\d+(?:\.\d+)?\s*万"
    r"|\$\d+(?:\.\d+)?"
    r"|(?<!\d)\d{4,8}(?!\d)"
)
_CLAIM_WORDS = ("薪资", "工资", "年薪", "录用", "已发送", "已发出")
_BAD_TONE = ("哈哈", "嘿嘿", "lol", "LOL", "哥们", "兄弟")


def build_reflect_messages(
    email: dict[str, Any],
    draft: DraftReply,
    *,
    needs_reply: bool,
    slots: dict[str, Any] | None,
) -> list[dict[str, str]]:
    payload = {
        "email": {
            "from": email.get("from", ""),
            "subject": email.get("subject", ""),
            "body": email.get("body", ""),
        },
        "needs_reply": needs_reply,
        "slots": slots or {},
        "draft": {"subject": draft.subject, "body": draft.body},
    }
    return [
        {"role": "system", "content": REFLECT_SYSTEM},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def parse_reflection(model_text: str) -> ReflectionResult:
    return ReflectionResult.model_validate(extract_json_object(model_text))


def reflect_once(
    llm: LLM,
    email: dict[str, Any],
    draft: DraftReply,
    *,
    needs_reply: bool,
    slots: dict[str, Any] | None,
) -> ReflectionResult:
    """调用模型恰好一次。不做第二轮审稿。"""
    response = llm.complete(
        build_reflect_messages(email, draft, needs_reply=needs_reply, slots=slots),
        temperature=0.0,
        json_mode=True,
    )
    if not response.content:
        raise RuntimeError("审稿模型没有返回内容")
    return parse_reflection(response.content)


def reply_allowed(email: dict[str, Any], needs_reply: bool) -> tuple[bool, str | None]:
    if is_otp_email(email):
        return False, "验证码邮件不应该回复"
    text = f"{email.get('subject', '')}\n{email.get('body', '')}"
    lowered = text.lower()
    if any(word in text for word in ("退订", "礼品卡", "限时优惠")) or "unsubscribe" in lowered:
        return False, "营销邮件不应该回复"
    if any(word in lowered for word in ("receipt", "invoice", "amount paid")) or any(
        word in text for word in ("账单", "发票")
    ):
        return False, "账单或收据不需要回复"
    if "宣讲会" in text:
        return False, "宣讲会通知不需要回复"
    if not needs_reply:
        return False, "没有回复请求"
    return True, None


def corpus_for(email: dict[str, Any], slots: dict[str, Any] | None) -> str:
    parts = [
        str(email.get("from", "")),
        str(email.get("subject", "")),
        str(email.get("body", "")),
    ]
    if slots:
        parts.extend(str(line) for line in slots.get("slot_lines") or [])
        proposed = slots.get("proposed")
        if isinstance(proposed, dict):
            parts.append(str(proposed.get("date") or ""))
            parts.append(str(proposed.get("time") or ""))
    return "\n".join(parts)


def _compact(text: str) -> str:
    return re.sub(r"\s+", "", text)


def invented_facts(draft_body: str, corpus: str) -> list[str]:
    compact_corpus = _compact(corpus)
    found: list[str] = []
    for match in _FACT.finditer(draft_body):
        token = match.group(0).strip()
        if _compact(token) not in compact_corpus:
            found.append(token)
    return found


def extra_claims(draft_body: str, corpus: str) -> list[str]:
    return [word for word in _CLAIM_WORDS if word in draft_body and word not in corpus]


def tone_issues(draft_body: str) -> list[str]:
    issues: list[str] = []
    if any(word in draft_body for word in _BAD_TONE):
        issues.append("语气过于随意")
    if re.search(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", draft_body):
        issues.append("含有表情符号")
    return issues


def safe_revised_body(email: dict[str, Any], slots: dict[str, Any] | None) -> str:
    """修订稿只引用主题和已经给出的空闲时间。"""
    lines = [
        "您好，",
        "",
        f"我已收到主题为「{str(email.get('subject', '')).strip()}」的邮件。",
    ]
    proposed = (slots or {}).get("proposed") if slots else None
    if isinstance(proposed, dict) and proposed.get("free") is False:
        lines.append("来信里的时间在日历上已经有安排。")
    slot_lines = list((slots or {}).get("slot_lines") or []) if slots else []
    if slot_lines:
        lines.append("我目前有空的时间是：")
        lines.extend(f"- {line}" for line in slot_lines)
    else:
        lines.append("如需我确认下一步，请直接说明时间。")
    lines.extend(["", "此致", "傅公羽"])
    return "\n".join(lines)


def fallback_reflection(payload: dict[str, Any]) -> ReflectionResult:
    """离线 stub 使用的同一张检查清单。在线路径把清单交给 DeepSeek。"""
    email = payload.get("email") or {}
    draft = payload.get("draft") or {}
    slots = payload.get("slots") or {}
    if not isinstance(slots, dict):
        slots = {}
    needs_reply = bool(payload.get("needs_reply"))
    body = str(draft.get("body") or "")
    corpus = corpus_for(email, slots)

    invented = invented_facts(body, corpus)
    claims = extra_claims(body, corpus)
    tone = tone_issues(body)
    allowed, why = reply_allowed(email, needs_reply)

    issues: list[str] = []
    if invented:
        issues.append("草稿里有原邮件和空闲时间没有的事实：" + "、".join(invented))
    if claims:
        issues.append("草稿添加了原文没有的承诺：" + "、".join(claims))
    issues.extend(tone)
    if not allowed and why:
        issues.append(why)

    if invented or claims or tone:
        revised = safe_revised_body(email, slots)
    else:
        revised = body

    revised_ok = (
        not invented_facts(revised, corpus)
        and not extra_claims(revised, corpus)
        and not tone_issues(revised)
    )
    return ReflectionResult(
        tone_ok=not tone,
        faithful=not invented and not claims,
        should_send=allowed and revised_ok,
        issues=issues,
        revised_body=revised,
    )
