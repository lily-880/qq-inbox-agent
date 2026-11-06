"""M1 固定流水线：分类 → 动作策略 → 需要时起草。起草之后交给 M2 审一次。"""

from __future__ import annotations

from typing import Any

from .draft import request_draft
from .llm_types import LLM
from .pipeline import classify_with_llm
from .policy import decide_action
from .reflect import reflect_once
from .safety import is_otp_email, public_gist, public_otp_summary
from .schema import ActionDecision, DraftReply, EmailOutcome, PendingAction
from .tools import slots_for_email


def queue_pending(email: dict[str, Any], subject: str, body: str) -> PendingAction | None:
    if is_otp_email(email) or not body.strip():
        return None
    return PendingAction(
        id=f"pending-{email['id']}",
        email_id=email["id"],
        to=email["from"],
        subject=subject,
        body=body,
    )


def apply_action(
    email: dict[str, Any],
    decision: ActionDecision,
    llm: LLM,
    *,
    gist: str,
    reason: str | None = None,
) -> EmailOutcome:
    """把一个动作落到摘要、提问或「草稿 + 一次审稿 + 待确认」。不发送。"""
    safe_gist = public_gist(email, gist)
    outcome = EmailOutcome(
        email_id=email["id"],
        action=decision.action,
        needs_reply=decision.needs_reply,
        gist=safe_gist,
        reason=reason or decision.reason,
    )
    if is_otp_email(email):
        blocked = None
        if decision.action == "draft_followup":
            blocked = "验证码邮件不起草、不进入待确认发送。"
        return outcome.model_copy(
            update={
                "action": "ignore",
                "needs_reply": False,
                "gist": public_otp_summary(email),
                "blocked_reason": blocked,
            }
        )

    if decision.action == "ask_user":
        question = (
            f"要不要为「{email.get('subject', '')}」起草回复？"
            f"{safe_gist}在你点头之前不会写终稿，也不会发送。"
        )
        return outcome.model_copy(update={"question": question})

    if decision.action != "draft_followup":
        return outcome

    slots = slots_for_email(email)
    draft: DraftReply = request_draft(llm, email, slots=slots)
    reflection = reflect_once(
        llm,
        email,
        draft,
        needs_reply=decision.needs_reply,
        slots=slots,
    )
    pending = None
    blocked = None
    if reflection.should_send:
        body = (reflection.revised_body or draft.body).strip()
        pending = queue_pending(email, draft.subject, body)
        if pending is None:
            blocked = "没有生成可等待确认的回复。"
    else:
        blocked = "审稿认为这封邮件不该发送回复。"
    return outcome.model_copy(
        update={
            "draft": draft,
            "reflection": reflection,
            "pending": pending,
            "blocked_reason": blocked,
        }
    )


def run_workflow(email: dict[str, Any], llm: LLM) -> EmailOutcome:
    """intake（调用方已读入）→ classify → action policy → optional draft → reflect once."""
    classification = classify_with_llm(llm, email)
    decision = decide_action(email, classification)
    return apply_action(
        email,
        decision,
        llm,
        gist=classification.summary,
        reason=decision.reason,
    )
