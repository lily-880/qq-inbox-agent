"""M1 动作策略：分类之后的固定规则，不调用模型。"""

from __future__ import annotations

from .safety import is_otp_email
from .schema import ActionDecision, EmailClassification

_REPLY_HINTS = (
    "请回复",
    "请确认",
    "是否能出席",
    "方便的时间",
    "如需改期",
    "提前说一声",
    "回复本邮件",
)


def email_text(email: dict) -> str:
    return f"{email.get('subject', '')}\n{email.get('body', '')}"


def needs_human_reply(email: dict) -> bool:
    text = email_text(email)
    return any(hint in text for hint in _REPLY_HINTS)


def decide_action(email: dict, classification: EmailClassification) -> ActionDecision:
    """固定策略，供流水线使用，也供离线 stub 跟随。

    验证码以正文为准：就算标签错了，也不进入起草。
    """
    if is_otp_email(email) or classification.label == "otp":
        return ActionDecision(
            action="ignore",
            needs_reply=False,
            reason="验证码邮件只留一句不含数字的摘要，然后丢弃，不回复。",
        )

    text = email_text(email)
    asks = needs_human_reply(email)
    label = classification.label

    if label == "promo":
        return ActionDecision(
            action="ignore",
            needs_reply=False,
            reason="营销邮件，保留一行摘要后当作噪声丢弃。",
        )
    if label == "receipt":
        return ActionDecision(
            action="summarize_only",
            needs_reply=False,
            reason="账单或收据，摘要留档，无需回复。",
        )
    if label == "job":
        if asks:
            return ActionDecision(
                action="draft_followup",
                needs_reply=True,
                reason="求职邮件在请对方确认时间或能否出席，起草回复并等待人工确认。",
            )
        return ActionDecision(
            action="summarize_only",
            needs_reply=False,
            reason="求职进展通知，没有明确的回复请求，只做摘要。",
        )
    if "宣讲会" in text or "活动提醒" in text:
        return ActionDecision(
            action="summarize_only",
            needs_reply=False,
            reason="校园活动或宣讲会通知，与个人投递结果无关，只做摘要。",
        )
    if asks:
        return ActionDecision(
            action="ask_user",
            needs_reply=True,
            reason="这封邮件可能需要回复，但不是明确的事务确认，先问你要不要起草。",
        )
    return ActionDecision(
        action="summarize_only",
        needs_reply=False,
        reason="普通邮件，一行摘要即可。",
    )
