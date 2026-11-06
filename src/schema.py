"""Structured outputs: classification, actions, drafts, and the confirmation gate."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

EmailLabel = Literal["job", "receipt", "otp", "promo", "other"]
ActionName = Literal["ignore", "summarize_only", "draft_followup", "ask_user"]


class EmailClassification(BaseModel):
    """模型对一封邮件的分类结果（不含原始邮件正文）。"""

    label: EmailLabel = Field(
        description="五类之一：job / receipt / otp / promo / other"
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="模型自评置信度，0~1",
    )
    reason: str = Field(
        min_length=1,
        description="一句中文理由，说明为何判为该类",
    )
    summary: str = Field(
        min_length=1,
        description="根据正文提炼的一行中文要点，不是原始 subject",
    )


class ActionDecision(BaseModel):
    """下一步动作。发送不在这些动作里。"""

    action: ActionName
    needs_reply: bool
    reason: str = Field(min_length=1)


class AgentDecision(ActionDecision):
    """Agent 对一封邮件的决定。"""

    email_id: str = Field(min_length=1)
    gist: str = Field(min_length=1)


class DraftReply(BaseModel):
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class ReflectionResult(BaseModel):
    """一次审稿的结论。

    tone_ok / faithful 描述送审的那一稿。
    revised_body 是修订一次之后的正文。
    should_send 表示修订稿能不能进入人工确认，不表示已经发送。
    """

    tone_ok: bool
    faithful: bool
    should_send: bool
    issues: list[str] = Field(default_factory=list)
    revised_body: str = ""


class PendingAction(BaseModel):
    """等待人类确认的回复。没有 sent 这个状态。"""

    id: str = Field(min_length=1)
    email_id: str = Field(min_length=1)
    kind: Literal["send_reply"] = "send_reply"
    to: str = Field(min_length=1)
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)
    status: Literal["pending_confirmation"] = "pending_confirmation"


class ConfirmationRecord(BaseModel):
    """用户的选择。executed_send 只能是 False。"""

    pending_id: str
    email_id: str
    decision: Literal["approve", "reject"]
    executed_send: Literal[False] = False
    note: str


class EmailOutcome(BaseModel):
    email_id: str
    action: ActionName
    needs_reply: bool
    gist: str
    reason: str
    draft: DraftReply | None = None
    reflection: ReflectionResult | None = None
    pending: PendingAction | None = None
    question: str | None = None
    blocked_reason: str | None = None


class AgentStep(BaseModel):
    index: int
    kind: Literal["tool_call", "tool_result", "draft", "reflection", "gate"]
    name: str | None = None
    detail: str


class AgentRun(BaseModel):
    steps: list[AgentStep]
    outcomes: list[EmailOutcome]
    pending: list[PendingAction]
    stopped_at: Literal["human_confirmation"] = "human_confirmation"
    executed_send: Literal[False] = False
