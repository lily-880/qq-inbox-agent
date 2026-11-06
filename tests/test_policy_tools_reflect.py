"""Policy, tools, and the one-pass reflection. No network."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.eval_action import evaluate_policy, evaluate_stub_invariants
from src.mailstore import SampleMailStore
from src.policy import decide_action
from src.reflect import fallback_reflection, reflect_once
from src.safety import public_otp_summary
from src.schema import ActionDecision, ConfirmationRecord, DraftReply, EmailClassification, PendingAction
from src.stub_llm import StubLLM, keyword_classification
from src.tools import (
    build_registry,
    calendar_free_slots,
    extract_proposed_slot,
    job_application_status,
)
from src.workflow import apply_action, run_workflow

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"


def _gold(label: str, summary: str = "评测摘要") -> EmailClassification:
    return EmailClassification(label=label, confidence=1, reason="金标", summary=summary)


def test_policy_matches_action_fixture():
    report = evaluate_policy()
    assert report["passed"], report["rows"]
    assert report["total"] == 10


def test_job_without_reply_request_is_summary_only():
    email = {
        "id": "x",
        "from": "hr@example.com",
        "subject": "简历已收到",
        "body": "我们已收到你的简历，后续如有安排再联系。",
    }
    decision = decide_action(email, _gold("job", "对方只是确认收到简历。"))
    assert decision.action == "summarize_only"
    assert decision.needs_reply is False


def test_otp_guard_ignores_a_wrong_job_label_and_hides_the_code():
    email = {
        "id": "otp-mislabeled",
        "from": "noreply@example.com",
        "subject": "验证码",
        "body": "验证码 482917，请勿转发。",
    }

    class ExplodingLLM:
        def complete(self, *args, **kwargs):
            raise AssertionError("验证码邮件不应该进入起草")

    outcome = apply_action(
        email,
        ActionDecision(action="draft_followup", needs_reply=True, reason="错误地想回复"),
        ExplodingLLM(),
        gist="验证码是 482917",
    )
    assert outcome.action == "ignore"
    assert outcome.pending is None
    assert "482917" not in outcome.model_dump_json()
    assert outcome.gist == public_otp_summary(email)


def test_tools_cover_samples_and_refuse_unknown_names():
    store = SampleMailStore(SAMPLES)
    registry = build_registry(store, StubLLM())
    assert registry.names() == [
        "list_emails",
        "read_email",
        "search_emails",
        "calendar_free_slots",
        "job_application_status",
        "classify_email",
    ]
    assert all("send" not in name for name in registry.names())

    listed = json.loads(registry.call("list_emails", {}))
    emails = listed["result"]["emails"]
    assert len(emails) == 10
    assert "body" not in emails[0]
    assert "label" not in emails[0]

    hidden = json.loads(
        registry.call(
            "read_email",
            {"email_id": "05_otp_login", "redact_secrets": False},
        )
    )["result"]
    assert hidden["redacted"] is True
    assert "482917" not in hidden["body"]
    assert "******" in hidden["body"]

    visible = json.loads(registry.call("read_email", {"email_id": "10_other_personal"}))
    assert "label" not in visible["result"]
    assert "组会" in visible["result"]["body"]

    assert json.loads(registry.call("search_emails", {"query": "482917"}))["result"]["emails"] == []
    found = json.loads(registry.call("search_emails", {"query": "面试"}))["result"]["emails"]
    assert {row["id"] for row in found} >= {"01_job_hr_screen", "02_job_interview_invite"}

    missing = json.loads(registry.call("send_email", {"to": "hr@example.com"}))
    assert missing["ok"] is False
    assert "unknown tool" in missing["error"]

    conflict = calendar_free_slots("2026-11-12", "14:00")
    assert conflict["proposed"]["free"] is False
    assert calendar_free_slots("2026-11-12", "15:00")["proposed"]["free"] is True
    email = store.get("02_job_interview_invite")
    assert extract_proposed_slot(email) == ("2026-11-12", "14:00")

    status = job_application_status("Example Tech")
    assert status["matches"][0]["status"] == "初筛通过"
    assert job_application_status("没有这家公司")["matches"] == []


def test_reflection_revises_an_unfaithful_draft_once():
    store = SampleMailStore(SAMPLES, only_ids={"01_job_hr_screen"})
    email = store.get("01_job_hr_screen")
    slots = calendar_free_slots()
    bad = DraftReply(
        subject="Re: 面试",
        body="您好，我确认 11 月 99 日 25:00 可以，薪资期望 100万。哈哈",
    )

    class Counting(StubLLM):
        def __init__(self) -> None:
            super().__init__()
            self.n = 0

        def complete(self, messages, **kwargs):
            self.n += 1
            return super().complete(messages, **kwargs)

    llm = Counting()
    result = reflect_once(llm, email, bad, needs_reply=True, slots=slots)
    assert llm.n == 1
    assert result.faithful is False
    assert result.tone_ok is False
    assert any("100万" in issue or "25:00" in issue for issue in result.issues)
    assert "100万" not in result.revised_body
    assert "25:00" not in result.revised_body
    assert "哈哈" not in result.revised_body
    assert result.should_send is True

    blocked = fallback_reflection(
        {
            "email": {
                "from": "a@b.c",
                "subject": "Invoice",
                "body": "Invoice total USD 5.00. Status: Paid.",
            },
            "needs_reply": True,
            "slots": {},
            "draft": {"subject": "Re", "body": "您好，收到账单。\n\n此致\n傅公羽"},
        }
    )
    assert blocked.should_send is False


def test_stub_workflow_invariants_and_interview_conflict():
    report = evaluate_stub_invariants()
    assert report["passed"], report

    store = SampleMailStore(SAMPLES)
    outcome = run_workflow(store.get("02_job_interview_invite"), StubLLM())
    assert outcome.pending is not None
    assert outcome.pending.status == "pending_confirmation"
    assert "14:00" in outcome.pending.body
    assert "已经有安排" in outcome.pending.body
    assert "2026-11-11 10:00-12:00" in outcome.pending.body
    assert outcome.reflection is not None
    assert outcome.reflection.should_send is True

    campus = keyword_classification(store.get("09_other_event"))
    assert campus.label == "other"


def test_confirmation_cannot_set_sent():
    pending = PendingAction(
        id="pending-x",
        email_id="x",
        to="a@example.com",
        subject="Re: hi",
        body="您好",
    )
    with pytest.raises(ValidationError):
        PendingAction(
            id="pending-x",
            email_id="x",
            to="a@example.com",
            subject="Re: hi",
            body="您好",
            status="sent",
        )
    with pytest.raises(ValidationError):
        ConfirmationRecord(
            pending_id=pending.id,
            email_id=pending.email_id,
            decision="approve",
            executed_send=True,
            note="no",
        )
