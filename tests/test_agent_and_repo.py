"""Agent loop, demo CLI, recorded score, and secret hygiene."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.agent import parse_agent_decisions, run_agent
from src.llm_types import LLMResponse, ToolCall
from src.mailstore import SampleMailStore
from src.safety import is_otp_email
from src.schema import AgentRun
from src.stub_llm import StubLLM
from src.tools import build_registry

ROOT = Path(__file__).resolve().parents[1]
SAMPLES = ROOT / "data" / "samples"
OTP_CODES = ("482917", "719304")


def test_parse_agent_decisions_from_a_fence():
    text = """好的。
```json
{"decisions":[{"email_id":"a","action":"ignore","gist":"摘要","reason":"原因"}]}
```
"""
    rows = parse_agent_decisions(text)
    assert rows[0].action == "ignore"
    assert rows[0].needs_reply is False


def test_agent_on_samples_stops_at_the_gate():
    store = SampleMailStore(SAMPLES)
    run = run_agent(StubLLM(), store)
    assert run.stopped_at == "human_confirmation"
    assert run.executed_send is False
    assert len(run.outcomes) == 10

    by_id = {outcome.email_id: outcome for outcome in run.outcomes}
    assert by_id["01_job_hr_screen"].action == "draft_followup"
    assert by_id["02_job_interview_invite"].action == "draft_followup"
    assert by_id["01_job_hr_screen"].pending is not None
    assert by_id["02_job_interview_invite"].pending is not None
    assert "初筛通过" in by_id["01_job_hr_screen"].reason
    assert "待面试" in by_id["02_job_interview_invite"].reason
    assert by_id["09_other_event"].action == "summarize_only"
    assert by_id["09_other_event"].pending is None
    assert by_id["10_other_personal"].action == "ask_user"
    assert by_id["10_other_personal"].question
    assert by_id["10_other_personal"].pending is None
    assert by_id["05_otp_login"].action == "ignore"
    assert by_id["07_promo_course"].action == "ignore"
    assert by_id["03_receipt_stripe"].action == "summarize_only"

    called = {step.name for step in run.steps if step.kind == "tool_call"}
    assert called == {
        "list_emails",
        "read_email",
        "classify_email",
        "calendar_free_slots",
        "job_application_status",
        "search_emails",
    }
    blob = run.model_dump_json()
    for code in OTP_CODES:
        assert code not in blob
    for email in store.emails:
        if is_otp_email(email):
            for code in re.findall(r"(?<!\d)\d{4,8}(?!\d)", email["body"]):
                assert code not in blob


def test_unknown_send_tool_is_not_executed():
    store = SampleMailStore(SAMPLES, only_ids={"07_promo_course"})

    class Scripted:
        def complete(self, messages, *, tools=None, temperature=0.0, json_mode=False):
            del tools, temperature, json_mode
            tool_msgs = [message for message in messages if message.get("role") == "tool"]
            if not tool_msgs:
                return LLMResponse(
                    content=None,
                    tool_calls=[
                        ToolCall(id="bad", name="send_email", arguments={"to": "a@b.c"})
                    ],
                )
            assert "unknown tool: send_email" in tool_msgs[-1]["content"]
            return LLMResponse(
                content=json.dumps(
                    {
                        "decisions": [
                            {
                                "email_id": "07_promo_course",
                                "action": "ignore",
                                "needs_reply": False,
                                "gist": "促销，丢弃。",
                                "reason": "发送工具不存在。",
                            }
                        ]
                    },
                    ensure_ascii=False,
                )
            )

    run = run_agent(Scripted(), store)
    assert run.pending == []
    assert run.executed_send is False
    assert any(step.name == "send_email" and step.kind == "tool_call" for step in run.steps)
    assert any("unknown tool: send_email" in step.detail for step in run.steps)


def test_agent_stops_when_the_model_never_finishes():
    store = SampleMailStore(SAMPLES, only_ids={"10_other_personal"})

    class Loop:
        def complete(self, messages, *, tools=None, temperature=0.0, json_mode=False):
            del messages, tools, temperature, json_mode
            return LLMResponse(
                content=None,
                tool_calls=[ToolCall(id="again", name="list_emails", arguments={})],
            )

    with pytest.raises(RuntimeError, match="没有发送"):
        run_agent(Loop(), store, max_steps=2)


def test_registry_classify_does_not_echo_otp_digits():
    store = SampleMailStore(SAMPLES, only_ids={"06_otp_password_reset"})
    registry = build_registry(store, StubLLM())
    payload = json.loads(registry.call("classify_email", {"email_id": "06_otp_password_reset"}))
    assert "719304" not in json.dumps(payload, ensure_ascii=False)


def test_recorded_v1_score_and_known_miss():
    data = json.loads((ROOT / "data" / "recorded" / "classify_v1.json").read_text(encoding="utf-8"))
    assert data["total"] == 10
    assert data["correct"] == 9
    assert data["accuracy"] == 0.9
    misses = [row for row in data["rows"] if not row["ok"]]
    assert len(misses) == 1
    assert misses[0]["id"] == "09_other_event"
    assert misses[0]["gold"] == "other"
    assert misses[0]["pred"] == "job"


def test_demo_cli_and_confirm_do_not_send(tmp_path: Path):
    env = os.environ.copy()
    env.pop("DEEPSEEK_API_KEY", None)
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "demo.py"),
            "--json",
            "--outbox",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    run = AgentRun.model_validate_json(proc.stdout)
    assert run.executed_send is False
    assert run.stopped_at == "human_confirmation"
    assert len(run.pending) == 2
    for code in OTP_CODES:
        assert code not in proc.stdout

    workflow = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "demo.py"),
            "--workflow",
            "--email",
            "09_other_event",
            "--json",
            "--outbox",
            str(tmp_path / "wf"),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert workflow.returncode == 0, workflow.stderr
    campus = AgentRun.model_validate_json(workflow.stdout)
    assert campus.outcomes[0].action == "summarize_only"
    assert campus.pending == []

    confirm = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "demo.py"),
            "--confirm",
            "pending-01_job_hr_screen",
            "--decision",
            "approve",
            "--outbox",
            str(tmp_path),
        ],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert confirm.returncode == 0, confirm.stderr
    assert "executed_send=false" in confirm.stdout
    assert "不会被发送" in confirm.stdout
    saved = json.loads((tmp_path / "confirmations.json").read_text(encoding="utf-8"))
    assert saved[0]["decision"] == "approve"
    assert saved[0]["executed_send"] is False


def test_tree_has_no_secrets_or_mail_transport():
    skip = {".git", ".venv", "venv", "uploads", "__pycache__", ".pytest_cache", "outbox"}
    pattern = re.compile(r"sk-[A-Za-z0-9]{8,}")
    for path in ROOT.rglob("*"):
        if any(part in skip for part in path.parts):
            continue
        if not path.is_file():
            continue
        assert path.name != ".env"
        data = path.read_bytes()
        if b"\0" in data:
            continue
        text = data.decode("utf-8", "ignore")
        assert ("DEEPSEEK_API_KEY=" + "sk-") not in text
        assert pattern.search(text) is None, path
    assert not (ROOT / ".env").exists()
    example = (ROOT / ".env.example").read_text(encoding="utf-8")
    assert "DEEPSEEK_API_KEY=" in example
    ignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for line in (".env", ".venv/", "__pycache__/", "data/outbox/"):
        assert line in ignore

    for folder in ("src", "scripts"):
        for path in (ROOT / folder).rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            assert "import smtplib" not in text
            assert "import imaplib" not in text
            assert "smtp" not in text.lower()


def test_pending_status_rejects_sent():
    with pytest.raises(ValidationError):
        from src.schema import PendingAction

        PendingAction(
            id="p",
            email_id="e",
            to="a@b.c",
            subject="s",
            body="b",
            status="sent",
        )
