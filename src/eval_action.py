"""M4 extra eval: action policy and draft/reflect invariants on fixtures.

This does not call DeepSeek and it is not a classification accuracy.
The classification score stays in data/recorded/classify_v1.json.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .mailstore import SampleMailStore, load_sample_file
from .policy import decide_action
from .safety import is_otp_email
from .schema import EmailClassification
from .stub_llm import StubLLM
from .workflow import run_workflow

ROOT = Path(__file__).resolve().parents[1]


def load_expected(path: Path | None = None) -> list[dict]:
    expected_path = path or (ROOT / "data" / "action_eval_expected.json")
    data = json.loads(expected_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("action_eval_expected.json 应该是数组")
    return data


def evaluate_policy(
    sample_dir: Path | None = None,
    expected_path: Path | None = None,
) -> dict:
    """Gold label in, action out. No model call."""
    sample_dir = sample_dir or (ROOT / "data" / "samples")
    by_id = {
        email["id"]: email
        for email in (load_sample_file(path) for path in sorted(sample_dir.glob("*.json")))
    }
    rows = []
    correct = 0
    for spec in load_expected(expected_path):
        email = by_id[spec["id"]]
        if email.get("label") != spec["label"]:
            raise ValueError(f"{spec['id']} 的金标和样例文件不一致")
        classification = EmailClassification(
            label=spec["label"],
            confidence=1.0,
            reason="评测使用金标标签，不调用模型。",
            summary="评测摘要",
        )
        decision = decide_action(email, classification)
        ok = (
            decision.action == spec["action"]
            and decision.needs_reply is spec["needs_reply"]
        )
        correct += int(ok)
        rows.append(
            {
                "id": spec["id"],
                "label": spec["label"],
                "expected_action": spec["action"],
                "action": decision.action,
                "expected_needs_reply": spec["needs_reply"],
                "needs_reply": decision.needs_reply,
                "ok": ok,
                "reason": decision.reason,
            }
        )
    total = len(rows)
    return {
        "name": "action_policy_given_gold_label",
        "total": total,
        "correct": correct,
        "passed": correct == total and total > 0,
        "rows": rows,
    }


def evaluate_stub_invariants(sample_dir: Path | None = None) -> dict:
    """Run the fixed pipeline with the offline stub. Checks the gate, not accuracy."""
    sample_dir = sample_dir or (ROOT / "data" / "samples")
    store = SampleMailStore(sample_dir)
    llm = StubLLM()
    expected = {row["id"]: row for row in load_expected()}
    rows = []
    leaks: list[str] = []
    for email in store.emails:
        outcome = run_workflow(email, llm)
        blob = outcome.model_dump_json()
        leaked: list[str] = []
        if is_otp_email(email):
            for code in re.findall(r"(?<!\d)\d{4,8}(?!\d)", email["body"]):
                if code in blob:
                    leaked.append(code)
                    leaks.append(f"{email['id']}:{code}")
        spec = expected[email["id"]]
        pending = outcome.pending is not None
        rows.append(
            {
                "id": email["id"],
                "action": outcome.action,
                "action_matches_policy_fixture": outcome.action == spec["action"],
                "needs_reply": outcome.needs_reply,
                "has_reflection": outcome.reflection is not None,
                "pending": outcome.pending.id if outcome.pending else None,
                "should_send": (
                    outcome.reflection.should_send if outcome.reflection else None
                ),
                "leaked_codes": leaked,
                "executed_send": False,
            }
        )
    actions_ok = all(row["action_matches_policy_fixture"] for row in rows)
    reflections_ok = all(
        (row["action"] == "draft_followup") == row["has_reflection"] for row in rows
    )
    pending_ok = all(
        (row["pending"] is not None) == (row["should_send"] is True) for row in rows
    )
    drafts_queued = all(row["action"] != "draft_followup" or row["pending"] for row in rows)
    return {
        "name": "stub_workflow_invariants",
        "passed": actions_ok and reflections_ok and pending_ok and drafts_queued and not leaks,
        "actions_match_fixture": actions_ok,
        "reflection_only_on_drafts": reflections_ok,
        "pending_only_when_should_send": pending_ok,
        "drafts_are_queued": drafts_queued,
        "leaked_otp_codes": leaks,
        "rows": rows,
    }
