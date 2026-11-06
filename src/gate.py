"""Human confirmation gate. Approving a pending reply does not send it."""

from __future__ import annotations

import json
from pathlib import Path

from .schema import ConfirmationRecord, PendingAction


def record_confirmation(pending: PendingAction, decision: str) -> ConfirmationRecord:
    if decision not in {"approve", "reject"}:
        raise ValueError("decision 只能是 approve 或 reject")
    return ConfirmationRecord(
        pending_id=pending.id,
        email_id=pending.email_id,
        decision=decision,  # type: ignore[arg-type]
        executed_send=False,
        note="已记下你的选择。本程序没有发信功能，邮件不会被发送。",
    )


def load_confirmations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("confirmations.json 应该是数组")
    return data


def append_confirmation(path: Path, record: ConfirmationRecord) -> None:
    existing = load_confirmations(path)
    existing.append(record.model_dump())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(existing, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
