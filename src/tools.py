"""M3 tool registry.

The agent can call these functions. There is no send tool: a reply becomes a
PendingAction and stops at the human confirmation gate.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .llm_types import LLM
from .mailstore import SampleMailStore
from .pipeline import classify_with_llm

ROOT = Path(__file__).resolve().parents[1]
_PROPOSE = re.compile(r"(\d{1,2})\s*月\s*(\d{1,2})\s*日[^\d]{0,12}(\d{1,2}:\d{2})")


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict[str, Any]
    handler: Callable[[dict[str, Any]], Any]


class ToolRegistry:
    def __init__(self, specs: list[ToolSpec]) -> None:
        self._specs = {spec.name: spec for spec in specs}

    def names(self) -> list[str]:
        return list(self._specs)

    def openai_tools(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": spec.name,
                    "description": spec.description,
                    "parameters": spec.parameters,
                },
            }
            for spec in self._specs.values()
        ]

    def call(self, name: str, arguments: dict[str, Any] | None = None) -> str:
        spec = self._specs.get(name)
        if spec is None:
            # Unknown names, including anything that would send mail, are refused.
            payload = {"ok": False, "error": f"unknown tool: {name}"}
            return json.dumps(payload, ensure_ascii=False)
        try:
            result = spec.handler(arguments or {})
            payload = {"ok": True, "result": result}
        except Exception as exc:  # noqa: BLE001
            payload = {"ok": False, "error": str(exc)}
        return json.dumps(payload, ensure_ascii=False)


def _object_schema(properties: dict[str, Any], required: list[str] | None = None) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": required or [],
        "additionalProperties": False,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_hhmm(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)


def calendar_free_slots(
    propose_date: str | None = None,
    propose_time: str | None = None,
) -> dict[str, Any]:
    """Mock calendar. Not a live calendar."""
    data = load_json(ROOT / "data" / "mocks" / "calendar.json")
    slot_lines = [
        f"{slot['date']} {slot['start']}-{slot['end']}" for slot in data["free_slots"]
    ]
    proposed = None
    if propose_date and propose_time:
        free = True
        minutes = _parse_hhmm(propose_time)
        for busy in data["busy"]:
            if busy["date"] != propose_date:
                continue
            start = _parse_hhmm(busy["start"])
            end = _parse_hhmm(busy["end"])
            if start <= minutes < end:
                free = False
                break
        proposed = {"date": propose_date, "time": propose_time, "free": free}
    return {
        "timezone": data["timezone"],
        "free_slots": data["free_slots"],
        "busy": data["busy"],
        "slot_lines": slot_lines,
        "proposed": proposed,
        "note": "模拟日历，不是真的日程。",
    }


def job_application_status(query: str = "") -> dict[str, Any]:
    """Mock application tracker. Not a live recruiting system."""
    data = load_json(ROOT / "data" / "mocks" / "applications.json")
    needle = query.strip().casefold()
    matches = []
    for row in data["applications"]:
        haystack = " ".join(
            [row["company"], row["role"], row["related_email_id"], row["status"]]
        ).casefold()
        if not needle or needle in haystack:
            matches.append(row)
    return {"matches": matches, "note": "模拟投递进度，不是招聘系统。"}


def extract_proposed_slot(email: dict[str, Any], *, year: int = 2026) -> tuple[str, str] | None:
    text = f"{email.get('subject', '')}\n{email.get('body', '')}"
    match = _PROPOSE.search(text)
    if not match:
        return None
    month, day, clock = match.groups()
    return f"{year}-{int(month):02d}-{int(day):02d}", clock


def wants_schedule(email: dict[str, Any]) -> bool:
    text = f"{email.get('subject', '')}\n{email.get('body', '')}"
    return any(word in text for word in ("时间", "出席", "改期", "有空"))


def slots_for_email(email: dict[str, Any]) -> dict[str, Any] | None:
    if not wants_schedule(email):
        return None
    proposed = extract_proposed_slot(email)
    if proposed:
        return calendar_free_slots(proposed[0], proposed[1])
    return calendar_free_slots()


def build_registry(store: SampleMailStore, llm: LLM | None = None) -> ToolRegistry:
    def list_emails(args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit") or 20)
        limit = max(1, min(limit, 50))
        return {"emails": store.list_emails(limit=limit)}

    def read_email(args: dict[str, Any]) -> dict[str, Any]:
        if "email_id" not in args:
            raise KeyError("email_id")
        redact = bool(args.get("redact_secrets", True))
        return store.read_email(str(args["email_id"]), redact_secrets=redact)

    def search_emails(args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit") or 20)
        limit = max(1, min(limit, 50))
        return {
            "emails": store.search_emails(str(args.get("query") or ""), limit=limit)
        }

    def calendar(args: dict[str, Any]) -> dict[str, Any]:
        return calendar_free_slots(args.get("propose_date"), args.get("propose_time"))

    def applications(args: dict[str, Any]) -> dict[str, Any]:
        return job_application_status(str(args.get("query") or ""))

    def classify_email(args: dict[str, Any]) -> dict[str, Any]:
        if llm is None:
            raise RuntimeError("分类器不可用")
        if "email_id" not in args:
            raise KeyError("email_id")
        email = store.get(str(args["email_id"]))
        classification = classify_with_llm(llm, email)
        return {"email_id": email["id"], **classification.model_dump()}

    return ToolRegistry(
        [
            ToolSpec(
                name="list_emails",
                description="列出样例收件箱里的邮件，不含正文，也不含人工标注。",
                parameters=_object_schema(
                    {"limit": {"type": "integer", "minimum": 1, "maximum": 50}}
                ),
                handler=list_emails,
            ),
            ToolSpec(
                name="read_email",
                description="按 id 读取一封邮件。验证码数字会被隐藏，即使用户要求原文。",
                parameters=_object_schema(
                    {
                        "email_id": {"type": "string"},
                        "redact_secrets": {"type": "boolean"},
                    },
                    required=["email_id"],
                ),
                handler=read_email,
            ),
            ToolSpec(
                name="search_emails",
                description="在发件人、主题和正文里查找。纯数字验证码不能当作查询词。",
                parameters=_object_schema(
                    {
                        "query": {"type": "string"},
                        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
                    },
                    required=["query"],
                ),
                handler=search_emails,
            ),
            ToolSpec(
                name="calendar_free_slots",
                description="返回模拟的空闲时间。可传入 propose_date（YYYY-MM-DD）和 propose_time（HH:MM）检查来信里的时间是否空闲。",
                parameters=_object_schema(
                    {
                        "propose_date": {"type": "string"},
                        "propose_time": {"type": "string"},
                    }
                ),
                handler=calendar,
            ),
            ToolSpec(
                name="job_application_status",
                description="用公司、岗位或邮件 id 查询模拟的投递进度。",
                parameters=_object_schema({"query": {"type": "string"}}),
                handler=applications,
            ),
            ToolSpec(
                name="classify_email",
                description="对一封邮件做五类分类并给一行摘要。这是分类流水线，不是人工金标。",
                parameters=_object_schema(
                    {"email_id": {"type": "string"}},
                    required=["email_id"],
                ),
                handler=classify_email,
            ),
        ]
    )
