"""Offline LLM.

It speaks the same tool-calling and JSON contracts as DeepSeek so the agent
loop can be demoed without a key. Keyword classification is for that demo
only. It is not the M4 score. The recorded DeepSeek run is
data/recorded/classify_v1.json (90%, 9/10).
"""

from __future__ import annotations

import json
from typing import Any

from .llm_types import LLMResponse, ToolCall
from .policy import decide_action
from .reflect import fallback_reflection
from .safety import public_otp_summary
from .schema import DraftReply, EmailClassification
from .tools import extract_proposed_slot


def keyword_classification(email: dict[str, Any]) -> EmailClassification:
    """Demo classifier. Do not report its hit rate as the DeepSeek accuracy."""
    text = f"{email.get('from', '')}\n{email.get('subject', '')}\n{email.get('body', '')}"
    lowered = text.lower()
    if "验证码" in text:
        label = "otp"
        reason = "正文在提供验证码。"
        summary = public_otp_summary(email)
    elif any(word in lowered for word in ("receipt", "invoice", "amount paid")) or any(
        word in text for word in ("账单", "发票", "付款")
    ):
        label = "receipt"
        reason = "邮件是付款收据或账单。"
        if "amount paid" in lowered or "receipt" in lowered:
            summary = "付款收据，留档即可，无需操作。"
        else:
            summary = "账单已支付，留档即可，无需回复。"
    elif "宣讲会" in text or "就业指导" in text:
        label = "other"
        reason = "这是校园活动通知，不是个人的投递结果或面试安排。"
        summary = "校园宣讲会提醒，与岗位投递结果无关。"
    elif "组会" in text:
        label = "other"
        reason = "同学之间的组会通知，不属于招聘、账单、验证码或营销。"
        summary = "组会改期，如果时间冲突需要提前说一声。"
    elif any(word in text for word in ("退订", "礼品卡", "限时")) or "unsubscribe" in lowered:
        label = "promo"
        reason = "营销或课程推销。"
        summary = "礼品卡广告，摘要后丢弃。" if "礼品卡" in text else "课程促销，摘要后丢弃。"
    elif any(word in text for word in ("初筛", "面试", "实习", "Offer", "offer")):
        label = "job"
        reason = "邮件在处理收件人自己的招聘流程。"
        if "初筛" in text:
            summary = "初筛通过，对方想约线上面试，需要回复方便的时间。"
        else:
            summary = "对方发出面试邀请，需要确认能否出席。"
    else:
        label = "other"
        reason = "不属于招聘、账单、验证码或营销。"
        summary = "普通邮件，先留一行摘要。"
    return EmailClassification(label=label, confidence=0.66, reason=reason, summary=summary)


def _kind(messages: list[dict[str, Any]]) -> str:
    system = ""
    for message in messages:
        if message.get("role") == "system":
            system = message.get("content") or ""
            break
    if system.startswith("你是邮件分类助手"):
        return "classify"
    if system.startswith("你是邮件回复起草助手"):
        return "draft"
    if system.startswith("你是回复审稿人"):
        return "reflect"
    return "agent"


def _email_from_classify_user(content: str) -> dict[str, str]:
    sender = ""
    subject = ""
    body = ""
    if content.startswith("发件人:"):
        lines = content.splitlines()
        sender = lines[0].split(":", 1)[1].strip()
        if len(lines) > 1 and lines[1].startswith("主题:"):
            subject = lines[1].split(":", 1)[1].strip()
        marker = "正文:\n"
        index = content.find(marker)
        if index >= 0:
            body = content[index + len(marker) :]
    return {"from": sender, "subject": subject, "body": body}


def stub_draft(payload: dict[str, Any]) -> DraftReply:
    subject = str(payload.get("subject") or "")
    body = str(payload.get("body") or "")
    proposed = payload.get("proposed")
    slot_lines = list(payload.get("free_slots") or [])
    lines = ["您好，", ""]
    if isinstance(proposed, dict) and proposed.get("time"):
        clock = str(proposed["time"])
        if proposed.get("free") is False:
            lines.append(f"感谢邀请。来信中的时间（{clock}）在日历上已经有安排。")
        else:
            lines.append(f"感谢邀请。我可以按来信中的时间（{clock}）出席。")
    elif "方便的时间" in body or "请回复" in body:
        lines.append("感谢来信。我可以配合邮件里提到的安排。")
    else:
        lines.append(f"我已收到主题为「{subject}」的邮件。")
    if slot_lines and (
        "时间" in body or (isinstance(proposed, dict) and proposed.get("free") is False)
    ):
        lines.append("我目前有空的时间是：")
        lines.extend(f"- {line}" for line in slot_lines)
    lines.extend(["", "此致", "傅公羽"])
    return DraftReply(subject=f"Re: {subject}", body="\n".join(lines))


def _unwrap(result_text: str) -> dict[str, Any]:
    data = json.loads(result_text)
    if isinstance(data, dict) and data.get("ok") is True and "result" in data:
        return data["result"]
    if isinstance(data, dict) and data.get("ok") is False:
        raise RuntimeError(data.get("error") or "tool failed")
    if not isinstance(data, dict):
        raise RuntimeError("工具结果不是 JSON 对象")
    return data


def _completed_calls(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: dict[str, str] = {}
    for message in messages:
        if message.get("role") == "tool":
            results[str(message.get("tool_call_id"))] = str(message.get("content") or "")
    done: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") != "assistant":
            continue
        for tool_call in message.get("tool_calls") or []:
            function = tool_call["function"]
            raw_args = function.get("arguments") or "{}"
            try:
                arguments = json.loads(raw_args)
            except json.JSONDecodeError:
                arguments = {}
            if not isinstance(arguments, dict):
                arguments = {}
            done.append(
                {
                    "id": tool_call["id"],
                    "name": function["name"],
                    "arguments": arguments,
                    "result": results.get(tool_call["id"]),
                }
            )
    return done


class StubLLM:
    def __init__(self) -> None:
        self._call_n = 0

    def complete(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
        json_mode: bool = False,
    ) -> LLMResponse:
        del tools, temperature, json_mode
        kind = _kind(messages)
        if kind == "classify":
            email = _email_from_classify_user(str(messages[-1]["content"]))
            return LLMResponse(content=keyword_classification(email).model_dump_json())
        if kind == "draft":
            payload = json.loads(str(messages[-1]["content"]))
            return LLMResponse(content=stub_draft(payload).model_dump_json())
        if kind == "reflect":
            payload = json.loads(str(messages[-1]["content"]))
            return LLMResponse(content=fallback_reflection(payload).model_dump_json())
        return self._agent_turn(messages)

    def _next_id(self) -> str:
        self._call_n += 1
        return f"stub-{self._call_n}"

    def _tools(self, pairs: list[tuple[str, dict[str, Any]]]) -> LLMResponse:
        calls = [
            ToolCall(id=self._next_id(), name=name, arguments=arguments)
            for name, arguments in pairs
        ]
        return LLMResponse(content=None, tool_calls=calls)

    def _agent_turn(self, messages: list[dict[str, Any]]) -> LLMResponse:
        finished = [call for call in _completed_calls(messages) if call["result"] is not None]
        names = [call["name"] for call in finished]

        if "list_emails" not in names:
            return self._tools([("list_emails", {})])

        listed = _unwrap(next(call["result"] for call in finished if call["name"] == "list_emails"))
        emails = listed["emails"]
        read_ids: set[str] = set()
        for call in finished:
            if call["name"] == "read_email":
                read_ids.add(str(call["arguments"].get("email_id") or ""))
        unread = [email["id"] for email in emails if email["id"] not in read_ids]
        if unread:
            return self._tools([("read_email", {"email_id": email_id}) for email_id in unread])

        classified: set[str] = set()
        for call in finished:
            if call["name"] == "classify_email":
                classified.add(str(call["arguments"].get("email_id") or ""))
        pending = [email["id"] for email in emails if email["id"] not in classified]
        if pending:
            return self._tools(
                [("classify_email", {"email_id": email_id}) for email_id in pending]
            )

        classes = [
            _unwrap(call["result"])
            for call in finished
            if call["name"] == "classify_email"
        ]
        job_ids = [row["email_id"] for row in classes if row.get("label") == "job"]
        extras: list[tuple[str, dict[str, Any]]] = []
        if job_ids and "calendar_free_slots" not in names:
            propose: dict[str, Any] = {}
            for call in finished:
                if call["name"] != "read_email":
                    continue
                row = _unwrap(call["result"])
                found = extract_proposed_slot(row)
                if found:
                    propose = {"propose_date": found[0], "propose_time": found[1]}
                    break
            extras.append(("calendar_free_slots", propose))
        if job_ids and "job_application_status" not in names:
            extras.append(("job_application_status", {"query": "实习"}))
        if "search_emails" not in names:
            extras.append(("search_emails", {"query": "面试"}))
        if extras:
            return self._tools(extras)

        return LLMResponse(content=self._decisions_json(finished, emails))

    def _decisions_json(
        self,
        finished: list[dict[str, Any]],
        emails: list[dict[str, Any]],
    ) -> str:
        classes: dict[str, dict[str, Any]] = {}
        reads: dict[str, dict[str, Any]] = {}
        status: dict[str, Any] | None = None
        for call in finished:
            if call["name"] == "classify_email":
                row = _unwrap(call["result"])
                classes[row["email_id"]] = row
            elif call["name"] == "read_email":
                row = _unwrap(call["result"])
                reads[row["id"]] = row
            elif call["name"] == "job_application_status":
                status = _unwrap(call["result"])

        decisions = []
        for item in emails:
            read = reads[item["id"]]
            email = {
                "id": item["id"],
                "from": read["from"],
                "subject": read["subject"],
                "body": read["body"],
            }
            row = classes[item["id"]]
            classification = EmailClassification.model_validate(
                {key: row[key] for key in ("label", "confidence", "reason", "summary")}
            )
            decision = decide_action(email, classification)
            reason = decision.reason
            if status and decision.action == "draft_followup":
                for match in status.get("matches") or []:
                    if match.get("related_email_id") == item["id"]:
                        reason = f"{reason} 模拟投递状态：{match['status']}。"
                        break
            decisions.append(
                {
                    "email_id": item["id"],
                    "action": decision.action,
                    "needs_reply": decision.needs_reply,
                    "gist": classification.summary,
                    "reason": reason,
                }
            )
        return json.dumps({"decisions": decisions}, ensure_ascii=False)
