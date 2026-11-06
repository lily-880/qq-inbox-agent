"""对一封邮件跑分类：messages → 模型 → EmailClassification。"""

from __future__ import annotations

from typing import Any

from .classify import email_to_classify_messages, parse_classification
from .llm import chat_complete
from .llm_types import LLM
from .safety import is_otp_email, public_otp_summary
from .schema import EmailClassification


def classify_one(email: dict[str, Any], *, temperature: float = 0.0) -> EmailClassification:
    """在线评测入口。不改写模型原文，方便和 V1 记录对照。"""
    messages = email_to_classify_messages(email)
    raw = chat_complete(messages, temperature=temperature)
    return parse_classification(raw)


def classify_with_llm(
    llm: LLM,
    email: dict[str, Any],
    *,
    temperature: float = 0.0,
) -> EmailClassification:
    """流水线和工具用的分类。验证码的摘要和理由会被换成不含数字的句子。"""
    response = llm.complete(
        email_to_classify_messages(email),
        temperature=temperature,
    )
    if not response.content:
        raise RuntimeError("分类模型没有返回内容")
    classification = parse_classification(response.content)
    if is_otp_email(email) or classification.label == "otp":
        return classification.model_copy(
            update={
                "label": "otp",
                "summary": public_otp_summary(email),
                "reason": "验证码邮件。数字已隐藏，不在理由里复述。",
            }
        )
    return classification
