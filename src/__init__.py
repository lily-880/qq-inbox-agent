"""QQ inbox agent. Import submodules for the workflow, tools, and agent."""

from .classify import (
    CLASSIFY_SYSTEM,
    build_classify_messages,
    email_to_classify_messages,
    parse_classification,
)
from .schema import EmailClassification, EmailLabel

__all__ = [
    "EmailClassification",
    "EmailLabel",
    "CLASSIFY_SYSTEM",
    "build_classify_messages",
    "email_to_classify_messages",
    "parse_classification",
]
