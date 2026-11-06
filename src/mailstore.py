"""Mailbox boundary.

Tools call a MailStore. SampleMailStore reads data/samples and never opens a
socket. A later IMAP adapter, or an adapter in front of the cn-mail MCP, can
implement the same three operations: list_emails, read_email, search_emails.
This file stays on the fixture side of that boundary.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .safety import is_otp_email, mask_digit_codes

_CODE_QUERY = re.compile(r"\d{4,8}")


def load_sample_file(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


class SampleMailStore:
    """Fixture inbox. Gold labels stay out of the model-facing methods."""

    def __init__(self, sample_dir: Path, *, only_ids: set[str] | None = None) -> None:
        self.sample_dir = Path(sample_dir)
        emails: list[dict] = []
        for path in sorted(self.sample_dir.glob("*.json")):
            raw = load_sample_file(path)
            item = {
                "id": raw["id"],
                "from": raw["from"],
                "subject": raw["subject"],
                "body": raw["body"],
            }
            if only_ids is not None and item["id"] not in only_ids:
                continue
            emails.append(item)
        self.emails = emails

    def get(self, email_id: str) -> dict:
        for email in self.emails:
            if email["id"] == email_id:
                return email
        raise KeyError(f"没有这封邮件: {email_id}")

    def list_emails(self, *, limit: int = 20) -> list[dict]:
        rows = []
        for email in self.emails[:limit]:
            rows.append(
                {
                    "id": email["id"],
                    "from": email["from"],
                    "subject": email["subject"],
                    "unread": True,
                }
            )
        return rows

    def read_email(self, email_id: str, *, redact_secrets: bool = True) -> dict:
        email = self.get(email_id)
        body = email["body"]
        redacted = False
        # OTP digits stay masked even if redact_secrets is false.
        # Other fixture mail is already anonymized, so this flag cannot reveal a code.
        del redact_secrets
        if is_otp_email(email):
            body = mask_digit_codes(body)
            redacted = True
        return {
            "id": email["id"],
            "from": email["from"],
            "subject": email["subject"],
            "body": body,
            "redacted": redacted,
        }

    def search_emails(self, query: str, *, limit: int = 20) -> list[dict]:
        needle = query.casefold().strip()
        if not needle or _CODE_QUERY.fullmatch(needle):
            return []
        hits: list[dict] = []
        for email in self.emails:
            body = mask_digit_codes(email["body"]) if is_otp_email(email) else email["body"]
            haystack = "\n".join([email["from"], email["subject"], body]).casefold()
            if needle in haystack:
                hits.append(
                    {
                        "id": email["id"],
                        "from": email["from"],
                        "subject": email["subject"],
                        "unread": True,
                    }
                )
            if len(hits) >= limit:
                break
        return hits
