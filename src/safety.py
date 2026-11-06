"""OTP handling for anything a person or the agent would read.

Sample files keep their original digits so the classification eval is unchanged.
Agent-facing summaries and tool results do not.
"""

from __future__ import annotations

import re

_CODE = re.compile(r"(?<!\d)\d{4,8}(?!\d)")


def is_otp_email(email: dict) -> bool:
    text = f"{email.get('subject', '')}\n{email.get('body', '')}"
    return "验证码" in text


def mask_digit_codes(text: str) -> str:
    """Replace 4–8 digit runs. Used on OTP bodies, not on ordinary invoices."""
    return _CODE.sub("******", text)


def public_otp_summary(email: dict) -> str:
    text = f"{email.get('subject', '')}\n{email.get('body', '')}"
    if "密码" in text or "重设" in text:
        return "密码重设验证码邮件，数字已隐藏，不需要回复。"
    return "登录验证码邮件，数字已隐藏，不需要回复。"


def public_gist(email: dict, gist: str) -> str:
    if is_otp_email(email):
        return public_otp_summary(email)
    return gist
