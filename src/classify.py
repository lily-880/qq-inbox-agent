"""邮件分类：提示词 + Pydantic 输出解析。"""

from __future__ import annotations

from typing import Any

from .jsonutil import extract_json_object
from .schema import EmailClassification

CLASSIFY_SYSTEM = """你是邮件分类助手。根据发件人、主题和正文，输出一个 JSON 对象，字段必须严格符合：
- label: 只能是 job / receipt / otp / promo / other 之一
- confidence: 0 到 1 的小数
- reason: 一句中文理由
- summary: 根据正文提炼的一行中文要点（不要照抄 subject）

类别说明：
- job: 收件人本人的招聘进展，例如投递结果、初筛、面试邀请、Offer、HR 约时间。必须是在处理「你的申请」。
- receipt: 付款、发票、订阅账单
- otp: 验证码/登录安全。summary 和 reason 都不要写出验证码数字。
- promo: 营销、课程推销、礼品卡广告
- other: 其余

特别注意：
- 校园宣讲会、双选会、就业中心的活动通知，即使出现公司名或「招聘」「岗位」，只要不是收件人本人的投递结果、面试邀请或 Offer，一律标 other，不要标 job。
- 例如「某公司将在校区举行宣讲会，与岗位投递结果无关」是 other。
- 同学改期、组会、私事也是 other。

只输出 JSON，不要 Markdown，不要额外解释。"""

CLASSIFY_USER_TEMPLATE = """发件人: {from_}
主题: {subject}
正文:
{body}
"""


def build_classify_messages(
    *,
    from_: str,
    subject: str,
    body: str,
) -> list[dict[str, str]]:
    """构造发给聊天模型的 messages（不含 label）。"""
    return [
        {"role": "system", "content": CLASSIFY_SYSTEM},
        {
            "role": "user",
            "content": CLASSIFY_USER_TEMPLATE.format(
                from_=from_,
                subject=subject,
                body=body,
            ),
        },
    ]


def parse_classification(model_text: str) -> EmailClassification:
    """把模型原始回复解析成 EmailClassification；失败抛 ValidationError / ValueError。"""
    data = extract_json_object(model_text)
    return EmailClassification.model_validate(data)


def email_to_classify_messages(email: dict[str, Any]) -> list[dict[str, str]]:
    """把邮件 dict（from/subject/body）转成可发给模型的 messages；忽略 label。"""
    return build_classify_messages(
        from_=email.get("from", ""),
        subject=email.get("subject", ""),
        body=email.get("body", ""),
    )
