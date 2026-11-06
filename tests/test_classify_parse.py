"""解析器烟雾测试：不调真实模型。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classify import CLASSIFY_SYSTEM, email_to_classify_messages, parse_classification
from src.schema import EmailClassification


def test_email_to_messages_omits_label():
    email = {
        "id": "x",
        "label": "job",
        "from": "hr@example.com",
        "subject": "面试邀请",
        "body": "请确认本周四下午是否方便。",
    }
    msgs = email_to_classify_messages(email)
    user = msgs[1]["content"]
    assert "面试邀请" in user
    assert "hr@example.com" in user
    assert "请确认本周四下午是否方便" in user
    assert "label" not in user.lower()
    assert "job" not in user
    assert msgs[0]["role"] == "system"


def test_parse_valid_json():
    raw = """{
      "label": "job",
      "confidence": 0.9,
      "reason": "HR 邀请面试",
      "summary": "邀请本周线上面试并请回复时间"
    }"""
    obj = parse_classification(raw)
    assert isinstance(obj, EmailClassification)
    assert obj.label == "job"


def test_parse_fenced_json():
    raw = """```json
{"label":"promo","confidence":0.8,"reason":"课程促销","summary":"推销进阶课打折"}
```"""
    obj = parse_classification(raw)
    assert obj.label == "promo"


def test_prompt_campus_talk_is_other_not_job():
    assert "宣讲会" in CLASSIFY_SYSTEM
    assert "不要标 job" in CLASSIFY_SYSTEM
    assert "other" in CLASSIFY_SYSTEM


if __name__ == "__main__":
    test_email_to_messages_omits_label()
    test_parse_valid_json()
    test_parse_fenced_json()
    test_prompt_campus_talk_is_other_not_job()
    print("ok")
