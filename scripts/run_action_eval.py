"""动作 / 草稿路径的夹具评测。不需要 API key，也不代表分类准确率。"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval_action import evaluate_policy, evaluate_stub_invariants


def main() -> None:
    policy = evaluate_policy()
    stub = evaluate_stub_invariants()
    print("动作策略（金标标签 → 动作，不调用模型）")
    for row in policy["rows"]:
        mark = "OK" if row["ok"] else "MISS"
        print(
            f"[{mark}] {row['id']}: label={row['label']} "
            f"action={row['action']} needs_reply={row['needs_reply']}"
        )
    print(f"policy {policy['correct']}/{policy['total']} passed={policy['passed']}")
    print()
    print("离线流水线不变量（stub，不是 DeepSeek 分数）")
    print(
        "actions_match_fixture="
        f"{stub['actions_match_fixture']} "
        f"reflection_only_on_drafts={stub['reflection_only_on_drafts']} "
        f"pending_only_when_should_send={stub['pending_only_when_should_send']} "
        f"drafts_are_queued={stub['drafts_are_queued']} "
        f"leaked_otp_codes={stub['leaked_otp_codes'] or '[]'}"
    )
    report = {"policy": policy, "stub_invariants": stub}
    out = ROOT / "data" / "action_eval_results.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    print("分类准确率不在这里。记录是 data/recorded/classify_v1.json：90%（9/10）。")
    if not policy["passed"] or not stub["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
