"""跑 data/samples 全部样例并输出准确率。需要 DEEPSEEK_API_KEY。

分类提示词里已经写明校园宣讲会算 other。在真正跑完之前，不要把
data/recorded/classify_v1.json 里的 90%（9/10）改成更高的数。
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.mailstore import load_sample_file
from src.pipeline import classify_one


def main() -> None:
    if not os.getenv("DEEPSEEK_API_KEY", "").strip():
        raise SystemExit(
            "缺少 DEEPSEEK_API_KEY，无法跑在线分类评测。"
            "动作夹具请用 python scripts/run_action_eval.py。"
        )

    sample_dir = ROOT / "data" / "samples"
    paths = sorted(sample_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"没有样例: {sample_dir}")

    rows = []
    correct = 0
    for path in paths:
        email = load_sample_file(path)
        gold = email.get("label")
        try:
            pred = classify_one(email)
            pred_label = pred.label
            ok = pred_label == gold
            if ok:
                correct += 1
            rows.append(
                {
                    "id": email.get("id", path.stem),
                    "gold": gold,
                    "pred": pred_label,
                    "ok": ok,
                    "confidence": pred.confidence,
                    "reason": pred.reason,
                    "summary": pred.summary,
                }
            )
            mark = "OK" if ok else "MISS"
            print(f"[{mark}] {path.name}: gold={gold} pred={pred_label}")
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "id": email.get("id", path.stem),
                    "gold": gold,
                    "pred": None,
                    "ok": False,
                    "error": str(exc),
                }
            )
            print(f"[ERR] {path.name}: {exc}")

    total = len(rows)
    acc = correct / total if total else 0.0
    report = {
        "total": total,
        "correct": correct,
        "accuracy": round(acc, 4),
        "rows": rows,
    }
    out = ROOT / "data" / "eval_results.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\naccuracy={acc:.0%} ({correct}/{total})")
    print(f"wrote {out}")
    print(
        "记录在案的 V1 结果仍是 90%（9/10），已知失误 09_other_event"
        "（gold=other，pred=job）。本次文件不会覆盖 data/recorded/classify_v1.json。"
    )


if __name__ == "__main__":
    main()
