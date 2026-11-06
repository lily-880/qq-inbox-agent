"""收件箱演示。默认用离线 stub，不需要 DEEPSEEK_API_KEY，也不会发信。"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.agent import run_agent, seal_workflow, tool_names_called
from src.gate import append_confirmation, record_confirmation
from src.llm import DeepSeekLLM
from src.mailstore import SampleMailStore, load_sample_file
from src.schema import AgentRun
from src.stub_llm import StubLLM
from src.workflow import run_workflow

ACTION_LABEL = {
    "ignore": "丢弃",
    "summarize_only": "只摘要",
    "draft_followup": "起草，等待确认",
    "ask_user": "先问你",
}


def recorded_blurb() -> str:
    path = ROOT / "data" / "recorded" / "classify_v1.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    miss = next(row for row in data["rows"] if not row["ok"])
    accuracy = f"{data['accuracy']:.0%}"
    return (
        f"记录在案的 DeepSeek 分类：{accuracy}（{data['correct']}/{data['total']}），"
        f"已知失误 {miss['id']}（gold={miss['gold']}，pred={miss['pred']}）。"
        "离线 stub 不是这个分数。"
    )


def make_llm(mode: str):
    has_key = bool(os.getenv("DEEPSEEK_API_KEY", "").strip())
    if mode == "auto":
        mode = "deepseek" if has_key else "stub"
    if mode == "stub":
        return StubLLM(), "stub"
    if not has_key:
        raise SystemExit("没有 DEEPSEEK_API_KEY。离线演示请用 --llm stub。")
    return DeepSeekLLM(), "deepseek"


def confirm(outbox: Path, pending_id: str, decision: str) -> None:
    run_path = outbox / "last_run.json"
    if not run_path.exists():
        raise SystemExit(f"没有找到 {run_path}。先跑一次 python scripts/demo.py。")
    run = AgentRun.model_validate_json(run_path.read_text(encoding="utf-8"))
    pending = next((item for item in run.pending if item.id == pending_id), None)
    if pending is None:
        known = ", ".join(item.id for item in run.pending) or "（无）"
        raise SystemExit(f"没有找到 {pending_id}。当前待确认：{known}")
    record = record_confirmation(pending, decision)
    append_confirmation(outbox / "confirmations.json", record)
    print(f"已记录 {record.decision}：{record.pending_id}")
    print(record.note)
    print("executed_send=false")


def format_run(run: AgentRun, *, backend: str, mode: str) -> str:
    lines = [
        f"QQ 收件箱 Agent · {backend} · {mode}",
        recorded_blurb(),
        "",
        "步骤",
    ]
    current_name: str | None = None
    count = 0

    def flush() -> None:
        nonlocal current_name, count
        if current_name is None:
            return
        suffix = f" × {count}" if count > 1 else ""
        lines.append(f"  tool {current_name}{suffix}")
        current_name = None
        count = 0

    for step in run.steps:
        if step.kind == "tool_result":
            continue
        if step.kind == "tool_call":
            if step.name != current_name:
                flush()
                current_name = step.name
            count += 1
            continue
        flush()
        if step.kind == "draft":
            lines.append(f"  draft {step.name}")
        elif step.kind == "reflection":
            lines.append(f"  reflect {step.name}: {step.detail}")
        elif step.kind == "gate":
            lines.append(f"  gate {step.detail}")
    flush()

    lines.extend(["", "结果"])
    for outcome in run.outcomes:
        label = ACTION_LABEL.get(outcome.action, outcome.action)
        reply = "需要回复" if outcome.needs_reply else "不必回复"
        lines.append(f"  {outcome.email_id}  {label}  {reply}")
        lines.append(f"    摘要：{outcome.gist}")
        if outcome.question:
            lines.append(f"    提问：{outcome.question}")
        if outcome.blocked_reason and outcome.pending is None:
            lines.append(f"    未进入待发送：{outcome.blocked_reason}")
        if outcome.pending:
            lines.append(
                f"    待确认 {outcome.pending.id} → {outcome.pending.to} "
                f"（{outcome.pending.status}）"
            )
            for paragraph in outcome.pending.body.splitlines():
                lines.append(f"      {paragraph}" if paragraph else "")
    lines.extend(
        [
            "",
            "人工确认门",
            f"  待确认 {len(run.pending)} 封。程序没有发送任何邮件。",
        ]
    )
    if run.pending:
        sample = run.pending[0]
        lines.append(
            "  记下选择（仍然不会发送）："
            f"python scripts/demo.py --confirm {sample.id} --decision approve"
        )
    called = [name for name in tool_names_called(run) if name]
    if called:
        lines.append("  本轮工具：" + ", ".join(dict.fromkeys(called)))
    lines.append("  executed_send=false")
    return "\n".join(lines)


def build_run(args: argparse.Namespace) -> tuple[AgentRun, str, str]:
    llm, backend = make_llm(args.llm)
    only_ids = {args.email} if args.email else None
    if args.workflow:
        if not args.email:
            raise SystemExit("--workflow 需要 --email <样例 id>")
        matches = list(Path(args.sample_dir).glob(f"{args.email}.json"))
        if not matches:
            matches = [
                path
                for path in Path(args.sample_dir).glob("*.json")
                if load_sample_file(path).get("id") == args.email
            ]
        if not matches:
            raise SystemExit(f"没有样例 {args.email}")
        raw = load_sample_file(matches[0])
        email = {
            "id": raw["id"],
            "from": raw["from"],
            "subject": raw["subject"],
            "body": raw["body"],
        }
        outcome = run_workflow(email, llm)
        return seal_workflow(outcome), backend, "固定流水线"
    store = SampleMailStore(args.sample_dir, only_ids=only_ids)
    if not store.emails:
        raise SystemExit(f"没有样例：{args.sample_dir}")
    return run_agent(llm, store), backend, "多步 agent"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="QQ 收件箱 Agent 演示（默认不发信、不需要密钥）")
    parser.add_argument("--llm", choices=["stub", "deepseek", "auto"], default="stub")
    parser.add_argument("--email", help="只处理这一封样例 id")
    parser.add_argument("--workflow", action="store_true", help="走固定流水线而不是多步 agent")
    parser.add_argument("--json", action="store_true", dest="as_json", help="只把结果打印成 JSON")
    parser.add_argument("--sample-dir", type=Path, default=ROOT / "data" / "samples")
    parser.add_argument("--outbox", type=Path, default=ROOT / "data" / "outbox")
    parser.add_argument("--confirm", help="待确认动作 id，例如 pending-01_job_hr_screen")
    parser.add_argument("--decision", choices=["approve", "reject"])
    args = parser.parse_args(argv)

    if args.confirm:
        if not args.decision:
            raise SystemExit("--confirm 需要 --decision approve 或 reject")
        confirm(args.outbox, args.confirm, args.decision)
        return

    run, backend, mode = build_run(args)
    args.outbox.mkdir(parents=True, exist_ok=True)
    (args.outbox / "last_run.json").write_text(
        run.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    if args.as_json:
        print(run.model_dump_json(indent=2))
        return
    print(format_run(run, backend=backend, mode=mode))


if __name__ == "__main__":
    main()
