# QQ 收件箱 Agent

傅公羽（lily-880）的作品集项目，对应 DeepLearning.AI **Agentic AI** 的五个模块。日常目标是处理 QQ 邮箱里的新邮件；这个仓库用 10 封脱敏样例把整条链路跑通，不连接真邮箱。

## 每天想做的事

1. 看新邮件。
2. 分成 `job` / `receipt` / `otp` / `promo` / `other`。
3. 营销和验证码留一句摘要后丢掉。验证码数字不进入摘要，也不进入回复。
4. 重要邮件给出一行摘要，并判断要不要回复。
5. 需要回复时起草一封，审一次，然后停住。
6. 人确认之后才算数。**这个程序没有发信功能。** 确认只是记一笔。

## V1 做了什么，现在做了什么

V1 只做「一封邮件 → 分类 + 一行中文摘要」。

| | V1 | 现在 |
| --- | --- | --- |
| 分类 | `EmailClassification`，DeepSeek，10 封样例 | 同一套模型和样例。提示词补充了：校园宣讲会算 `other`，不是 `job` |
| 分类分数 | 90%（9/10） | **仍是这 90%。** 没有重新调用 DeepSeek，不能写成更高 |
| 已知失误 | `09_other_event`：宣讲会，金标 `other`，模型判成 `job` | 原样记在 `data/recorded/classify_v1.json` |
| 动作 | 无 | 固定策略：丢弃 / 只摘要 / 起草 / 先问人 |
| 草稿 | 无 | 起草后做一次语气、事实、该不该发的审稿，并修订一稿 |
| 工具 | 无 | 样例邮箱上的 list / read / search，外加模拟日历和模拟投递进度 |
| Agent | 无 | 多步工具调用，最后停在人工确认门 |
| 离线演示 | 无 | 没有密钥也能跑控制流 |

## 五个模块

| 模块 | 在仓库里的位置 | 做什么 |
| --- | --- | --- |
| M1 Workflow | `src/workflow.py`，`src/policy.py`，`src/pipeline.py`，`src/classify.py` | 固定流水线：读入 → 分类 → 动作策略 → 需要时才起草 |
| M2 Reflection | `src/reflect.py` | 草稿之后检查语气、是否只使用原信和空闲时间里的事实、到底该不该发，然后只修订一次 |
| M3 Tools | `src/tools.py`，`src/mailstore.py`，`data/mocks/` | 显式工具注册表，能被 agent 调用。邮箱边界是 `SampleMailStore` 的 `list_emails` / `read_email` / `search_emails`，以后可以换成 IMAP 或 cn-mail，而不是在工具里直接连邮箱 |
| M4 Eval | `scripts/run_eval.py`，`data/recorded/classify_v1.json`，`scripts/run_action_eval.py`，`data/action_eval_expected.json` | 在线分类评测保持 V1 的 10 封样例；动作路径用金标标签和离线 stub 做夹具评测，不需要密钥 |
| M5 Agents | `src/agent.py`，`src/gate.py`，`scripts/demo.py` | 多步 agent 自己决定 `ignore` / `summarize_only` / `draft_followup` / `ask_user`，可以调用工具，然后必须停在人工确认门。发送只是状态为 `pending_confirmation` 的待确认动作 |

离线模型在 `src/stub_llm.py`。它走同一套工具调用和 JSON 约定，方便没有密钥时演示。DeepSeek 在 `src/llm.py`，密钥只从环境变量或 `.env` 读取。

## 分类评测（不要和 stub 混在一起）

`data/recorded/classify_v1.json` 是 V1 用 DeepSeek 跑出来的结果：

- 10 封，对 9 封，准确率 **90%**
- 唯一的失误是 `09_other_event`
- 金标 `other`，模型预测 `job`（校园宣讲会 / 就业中心活动通知）

`src/classify.py` 的系统提示已经写明：宣讲会、双选会、就业中心活动通知一律 `other`，除非信是在处理收件人自己的投递、面试或 Offer。**这只是提示词改动，没有新的在线分数。**

`python scripts/demo.py` 里的 stub 会把这封宣讲会标成 `other`。那是关键字演示，不是 DeepSeek 的新准确率。

## 动作夹具（M4 的补充，不调用模型）

金标标签进，动作出。期望在 `data/action_eval_expected.json`：

| 样例 | 标签 | 动作 |
| --- | --- | --- |
| 01、02 求职 | `job` | `draft_followup` |
| 03、04 收据 / 账单 | `receipt` | `summarize_only` |
| 05、06 验证码 | `otp` | `ignore` |
| 07、08 营销 | `promo` | `ignore` |
| 09 宣讲会 | `other` | `summarize_only` |
| 10 组会改期 | `other` | `ask_user` |

02 的面试时间是 11 月 12 日 14:00。模拟日历在 `data/mocks/calendar.json` 里把这一小时标成已有安排，草稿会改提空闲时间，并停在待确认。

## 怎么跑

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m pytest
python scripts/run_action_eval.py
python scripts/demo.py
```

常用演示：

```bash
python scripts/demo.py --email 02_job_interview_invite
python scripts/demo.py --workflow --email 01_job_hr_screen
python scripts/demo.py --confirm pending-01_job_hr_screen --decision approve
```

最后一条只把选择写进 `data/outbox/confirmations.json`，并打印 `executed_send=false`。

有 DeepSeek 密钥时：

```bash
cp .env.example .env
# 在 .env 里填写 DEEPSEEK_API_KEY，不要把 .env 提交进来

python scripts/run_eval.py
python scripts/demo.py --llm deepseek
python scripts/demo.py --llm auto   # 有密钥用 DeepSeek，没有就用 stub
```

`run_eval.py` 把新的一次运行写到 `data/eval_results.json`（已 gitignore）。它不会改写 `data/recorded/classify_v1.json`。

## 明确不做

- 不自动发信，也没有 SMTP。同意一封草稿不会把它发出去。
- 不把验证码数字读进摘要、工具结果或回复。样例文件里仍留着原始数字，只为了分类评测能对照原文。
- 不同步微信。
- 不连接真的 QQ / IMAP。`src/mailstore.py` 是以后替换邮箱来源的边界。

## 目录

```
src/classify.py          分类提示词和解析
src/pipeline.py          单封分类
src/policy.py            动作策略
src/workflow.py          M1 固定流水线
src/reflect.py           M2 审稿一次
src/mailstore.py         样例邮箱
src/tools.py             M3 工具注册表
src/agent.py             M5 多步 agent
src/gate.py              人工确认，不发送
src/stub_llm.py          离线模型
src/llm.py               DeepSeek
scripts/demo.py          演示
scripts/run_eval.py      在线分类评测
scripts/run_action_eval.py
data/samples/            10 封脱敏邮件
data/recorded/           V1 的 90% 记录
data/mocks/              模拟日历和投递进度
```
