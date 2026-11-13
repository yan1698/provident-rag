"""Freeze actual public-benchmark runs into a candid baseline and candidate-badcase report."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.eval.benchmarks.data import write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=ROOT / "benchmarks/reports/baseline")
    parser.add_argument("--continuation-plan", type=Path, help="saved reviewed continuation plan, if used")
    args = parser.parse_args()
    report = {
        "scope": "Provident V1 Chinese",
        "official_score_comparable": False,
        "status": "partial_real_execution",
        "runs": [],
        "candidate_badcases": [],
        "diagnostics": [],
        "known_tokens": 0,
        "known_priced_cost_cny": 0.0,
        "actual_invoice_cost": None,
        "total_cost_status": "configured_price_estimate",
        "core_not_executed": [],
        "attempted_cases": 0,
        "completed_cases": 0,
        "failed_cases": 0,
        "coverage": {},
        "duplicate_attempts": [],
        "unknown_usage_requests": 0,
    }
    lines = [
        "# Provident V1 中文真实试跑报告",
        "",
        "这是中文固定上下文受控试跑基线，尚未形成完整检索或端到端基线。程序结束不表示质量评测通过。",
        "",
        "以下按实际执行配置分别报告，不合并不同输出上限、超时或代码快照的分数。",
        "",
    ]
    active = set()
    seen_attempts = set()
    seen_runs = set()
    for directory in args.runs:
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        if manifest["run_id"] in seen_runs:
            raise ValueError("duplicate run directory would double-count usage")
        seen_runs.add(manifest["run_id"])
        if not manifest["execution_requested"]:
            raise ValueError("offline preparation is not a real model run")
        results = json.loads((directory / "results.json").read_text(encoding="utf-8"))
        summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
        attempted = [x for x in results if x["status"] in {"executed", "error"}]
        completed = [x for x in results if x["status"] == "executed"]
        active.add(manifest["benchmark"])
        coverage = report["coverage"].setdefault(
            manifest["benchmark"], {"attempted": 0, "completed": 0, "failed": 0, "modes": [], "tasks": []}
        )
        coverage["attempted"] += len(attempted)
        coverage["completed"] += len(completed)
        coverage["failed"] += len(attempted) - len(completed)
        for key, values in (("modes", [manifest["mode"]]), ("tasks", [x["input"]["task_type"] for x in attempted])):
            coverage[key] = sorted(set(coverage[key]) | set(values))
        report["attempted_cases"] += len(attempted)
        report["completed_cases"] += len(completed)
        report["failed_cases"] += len(attempted) - len(completed)
        for trace in attempted:
            identity = (manifest["benchmark"], manifest["snapshot_sha256"], trace["sample_id"])
            if identity in seen_attempts:
                report["duplicate_attempts"].append(list(identity))
            seen_attempts.add(identity)
            if trace.get("token_usage") is None:
                report["unknown_usage_requests"] += 1
        row = {
            "benchmark": manifest["benchmark"],
            "run_id": manifest["run_id"],
            "mode": manifest["mode"],
            "dataset_revision": manifest["dataset_revision"],
            "snapshot_sha256": manifest["snapshot_sha256"],
            "code": {key: manifest["code"][key] for key in ("git_commit", "dirty", "source_sha256")},
            "selected": len(results),
            "attempted_cases": len(attempted),
            "completed": len(completed),
            "not_executed": len(results) - len(attempted),
            "summary": summary,
            "known_tokens": manifest["observed_tokens"],
            "known_priced_cost_cny": manifest["priced_observed_cost"],
            "cost_status": manifest["cost_status"],
            "config": manifest["config"],
            "directory": str(directory.resolve()),
            "trace_ids": [x["trace_id"] for x in attempted],
        }
        report["runs"].append(row)
        report["known_tokens"] += row["known_tokens"]
        report["known_priced_cost_cny"] += row["known_priced_cost_cny"]
        config = manifest["config"]
        lines += [
            f"## {manifest['benchmark']} / {manifest['run_id']}",
            "",
            f"模式 `{manifest['mode']}`；本次选中 {len(results)} 条，实际尝试 {len(attempted)} 条，返回回答 {len(completed)} 条，本次未执行 {row['not_executed']} 条。",
            f"模型 `{config['model']}`，单条硬超时 {config['timeout_seconds']} 秒，重试 {config['retries']} 次；完整输出上限 {config.get('max_total_output_tokens', '未设置，仅最终文本 max_tokens=512')} Token。",
            f"数据 revision：`{manifest['dataset_revision']}`。",
            f"运行：[{manifest['run_id']}]({directory.resolve().as_posix()}/report.md)。",
            "",
        ]
        for trace in attempted:
            if trace["badcase_types"]:
                report["candidate_badcases"].append(
                    {
                        "status": "candidate_badcase",
                        "benchmark": manifest["benchmark"],
                        "sample_id": trace["sample_id"],
                        "trace_id": trace["trace_id"],
                        "types": trace["badcase_types"],
                        "reason": trace.get("reason"),
                        "question": trace["input"]["question"],
                        "actual_answer": (trace.get("output") or {}).get("answer"),
                        "evidence": trace["input"].get("contexts"),
                        "trace_path": str((directory / "traces" / f"{trace['trace_id']}.json").resolve()),
                    }
                )
            for usage in trace.get("token_usage") or []:
                cap = manifest["config"].get("max_total_output_tokens")
                if cap is None and usage["completion_tokens"] > manifest["config"]["max_output_tokens"]:
                    report["diagnostics"].append(
                        {
                            "type": "legacy_final_text_limit_does_not_cap_thinking",
                            "trace_id": trace["trace_id"],
                            "requested_max_tokens": manifest["config"]["max_output_tokens"],
                            "returned_completion_tokens": usage["completion_tokens"],
                            "source": "https://help.aliyun.com/zh/model-studio/deep-thinking/",
                        }
                    )
        if manifest["benchmark"] == "rgb" and completed:
            matches = sum(x["metrics"]["rgb_reference_match"]["value"] for x in completed)
            lines += [
                f"已返回回答的官方参考答案子串匹配：{int(matches)}/{len(completed)}。不是语义正确率，样本量不足以形成稳定分数。",
                f"可回答样本错误拒答模板命中率：{summary['metrics']['false_refusal_rate']['value']:.4f}。本次没有纯噪声拒答、跨证据整合或反事实模型执行，不能据此判断这些能力。",
                "",
            ]
        if manifest["benchmark"] == "crud" and completed:
            for name in ("crud_bleu_no_bp", "crud_rougeL"):
                value = summary["metrics"][name]
                if value["status"] == "evaluated":
                    lines += [f"`{name}`：{value['value']:.6f}，实际可评分 {value['evaluated_samples']} 条。"]
            lines += [
                "",
                "仅执行 `questanswer_1doc`，使用官方新闻与任务 Prompt，经 Provident 模型适配器运行；这些是文本重合分数，不是语义正确率或 Agent 能力分数。",
                "",
            ]
        if completed:
            lines += [
                f"成功样本 wall time P50/P95：{summary['p50_ms'] / 1000:.3f}/{summary['p95_ms'] / 1000:.3f} 秒，包含独立 worker 启动与导入；超时失败另计，非稳态服务延迟。",
                f"本次已知 usage：{row['known_tokens']} Token；按配置单价估算 {row['known_priced_cost_cny']:.6f} 元。",
                "",
            ]
        if manifest["cost_status"] == "unknown_failed_attempt_usage":
            lines += [
                "超时后触发停止策略。原始记录中后续样本的 `budget_stopped` 表示策略关停，并非实际已耗尽总请求数或费用。失败请求的服务商最终用量未知；原 `budget_reserved` 也不能当作实际费用。",
                "",
            ]
    for name, reason in (
        ("crud", "前一批停止，未开始模型调用"),
        ("cmteb_du", "完整官方语料与隔离索引未准备；当前 not_supported"),
        ("cmteb_t2", "20 条完整候选已准备；当前轻量运行环境缺 Cross-Encoder 依赖，本次未授权执行"),
    ):
        if name not in active:
            report["core_not_executed"].append({"benchmark": name, "attempted": 0, "reason": reason})
            lines += [f"## {name}", "", f"实际模型样本 0 条：{reason}。", ""]
    report["unique_attempted_cases"] = len(seen_attempts)
    if report["unknown_usage_requests"]:
        report["total_cost_status"] = "unknown_failed_attempt_usage"
    lines[4:4] = [
        f"实际累计尝试 {report['attempted_cases']} 次（{len(seen_attempts)} 个不同样本），返回 {report['completed_cases']} 条回答，失败 {report['failed_cases']} 条。",
        "历史首轮中停止的样本可能在后续授权运行完成；累计统计只计实际请求，不把历史未执行记录重复计入。",
        "",
    ]
    lines += [
        "## 用量与成本",
        "",
        f"已收到的真实 usage 共 {report['known_tokens']} Token，按固定公开单价估算 {report['known_priced_cost_cny']:.6f} 元。",
        f"缺少 usage 的实际请求 {report['unknown_usage_requests']} 次；其用量和最终账单未知，总费用不能填成已知部分或零。",
        "单价来源：[百炼官方价格](https://help.aliyun.com/zh/model-studio/model-pricing)，配置记录核查日、地域及非缓存价格；最终账单含折扣等差异。",
        "",
        "## 失败样本与覆盖空白",
        "",
    ]
    for badcase in report["candidate_badcases"]:
        lines += [
            f"- `{badcase['sample_id']}`，Trace `{badcase['trace_id']}`：{badcase['reason']}；状态 `candidate_badcase`。"
        ]
    lines += [
        "",
        "语义答案正确率、Verifier Unsupported Precision/Recall/F1、引用语义 Precision/Coverage、多轮质量及未运行检索指标均为 `not_evaluable`。没有运行 LLM Judge。",
        "",
        "所有分数均不能作为完整官方基准成绩；固定上下文结果不代表端到端检索能力。",
    ]
    if report["diagnostics"]:
        lines += [
            "",
            "## 执行器预算修正",
            "",
            "首轮设置 `max_tokens=512`，真实 usage 返回 742 个 completion Token，原预算实现漏计了默认思考 Token。[官方说明](https://help.aliyun.com/zh/model-studio/deep-thinking/)明确 `max_tokens` 仅限制最终回复，`max_completion_tokens` 限制完整输出。评测侧新增传输包装并留痕，未修改生产 RAG 逻辑。",
            "",
        ]
    if args.continuation_plan:
        plan = json.loads(args.continuation_plan.read_text(encoding="utf-8"))
        config = json.loads((ROOT / plan["config"]).read_text(encoding="utf-8"))
        if not any(row["config"] == config for row in report["runs"]):
            raise ValueError("continuation plan configuration does not match supplied actual runs")
        report["continuation_plan"] = {
            "path": str(args.continuation_plan.resolve()),
            "plan": plan,
            "config": config,
            "final_shared_reservation": manifest["budget_reserved"],
        }
        hold = plan["initial_reserved"]
        lines += [
            "## 已授权续跑范围",
            "",
            f"续跑计划包含 {sum(job['limit'] for job in plan['runs'])} 个样本；累计上限 {config['max_requests']} 次请求、{config['max_tokens']} Token、{config['max_cost']} 元，单条 {config['timeout_seconds']} 秒、完整输出 {config['max_total_output_tokens']} Token、重试 {config['retries']} 次。",
            f"历史风险预留：{hold['requests']} 次请求、{hold['tokens']} Token、{hold['cost']} 元；不是实际用量或账单。",
            f"最终共享预留：{manifest['budget_reserved']}；不是实际消费。未执行计划 pending_runs 中的 T2 或全库检索。",
            "",
        ]
    reproduce = "python scripts/report_benchmark.py " + " ".join(
        str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p) for p in (d.resolve() for d in args.runs)
    )
    if args.continuation_plan:
        reproduce += " --continuation-plan " + str(args.continuation_plan)
    lines += ["复现本报告（只读已保存结果，不调用模型）：", "", "```powershell", reproduce, "```"]
    write_json(args.output.with_suffix(".json"), report)
    write_json(args.output.with_name(args.output.name + "-badcases.json"), report["candidate_badcases"])
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(args.output.with_suffix(".md"))


if __name__ == "__main__":
    main()
