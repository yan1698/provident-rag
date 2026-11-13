"""Chinese public benchmark CLI. Validation/preparation never import models or connect to DB."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.eval.benchmarks.adapters import MODES, validate_cases
from app.eval.benchmarks.data import convert, fetch, load_snapshot, registry
from app.eval.benchmarks.runner import compare, run, select_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    download = sub.add_parser("fetch")
    download.add_argument(
        "benchmark", choices=("rgb", "crud", "cmteb_du", "cmteb_du_qrels", "cmteb_t2", "cmteb_protocol")
    )
    download.add_argument("--max-file-bytes", type=int, default=16_000_000)
    download.add_argument("--file", action="append", help="explicit registered files; may include optional corpus")
    download.add_argument("--retry-invalid", action="store_true", help="preserve incomplete downloaded bytes and retry")
    conversion = sub.add_parser("convert")
    sample = sub.add_parser("fetch-sample")
    sample.add_argument("benchmark", choices=("cmteb_t2",))
    sample.add_argument("--limit", type=int, default=20)
    sample.add_argument("--max-bytes", type=int, default=32_000_000)
    conversion.add_argument("benchmark", choices=("rgb", "crud", "cmteb_du", "cmteb_t2"))
    conversion.add_argument("--file", required=True)
    conversion.add_argument("--limit", type=int, help="convert only first N official rows; never changes split")
    conversion.add_argument("--task", help="CRUD official task name")
    conversion.add_argument("--noise-rate", type=float, default=0)
    conversion.add_argument("--correct-rate", type=float, default=0)
    conversion.add_argument("--passage-num", type=int, default=5)
    conversion.add_argument("--seed", type=int, default=2333)
    validation = sub.add_parser("validate")
    validation.add_argument("snapshot", type=Path)
    for command in ("dry-run", "evaluate"):
        execution = sub.add_parser(command)
        execution.add_argument("snapshot", type=Path)
        execution.add_argument("--config", type=Path, required=True)
        execution.add_argument("--mode", choices=sorted(MODES), required=True)
        execution.add_argument("--limit", type=int, required=True)
        execution.add_argument("--offset", type=int, default=0)
        execution.add_argument("--ids", help="comma-separated source IDs or row locators")
        execution.add_argument("--authorize-models", action="store_true")
        execution.add_argument("--worker-python", default=sys.executable)
        if command == "dry-run":
            execution.add_argument(
                "--execute", action="store_true", help="real controlled execution; requires authorization"
            )
    comparison = sub.add_parser("compare")
    comparison.add_argument("left", type=Path)
    comparison.add_argument("right", type=Path)
    args = parser.parse_args()
    if args.command == "fetch":
        print(
            json.dumps(
                fetch(args.benchmark, max_bytes=args.max_file_bytes, files=args.file, retry_invalid=args.retry_invalid),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "fetch-sample":
        from app.eval.benchmarks.ranges import sample_t2

        print(sample_t2(limit=args.limit, max_bytes=args.max_bytes))
    elif args.command == "convert":
        options = {"limit": args.limit, "task": args.task}
        if args.benchmark == "rgb":
            options.update(
                noise_rate=args.noise_rate, correct_rate=args.correct_rate, passage_num=args.passage_num, seed=args.seed
            )
        print(convert(args.benchmark, args.file, **options))
    elif args.command == "validate":
        snapshot, cases = load_snapshot(args.snapshot)
        print(json.dumps({**validate_cases(cases), "snapshot_sha256": snapshot["snapshot_sha256"]}))
    elif args.command in {"dry-run", "evaluate"}:
        if args.command == "dry-run" and args.limit > 20:
            parser.error("dry-run supports at most 20 samples")
        _, cases = load_snapshot(args.snapshot)
        selected = select_cases(
            cases, limit=args.limit, offset=args.offset, ids=args.ids.split(",") if args.ids else None
        )
        folder = run(
            args.snapshot,
            json.loads(args.config.read_text(encoding="utf-8")),
            mode=args.mode,
            selected=selected,
            authorize_models=args.authorize_models,
            execute=args.command == "evaluate" or args.execute,
            worker_python=args.worker_python,
        )
        print(folder)
        summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        if (args.command == "evaluate" or args.execute) and summary["executed"] != summary["selected"]:
            raise SystemExit(2)
    else:
        print(json.dumps(compare(args.left, args.right), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, RuntimeError, KeyError, OSError) as exc:
        raise SystemExit(f"benchmark error: {exc}") from exc
