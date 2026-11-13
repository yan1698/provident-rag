"""Offline regression checks; official adapter fixture, arithmetic, isolation and authorization only."""

from __future__ import annotations

from lib.test_materials import load_test_materials

_TEST_MATERIALS = load_test_materials(__file__)

import ast
import contextlib
import io
import json
import math
import random
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.eval.benchmarks.adapters import adapt, rgb_contexts, validate_cases
from app.eval.benchmarks.data import canonical, digest, fetch, load_snapshot, registry, write_json
from app.eval.benchmarks.metrics import aggregate, metric, percentile, ranking, rgb_match
from app.eval.benchmarks.runner import compare, run, select_cases, validate_config
from app.eval.benchmarks.worker import execute


def fixture():
    obj = json.loads((ROOT / _TEST_MATERIALS["text_004"]).read_text(encoding="utf-8"))
    assert digest(canonical(obj["row"])) == obj["row_sha256"]
    return obj["row"]


def snapshot(path):
    cases = adapt(
        "rgb",
        [fixture()],
        relative="data/zh_refine.json",
        spec=registry("rgb"),
        options={"noise_rate": 0.6, "passage_num": 5, "seed": 2333},
    )
    obj = {"schema_version": 1, "benchmark": "rgb", "dataset_revision": registry("rgb")["revision"], "cases": cases}
    obj["snapshot_sha256"] = digest(canonical(obj))
    write_json(path, obj)
    return cases


class PublicBenchmarks(unittest.TestCase):
    def setUp(self):
        self.config = json.loads((ROOT / _TEST_MATERIALS["text_014"]).read_text())

    def test_source_archive_without_git_retains_hashes_and_unknown_status(self):
        from app.eval.benchmarks.runner import code_snapshot

        with patch(_TEST_MATERIALS["text_001"], side_effect=FileNotFoundError):
            snapshot = code_snapshot()
        self.assertIsNone(snapshot["git_commit"])
        self.assertIsNone(snapshot["dirty"])
        self.assertEqual(len(snapshot["source_sha256"]), 64)
        self.assertIn("benchmarks/registry.json", snapshot["files"])
        self.assertNotIn(".env", snapshot["files"])

    def test_scope_is_chinese_v1(self):
        all_specs = json.loads((ROOT / "benchmarks/registry.json").read_text())
        self.assertEqual(
            set(all_specs) - {"schema_version"},
            {"rgb", "crud", "cmteb_du", "cmteb_t2", "cmteb_du_qrels", "cmteb_protocol"},
        )
        for name in ("rgb", "crud", "cmteb_du", "cmteb_t2", "cmteb_du_qrels"):
            self.assertEqual(len(registry(name)["revision"]), 40)

    def test_rgb_preserves_raw_row_and_original_split(self):
        row = fixture()
        original = deepcopy(row)
        cases = adapt("rgb", [row], relative="data/zh_refine.json", spec=registry("rgb"), options={"noise_rate": 0.6})
        self.assertEqual(validate_cases(cases)["samples"], 1)
        self.assertEqual(row, original)
        self.assertEqual(cases[0]["original"], original)
        self.assertIsNone(cases[0]["official_split"])
        self.assertEqual(len(cases[0]["contexts"]), 5)

    def test_rgb_refusal_has_negative_contexts_only(self):
        row = fixture()
        actual = rgb_contexts(row, "zh_refine", {"noise_rate": 1})
        self.assertTrue(all(text in row["negative"] for text in actual))

    def test_reference_protocol_and_aliases(self):
        self.assertTrue(rgb_match("A B c", ["ab", ["missing", "c"]]))
        self.assertFalse(rgb_match("A B", ["ab", "c"]))
        self.assertFalse(rgb_match("a", [["b", "c"]]))

    def test_rgb_official_selection_alignment_if_downloaded(self):
        from app.eval.benchmarks.data import DATA_ROOT, verified_raw, read_rows

        folder = DATA_ROOT / "raw/rgb" / registry("rgb")["revision"]
        if not (folder / "evalue.py").exists():
            self.skipTest(_TEST_MATERIALS["text_002"])
        # Load only the inspected pure selector, not upstream model imports or executable CLI.
        tree = ast.parse(verified_raw("rgb", "evalue.py")[0].read_text(encoding="utf-8"))
        function = next(x for x in tree.body if isinstance(x, ast.FunctionDef) and x.name == "processdata")
        namespace = {"math": math, "random": random}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "official_RGB_processdata", "exec"), namespace)
        for variant in ("zh_refine", "zh_int", "zh_fact"):
            rows = read_rows(verified_raw("rgb", f"data/{variant}.json")[0])[:10]
            for noise in (0, 0.6, 1) if variant == "zh_refine" else (0, 0.6):
                for row in rows:
                    random.seed(2333)
                    with contextlib.redirect_stdout(io.StringIO()):
                        expected = namespace["processdata"](deepcopy(row), noise, 5, variant, 0)[2]
                    self.assertEqual(rgb_contexts(row, variant, {"noise_rate": noise}), expected)

    def test_ranking_math_and_no_labels(self):
        result = ranking(["b", "a", "c"], {"a": 1, "c": 1}, complete_ranking=True)
        self.assertEqual(result["mrr@10"]["value"], 0.5)
        self.assertAlmostEqual(result["map"]["value"], (1 / 2 + 2 / 3) / 2)
        self.assertEqual(ranking([], None)["ndcg@10"]["status"], "not_evaluable")
        self.assertEqual(ranking([], {"a": 1})["recall@5"]["value"], 0)
        with self.assertRaises(ValueError):
            ranking(["a", "a"], {"a": 1})

    def test_tied_map_matches_threshold_ap(self):
        out = ranking(
            ["a", "b"],
            {"a": 1, "b": 0},
            complete_ranking=True,
            scored=[{"id": "a", "score": 0.5}, {"id": "b", "score": 0.5}],
        )
        self.assertEqual(out["map"]["value"], 0.5)

    def test_configuration_rejects_unbounded_or_stub(self):
        validate_config(self.config)
        for key, value in (
            ("max_tokens", 0),
            ("timeout_seconds", float("inf")),
            ("max_cost", float("nan")),
            ("retries", 1),
            ("provider", "stub"),
            ("input_price_per_million", 0),
        ):
            with self.assertRaises(ValueError):
                validate_config({**self.config, key: value})

    def test_authorization_before_model_import_or_worker(self):
        with self.assertRaises(ValueError):
            execute({"authorize_models": False})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            cases = snapshot(path)
            with patch(_TEST_MATERIALS["text_005"], side_effect=AssertionError("must not spawn")):
                with self.assertRaisesRegex(ValueError, "authorize"):
                    run(
                        path,
                        self.config,
                        mode="generation_with_context",
                        selected=cases,
                        execute=True,
                        root=Path(directory),
                    )

    def test_dry_run_persists_no_fabricated_output(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            cases = snapshot(path)
            folder = run(
                path, self.config, mode="generation_with_context", selected=cases, execute=False, root=Path(directory)
            )
            result = json.loads((folder / "results.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(result["status"], "prepared_not_executed")
            self.assertIsNone(result["output"])
            self.assertIsNone(result["application_trace_id"])
            summary = json.loads((folder / "summary.json").read_text())
            self.assertEqual(summary["executed"], 0)
            self.assertIsNone(summary["p50_ms"])
            with self.assertRaisesRegex(ValueError, "prepared"):
                compare(folder, folder)

    def test_unsupported_mode_never_spawns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            cases = snapshot(path)
            folder = run(
                path,
                self.config,
                mode="multi_turn_rag",
                selected=cases,
                execute=True,
                authorize_models=True,
                root=Path(directory),
            )
            results = json.loads((folder / "results.json").read_text(encoding="utf-8"))
            self.assertEqual(results[0]["status"], "not_supported")
            self.assertIsNone(results[0]["output"])

    def test_checksum_and_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            cases = snapshot(path)
            self.assertEqual(load_snapshot(path)[1], cases)
            obj = json.loads(path.read_text(encoding="utf-8"))
            obj["cases"][0]["contexts"][0]["text"] += "!"
            write_json(path, obj)
            with self.assertRaisesRegex(ValueError, "checksum"):
                load_snapshot(path)
            for kwargs in ({"limit": 0}, {"limit": 1, "offset": 3}, {"limit": 1, "ids": ["unknown"]}):
                with self.assertRaises(ValueError):
                    select_cases(cases, **kwargs)

    def test_raw_tampering_rejected_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw/rgb" / registry("rgb")["revision"]
            raw.mkdir(parents=True)
            (raw / "readme.md").write_text("corrupt")
            write_json(raw / "manifest.json", {"files": {"readme.md": {"sha256": "incorrect"}}})
            with patch(_TEST_MATERIALS["text_006"], side_effect=AssertionError("no network")):
                with self.assertRaisesRegex(ValueError, "checksum"):
                    fetch("rgb", root=root, files=["readme.md"])

    def test_metric_denominators_and_percentiles(self):
        result = aggregate(
            [
                {"status": "error", "badcase_types": ["execution_failure"], "metrics": {}},
                {"status": "executed", "badcase_types": [], "metrics": {"x": metric(0.5)}, "latency_ms": 10},
            ]
        )
        self.assertEqual(result["metrics"]["x"]["selected_samples"], 2)
        self.assertEqual(result["metrics"]["x"]["evaluated_samples"], 1)
        self.assertEqual(percentile([10, 30], 0.95), 29)
        self.assertIsNone(percentile([], 0.5))

    def test_jsonl_conversion_limit(self):
        from app.eval.benchmarks.data import read_rows

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "official.json"
            path.write_bytes(canonical(fixture()) + b"\n" + canonical(fixture()) + b"\n")
            self.assertEqual(len(read_rows(path, limit=1)), 1)

    def test_truncated_transfer_not_accepted_as_raw(self):
        class Response(io.BytesIO):
            headers = {"Content-Length": "100"}

        with tempfile.TemporaryDirectory() as directory:
            with patch(_TEST_MATERIALS["text_007"], return_value=Response(b"short")):
                with self.assertRaisesRegex(RuntimeError, "incomplete HTTP"):
                    fetch("rgb", root=Path(directory), files=["readme.md"])
            raw = Path(directory) / "raw/rgb" / registry("rgb")["revision"]
            self.assertFalse((raw / "readme.md").exists())
            self.assertIn("error", json.loads((raw / "manifest.json").read_text())["files"]["readme.md"])

    def test_range_reader_refuses_full_response(self):
        from app.eval.benchmarks.ranges import RangeFile

        class Response(io.BytesIO):
            status = 200
            headers = {}

        with tempfile.TemporaryDirectory() as directory:
            reader = RangeFile("https://official.example/fixed-revision", 100, Path(directory), max_bytes=8)
            with patch(_TEST_MATERIALS["text_008"], return_value=Response(b"12345678")):
                with self.assertRaisesRegex(ValueError, _TEST_MATERIALS["text_015"]):
                    reader.read(8)
            with self.assertRaisesRegex(ValueError, "byte cap"):
                reader.read(9)

    def test_shared_budget_prevents_another_call(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            cases = snapshot(path)
            folder = run(
                path,
                self.config,
                mode="generation_with_context",
                selected=cases,
                execute=True,
                authorize_models=True,
                root=Path(directory),
                initial_reserved={"requests": self.config["max_requests"], "tokens": 0, "cost": 0},
            )
            result = json.loads((folder / "results.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(result["status"], "budget_stopped")
            self.assertIsNone(result["output"])

    def test_confirmed_usage_releases_hold_and_scoring_failure_is_separate(self):
        # Synthetic transport fixture only; never stored as a public benchmark run.
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "snapshot.json"
            cases = snapshot(path)

            def child(argv, **kwargs):
                write_json(
                    Path(argv[-1]),
                    {
                        "usage": [{"prompt_tokens": 11, "completion_tokens": 7}],
                        "output": {"answer": _TEST_MATERIALS["text_016"]},
                        "effective_config": {},
                        "stages_ms": {},
                        "timing_meta": {},
                    },
                )
                return SimpleNamespace(returncode=0)

            with (
                patch(_TEST_MATERIALS["text_009"], return_value={}),
                patch(
                    _TEST_MATERIALS["text_010"],
                    return_value={"archive_sha256": "test", "archive": "source.zip"},
                ),
                patch(_TEST_MATERIALS["text_011"], side_effect=child),
                patch(_TEST_MATERIALS["text_012"], side_effect=ValueError("unit scoring error")),
            ):
                folder = run(
                    path,
                    self.config,
                    mode="generation_with_context",
                    selected=cases,
                    execute=True,
                    authorize_models=True,
                    root=Path(directory),
                )
            manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
            trace = json.loads((folder / "results.json").read_text(encoding="utf-8"))[0]
            self.assertEqual(manifest["budget_reserved"]["tokens"], 18)
            self.assertEqual(manifest["budget_reserved"]["requests"], 1)
            self.assertAlmostEqual(manifest["budget_reserved"]["cost"], (11 * 12 + 7 * 36) / 1_000_000)
            self.assertEqual(trace["inference_status"], "completed")
            self.assertEqual(trace["badcase_types"], ["scoring_failure"])
            self.assertEqual(manifest["status"], "completed_with_failures")

    def test_downloaded_chinese_tasks_validate_if_present(self):
        from app.eval.benchmarks.data import verified_raw, read_rows

        try:
            path, _ = verified_raw("crud", "data/crud_split/split_merged.json")
        except OSError:
            self.skipTest(_TEST_MATERIALS["text_013"])
        groups = read_rows(path)
        for task in groups:
            cases = adapt(
                "crud",
                groups,
                relative="data/crud_split/split_merged.json",
                spec=registry("crud"),
                options={"task": task, "limit": 20},
            )
            self.assertEqual(validate_cases(cases)["samples"], 20)
            self.assertEqual(cases[0]["original"], groups[task][0])

    def test_benchmark_transport_caps_total_not_only_final_text(self):
        from app.eval.benchmarks.transport import bounded_client

        bodies = []

        def create(**body):
            bodies.append(body)
            return SimpleNamespace(usage=SimpleNamespace(prompt_tokens=3, completion_tokens=12), choices=[])

        resource = SimpleNamespace(create=create)
        client = SimpleNamespace(
            _inner=SimpleNamespace(_client=SimpleNamespace(chat=SimpleNamespace(completions=resource)))
        )
        calls = []
        with patch(_TEST_MATERIALS["text_003"], return_value=client):
            actual = bounded_client(object(), total_output_tokens=2048, calls=calls)
            actual._inner._client.chat.completions.create(model="qwen3.8-max", max_tokens=512)
        self.assertNotIn("max_tokens", bodies[0])
        self.assertEqual(bodies[0]["max_completion_tokens"], 2048)
        self.assertEqual(calls[0]["completion_tokens"], 12)


if __name__ == "__main__":
    unittest.main(verbosity=2)
