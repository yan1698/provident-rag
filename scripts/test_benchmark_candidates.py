"""Synthetic ranking input contracts, independent of private remote-run reports."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.eval.benchmarks.local import validate_candidate_scores


class CandidateScores(unittest.TestCase):
    def setUp(self):
        self.case = {"contexts": [{"id": "a"}, {"id": "b"}], "relevance_labels": {"a": 1, "b": 0}}

    def test_missing_candidate_rejected(self):
        with self.assertRaises(ValueError):
            validate_candidate_scores(self.case, [0.1])

    def test_invalid_scores_rejected(self):
        for value in (float("nan"), float("inf"), True, "0.5", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_candidate_scores(self.case, [0.1, value])

    def test_missing_label_rejected(self):
        self.case["relevance_labels"].pop("b")
        with self.assertRaises(ValueError):
            validate_candidate_scores(self.case, [0.1, 0.2])

    def test_duplicate_candidate_id_rejected(self):
        self.case["contexts"][1]["id"] = "a"
        with self.assertRaises(ValueError):
            validate_candidate_scores(self.case, [0.1, 0.2])

    def test_finite_ties_preserved(self):
        validate_candidate_scores(self.case, [0.5, 0.5])


if __name__ == "__main__":
    unittest.main()
