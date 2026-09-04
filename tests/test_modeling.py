from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from marginshield.data_builder import MODEL_FEATURE_COLUMNS, TARGET_ONLY_COLUMNS
from marginshield.tournament import classification_metrics, load_dataset, ring_level_metrics, select_threshold


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "data" / "reports"


class ModelingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = json.loads((REPORTS / "ring_model_report.json").read_text())
        cls.splits = load_dataset()

    def test_exact_feature_contract_excludes_all_target_metadata(self) -> None:
        self.assertEqual(self.report["target"], "ring_label")
        self.assertEqual(self.report["feature_contract"], MODEL_FEATURE_COLUMNS)
        self.assertFalse(set(MODEL_FEATURE_COLUMNS) & set(TARGET_ONLY_COLUMNS))
        self.assertNotIn("ring_id", MODEL_FEATURE_COLUMNS)
        self.assertNotIn("scenario_type", MODEL_FEATURE_COLUMNS)

    def test_validation_threshold_meets_declared_safety_constraint(self) -> None:
        validation = self.report["policy_validation"]
        self.assertGreaterEqual(validation["precision"], 0.85)
        self.assertGreaterEqual(validation["review_volume"], 30)
        self.assertGreater(validation["recall"], 0.0)

    def test_final_report_has_ring_and_uncertainty_metrics(self) -> None:
        test = self.report["test"]
        self.assertIn("ring_level", test)
        self.assertIn("ring_precision", test["ring_level"])
        self.assertIn("ring_recall", test["ring_level"])
        self.assertIn("precision", self.report["test_temporal_bootstrap_95pct_ci"])
        self.assertGreater(test["precision"], 0.0)
        costs = test["costs"]
        self.assertEqual(
            costs["net_preventable_value_inr"],
            costs["preventable_loss_inr"] - costs["false_positive_cost_inr"] - costs["true_positive_review_cost_inr"],
        )

    def test_context_diagnostic_does_not_explain_ring_target(self) -> None:
        diagnostics = self.report["diagnostics"]["context_only"]["test"]
        self.assertLess(diagnostics["pr_auc"], 0.08)
        self.assertGreater(self.report["diagnostics"]["graph_velocity_only"]["test"]["pr_auc"], diagnostics["pr_auc"])

    def test_pair_rule_shortcut_is_reported(self) -> None:
        diagnostic = self.report["diagnostics"]["device_payment_pair_rule"]["test"]
        self.assertEqual(diagnostic["rule"], "device_payment_pair_accounts_30d >= 2")
        self.assertGreater(diagnostic["precision"], 0.8)
        self.assertIn("recoverability", diagnostic["interpretation"])

    def test_threshold_selection_enforces_precision_and_review_volume(self) -> None:
        frame = pd.DataFrame({
            "ring_label": [1, 1, 1, 0, 0, 0],
            "expected_loss_if_ring_inr": [500, 500, 500, 0, 0, 0],
            "false_positive_cost_inr": [0, 0, 0, 50, 50, 50],
            "review_cost_inr": [50] * 6,
            "ring_id": ["A", "A", "B", "", "", ""],
            "device_id": ["d"] * 6,
            "payment_token_id": ["p"] * 6,
            "device_payment_pair_accounts_30d": [2] * 6,
            "event_timestamp": pd.date_range("2025-01-01", periods=6, tz="UTC"),
        })
        scores = np.array([0.95, 0.90, 0.85, 0.80, 0.20, 0.10])
        threshold, _ = select_threshold(frame, scores, precision_floor=0.85, min_flags=3)
        self.assertGreaterEqual(frame.loc[scores >= threshold, "ring_label"].mean(), 0.85)
        self.assertGreaterEqual((scores >= threshold).sum(), 3)

    def test_first_request_is_not_an_observed_multi_account_ring(self) -> None:
        frame = pd.DataFrame({
            "ring_label": [1, 1], "ring_id": ["A", "A"],
            "expected_loss_if_ring_inr": [100, 100], "false_positive_cost_inr": [0, 0],
            "review_cost_inr": [10, 10], "device_id": ["d", "d"], "payment_token_id": ["p", "p"],
            "device_payment_pair_accounts_30d": [1, 2],
            "event_timestamp": pd.date_range("2025-01-01", periods=2, tz="UTC"),
        })
        metric = ring_level_metrics(frame, np.array([0.9, 0.1]), threshold=0.5)
        self.assertEqual(metric["detected_before_half_loss"], 0)


if __name__ == "__main__":
    unittest.main()
