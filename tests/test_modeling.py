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

    def test_report_and_dataset_are_the_same_generation(self) -> None:
        """A dataset rebuild without a retrain must fail loudly, not score silently."""
        manifest = json.loads((ROOT / "data" / "processed" / "dataset_manifest.json").read_text())
        self.assertEqual(self.report["dataset_version"], manifest["version"])
        for split in self.splits.values():
            self.assertEqual(set(split["generator_version"].unique()), {manifest["version"]})

    def test_validation_policy_meets_declared_capacity_and_value_constraint(self) -> None:
        validation = self.report["policy_validation"]
        calibration = self.report["calibration"]
        # The declared policy is value-maximising under an explicit review capacity.
        self.assertGreaterEqual(validation["review_volume"], 30)
        self.assertLessEqual(validation["review_volume"], 100)
        self.assertGreater(validation["recall"], 0.0)
        self.assertGreater(validation["costs"]["net_preventable_value_inr"], 0.0)
        self.assertIn("Neither tier imposes a precision floor", calibration["policy_selection"])

    def test_value_policy_beats_the_precision_floor_it_replaced(self) -> None:
        """The floor was removed on evidence, so the evidence stays in the report."""
        comparison = self.report["precision_floor_comparison"]
        self.assertEqual(comparison["compared_precision_floor"], 0.85)
        self.assertFalse(self.report["policy_validation"]["precision"] >= 0.85 and not comparison["precision_floor_feasible"])
        if comparison["precision_floor_feasible"]:
            self.assertGreaterEqual(comparison["net_value_gained_inr"], 0.0)
            self.assertGreaterEqual(
                comparison["locked_net_value_inr"], comparison["precision_floor_net_value_inr"]
            )

    def test_both_tiers_are_locked_and_reported(self) -> None:
        calibration = self.report["calibration"]
        self.assertIsNotNone(calibration["verify_evidence_threshold"])
        for key in ("policy_tiers", "test_policy_tiers"):
            tiers = self.report[key]
            self.assertIn("verify_evidence", tiers)
            self.assertIn("manual_review", tiers)
            # Verification is gated on structural evidence, so it never exceeds any intervention.
            self.assertLessEqual(tiers["verify_evidence"]["volume"], tiers["any_intervention"]["volume"])

    def test_final_report_has_ring_and_uncertainty_metrics(self) -> None:
        test = self.report["test"]
        self.assertIn("ring_level", test)
        self.assertIn("ring_precision", test["ring_level"])
        self.assertIn("ring_recall", test["ring_level"])
        self.assertIn("precision", self.report["test_temporal_bootstrap_95pct_ci"])
        self.assertGreater(test["precision"], 0.0)
        self.assertIn("expected_calibration_error", test["calibration"])
        self.assertTrue(test["calibration"]["bins"])
        self.assertIn("above_1pct", test["calibration"])
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
        diagnostic = self.report["diagnostics"]["device_payment_pair_rule"]["validation"]
        self.assertEqual(diagnostic["rule"], "device_payment_pair_accounts_30d >= 2")
        self.assertLess(diagnostic["precision"], 0.70)
        self.assertIn("recoverability", diagnostic["interpretation"])

    def test_within_shared_shortcut_and_topology_diagnostics_are_reported(self) -> None:
        diagnostic = self.report["diagnostics"]["within_shared_identity"]["validation"]
        self.assertLess(diagnostic["maximum_univariate_ap_lift"], 2.0)
        self.assertGreater(diagnostic["rows"], 0)
        topology = self.report["test_topology_recall"]["by_topology"]
        self.assertGreaterEqual(len(topology), 4)
        self.assertTrue(all("early_ring_recall" in row for row in topology))
        rates = [row["refund_value_rate"] for row in self.report["false_positive_cost_sensitivity"]]
        self.assertEqual(rates, [0.05, 0.075, 0.15])

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
            "case_id": ["case-1", "case-2"],
            "ring_label": [1, 1], "ring_id": ["A", "A"],
            "expected_loss_if_ring_inr": [100, 100], "false_positive_cost_inr": [0, 0],
            "review_cost_inr": [10, 10], "customer_id": ["c1", "c2"],
            "device_id": ["d", "d"], "address_id": ["a1", "a2"], "payment_token_id": ["p", "p"],
            "device_payment_pair_accounts_30d": [1, 2],
            "shared_device_accounts_30d": [1, 2],
            "shared_address_accounts_90d": [1, 1],
            "shared_payment_accounts_30d": [1, 2],
            "event_timestamp": pd.date_range("2025-01-01", periods=2, tz="UTC"),
        })
        metric = ring_level_metrics(frame, np.array([0.9, 0.1]), threshold=0.5)
        self.assertEqual(metric["detected_before_half_loss"], 0)


if __name__ == "__main__":
    unittest.main()
