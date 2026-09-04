from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from marginshield.data_builder import (
    ENTITY_WINDOWS,
    MODEL_FEATURE_COLUMNS,
    TARGET_ONLY_COLUMNS,
    apply_point_in_time_features,
    apply_temporal_entity_counts,
)


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "processed"
REPORTS = ROOT / "data" / "reports"


class DatasetContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = pd.read_csv(DATA / "master_refund_cases.csv.gz")
        cls.links = pd.read_csv(DATA / "entity_links.csv.gz")

    def test_ring_native_dataset_contract(self) -> None:
        required = {"case_id", "event_timestamp", "split", "ring_label", "ring_id", "scenario_type", "device_payment_pair_accounts_30d"}
        self.assertTrue(required.issubset(self.cases.columns))
        self.assertTrue(self.cases["case_id"].is_unique)
        self.assertTrue(self.cases["is_synthetic"].eq(1).all())
        self.assertEqual({"train", "validation", "test"}, set(self.cases["split"]))
        self.assertTrue(self.cases.loc[self.cases["ring_label"].eq(1), "ring_id"].ne("").all())
        self.assertFalse(set(MODEL_FEATURE_COLUMNS) & set(TARGET_ONLY_COLUMNS))
        self.assertIn("merchant_refund_volume_index_30d", MODEL_FEATURE_COLUMNS)
        self.assertNotIn("merchant_refund_rate_30d", MODEL_FEATURE_COLUMNS)
        self.assertNotIn("shared_payment_accounts_30d", MODEL_FEATURE_COLUMNS)

    def test_basic_entity_counts_are_point_in_time(self) -> None:
        expected = apply_temporal_entity_counts(self.cases)
        columns = [feature for feature, _ in ENTITY_WINDOWS.values()]
        pd.testing.assert_frame_equal(
            self.cases[columns].reset_index(drop=True), expected[columns].reset_index(drop=True), check_dtype=False
        )
        self.assertEqual(len(self.links), len(self.cases) * 3)

    def test_future_entity_change_does_not_change_past_features(self) -> None:
        sample = self.cases.sort_values(["event_timestamp", "case_id"]).head(80).copy().reset_index(drop=True)
        baseline = apply_point_in_time_features(sample)
        changed = sample.copy()
        changed.loc[len(changed) - 1, ["device_id", "payment_token_id"]] = changed.loc[0, ["device_id", "payment_token_id"]].to_numpy()
        recomputed = apply_point_in_time_features(changed)
        columns = [name for name in MODEL_FEATURE_COLUMNS if name not in {"vertical", "payment_method", "refund_amount_inr", "refund_share", "account_age_days"}]
        pd.testing.assert_series_equal(baseline.loc[0, columns], recomputed.loc[0, columns], check_names=False)

    def test_hard_benign_identity_scenarios_are_negative(self) -> None:
        benign = self.cases.loc[self.cases["scenario_type"].str.startswith("benign_")]
        self.assertGreater(len(benign), 1_000)
        self.assertTrue(benign["ring_label"].eq(0).all())
        self.assertGreater((benign["shared_identifier_types_30d"] >= 2).sum(), 0)
        self.assertGreater((benign["linked_refund_burst_72h"] >= 2).sum(), 0)

    def test_locked_test_rings_are_temporally_slower(self) -> None:
        test_rings = self.cases.loc[self.cases["split"].eq("test") & self.cases["ring_label"].eq(1)]
        validation_rings = self.cases.loc[self.cases["split"].eq("validation") & self.cases["ring_label"].eq(1)]
        self.assertTrue(test_rings["ring_topology"].eq("slower_two_identifier").all())
        spans = lambda frame: (
            frame.assign(event_timestamp=pd.to_datetime(frame["event_timestamp"], utc=True, format="mixed"))
            .groupby("ring_id")["event_timestamp"]
            .agg(lambda values: (values.max() - values.min()).total_seconds() / 3600)
        )
        self.assertGreater(spans(test_rings).median(), spans(validation_rings).median())
        validation = json.loads((REPORTS / "validation_report.json").read_text())
        self.assertTrue(validation["passed"])
        self.assertTrue(all(validation["checks"].values()))


if __name__ == "__main__":
    unittest.main()
