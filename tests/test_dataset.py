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
from marginshield.feature_engine import PointInTimeFeatureEngine, RawRefundEvent, apply_features_via_engine


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
        self.assertIn("shared_payment_accounts_30d", MODEL_FEATURE_COLUMNS)
        self.assertIn("connected_accounts_max_window", MODEL_FEATURE_COLUMNS)
        self.assertIn("multi_identifier_neighbor_accounts", MODEL_FEATURE_COLUMNS)

    def test_basic_entity_counts_are_point_in_time(self) -> None:
        expected = apply_temporal_entity_counts(self.cases)
        columns = [feature for feature, _ in ENTITY_WINDOWS.values()]
        pd.testing.assert_frame_equal(
            self.cases[columns].reset_index(drop=True), expected[columns].reset_index(drop=True), check_dtype=False
        )
        self.assertEqual(len(self.links), len(self.cases) * 3)

    def test_offline_table_matches_incremental_live_engine(self) -> None:
        replayed, _ = apply_features_via_engine(self.cases)
        pd.testing.assert_frame_equal(
            self.cases[MODEL_FEATURE_COLUMNS].reset_index(drop=True),
            replayed[MODEL_FEATURE_COLUMNS].reset_index(drop=True),
            check_dtype=False,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_seven_day_window_uses_seconds_not_datetime_storage_unit(self) -> None:
        engine = PointInTimeFeatureEngine()
        base = dict(
            customer_id="customer", merchant_id="merchant", device_id="device", address_id="address",
            payment_token_id="payment", product_id="product", vertical="home", payment_method="upi", refund_amount_inr=100,
            refund_share=1.0, account_age_days=100,
        )
        first = engine.observe(RawRefundEvent(case_id="one", event_seconds=0, **base))
        second = engine.observe(RawRefundEvent(case_id="two", event_seconds=6 * 86_400, **base))
        third = engine.observe(RawRefundEvent(case_id="three", event_seconds=14 * 86_400, **base))
        self.assertEqual(first["refund_velocity_7d"], 1)
        self.assertEqual(second["refund_velocity_7d"], 2)
        self.assertEqual(third["refund_velocity_7d"], 1)

    def test_entity_type_namespaces_prevent_identifier_collision(self) -> None:
        engine = PointInTimeFeatureEngine()
        common = dict(merchant_id="merchant", vertical="home", payment_method="upi", refund_amount_inr=100, refund_share=1.0, account_age_days=100)
        engine.observe(RawRefundEvent(case_id="one", event_seconds=0, customer_id="c1", device_id="SAME", address_id="a1", payment_token_id="p1", product_id="product-1", **common))
        observed = engine.observe(RawRefundEvent(case_id="two", event_seconds=1, customer_id="c2", device_id="d2", address_id="SAME", payment_token_id="p2", product_id="product-2", **common))
        self.assertEqual(observed["shared_address_accounts_90d"], 1)
        self.assertEqual(observed["connected_accounts_max_window"], 1)

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
        expected = {
            "benign_family_wallet", "benign_group_purchase", "benign_corporate_card",
            "benign_hostel_kiosk", "benign_office_network", "benign_support_migration",
        }
        self.assertTrue(expected.issubset(set(benign["scenario_type"])))

    def test_locked_test_rings_are_temporally_slower(self) -> None:
        test_rings = self.cases.loc[self.cases["split"].eq("test") & self.cases["ring_label"].eq(1)]
        validation_rings = self.cases.loc[self.cases["split"].eq("validation") & self.cases["ring_label"].eq(1)]
        test_topologies = set(test_rings["ring_topology"])
        validation_topologies = set(validation_rings["ring_topology"])
        self.assertGreaterEqual(len(test_topologies), 4)
        self.assertGreaterEqual(len(validation_topologies), 4)
        self.assertTrue(test_topologies.isdisjoint(validation_topologies))
        spans = lambda frame: (
            frame.assign(event_timestamp=pd.to_datetime(frame["event_timestamp"], utc=True, format="mixed"))
            .groupby("ring_id")["event_timestamp"]
            .agg(lambda values: (values.max() - values.min()).total_seconds() / 3600)
        )
        self.assertGreater(spans(test_rings).median(), spans(validation_rings).median())
        validation = json.loads((REPORTS / "validation_report.json").read_text())
        self.assertTrue(validation["passed"])
        self.assertTrue(all(validation["checks"].values()))
        shortcut = validation["shortcut_gate"]
        self.assertLess(shortcut["validation_pair_rule_precision"], 0.70)
        self.assertGreaterEqual(shortcut["legitimate_pair_reuse_rows"], shortcut["positive_pair_reuse_rows"])
        self.assertLess(shortcut["maximum_within_shared_univariate_ap_lift"], 2.0)
        self.assertGreaterEqual(shortcut["independent_legitimate_shared_identity_rows"], 25)
        self.assertLess(shortcut["shared_identifier_rule_precision"], 0.50)


if __name__ == "__main__":
    unittest.main()
