from __future__ import annotations

import json
import sqlite3
import unittest
import uuid
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from fastapi.testclient import TestClient

import server
from server import MarginShieldService, app


class LiveApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temp_dir = TemporaryDirectory()
        cls.original_service = server.service
        cls.service = MarginShieldService(Path(cls.temp_dir.name) / "decisions.sqlite")
        server.service = cls.service
        cls.client = TestClient(app)

    @classmethod
    def tearDownClass(cls) -> None:
        server.service = cls.original_service
        cls.temp_dir.cleanup()

    def feature_payload(self) -> dict:
        row = self.service.cases.iloc[-1]
        return {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in self.service.bundle.features}

    def test_runtime_state_from_an_older_build_is_discarded_not_fatal(self) -> None:
        """A dataset rebuild must not leave the service unable to start."""
        with TemporaryDirectory() as directory:
            db_path = Path(directory) / "decisions.sqlite"
            stale_event = {
                "case_id": "STALE-0001", "event_timestamp": "2020-01-01T00:00:00+00:00",
                "customer_id": "c1", "merchant_id": "m1", "device_id": "d1", "address_id": "a1",
                "payment_token_id": "p1", "product_id": "pr1", "vertical": "home",
                "payment_method": "upi", "refund_amount_inr": 1000.0, "refund_share": 0.5,
                "account_age_days": 100.0,
            }
            connection = sqlite3.connect(db_path)
            connection.executescript("""
                CREATE TABLE decisions (
                    case_id TEXT PRIMARY KEY, event_timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL, threshold REAL NOT NULL,
                    raw_event_json TEXT NOT NULL, features_json TEXT NOT NULL,
                    probability REAL NOT NULL, recommendation TEXT NOT NULL,
                    evidence_json TEXT NOT NULL, linked_state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL, replayable INTEGER NOT NULL
                );
                CREATE TABLE analyst_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT, case_id TEXT NOT NULL,
                    action TEXT NOT NULL, note TEXT NOT NULL, created_at TEXT NOT NULL
                );
            """)
            connection.execute(
                "INSERT INTO decisions VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                ("STALE-0001", stale_event["event_timestamp"], "marginshield-0.0.0-old", 0.5,
                 json.dumps(stale_event), json.dumps({}), 0.5, "approve", "{}", "{}",
                 "2020-01-01T00:00:00+00:00", 1),
            )
            connection.commit()
            connection.close()

            service = MarginShieldService(db_path)

            self.assertTrue(service.dashboard["cases"])
            with sqlite3.connect(db_path) as check:
                remaining = check.execute("SELECT COUNT(*) FROM decisions").fetchone()[0]
            self.assertEqual(remaining, 0)

    def test_health_and_model_scope(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target"], "ring_label")
        model = self.client.get("/api/model").json()
        self.assertEqual(model["target"], "ring_label")
        # Derived from the dataset actually loaded, so a rebuild cannot silently
        # leave the model and the data on different generations.
        generator = str(self.service.cases["generator_version"].iloc[0])
        self.assertEqual(model["dataset_version"], generator)
        self.assertIn(generator, self.service.bundle.version)

    def test_static_routes_do_not_expose_project_artifacts(self) -> None:
        for path in ("/", "/index.html", "/rings.html", "/styles.css", "/app.js", "/rings.js", "/chat.js"):
            response = self.client.get(path)
            self.assertEqual(response.status_code, 200, path)
            self.assertIn("no-store", response.headers["cache-control"])
        rings_html = self.client.get("/rings.html").text
        for label in ("Casework", "Portfolio", "Policy lab", "Abuse rings"):
            self.assertIn(label, rings_html)
        for path in (
            "/server.py",
            "/requirements-ml.txt",
            "/PROJECT_STATUS.md",
            "/data/processed/master_refund_cases.csv.gz",
            "/data/model/live_refund_ring_model.joblib",
            "/data/reports/ring_model_report.json",
        ):
            self.assertEqual(self.client.get(path).status_code, 404, path)

    def test_score_endpoint_supports_unseen_categories(self) -> None:
        payload = self.feature_payload()
        payload["vertical"] = "new_vertical"
        payload["payment_method"] = "new_rail"
        response = self.client.post("/api/score", json={"features": payload})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertTrue(0 <= body["ring_probability"] <= 1)
        self.assertIn(body["action"], {"approve", "manual_review", "verify_evidence"})
        self.assertTrue(body["model_evidence"])
        self.assertTrue(all("contribution" in item for item in body["model_evidence"]))
        self.assertNotEqual(body["model_evidence"][0]["name"], "Model explanation unavailable")
        self.assertIn("SHAP", body["explanation_basis"])
        # Both tiers are locked by value, so both boundaries are published.
        self.assertIn("verify_evidence_threshold", body["policy"])
        self.assertEqual(
            body["policy"]["verify_evidence_threshold"],
            round(float(self.service.bundle.verify_evidence_threshold), 6),
        )

    def test_score_endpoint_rejects_missing_contract_fields(self) -> None:
        response = self.client.post("/api/score", json={"features": {}})
        self.assertEqual(response.status_code, 422)
        missing = {item["loc"][-1] for item in response.json()["detail"] if item["type"] == "missing"}
        self.assertEqual(missing, set(self.service.bundle.features))

    def test_dashboard_exposes_probability_and_percentage_separately(self) -> None:
        body = self.client.get("/api/dashboard").json()
        case = body["cases"][0]
        self.assertTrue(0 <= case["ring_probability"] <= 1)
        self.assertAlmostEqual(case["risk_percent"], round(case["ring_probability"] * 100, 3), places=3)
        self.assertNotIn("risk_score", case)
        self.assertNotIn("impact", case["evidence"][0])
        self.assertIn("contribution", case["evidence"][0])
        self.assertFalse(body["policy_metadata"]["test_labels_used"])
        # The policy is selected on economic value, not a fixed precision floor.
        self.assertFalse(body["policy_metadata"]["precision_floor_applied"])
        self.assertEqual(body["policy_metadata"]["selection_objective"], "synthetic net preventable value")
        self.assertEqual(body["policy_metadata"]["minimum_manual_reviews"], 30)
        self.assertEqual(body["policy_metadata"]["maximum_manual_reviews"], 100)
        comparison = body["policy_metadata"]["precision_floor_comparison"]
        self.assertEqual(comparison["compared_precision_floor"], 0.85)
        self.assertIn("locked_net_value_inr", comparison)
        self.assertEqual(len(body["metrics"]), len({item["threshold"] for item in body["metrics"]}))
        self.assertEqual(1, sum(item["is_locked"] for item in body["metrics"]))
        self.assertEqual([item["threshold"] for item in body["metrics"]], sorted(item["threshold"] for item in body["metrics"]))
        validation = body["evaluation"]["validation"]
        test = body["evaluation"]["test"]
        self.assertEqual(validation["true_positives"] + validation["false_negatives"], validation["positive_requests"])
        self.assertEqual(test["true_positives"] + test["false_negatives"], test["positive_requests"])
        self.assertEqual(validation["window"], "later validation policy window")
        self.assertEqual(test["window"], "final synthetic test")
        # Derived from the locked report rather than hardcoded, so a retrain
        # updates the expectation instead of breaking the suite.
        confusion = self.service.report["policy_validation"]["confusion_matrix"]
        self.assertAlmostEqual(
            validation["precision"], confusion["tp"] / (confusion["tp"] + confusion["fp"])
        )
        self.assertAlmostEqual(
            validation["recall"], confusion["tp"] / (confusion["tp"] + confusion["fn"])
        )

    def test_dashboard_returns_latest_persisted_analyst_action(self) -> None:
        case_id = self.client.get("/api/dashboard").json()["cases"][0]["case_id"]
        action = self.client.post(
            f"/api/decisions/{case_id}/action",
            json={"action": "escalated", "note": "dashboard state test"},
        )
        self.assertEqual(action.status_code, 200, action.text)
        cases = self.client.get("/api/dashboard").json()["cases"]
        persisted = next(item for item in cases if item["case_id"] == case_id)
        self.assertEqual(persisted["analyst_action"], "escalated")
        self.assertEqual(persisted["analyst_action_at"], action.json()["created_at"])

    def test_ring_endpoints_expose_only_suspected_graph_data(self) -> None:
        response = self.client.get("/api/rings")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertIn("device, address or payment-token links", body["method"])
        self.assertTrue(body["rings"])
        self.assertEqual(list(range(1, len(body["rings"]) + 1)), [ring["queue_rank"] for ring in body["rings"]])
        self.assertTrue(all("priority" not in ring for ring in body["rings"]))
        self.assertGreater(len({ring["case_count"] for ring in body["rings"]}), 1)
        exposures = [ring["model_weighted_exposure_inr"] for ring in body["rings"]]
        self.assertEqual(exposures, sorted(exposures, reverse=True))
        self.assertTrue(all(ring["entity_count"] >= 1 for ring in body["rings"]))
        self.assertIn("P1 is the highest queue rank", body["method"])
        self.assertIn("final synthetic test and replayed live or demo events", body["data_scope"])
        self.assertGreaterEqual(body["live_event_count"], 0)
        self.assertNotIn("ring_id", body["rings"][0].get("reasons", []))
        detail = self.client.get(f"/api/rings/{body['rings'][0]['ring_id']}")
        self.assertEqual(detail.status_code, 200)
        self.assertTrue(detail.json()["nodes"])

    def test_raw_event_is_featured_scored_and_audited(self) -> None:
        suffix = uuid.uuid4().hex[:10]
        latest = datetime.fromtimestamp(self.service.engine.last_seconds + 1, tz=timezone.utc)
        payload = {
            "case_id": f"LIVE-{suffix}", "event_timestamp": latest.isoformat(),
            "customer_id": f"CUS-{suffix}", "merchant_id": "MER-LIVE",
            "device_id": f"DEV-{suffix}", "address_id": f"ADR-{suffix}",
            "payment_token_id": f"PAY-{suffix}", "product_id": f"PRD-{suffix}",
            "vertical": "home", "payment_method": "upi",
            "refund_amount_inr": 1250, "refund_share": 0.8, "account_age_days": 90,
        }
        response = self.client.post("/api/events", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertTrue(body["audit_recorded"])
        self.assertEqual(body["features"]["refund_velocity_7d"], 1)
        self.assertIn(body["action"], {"approve", "manual_review", "verify_evidence"})
        self.assertEqual(self.client.post("/api/events", json=payload).status_code, 409)
        action = self.client.post(
            f"/api/decisions/{payload['case_id']}/action",
            json={"action": "accepted_recommendation", "note": "integration test"},
        )
        self.assertEqual(action.status_code, 200, action.text)
        audit = self.client.get(f"/api/audit/{payload['case_id']}")
        self.assertEqual(audit.status_code, 200)
        self.assertEqual(audit.json()["decision"]["model_version"], self.service.bundle.version)
        self.assertEqual(audit.json()["analyst_actions"][-1]["action"], "accepted_recommendation")

    def test_chat_fallback_is_grounded_in_locked_report(self) -> None:
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            response = self.client.post(
                "/api/chat",
                json={"message": "Why are precision and recall so different?", "view": "policy"},
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["provider"], "local_grounded_fallback")
        # Derived from the locked report so the assertion cannot go stale on retrain.
        validation = self.service.report["policy_validation"]
        confusion = validation["confusion_matrix"]
        self.assertIn(f"{confusion['tp']} of {confusion['tp'] + confusion['fn']}", body["answer"])
        self.assertIn(f"{validation['precision']:.1%}", body["answer"])
        self.assertIn("synthetic", body["answer"].lower())

    def test_chat_uses_selected_case_context(self) -> None:
        selected = self.client.get("/api/dashboard").json()["cases"][0]
        with patch.dict("os.environ", {"GEMINI_API_KEY": ""}):
            response = self.client.post(
                "/api/chat",
                json={"message": "Why this case?", "view": "casework", "case_id": selected["case_id"]},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertIn(selected["case_id"], response.json()["answer"])

    def test_typed_feature_payload_rejects_impossible_values(self) -> None:
        payload = self.feature_payload()
        payload["refund_share"] = 1.5
        response = self.client.post("/api/score", json={"features": payload})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
