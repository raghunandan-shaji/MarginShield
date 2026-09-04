from __future__ import annotations

import unittest
import uuid
from datetime import datetime, timezone

from fastapi.testclient import TestClient

from server import app, service


class LiveApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.client = TestClient(app)

    def feature_payload(self) -> dict:
        row = service.cases.iloc[-1]
        return {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in service.bundle.features}

    def test_health_and_model_scope(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["target"], "ring_label")
        model = self.client.get("/api/model").json()
        self.assertEqual(model["target"], "ring_label")
        self.assertEqual(model["dataset_version"], "3.0.0-locked")

    def test_static_routes_do_not_expose_project_artifacts(self) -> None:
        for path in ("/", "/index.html", "/rings.html", "/styles.css", "/app.js", "/rings.js"):
            self.assertEqual(self.client.get(path).status_code, 200, path)
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
        self.assertNotIn("verify_evidence_threshold", body["policy"])

    def test_score_endpoint_rejects_missing_contract_fields(self) -> None:
        response = self.client.post("/api/score", json={"features": {}})
        self.assertEqual(response.status_code, 422)
        missing = {item["loc"][-1] for item in response.json()["detail"] if item["type"] == "missing"}
        self.assertEqual(missing, set(service.bundle.features))

    def test_dashboard_exposes_probability_and_percentage_separately(self) -> None:
        body = self.client.get("/api/dashboard").json()
        case = body["cases"][0]
        self.assertTrue(0 <= case["ring_probability"] <= 1)
        self.assertAlmostEqual(case["risk_percent"], round(case["ring_probability"] * 100, 3), places=3)
        self.assertNotIn("risk_score", case)
        self.assertNotIn("impact", case["evidence"][0])
        self.assertIn("contribution", case["evidence"][0])
        self.assertFalse(body["policy_metadata"]["test_labels_used"])
        self.assertEqual(len(body["metrics"]), len({item["threshold"] for item in body["metrics"]}))
        self.assertEqual(1, sum(item["is_locked"] for item in body["metrics"]))

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
        self.assertNotIn("ring_id", body["rings"][0].get("reasons", []))
        detail = self.client.get(f"/api/rings/{body['rings'][0]['ring_id']}")
        self.assertEqual(detail.status_code, 200)
        self.assertTrue(detail.json()["nodes"])

    def test_raw_event_is_featured_scored_and_audited(self) -> None:
        suffix = uuid.uuid4().hex[:10]
        latest = datetime.fromtimestamp(service.engine.last_seconds + 1, tz=timezone.utc)
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
        self.assertEqual(audit.json()["decision"]["model_version"], service.bundle.version)
        self.assertEqual(audit.json()["analyst_actions"][-1]["action"], "accepted_recommendation")

    def test_typed_feature_payload_rejects_impossible_values(self) -> None:
        payload = self.feature_payload()
        payload["refund_share"] = 1.5
        response = self.client.post("/api/score", json={"features": payload})
        self.assertEqual(response.status_code, 422)


if __name__ == "__main__":
    unittest.main()
