from __future__ import annotations

import unittest

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

    def test_score_endpoint_rejects_missing_contract_fields(self) -> None:
        response = self.client.post("/api/score", json={"features": {}})
        self.assertEqual(response.status_code, 422)
        self.assertIn("fields", response.json()["detail"])

    def test_dashboard_exposes_probability_and_percentage_separately(self) -> None:
        body = self.client.get("/api/dashboard").json()
        case = body["cases"][0]
        self.assertTrue(0 <= case["ring_probability"] <= 1)
        self.assertAlmostEqual(case["risk_percent"], round(case["ring_probability"] * 100, 1), places=1)
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
        self.assertIn("shared device-payment relationship", body["method"])
        self.assertTrue(body["rings"])
        self.assertEqual(list(range(1, len(body["rings"]) + 1)), [ring["queue_rank"] for ring in body["rings"]])
        self.assertTrue(all("priority" not in ring for ring in body["rings"]))
        self.assertGreater(len({ring["case_count"] for ring in body["rings"]}), 1)
        exposures = [ring["model_weighted_exposure_inr"] for ring in body["rings"]]
        self.assertEqual(exposures, sorted(exposures, reverse=True))
        self.assertTrue(all(ring["entity_count"] >= 2 for ring in body["rings"]))
        self.assertIn("P1 is the highest queue rank", body["method"])
        self.assertNotIn("ring_id", body["rings"][0].get("reasons", []))
        detail = self.client.get(f"/api/rings/{body['rings'][0]['ring_id']}")
        self.assertEqual(detail.status_code, 200)
        self.assertTrue(detail.json()["nodes"])


if __name__ == "__main__":
    unittest.main()
