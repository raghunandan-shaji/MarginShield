from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from marginshield.rings import build_ring_catalog
from marginshield.tournament import LiveModelBundle, classification_metrics, score_live_bundle


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


class ScoreRequest(BaseModel):
    features: dict[str, Any] = Field(description="Pre-decision refund-request features matching the ring model contract")


def score_action(probability: float, bundle: LiveModelBundle) -> tuple[str, str, str]:
    if probability >= bundle.verify_threshold:
        return "verify_evidence", "Verify evidence", "Verify threshold met"
    if probability >= bundle.threshold:
        return "manual_review", "Manual review", "Review threshold met"
    return "approve", "Approve", "Below review threshold"


def operational_signals(features: dict[str, Any]) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    pair_accounts = int(features.get("device_payment_pair_accounts_30d", 0))
    if pair_accounts >= 2:
        signals.append({"direction": "risk", "text": f"The device-payment relationship links {pair_accounts} accounts in the trailing 30 days."})
    burst = int(features.get("linked_refund_burst_72h", 0))
    if burst >= 2:
        signals.append({"direction": "risk", "text": f"{burst} linked refund requests occurred in the trailing 72 hours."})
    types = int(features.get("shared_identifier_types_30d", 0))
    if types >= 2:
        signals.append({"direction": "risk", "text": f"{types} identifier types are reused across customer accounts."})
    if float(features.get("graph_overlap_surprisal", 0.0)) >= 1.0:
        signals.append({"direction": "risk", "text": "The combined identifier overlap is unusual relative to recent request history."})
    return signals[:4] or [{"direction": "neutral", "text": "No pre-decision coordinated-ring pattern meets the intervention policy."}]


FEATURE_LABELS = {
    "vertical": "Merchant vertical",
    "payment_method": "Payment method",
    "refund_amount_inr": "Refund amount",
    "refund_share": "Refund share of order",
    "account_age_days": "Account age",
    "refund_velocity_24h": "Customer refund velocity / 24h",
    "refund_velocity_7d": "Customer refund velocity / 7d",
    "merchant_refund_volume_index_30d": "Merchant refund-volume index / 30d",
    "shared_device_accounts_30d": "Device-linked accounts / 30d",
    "shared_address_accounts_90d": "Address-linked accounts / 90d",
    "shared_payment_accounts_30d": "Payment-linked accounts / 30d",
    "shared_identifier_types_30d": "Shared identifier types / 30d",
    "device_payment_pair_accounts_30d": "Device-payment linked accounts / 30d",
    "linked_refund_burst_72h": "Linked refund requests / 72h",
    "linked_merchants_30d": "Linked merchants / 30d",
    "graph_overlap_surprisal": "Multi-identifier overlap rarity",
}


def _source_feature(encoded_name: str, features: list[str]) -> str:
    transformed = encoded_name.split("__", 1)[-1]
    for feature in sorted(features, key=len, reverse=True):
        if transformed == feature or transformed.startswith(f"{feature}_"):
            return feature
    return transformed


def model_evidence(row: pd.Series | dict[str, Any], bundle: LiveModelBundle) -> list[dict[str, Any]]:
    """Return per-case logistic log-odds contributions, not hand-authored impacts."""
    if bundle.model_name != "logistic_l2":
        return [{"feature": "model", "name": "Model explanation unavailable", "observed": None, "contribution": 0.0, "direction": "neutral"}]
    prepared = pd.DataFrame([{name: row[name] for name in bundle.features}])
    for name in bundle.categorical_features:
        prepared[name] = prepared[name].fillna("__missing__").astype(str)
    for name, median in bundle.numeric_medians.items():
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce").fillna(median)
    preprocessor = bundle.model.named_steps["preprocessor"]
    classifier = bundle.model.named_steps["classifier"]
    transformed = preprocessor.transform(prepared)
    values = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
    calibrator_scale = float(bundle.calibrator.coef_[0, 0])
    contributions: dict[str, float] = {}
    for encoded_name, value, coefficient in zip(preprocessor.get_feature_names_out(), values, classifier.coef_[0]):
        feature = _source_feature(str(encoded_name), bundle.features)
        contributions[feature] = contributions.get(feature, 0.0) + float(value * coefficient * calibrator_scale)
    ranked = sorted(contributions.items(), key=lambda item: abs(item[1]), reverse=True)[:5]
    evidence = []
    for feature, contribution in ranked:
        observed = row[feature]
        if hasattr(observed, "item"):
            observed = observed.item()
        evidence.append({
            "feature": feature,
            "name": FEATURE_LABELS.get(feature, feature.replace("_", " ").title()),
            "observed": observed,
            "contribution": round(contribution, 3),
            "direction": "raises" if contribution > 0 else "lowers" if contribution < 0 else "neutral",
        })
    return evidence


def build_policy_metrics(cases: pd.DataFrame, bundle: LiveModelBundle) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation = cases.loc[cases["split"].eq("validation")].copy().sort_values(["event_timestamp", "case_id"])
    validation["event_timestamp"] = pd.to_datetime(validation["event_timestamp"], utc=True, format="mixed")
    cutoff = validation["event_timestamp"].median()
    policy = validation.loc[validation["event_timestamp"] > cutoff].reset_index(drop=True)
    scores = score_live_bundle(bundle, policy)
    ordered = np.sort(scores)[::-1]
    review_volumes = [max(30, round(len(policy) * share)) for share in (0.01, 0.015, 0.025, 0.035, 0.05)]
    thresholds = {float(bundle.verify_threshold), float(bundle.threshold)}
    thresholds.update(float(ordered[min(volume - 1, len(ordered) - 1)]) for volume in review_volumes)
    metrics: list[dict[str, Any]] = []
    for threshold in sorted(thresholds, reverse=True):
        metric = classification_metrics(policy, scores, threshold)
        metrics.append({
            "threshold": round(threshold, 6),
            "threshold_percent": round(threshold * 100, 2),
            "is_locked": bool(abs(threshold - bundle.threshold) < 1e-9),
            "net_value": metric["costs"]["net_preventable_value_inr"],
            "precision": metric["precision"], "recall": metric["recall"],
            "review_volume": metric["review_volume"],
            "false_positive_cost": metric["costs"]["false_positive_cost_inr"],
            "review_cost": metric["costs"]["total_review_cost_inr"],
            "meets_precision_floor": metric["precision"] >= 0.85,
        })
    return metrics, {
        "source": "later half of validation split",
        "rows": len(policy),
        "cutoff": cutoff.isoformat(),
        "test_labels_used": False,
    }


def build_dashboard(cases: pd.DataFrame, bundle: LiveModelBundle) -> dict[str, Any]:
    held_out = cases.loc[cases["split"].eq("test")].copy().sort_values(["event_timestamp", "case_id"])
    held_out["ring_probability"] = score_live_bundle(bundle, held_out)
    actions = [score_action(float(score), bundle) for score in held_out["ring_probability"]]
    held_out["action_key"], held_out["action"], held_out["status"] = zip(*actions)
    queue = held_out.nlargest(520, "ring_probability").sort_values("ring_probability", ascending=False)
    case_payloads = []
    for _, row in queue.iterrows():
        evidence = model_evidence(row, bundle)
        signals = operational_signals(row.to_dict())
        case_payloads.append({
            "case_id": row["case_id"], "merchant": row["merchant_id"], "customer": row["customer_id"],
            "vertical": row["vertical"], "event_timestamp": pd.Timestamp(row["event_timestamp"]).isoformat(),
            "ring_probability": round(float(row["ring_probability"]), 6),
            "risk_percent": round(float(row["ring_probability"]) * 100, 1),
            "status": row["status"],
            "reason": signals[0]["text"], "action": row["action"],
            "notes": " ".join(signal["text"] for signal in signals),
            "refund_amount": int(row["refund_amount_inr"]),
            "conditional_loss_if_ring": int(row["expected_loss_if_ring_inr"]),
            "estimated_false_positive_cost": int(row["false_positive_cost_inr"]),
            "refund_method": row["refund_method"], "evidence": evidence,
            "explanation_basis": "Directional per-case contributions to the calibrated logistic model's log-odds.",
            "linked_entities": [
                {"label": "Device-linked accounts / 30d", "value": int(row["shared_device_accounts_30d"]), "severity": "high" if int(row["shared_device_accounts_30d"]) >= 3 else "medium"},
                {"label": "Address-linked accounts / 90d", "value": int(row["shared_address_accounts_90d"]), "severity": "high" if int(row["shared_address_accounts_90d"]) >= 3 else "medium"},
                {"label": "Payment-linked accounts / 30d", "value": int(row["shared_payment_accounts_30d"]), "severity": "high" if int(row["shared_payment_accounts_30d"]) >= 3 else "medium"},
            ],
            "timeline": [
                {"step": "Scored at", "value": pd.Timestamp(row["event_timestamp"]).isoformat(), "value_type": "timestamp"},
                {"step": "Linked activity", "value": f"{int(row['linked_refund_burst_72h'])} requests / 72h"},
                {"step": "Ring probability", "value": round(float(row["ring_probability"]), 6), "value_type": "probability"},
                {"step": "Policy action", "value": row["action"]},
            ],
        })
    policy_metrics, policy_metadata = build_policy_metrics(cases, bundle)
    by_vertical = []
    for vertical, group in held_out.groupby("vertical"):
        flagged = group["ring_probability"] >= bundle.threshold
        by_vertical.append({
            "vertical": vertical,
            "refund_exposure": int(group["refund_amount_inr"].sum()),
            "flagged_loss_exposure": int(group.loc[flagged, "expected_loss_if_ring_inr"].sum()),
        })
    return {
        "as_of": pd.to_datetime(held_out["event_timestamp"], utc=True, format="mixed").max().isoformat(),
        "cases": case_payloads,
        "summary": {
            "queue_refund_exposure": int(queue["refund_amount_inr"].sum()),
            "flagged_loss_exposure": int(queue.loc[queue["ring_probability"] >= bundle.threshold, "expected_loss_if_ring_inr"].sum()),
            "verify_evidence_cases": int((queue["ring_probability"] >= bundle.verify_threshold).sum()),
            "by_vertical": sorted(by_vertical, key=lambda item: item["refund_exposure"], reverse=True),
            "risk_bands": {
                "approve": int((held_out["ring_probability"] < bundle.threshold).sum()),
                "manual_review": int(((held_out["ring_probability"] >= bundle.threshold) & (held_out["ring_probability"] < bundle.verify_threshold)).sum()),
                "verify_evidence": int((held_out["ring_probability"] >= bundle.verify_threshold).sum()),
            },
            "by_action": held_out["action"].value_counts().to_dict(),
        },
        "metrics": policy_metrics,
        "policy_metadata": policy_metadata,
        "active_threshold": round(float(bundle.threshold), 6),
        "active_threshold_percent": round(float(bundle.threshold) * 100, 1),
        "model_scope": "Coordinated refund-abuse rings only. Synthetic benchmark performance is not production performance.",
    }


class MarginShieldService:
    def __init__(self) -> None:
        self.bundle: LiveModelBundle = joblib.load(DATA / "model" / "live_refund_ring_model.joblib")
        self.cases = pd.read_csv(DATA / "processed" / "master_refund_cases.csv.gz")
        self.report = json.loads((DATA / "reports" / "ring_model_report.json").read_text(encoding="utf-8"))
        self.rings = build_ring_catalog(self.cases, self.bundle)
        self.dashboard = build_dashboard(self.cases, self.bundle)

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        missing = sorted(set(self.bundle.features).difference(features))
        if missing:
            raise HTTPException(status_code=422, detail={"message": "Missing required model features", "fields": missing})
        probability = float(score_live_bundle(self.bundle, pd.DataFrame([features]))[0])
        action_key, action, _ = score_action(probability, self.bundle)
        return {
            "ring_probability": round(probability, 4), "action": action_key, "action_label": action,
            "policy": {"manual_review_threshold": round(self.bundle.threshold, 4), "verify_evidence_threshold": round(self.bundle.verify_threshold, 4)},
            "model": {"name": self.bundle.model_name, "version": self.bundle.version, "target": self.bundle.target},
            "signals": operational_signals(features),
            "model_evidence": model_evidence(features, self.bundle),
            "explanation_basis": "Directional per-case contributions to the calibrated logistic model's log-odds.",
        }


service = MarginShieldService()
app = FastAPI(title="MarginShield Coordinated Ring Risk API", version=service.bundle.version)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {"status": "online", "model": service.bundle.model_name, "version": service.bundle.version, "target": service.bundle.target, "candidate_rings": len(service.rings["candidate_rings"])}


@app.get("/api/model")
def model_card() -> dict[str, Any]:
    return service.report


@app.get("/api/dashboard")
def dashboard() -> dict[str, Any]:
    return service.dashboard


@app.post("/api/score")
def score_refund(request: ScoreRequest) -> dict[str, Any]:
    return service.score(request.features)


@app.get("/api/rings")
def list_rings() -> dict[str, Any]:
    return {
        "as_of": service.rings["as_of"], "window_days": service.rings["window_days"],
        "ranking_note": service.rings["ranking_note"], "method": service.rings["method"],
        "manual_review_threshold": service.bundle.threshold, "verify_evidence_threshold": service.bundle.verify_threshold,
        "rings": [{key: value for key, value in ring.items() if key not in {"nodes", "edges"}} for ring in service.rings["candidate_rings"]],
    }


@app.get("/api/rings/{ring_id}")
def ring_detail(ring_id: str) -> dict[str, Any]:
    for ring in service.rings["candidate_rings"]:
        if ring["ring_id"] == ring_id:
            return ring
    raise HTTPException(status_code=404, detail="Ring not found")


@app.get("/", include_in_schema=False)
@app.get("/index.html", include_in_schema=False)
def index_page() -> FileResponse:
    return FileResponse(ROOT / "index.html")


@app.get("/rings.html", include_in_schema=False)
def rings_page() -> FileResponse:
    return FileResponse(ROOT / "rings.html")


@app.get("/styles.css", include_in_schema=False)
def stylesheet() -> FileResponse:
    return FileResponse(ROOT / "styles.css", media_type="text/css")


@app.get("/app.js", include_in_schema=False)
def casework_script() -> FileResponse:
    return FileResponse(ROOT / "app.js", media_type="text/javascript")


@app.get("/rings.js", include_in_schema=False)
def rings_script() -> FileResponse:
    return FileResponse(ROOT / "rings.js", media_type="text/javascript")
