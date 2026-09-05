from __future__ import annotations

import copy
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import joblib
import httpx
import numpy as np
import pandas as pd
from catboost import Pool
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from marginshield.feature_engine import PointInTimeFeatureEngine, RawRefundEvent, apply_features_via_engine
from marginshield.rings import build_ring_catalog
from marginshield.tournament import LiveModelBundle, classification_metrics, score_live_bundle


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


class FeaturePayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    vertical: str = Field(min_length=1)
    payment_method: str = Field(min_length=1)
    refund_amount_inr: float = Field(ge=0)
    refund_share: float = Field(ge=0, le=1)
    account_age_days: float = Field(ge=0)
    refund_velocity_24h: int = Field(ge=1)
    refund_velocity_7d: int = Field(ge=1)
    merchant_refund_volume_index_30d: float = Field(ge=0, le=1)
    shared_device_accounts_30d: int = Field(ge=1)
    shared_address_accounts_90d: int = Field(ge=1)
    shared_payment_accounts_30d: int = Field(ge=1)
    shared_identifier_types_30d: int = Field(ge=0, le=3)
    device_payment_pair_accounts_30d: int = Field(ge=1)
    connected_accounts_max_window: int = Field(ge=1)
    multi_identifier_neighbor_accounts: int = Field(ge=0)
    identifier_reuse_balance: float = Field(ge=0, le=1)
    linked_refund_burst_72h: int = Field(ge=1)
    linked_merchants_30d: int = Field(ge=1)
    linked_same_product_accounts_30d: int = Field(ge=1)
    graph_overlap_surprisal: float = Field(ge=0)


class ScoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: FeaturePayload = Field(description="Pre-decision features matching the exact ring-model contract")


class RawRefundEventRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case_id: str = Field(min_length=1, max_length=120)
    event_timestamp: datetime
    customer_id: str = Field(min_length=1, max_length=120)
    merchant_id: str = Field(min_length=1, max_length=120)
    device_id: str = Field(min_length=1, max_length=200)
    address_id: str = Field(min_length=1, max_length=200)
    payment_token_id: str = Field(min_length=1, max_length=200)
    product_id: str = Field(min_length=1, max_length=200)
    vertical: str = Field(min_length=1, max_length=80)
    payment_method: str = Field(min_length=1, max_length=80)
    refund_amount_inr: float = Field(ge=0, le=10_000_000)
    refund_share: float = Field(ge=0, le=1)
    account_age_days: float = Field(ge=0, le=20_000)
    refund_method: str = Field(default="original_rail", min_length=1, max_length=80)

    @field_validator("event_timestamp")
    @classmethod
    def timezone_required(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("event_timestamp must include a timezone")
        return value.astimezone(timezone.utc)


class AnalystActionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["accepted_recommendation", "verified_legitimate", "confirmed_abuse", "escalated"]
    note: str = Field(default="", max_length=2_000)


class ChatTurn(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2_000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    message: str = Field(min_length=1, max_length=2_000)
    history: list[ChatTurn] = Field(default_factory=list, max_length=8)
    view: Literal["casework", "portfolio", "policy", "abuse_rings"] = "casework"
    case_id: str | None = Field(default=None, max_length=120)
    ring_id: str | None = Field(default=None, max_length=120)


def score_action(probability: float, features: dict[str, Any], bundle: LiveModelBundle) -> tuple[str, str, str]:
    if probability >= bundle.threshold:
        structural_evidence = (
            int(features.get("shared_identifier_types_30d", 0)) >= 2
            or int(features.get("multi_identifier_neighbor_accounts", 0)) >= 1
        )
        if structural_evidence:
            return "verify_evidence", "Verify evidence", "Review threshold and multi-identifier evidence met"
        return "manual_review", "Manual review", "Review threshold met without multi-identifier overlap"
    return "approve", "Approve", "Below review threshold"


def operational_signals(features: dict[str, Any]) -> list[dict[str, str]]:
    signals: list[dict[str, str]] = []
    entity_counts = {
        "device": int(features.get("shared_device_accounts_30d", 0)),
        "address": int(features.get("shared_address_accounts_90d", 0)),
        "payment token": int(features.get("shared_payment_accounts_30d", 0)),
    }
    strongest_entity, strongest_count = max(entity_counts.items(), key=lambda item: item[1])
    if strongest_count >= 2:
        signals.append({"direction": "risk", "text": f"The submitted {strongest_entity} links {strongest_count} customer accounts in its trailing observation window."})
    connected = int(features.get("connected_accounts_max_window", 0))
    if connected >= 3:
        signals.append({"direction": "risk", "text": f"Submitted identifiers connect this request to {connected} customer accounts in total."})
    burst = int(features.get("linked_refund_burst_72h", 0))
    if burst >= 2:
        signals.append({"direction": "risk", "text": f"{burst} linked refund requests occurred in the trailing 72 hours."})
    types = int(features.get("shared_identifier_types_30d", 0))
    if types >= 2:
        signals.append({"direction": "risk", "text": f"{types} identifier types are reused across customer accounts."})
    multi_neighbors = int(features.get("multi_identifier_neighbor_accounts", 0))
    if multi_neighbors >= 1:
        signals.append({"direction": "risk", "text": f"{multi_neighbors} linked account{'s' if multi_neighbors != 1 else ''} overlap on more than one identifier type."})
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
    "connected_accounts_max_window": "Connected accounts across identifiers",
    "multi_identifier_neighbor_accounts": "Multi-identifier neighbour accounts",
    "identifier_reuse_balance": "Identifier reuse balance",
    "linked_refund_burst_72h": "Linked refund requests / 72h",
    "linked_merchants_30d": "Linked merchants / 30d",
    "linked_same_product_accounts_30d": "Linked accounts refunding this product / 30d",
    "graph_overlap_surprisal": "Multi-identifier overlap rarity",
}


def _source_feature(encoded_name: str, features: list[str]) -> str:
    transformed = encoded_name.split("__", 1)[-1]
    for feature in sorted(features, key=len, reverse=True):
        if transformed == feature or transformed.startswith(f"{feature}_"):
            return feature
    return transformed


def _format_evidence(row: pd.Series | dict[str, Any], contributions: dict[str, float]) -> list[dict[str, Any]]:
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


def model_evidence(row: pd.Series | dict[str, Any], bundle: LiveModelBundle) -> list[dict[str, Any]]:
    """Return per-case calibrated log-odds contributions, not hand-authored impacts."""
    prepared = pd.DataFrame([{name: row[name] for name in bundle.features}])
    for name in bundle.categorical_features:
        prepared[name] = prepared[name].fillna("__missing__").astype(str)
    for name, median in bundle.numeric_medians.items():
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce").fillna(median)
    calibrator_scale = float(bundle.calibrator.coef_[0, 0])
    contributions: dict[str, float]
    if bundle.model_name == "logistic_l2":
        preprocessor = bundle.model.named_steps["preprocessor"]
        classifier = bundle.model.named_steps["classifier"]
        transformed = preprocessor.transform(prepared)
        values = transformed.toarray()[0] if hasattr(transformed, "toarray") else np.asarray(transformed)[0]
        contributions = {}
        for encoded_name, value, coefficient in zip(preprocessor.get_feature_names_out(), values, classifier.coef_[0]):
            feature = _source_feature(str(encoded_name), bundle.features)
            contributions[feature] = contributions.get(feature, 0.0) + float(value * coefficient * calibrator_scale)
    elif bundle.model_name == "catboost_calibrated":
        pool = Pool(prepared, cat_features=bundle.categorical_features)
        shap_values = bundle.model.get_feature_importance(pool, type="ShapValues")[0][:-1]
        contributions = {
            feature: float(value * calibrator_scale)
            for feature, value in zip(bundle.features, shap_values)
        }
    else:
        return [{"feature": "model", "name": "Model explanation unavailable", "observed": None, "contribution": 0.0, "direction": "neutral"}]
    return _format_evidence(row, contributions)


def model_evidence_batch(rows: pd.DataFrame, bundle: LiveModelBundle) -> list[list[dict[str, Any]]]:
    if bundle.model_name != "catboost_calibrated":
        return [model_evidence(row, bundle) for _, row in rows.iterrows()]
    prepared = rows[bundle.features].copy()
    for name in bundle.categorical_features:
        prepared[name] = prepared[name].fillna("__missing__").astype(str)
    for name, median in bundle.numeric_medians.items():
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce").fillna(median)
    pool = Pool(prepared, cat_features=bundle.categorical_features)
    values = bundle.model.get_feature_importance(pool, type="ShapValues")[:, :-1]
    scale = float(bundle.calibrator.coef_[0, 0])
    return [
        _format_evidence(row, {feature: float(value * scale) for feature, value in zip(bundle.features, vector)})
        for (_, row), vector in zip(rows.iterrows(), values)
    ]


def explanation_basis(bundle: LiveModelBundle) -> str:
    if bundle.model_name == "catboost_calibrated":
        return "Per-case CatBoost SHAP contributions scaled into the calibrated model's log-odds."
    if bundle.model_name == "logistic_l2":
        return "Directional per-case contributions to the calibrated logistic model's log-odds."
    return "Signals used by the transparent graph-rule baseline."


def build_policy_metrics(cases: pd.DataFrame, bundle: LiveModelBundle) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    validation = cases.loc[cases["split"].eq("validation")].copy().sort_values(["event_timestamp", "case_id"])
    validation["event_timestamp"] = pd.to_datetime(validation["event_timestamp"], utc=True, format="mixed")
    cutoff = validation["event_timestamp"].median()
    policy = validation.loc[validation["event_timestamp"] > cutoff].reset_index(drop=True)
    scores = score_live_bundle(bundle, policy)
    ordered = np.sort(scores)[::-1]
    review_volumes = [max(30, round(len(policy) * share)) for share in (0.01, 0.015, 0.025, 0.035, 0.05)]
    thresholds = {float(bundle.threshold)}
    thresholds.update(float(ordered[min(volume - 1, len(ordered) - 1)]) for volume in review_volumes)
    metrics: list[dict[str, Any]] = []
    for threshold in sorted(thresholds):
        metric = classification_metrics(policy, scores, threshold, include_ring_metrics=False)
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
            "true_positives": metric["confusion_matrix"]["tp"],
            "false_positives": metric["confusion_matrix"]["fp"],
            "false_negatives": metric["confusion_matrix"]["fn"],
            "positive_requests": metric["confusion_matrix"]["tp"] + metric["confusion_matrix"]["fn"],
        })
    return metrics, {
        "source": "later half of validation split",
        "rows": len(policy),
        "cutoff": cutoff.isoformat(),
        "test_labels_used": False,
        "selection_rule": "maximize recall subject to the validation precision floor and minimum flag count",
        "precision_floor": 0.85,
        "minimum_flags": 30,
    }


def benchmark_payload(report: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for source, name in (("policy_validation", "validation"), ("test", "test")):
        metric = report[source]
        confusion = metric["confusion_matrix"]
        ring_level = metric["ring_level"]
        payload[name] = {
            "window": "later validation policy window" if name == "validation" else "final synthetic test",
            "rows": metric["rows"],
            "positive_requests": confusion["tp"] + confusion["fn"],
            "true_positives": confusion["tp"],
            "false_positives": confusion["fp"],
            "false_negatives": confusion["fn"],
            "precision": metric["precision"],
            "recall": metric["recall"],
            "pr_auc": metric["pr_auc"],
            "roc_auc": metric["roc_auc"],
            "brier_score": metric["brier_score"],
            "flag_rate": metric["flag_rate"],
            "review_volume": metric["review_volume"],
            "net_preventable_value_inr": metric["costs"]["net_preventable_value_inr"],
            "actual_rings": ring_level["actual_rings"],
            "early_detected_rings": ring_level["detected_before_half_loss"],
            "early_ring_recall": ring_level["ring_recall"],
            "ring_candidate_precision": ring_level["ring_precision"],
        }
    return payload


def build_dashboard(cases: pd.DataFrame, bundle: LiveModelBundle, report: dict[str, Any]) -> dict[str, Any]:
    held_out = cases.loc[cases["split"].eq("test")].copy().sort_values(["event_timestamp", "case_id"])
    held_out["ring_probability"] = score_live_bundle(bundle, held_out)
    actions = [
        score_action(float(row.ring_probability), row._asdict(), bundle)
        for row in held_out.itertuples()
    ]
    held_out["action_key"], held_out["action"], held_out["status"] = zip(*actions)
    queue = held_out.nlargest(520, "ring_probability").sort_values("ring_probability", ascending=False)
    queue_evidence = model_evidence_batch(queue, bundle)
    case_payloads = []
    for (_, row), evidence in zip(queue.iterrows(), queue_evidence):
        signals = operational_signals(row.to_dict())
        case_payloads.append({
            "case_id": row["case_id"], "merchant": row["merchant_id"], "customer": row["customer_id"],
            "vertical": row["vertical"], "event_timestamp": pd.Timestamp(row["event_timestamp"]).isoformat(),
            "ring_probability": round(float(row["ring_probability"]), 6),
            "risk_percent": round(float(row["ring_probability"]) * 100, 3),
            "status": row["status"],
            "reason": signals[0]["text"], "action": row["action"],
            "notes": " ".join(signal["text"] for signal in signals),
            "refund_amount": int(row["refund_amount_inr"]),
            "conditional_loss_if_ring": int(row["expected_loss_if_ring_inr"]),
            "estimated_false_positive_cost": int(row["false_positive_cost_inr"]),
            "refund_method": row["refund_method"], "evidence": evidence,
            "explanation_basis": explanation_basis(bundle),
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
            "verify_evidence_cases": int(queue["action_key"].eq("verify_evidence").sum()),
            "held_out_requests": len(held_out),
            "held_out_refund_exposure": int(held_out["refund_amount_inr"].sum()),
            "flagged_requests": int((held_out["ring_probability"] >= bundle.threshold).sum()),
            "by_vertical": sorted(by_vertical, key=lambda item: item["refund_exposure"], reverse=True),
            "risk_bands": {
                "approve": int(held_out["action_key"].eq("approve").sum()),
                "manual_review": int(held_out["action_key"].eq("manual_review").sum()),
                "verify_evidence": int(held_out["action_key"].eq("verify_evidence").sum()),
            },
            "by_action": held_out["action"].value_counts().to_dict(),
        },
        "metrics": policy_metrics,
        "evaluation": benchmark_payload(report),
        "policy_metadata": policy_metadata,
        "active_threshold": round(float(bundle.threshold), 6),
        "active_threshold_percent": round(float(bundle.threshold) * 100, 1),
        "model_scope": "Coordinated refund-abuse rings only. Synthetic benchmark performance is not production performance.",
    }


class MarginShieldService:
    def __init__(self, db_path: Path | str = DATA / "decisions.sqlite") -> None:
        self.bundle: LiveModelBundle = joblib.load(DATA / "model" / "live_refund_ring_model.joblib")
        self.cases = pd.read_csv(DATA / "processed" / "master_refund_cases.csv.gz")
        self.report = json.loads((DATA / "reports" / "ring_model_report.json").read_text(encoding="utf-8"))
        self.db_path = str(db_path)
        self.lock = threading.RLock()
        self.engine = PointInTimeFeatureEngine()
        self.cases, self.engine = apply_features_via_engine(self.cases, self.engine)
        self._init_database()
        self._replay_live_events()
        self.rings = build_ring_catalog(self.cases, self.bundle)
        self.dashboard = build_dashboard(self.cases, self.bundle, self.report)

    def score(self, features: dict[str, Any]) -> dict[str, Any]:
        missing = sorted(set(self.bundle.features).difference(features))
        if missing:
            raise HTTPException(status_code=422, detail={"message": "Missing required model features", "fields": missing})
        probability = float(score_live_bundle(self.bundle, pd.DataFrame([features]))[0])
        action_key, action, rationale = score_action(probability, features, self.bundle)
        clipped = float(np.clip(probability, 1e-9, 1 - 1e-9))
        return {
            "ring_probability": round(probability, 6),
            "log_odds": round(float(np.log(clipped / (1 - clipped))), 6),
            "action": action_key, "action_label": action,
            "policy": {
                "manual_review_threshold": round(self.bundle.threshold, 6),
                "action_rule": "Above threshold: verify evidence when at least two identifier types or one neighbour overlap; otherwise manual review.",
            },
            "model": {"name": self.bundle.model_name, "version": self.bundle.version, "target": self.bundle.target},
            "decision_rationale": rationale,
            "signals": operational_signals(features),
            "model_evidence": model_evidence(features, self.bundle),
            "explanation_basis": explanation_basis(self.bundle),
        }

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_database(self) -> None:
        with self._connect() as connection:
            connection.executescript("""
                CREATE TABLE IF NOT EXISTS decisions (
                    case_id TEXT PRIMARY KEY,
                    event_timestamp TEXT NOT NULL,
                    model_version TEXT NOT NULL,
                    threshold REAL NOT NULL,
                    raw_event_json TEXT NOT NULL,
                    features_json TEXT NOT NULL,
                    probability REAL NOT NULL,
                    recommendation TEXT NOT NULL,
                    evidence_json TEXT NOT NULL,
                    linked_state_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    replayable INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS analyst_actions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    case_id TEXT NOT NULL,
                    action TEXT NOT NULL,
                    note TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (case_id) REFERENCES decisions(case_id)
                );
            """)

    @staticmethod
    def _raw_event(payload: dict[str, Any]) -> RawRefundEvent:
        timestamp = pd.Timestamp(payload["event_timestamp"])
        return RawRefundEvent(
            case_id=str(payload["case_id"]),
            event_seconds=int(timestamp.timestamp()),
            customer_id=str(payload["customer_id"]), merchant_id=str(payload["merchant_id"]),
            device_id=str(payload["device_id"]), address_id=str(payload["address_id"]),
            payment_token_id=str(payload["payment_token_id"]), vertical=str(payload["vertical"]),
            product_id=str(payload["product_id"]),
            payment_method=str(payload["payment_method"]), refund_amount_inr=float(payload["refund_amount_inr"]),
            refund_share=float(payload["refund_share"]), account_age_days=float(payload["account_age_days"]),
        )

    def _live_row(self, payload: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
        margin_rates = {"fashion": 0.24, "beauty": 0.31, "electronics": 0.12, "home": 0.19, "sports": 0.21, "general": 0.18}
        refund = float(payload["refund_amount_inr"])
        review_cost = 140 if refund > 5_000 else 65
        return {
            **payload, **features, "split": "live", "ring_label": 0, "ring_id": "",
            "scenario_type": "live_unlabelled", "ring_topology": "unknown", "is_synthetic": 0,
            "expected_loss_if_ring_inr": round(refund + refund * margin_rates.get(str(payload["vertical"]), 0.18) * 0.55),
            "review_cost_inr": review_cost,
            "false_positive_cost_inr": round(refund * 0.075 + review_cost),
        }

    def _replay_live_events(self) -> None:
        with self._connect() as connection:
            rows = connection.execute("SELECT raw_event_json, features_json FROM decisions WHERE replayable = 1 ORDER BY event_timestamp, case_id").fetchall()
        live_rows = []
        for row in rows:
            payload = json.loads(row["raw_event_json"])
            event = self._raw_event(payload)
            if self.engine.last_seconds is not None and event.event_seconds < self.engine.last_seconds:
                raise RuntimeError("Stored live event predates current feature-engine state")
            features = self.engine.observe(event)
            stored_features = json.loads(row["features_json"])
            if any(abs(float(features[name]) - float(stored_features[name])) > 1e-9 for name in self.bundle.features if name not in self.bundle.categorical_features):
                raise RuntimeError(f"Feature replay mismatch for {event.case_id}")
            live_rows.append(self._live_row(payload, features))
        if live_rows:
            self.cases = pd.concat([self.cases, pd.DataFrame(live_rows)], ignore_index=True)

    def score_event(self, request: RawRefundEventRequest) -> dict[str, Any]:
        payload = request.model_dump(mode="json")
        event = self._raw_event(payload)
        with self.lock:
            with self._connect() as connection:
                if connection.execute("SELECT 1 FROM decisions WHERE case_id = ?", (event.case_id,)).fetchone():
                    raise HTTPException(status_code=409, detail="case_id already scored")
            if self.engine.last_seconds is not None and event.event_seconds < self.engine.last_seconds:
                raise HTTPException(status_code=409, detail="event_timestamp is older than the latest committed event")
            features = self.engine.features(event)
            linked = self.engine.linked_state(event)
            result = self.score(features)
            now = datetime.now(timezone.utc).isoformat()
            evidence = {"signals": result["signals"], "model_evidence": result["model_evidence"], "rationale": result["decision_rationale"]}
            with self._connect() as connection:
                connection.execute(
                    "INSERT INTO decisions (case_id, event_timestamp, model_version, threshold, raw_event_json, features_json, probability, recommendation, evidence_json, linked_state_json, created_at, replayable) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        event.case_id, payload["event_timestamp"], self.bundle.version, self.bundle.threshold,
                        json.dumps(payload, sort_keys=True), json.dumps(features, sort_keys=True), result["ring_probability"],
                        result["action"], json.dumps(evidence, sort_keys=True), json.dumps(linked, sort_keys=True), now, 1,
                    ),
                )
            self.engine.commit(event)
            self.cases = pd.concat([self.cases, pd.DataFrame([self._live_row(payload, features)])], ignore_index=True)
            self.rings = build_ring_catalog(self.cases, self.bundle)
            return {**result, "case_id": event.case_id, "event_timestamp": payload["event_timestamp"], "features": features, "linked_prior_state": linked, "audit_recorded": True}

    def record_action(self, case_id: str, request: AnalystActionRequest) -> dict[str, Any]:
        with self._connect() as connection:
            if not connection.execute("SELECT 1 FROM decisions WHERE case_id = ?", (case_id,)).fetchone():
                historical = self.cases.loc[self.cases["case_id"].eq(case_id)]
                if historical.empty:
                    raise HTTPException(status_code=404, detail="Scored case not found")
                row = historical.iloc[0].to_dict()
                features = {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in self.bundle.features}
                result = self.score(features)
                raw = {
                    name: row[name].item() if hasattr(row[name], "item") else row[name]
                    for name in (
                        "case_id", "event_timestamp", "customer_id", "merchant_id", "device_id", "address_id",
                        "payment_token_id", "product_id", "vertical", "payment_method", "refund_amount_inr",
                        "refund_share", "account_age_days", "refund_method",
                    )
                }
                raw["event_timestamp"] = pd.Timestamp(raw["event_timestamp"]).isoformat()
                evidence = {"signals": result["signals"], "model_evidence": result["model_evidence"], "rationale": result["decision_rationale"]}
                connection.execute(
                    "INSERT INTO decisions (case_id, event_timestamp, model_version, threshold, raw_event_json, features_json, probability, recommendation, evidence_json, linked_state_json, created_at, replayable) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        case_id, raw["event_timestamp"], self.bundle.version, self.bundle.threshold,
                        json.dumps(raw, sort_keys=True), json.dumps(features, sort_keys=True), result["ring_probability"],
                        result["action"], json.dumps(evidence, sort_keys=True), json.dumps({}, sort_keys=True),
                        datetime.now(timezone.utc).isoformat(), 0,
                    ),
                )
            created_at = datetime.now(timezone.utc).isoformat()
            cursor = connection.execute(
                "INSERT INTO analyst_actions (case_id, action, note, created_at) VALUES (?, ?, ?, ?)",
                (case_id, request.action, request.note, created_at),
            )
        return {"id": cursor.lastrowid, "case_id": case_id, "action": request.action, "note": request.note, "created_at": created_at}

    def dashboard_payload(self) -> dict[str, Any]:
        payload = copy.deepcopy(self.dashboard)
        with self._connect() as connection:
            rows = connection.execute("""
                SELECT action.case_id, action.action, action.created_at
                FROM analyst_actions AS action
                INNER JOIN (
                    SELECT case_id, MAX(id) AS latest_id
                    FROM analyst_actions
                    GROUP BY case_id
                ) AS latest ON latest.latest_id = action.id
            """).fetchall()
        latest_actions = {row["case_id"]: dict(row) for row in rows}
        for case in payload["cases"]:
            latest = latest_actions.get(case["case_id"])
            if latest:
                case["analyst_action"] = latest["action"]
                case["analyst_action_at"] = latest["created_at"]
        return payload

    def chat_context(self, request: ChatRequest) -> dict[str, Any]:
        dashboard = self.dashboard_payload()
        context: dict[str, Any] = {
            "product": {
                "name": "MarginShield",
                "scope": dashboard["model_scope"],
                "target": self.report["target_definition"],
                "allowed_actions": ["approve", "manual review", "verify evidence"],
                "auto_reject": False,
            },
            "current_view": request.view,
            "policy": {
                "threshold": dashboard["active_threshold"],
                "selection": self.report["calibration"]["policy_selection"],
                "source": dashboard["policy_metadata"]["source"],
                "test_labels_used_for_selection": False,
            },
            "benchmark": dashboard["evaluation"],
            "portfolio": dashboard["summary"],
            "limitations": self.report["limitations"],
        }
        if request.case_id:
            selected_case = next((item for item in dashboard["cases"] if item["case_id"] == request.case_id), None)
            if selected_case:
                context["selected_case"] = selected_case
        if request.ring_id:
            selected_ring = next((item for item in self.rings["candidate_rings"] if item["ring_id"] == request.ring_id), None)
            if selected_ring:
                context["selected_ring"] = selected_ring
        return context

    @staticmethod
    def local_chat_answer(question: str, context: dict[str, Any]) -> str:
        query = question.lower()
        validation = context["benchmark"]["validation"]
        test = context["benchmark"]["test"]
        if "precision" in query or "recall" in query or "miss" in query or "performance" in query:
            return (
                f"On the validation policy window, precision is {validation['precision']:.1%} and request recall is "
                f"{validation['recall']:.1%}: {validation['true_positives']} of {validation['positive_requests']} abuse requests "
                f"were flagged and {validation['false_negatives']} were missed. On the final synthetic test, precision is "
                f"{test['precision']:.1%} and recall is {test['recall']:.1%}. This is a high-confidence triage policy with low "
                "coverage, not a comprehensive detector. All figures are synthetic benchmark results."
            )
        if "threshold" in query or "policy" in query:
            return (
                f"The manual-review threshold is {context['policy']['threshold']:.4f}. It was locked on the later validation "
                f"window by this rule: {context['policy']['selection']} Final-test labels were not used to choose it. Lower "
                "thresholds improve recall but fail the declared 85% validation precision constraint."
            )
        if "case" in query or "decision" in query or "why" in query:
            case = context.get("selected_case")
            if case:
                evidence = "; ".join(
                    f"{item['name']} ({item['contribution']:+.2f} log-odds)" for item in case["evidence"][:3]
                )
                return (
                    f"{case['case_id']} has a calibrated coordinated-ring probability of {case['risk_percent']:.1f}% and the "
                    f"recommended action is {case['action'].lower()}. The leading model contributions are {evidence}. The "
                    f"operational rationale is: {case['notes']} This is evidence for review, not proof of fraud."
                )
        if "ring" in query or "cluster" in query or "graph" in query:
            ring = context.get("selected_ring")
            if ring:
                return (
                    f"{ring['ring_id']} contains {ring['case_count']} requests linked through {ring['entity_count']} shared "
                    f"identifiers. {ring['high_risk_case_count']} requests meet the review policy. Its model-weighted conditional "
                    f"exposure is INR {ring['model_weighted_exposure_inr']:,}. It is a candidate connected component, not a "
                    "confirmed abuse ring."
                )
        if "data" in query or "synthetic" in query or "olist" in query or "real" in query:
            return (
                "The active 75,000-row benchmark and all fraud labels are synthetic. Olist has no refund-fraud labels and is not "
                "used as fraud truth. The simulator contains coordinated rings and difficult legitimate sharing, but benchmark "
                "performance is not real-world or production performance."
            )
        return (
            "MarginShield estimates whether a refund request belongs to a coordinated multi-account abuse ring using only "
            "pre-decision graph, velocity, account-history, merchant, product, and refund signals. Ask about the selected case or "
            "ring, the threshold, precision and recall, costs, or synthetic-data limitations."
        )

    def chat(self, request: ChatRequest) -> dict[str, Any]:
        context = self.chat_context(request)
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            return {
                "answer": self.local_chat_answer(request.message, context),
                "provider": "local_grounded_fallback",
                "grounded_in": sorted(context.keys()),
                "note": "Set GEMINI_API_KEY to enable the optional free-tier Gemini conversation layer.",
            }

        system_instruction = (
            "You are MarginShield's risk-analysis assistant. Answer only from the supplied JSON facts. Be concise, direct, and "
            "plain text. Never invent a metric, event, label, cause, or Razorpay capability. Distinguish validation from final "
            "synthetic test and from current operational data. Always disclose that benchmark labels are synthetic when discussing "
            "performance. Treat candidate rings as suspicions, not confirmed fraud. Never recommend auto-rejection. If the facts do "
            "not support an answer, say that the information is unavailable. Do not use markdown tables."
        )
        contents = [
            {"role": "user" if turn.role == "user" else "model", "parts": [{"text": turn.content}]}
            for turn in request.history
        ]
        contents.append({
            "role": "user",
            "parts": [{"text": f"Grounding facts:\n{json.dumps(context, sort_keys=True, default=str)}\n\nQuestion:\n{request.message}"}],
        })
        model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip() or "gemini-2.5-flash"
        try:
            response = httpx.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
                json={
                    "systemInstruction": {"parts": [{"text": system_instruction}]},
                    "contents": contents,
                    "generationConfig": {"temperature": 0.1, "maxOutputTokens": 500},
                },
                timeout=25.0,
            )
            response.raise_for_status()
            parts = response.json()["candidates"][0]["content"]["parts"]
            answer = "".join(part.get("text", "") for part in parts).strip()
            if not answer:
                raise ValueError("Gemini returned no text")
            return {"answer": answer, "provider": model, "grounded_in": sorted(context.keys())}
        except (httpx.HTTPError, KeyError, IndexError, ValueError):
            return {
                "answer": self.local_chat_answer(request.message, context),
                "provider": "local_grounded_fallback",
                "grounded_in": sorted(context.keys()),
                "note": "The configured Gemini request was unavailable; no ungrounded response was shown.",
            }

    def audit(self, case_id: str) -> dict[str, Any]:
        with self._connect() as connection:
            decision = connection.execute("SELECT * FROM decisions WHERE case_id = ?", (case_id,)).fetchone()
            if not decision:
                raise HTTPException(status_code=404, detail="Scored case not found")
            actions = connection.execute("SELECT id, action, note, created_at FROM analyst_actions WHERE case_id = ? ORDER BY id", (case_id,)).fetchall()
        item = dict(decision)
        for key in ("raw_event_json", "features_json", "evidence_json", "linked_state_json"):
            item[key.removesuffix("_json")] = json.loads(item.pop(key))
        return {"decision": item, "analyst_actions": [dict(action) for action in actions]}


service = MarginShieldService()
app = FastAPI(title="MarginShield Coordinated Ring Risk API", version=service.bundle.version)


@app.get("/api/health")
def health() -> dict[str, Any]:
    return {
        "status": "online", "model": service.bundle.model_name, "version": service.bundle.version,
        "target": service.bundle.target, "candidate_rings": len(service.rings["candidate_rings"]),
        "latest_event_timestamp": datetime.fromtimestamp(service.engine.last_seconds, tz=timezone.utc).isoformat(),
        "assistant": "gemini-2.5-flash" if os.getenv("GEMINI_API_KEY", "").strip() else "local_grounded_fallback",
    }


@app.get("/api/model")
def model_card() -> dict[str, Any]:
    return service.report


@app.get("/api/dashboard")
def dashboard() -> JSONResponse:
    return JSONResponse(service.dashboard_payload(), headers={"Cache-Control": "no-store"})


@app.post("/api/score")
def score_refund(request: ScoreRequest) -> dict[str, Any]:
    return service.score(request.features.model_dump())


@app.post("/api/events")
def score_raw_event(request: RawRefundEventRequest) -> dict[str, Any]:
    return service.score_event(request)


@app.post("/api/decisions/{case_id}/action")
def record_analyst_action(case_id: str, request: AnalystActionRequest) -> dict[str, Any]:
    return service.record_action(case_id, request)


@app.post("/api/chat")
def grounded_chat(request: ChatRequest) -> dict[str, Any]:
    return service.chat(request)


@app.get("/api/audit/{case_id}")
def audit_case(case_id: str) -> dict[str, Any]:
    return service.audit(case_id)


@app.get("/api/rings")
def list_rings() -> dict[str, Any]:
    return {
        "as_of": service.rings["as_of"], "window_days": service.rings["window_days"],
        "ranking_note": service.rings["ranking_note"], "method": service.rings["method"],
        "manual_review_threshold": service.bundle.threshold,
        "action_rule": "Verify evidence requires an above-threshold score plus multi-identifier overlap; other above-threshold cases go to manual review.",
        "data_scope": "Trailing 30-day graph over final synthetic test and replayed live or demo events. Labels are never used to form candidate components.",
        "live_event_count": int(service.cases["split"].eq("live").sum()),
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
    return FileResponse(ROOT / "index.html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/rings.html", include_in_schema=False)
def rings_page() -> FileResponse:
    return FileResponse(ROOT / "rings.html", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/styles.css", include_in_schema=False)
def stylesheet() -> FileResponse:
    return FileResponse(ROOT / "styles.css", media_type="text/css", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/app.js", include_in_schema=False)
def casework_script() -> FileResponse:
    return FileResponse(ROOT / "app.js", media_type="text/javascript", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/rings.js", include_in_schema=False)
def rings_script() -> FileResponse:
    return FileResponse(ROOT / "rings.js", media_type="text/javascript", headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/chat.js", include_in_schema=False)
def chat_script() -> FileResponse:
    return FileResponse(ROOT / "chat.js", media_type="text/javascript", headers={"Cache-Control": "no-store, max-age=0"})
