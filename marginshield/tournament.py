from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from marginshield.data_builder import (
    CONTEXT_FEATURE_COLUMNS,
    GRAPH_VELOCITY_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_ONLY_COLUMNS,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = ROOT / "data" / "processed" / "master_refund_cases.csv.gz"
DEFAULT_MODEL_DIR = ROOT / "data" / "model"
DEFAULT_REPORT_DIR = ROOT / "data" / "reports"
TARGET = "ring_label"


@dataclass(frozen=True)
class TournamentConfig:
    seed: int = 20260905
    minimum_precision: float = 0.85
    minimum_validation_flags: int = 30
    rolling_folds: int = 3
    bootstrap_samples: int = 300


@dataclass
class LiveModelBundle:
    model_name: str
    model: Any
    calibrator: LogisticRegression
    features: list[str]
    categorical_features: list[str]
    numeric_medians: dict[str, float]
    threshold: float
    version: str
    target: str = TARGET


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    return value


def load_dataset(path: Path = DEFAULT_DATASET) -> dict[str, pd.DataFrame]:
    frame = pd.read_csv(path)
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True, format="mixed")
    required = set(MODEL_FEATURE_COLUMNS + [TARGET, "case_id", "split", "event_timestamp", "ring_id", "expected_loss_if_ring_inr", "false_positive_cost_inr", "review_cost_inr"])
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    forbidden = set(MODEL_FEATURE_COLUMNS) & set(TARGET_ONLY_COLUMNS)
    if forbidden:
        raise ValueError(f"Target-only fields leaked into feature contract: {sorted(forbidden)}")
    if set(frame["split"].unique()) != {"train", "validation", "test"}:
        raise ValueError("Expected fixed chronological train, validation and test splits")
    return {part: frame.loc[frame["split"].eq(part)].sort_values(["event_timestamp", "case_id"], kind="stable").reset_index(drop=True) for part in ("train", "validation", "test")}


def feature_types(frame: pd.DataFrame, features: list[str] = MODEL_FEATURE_COLUMNS) -> tuple[list[str], list[str]]:
    categorical = [name for name in features if frame[name].dtype == "object"]
    return [name for name in features if name not in categorical], categorical


def build_logistic(frame: pd.DataFrame, features: list[str] = MODEL_FEATURE_COLUMNS, c_value: float = 0.35) -> Pipeline:
    numeric, categorical = feature_types(frame, features)
    return Pipeline([
        ("preprocessor", ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encode", OneHotEncoder(handle_unknown="ignore"))]), categorical),
        ])),
        ("classifier", LogisticRegression(max_iter=2_000, C=c_value, solver="lbfgs", class_weight=None)),
    ])


def build_catboost(seed: int) -> CatBoostClassifier:
    return CatBoostClassifier(
        iterations=360, depth=4, learning_rate=0.04, l2_leaf_reg=30.0,
        loss_function="Logloss", eval_metric="PRAUC", random_seed=seed,
        random_strength=0.7, verbose=False, allow_writing_files=False, thread_count=8,
    )


def catboost_frame(frame: pd.DataFrame, categorical: list[str], medians: dict[str, float], features: list[str] = MODEL_FEATURE_COLUMNS) -> pd.DataFrame:
    prepared = frame[features].copy()
    for name in categorical:
        prepared[name] = prepared[name].fillna("__missing__").astype(str)
    for name, median in medians.items():
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce").fillna(median)
    return prepared


def graph_rule_score(frame: pd.DataFrame) -> np.ndarray:
    """Transparent candidate rule. It is a baseline, not a latent simulator score."""
    entity_peak = np.maximum.reduce([
        frame["shared_device_accounts_30d"].to_numpy(float),
        frame["shared_address_accounts_90d"].to_numpy(float),
        frame["shared_payment_accounts_30d"].to_numpy(float),
    ]) - 1
    connected = np.maximum(frame["connected_accounts_max_window"].to_numpy(float) - 1, 0)
    burst = np.maximum(frame["linked_refund_burst_72h"].to_numpy(float) - 1, 0)
    identifiers = frame["shared_identifier_types_30d"].to_numpy(float)
    multi_neighbors = frame["multi_identifier_neighbor_accounts"].to_numpy(float)
    balance = frame["identifier_reuse_balance"].to_numpy(float)
    raw = (
        0.34 * np.log1p(np.maximum(entity_peak, 0))
        + 0.26 * np.log1p(connected)
        + 0.28 * np.log1p(burst)
        + 0.45 * identifiers
        + 0.22 * np.log1p(multi_neighbors)
        + 0.35 * balance
        - 5.0
    )
    return 1.0 / (1.0 + np.exp(-np.clip(raw, -20, 20)))


CandidateBuilder = Callable[[pd.DataFrame], Any]


def candidate_builders(seed: int) -> dict[str, CandidateBuilder]:
    return {
        "graph_rule_baseline": lambda _: None,
        "logistic_l2": lambda frame: build_logistic(frame),
        "catboost_calibrated": lambda _: build_catboost(seed),
    }


def fit_candidate(model_name: str, builder: CandidateBuilder, train: pd.DataFrame) -> Any:
    if model_name == "graph_rule_baseline":
        return None
    model = builder(train)
    if model_name == "catboost_calibrated":
        _, categorical = feature_types(train)
        medians = {name: float(train[name].median()) for name in MODEL_FEATURE_COLUMNS if name not in categorical}
        model.fit(catboost_frame(train, categorical, medians), train[TARGET], cat_features=categorical)
        model._marginshield_medians = medians
        return model
    model.fit(train[MODEL_FEATURE_COLUMNS], train[TARGET])
    return model


def predict_candidate(model_name: str, model: Any, frame: pd.DataFrame) -> np.ndarray:
    if model_name == "graph_rule_baseline":
        return graph_rule_score(frame)
    if model_name == "catboost_calibrated":
        _, categorical = feature_types(frame)
        return model.predict_proba(catboost_frame(frame, categorical, model._marginshield_medians))[:, 1]
    return model.predict_proba(frame[MODEL_FEATURE_COLUMNS])[:, 1]


def _log_odds(scores: np.ndarray) -> np.ndarray:
    clipped = np.clip(scores, 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped)).reshape(-1, 1)


def fit_platt_calibrator(scores: np.ndarray, labels: pd.Series, seed: int) -> LogisticRegression:
    calibrator = LogisticRegression(C=1.0, max_iter=1_000, random_state=seed)
    calibrator.fit(_log_odds(scores), labels.to_numpy(dtype=int))
    return calibrator


def calibrate(calibrator: LogisticRegression, scores: np.ndarray) -> np.ndarray:
    return calibrator.predict_proba(_log_odds(scores))[:, 1]


def score_live_bundle(bundle: LiveModelBundle, feature_frame: pd.DataFrame) -> np.ndarray:
    missing = sorted(set(bundle.features).difference(feature_frame.columns))
    if missing:
        raise ValueError(f"Missing required model features: {missing}")
    prepared = feature_frame[bundle.features].copy()
    for name in bundle.categorical_features:
        prepared[name] = prepared[name].fillna("__missing__").astype(str)
    for name, median in bundle.numeric_medians.items():
        prepared[name] = pd.to_numeric(prepared[name], errors="coerce").fillna(median)
    if bundle.model_name == "graph_rule_baseline":
        raw = graph_rule_score(prepared)
    elif bundle.model_name == "catboost_calibrated":
        raw = bundle.model.predict_proba(catboost_frame(prepared, bundle.categorical_features, bundle.numeric_medians))[:, 1]
    else:
        raw = bundle.model.predict_proba(prepared)[:, 1]
    return calibrate(bundle.calibrator, raw)


def raw_metrics(frame: pd.DataFrame, scores: np.ndarray) -> dict[str, float]:
    truth = frame[TARGET].to_numpy(dtype=int)
    return {
        "pr_auc": float(average_precision_score(truth, scores)),
        "roc_auc": float(roc_auc_score(truth, scores)) if len(np.unique(truth)) == 2 else float("nan"),
        "brier_score": float(brier_score_loss(truth, scores)),
    }


def calibration_summary(frame: pd.DataFrame, scores: np.ndarray, bins: int = 10) -> dict[str, Any]:
    truth = frame[TARGET].to_numpy(dtype=int)
    working = pd.DataFrame({"score": scores, "truth": truth})
    working["bin"] = pd.qcut(working["score"], q=bins, duplicates="drop")
    table = []
    weighted_error = 0.0
    for _, group in working.groupby("bin", observed=True, sort=True):
        mean_score = float(group["score"].mean())
        observed_rate = float(group["truth"].mean())
        weighted_error += len(group) * abs(mean_score - observed_rate)
        table.append({
            "rows": len(group),
            "mean_probability": mean_score,
            "observed_rate": observed_rate,
        })
    result = {
        "method": f"Equal-frequency reliability table with up to {bins} bins",
        "expected_calibration_error": float(weighted_error / len(working)),
        "bins": table,
    }
    tail = working.loc[working["score"] >= 0.01].copy()
    if len(tail) >= 30:
        tail["bin"] = pd.qcut(tail["score"], q=min(8, max(2, len(tail) // 30)), duplicates="drop")
        tail_table = []
        tail_error = 0.0
        for _, group in tail.groupby("bin", observed=True, sort=True):
            mean_score = float(group["score"].mean())
            observed_rate = float(group["truth"].mean())
            tail_error += len(group) * abs(mean_score - observed_rate)
            tail_table.append({"rows": len(group), "mean_probability": mean_score, "observed_rate": observed_rate})
        result["above_1pct"] = {
            "rows": len(tail),
            "expected_calibration_error": float(tail_error / len(tail)),
            "bins": tail_table,
        }
    return result


def rolling_folds(train: pd.DataFrame, count: int) -> list[tuple[pd.DataFrame, pd.DataFrame]]:
    horizon = max(2_000, len(train) // 9)
    initial = len(train) - horizon * count
    if count < 2 or initial < 12_000:
        raise ValueError("Training split is too small for configured rolling folds")
    return [(train.iloc[:initial + fold * horizon].copy(), train.iloc[initial + fold * horizon: initial + (fold + 1) * horizon].copy()) for fold in range(count)]


def constrained_recall(frame: pd.DataFrame, scores: np.ndarray, precision_floor: float, min_flags: int) -> float:
    truth = frame[TARGET].to_numpy(int)
    order = np.argsort(-scores, kind="stable")
    cumulative = np.cumsum(truth[order])
    counts = np.arange(1, len(frame) + 1)
    precision = cumulative / counts
    recall = cumulative / max(1, truth.sum())
    boundaries = np.r_[scores[order][:-1] != scores[order][1:], True]
    eligible = np.flatnonzero(boundaries & (counts >= min_flags) & (precision >= precision_floor))
    return float(recall[eligible].max()) if len(eligible) else 0.0


def select_threshold(frame: pd.DataFrame, scores: np.ndarray, precision_floor: float, min_flags: int) -> tuple[float, list[dict[str, float]]]:
    truth = frame[TARGET].to_numpy(int)
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_truth = truth[order]
    cumulative = np.cumsum(sorted_truth)
    count = np.arange(1, len(frame) + 1)
    precision = cumulative / count
    recall = cumulative / max(1, sorted_truth.sum())
    boundaries = np.r_[sorted_scores[:-1] != sorted_scores[1:], True]
    eligible = np.flatnonzero(boundaries & (count >= min_flags) & (precision >= precision_floor))
    if not len(eligible):
        raise ValueError(f"No validation threshold reaches {precision_floor:.0%} precision with at least {min_flags} flags")
    # Subject to the safety constraint, maximize recall; ties favour less review volume.
    best_recall = recall[eligible].max()
    best = eligible[recall[eligible] == best_recall][0]
    sampled = np.unique(np.linspace(0, len(sorted_scores) - 1, min(101, len(sorted_scores)), dtype=int))
    curve = [{"threshold": float(sorted_scores[index]), "precision": float(precision[index]), "recall": float(recall[index]), "flag_rate": float(count[index] / len(frame)), "review_volume": int(count[index])} for index in sampled]
    threshold = float(sorted_scores[best])
    if best + 1 < len(sorted_scores):
        upper = float(np.clip(sorted_scores[best], 1e-8, 1 - 1e-8))
        lower = float(np.clip(sorted_scores[best + 1], 1e-8, 1 - 1e-8))
        threshold = float(1.0 / (1.0 + np.exp(-(np.log(upper / (1 - upper)) + np.log(lower / (1 - lower))) / 2)))
    return threshold, curve


def costs(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, float]:
    flagged = scores >= threshold
    truth = frame[TARGET].to_numpy(int)
    true_positive_review_cost = float(frame.loc[flagged & (truth == 1), "review_cost_inr"].sum())
    return {
        "preventable_loss_inr": float(frame.loc[flagged & (truth == 1), "expected_loss_if_ring_inr"].sum()),
        "false_positive_cost_inr": float(frame.loc[flagged & (truth == 0), "false_positive_cost_inr"].sum()),
        "true_positive_review_cost_inr": true_positive_review_cost,
        "total_review_cost_inr": float(frame.loc[flagged, "review_cost_inr"].sum()),
        "missed_loss_inr": float(frame.loc[(~flagged) & (truth == 1), "expected_loss_if_ring_inr"].sum()),
    }


def _candidate_components(working: pd.DataFrame) -> list[list[str]]:
    """Find label-free connected case components from submitted entity links."""
    graph = nx.Graph()
    for column, entity_type in (
        ("device_id", "device"),
        ("address_id", "address"),
        ("payment_token_id", "payment"),
    ):
        for identifier, group in working.groupby(column, sort=False):
            if group["customer_id"].nunique() < 2:
                continue
            entity_node = f"entity:{entity_type}:{identifier}"
            for case_id in group["case_id"]:
                graph.add_edge(f"case:{case_id}", entity_node)

    indexed = working.set_index("case_id", drop=False)
    components: list[list[str]] = []
    for component in nx.connected_components(graph):
        case_ids = sorted(node.removeprefix("case:") for node in component if node.startswith("case:"))
        if len(case_ids) < 3:
            continue
        cases = indexed.loc[case_ids]
        if cases["customer_id"].nunique() < 3 or not cases["_flag"].any():
            continue
        components.append(case_ids)
    return components


def ring_level_metrics(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    working = frame.copy()
    working["_score"] = scores
    working["_flag"] = scores >= threshold
    working["event_timestamp"] = pd.to_datetime(working["event_timestamp"], utc=True, format="mixed")
    end_time = working["event_timestamp"].max()
    working = working.loc[working["event_timestamp"] >= end_time - pd.Timedelta(days=30)].copy()
    ring_rows = working.loc[working["ring_id"].ne("")].sort_values("event_timestamp")
    detected = 0
    all_rings = 0
    for _, group in ring_rows.groupby("ring_id", sort=False):
        all_rings += 1
        losses = group["expected_loss_if_ring_inr"].to_numpy(float)
        cumulative_before = np.r_[0.0, np.cumsum(losses)[:-1]]
        observed_link = np.maximum.reduce([
            group["shared_device_accounts_30d"].to_numpy(int),
            group["shared_address_accounts_90d"].to_numpy(int),
            group["shared_payment_accounts_30d"].to_numpy(int),
        ]) >= 2
        valid = group["_flag"].to_numpy(bool) & observed_link
        if np.any(valid & (cumulative_before < losses.sum() / 2)):
            detected += 1
    components = _candidate_components(working)
    indexed = working.set_index("case_id", drop=False)
    true_candidates = 0
    for case_ids in components:
        component = indexed.loc[case_ids]
        ring_counts = component.loc[component[TARGET].eq(1) & component["ring_id"].ne(""), "ring_id"].value_counts()
        true_candidates += int(bool(len(ring_counts) and ring_counts.iloc[0] >= 2))
    predicted = len(components)
    return {
        "actual_rings": all_rings,
        "detected_before_half_loss": detected,
        "ring_recall": float(detected / all_rings) if all_rings else 0.0,
        "predicted_ring_candidates": int(predicted),
        "ring_precision": float(true_candidates / predicted) if predicted else 0.0,
        "window_days": 30,
        "definition": "Within the same trailing 30-day window used by the Rings page, a true ring is detected only when its first flagged request has a pre-decision shared device, address or payment link and occurs before half of that ring's simulated loss. Candidate grouping is label-free and requires a connected component spanning at least three cases and three accounts; ring_id is evaluation-only.",
    }


def classification_metrics(frame: pd.DataFrame, scores: np.ndarray, threshold: float, include_ring_metrics: bool = True) -> dict[str, Any]:
    truth = frame[TARGET].to_numpy(int)
    flagged = scores >= threshold
    tn, fp, fn, tp = confusion_matrix(truth, flagged, labels=[0, 1]).ravel()
    result = {
        "rows": len(frame), "ring_rate": float(truth.mean()),
        **raw_metrics(frame, scores), "threshold": float(threshold),
        "precision": float(precision_score(truth, flagged, zero_division=0)),
        "recall": float(recall_score(truth, flagged, zero_division=0)),
        "flag_rate": float(flagged.mean()), "review_volume": int(flagged.sum()),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "costs": costs(frame, scores, threshold),
        "calibration": calibration_summary(frame, scores),
    }
    if include_ring_metrics:
        result["ring_level"] = ring_level_metrics(frame, scores, threshold)
    result["costs"]["net_preventable_value_inr"] = (
        result["costs"]["preventable_loss_inr"]
        - result["costs"]["false_positive_cost_inr"]
        - result["costs"]["true_positive_review_cost_inr"]
    )
    return result


def bootstrap_intervals(frame: pd.DataFrame, scores: np.ndarray, threshold: float, samples: int, seed: int) -> dict[str, list[float]]:
    working = frame.copy().reset_index(drop=True)
    working["_day"] = pd.to_datetime(working["event_timestamp"], utc=True).dt.floor("D")
    groups = [group.index.to_numpy() for _, group in working.groupby("_day", sort=True)]
    rng = np.random.default_rng(seed)
    observed: dict[str, list[float]] = {"precision": [], "recall": [], "pr_auc": [], "net_preventable_value_inr": []}
    for _ in range(samples):
        index = np.concatenate([groups[position] for position in rng.integers(0, len(groups), len(groups))])
        sampled = working.iloc[index].reset_index(drop=True)
        metric = classification_metrics(sampled, scores[index], threshold, include_ring_metrics=False)
        for key in observed:
            observed[key].append(metric["costs"][key] if key == "net_preventable_value_inr" else metric[key])
    return {name: [float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))] for name, values in observed.items()}


def diagnostic_model(train: pd.DataFrame, validation: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    model = build_logistic(train, features, c_value=0.25)
    model.fit(train[features], train[TARGET])
    val_scores = model.predict_proba(validation[features])[:, 1]
    test_scores = model.predict_proba(test[features])[:, 1]
    return {"features": features, "validation": raw_metrics(validation, val_scores), "test": raw_metrics(test, test_scores)}


def validation_diagnostic(train: pd.DataFrame, validation: pd.DataFrame, features: list[str]) -> dict[str, Any]:
    model = build_logistic(train, features, c_value=0.25)
    model.fit(train[features], train[TARGET])
    scores = model.predict_proba(validation[features])[:, 1]
    return {"features": features, "validation": raw_metrics(validation, scores)}


def pair_reuse_diagnostic(frame: pd.DataFrame) -> dict[str, Any]:
    """Expose how much of the synthetic target is recoverable from its core motif."""
    scores = frame["device_payment_pair_accounts_30d"].to_numpy(float)
    predictions = scores >= 2
    labels = frame[TARGET].to_numpy(int)
    return {
        "rule": "device_payment_pair_accounts_30d >= 2",
        "pr_auc": float(average_precision_score(labels, scores)),
        "roc_auc": float(roc_auc_score(labels, scores)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "review_volume": int(predictions.sum()),
        "false_positive_scenarios": frame.loc[
            predictions & frame[TARGET].eq(0), "scenario_type"
        ].value_counts().to_dict(),
        "interpretation": "A strong result here indicates simulator-rule recoverability, not model sophistication.",
    }


def single_feature_diagnostic(frame: pd.DataFrame) -> dict[str, Any]:
    """Measure whether one numeric feature trivially recovers the simulator label."""
    labels = frame[TARGET].to_numpy(int)
    rows = []
    for feature in MODEL_FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(frame[feature]):
            continue
        values = frame[feature].to_numpy(float)
        high = float(average_precision_score(labels, values))
        low = float(average_precision_score(labels, -values))
        rows.append({
            "feature": feature,
            "best_pr_auc": max(high, low),
            "direction": "high" if high >= low else "low",
        })
    rows.sort(key=lambda row: row["best_pr_auc"], reverse=True)
    return {
        "maximum_univariate_pr_auc": rows[0]["best_pr_auc"] if rows else 0.0,
        "features": rows,
        "interpretation": "Low univariate PR-AUC supports, but does not prove, that no single numeric simulator field defines the label.",
    }


def within_shared_diagnostic(frame: pd.DataFrame) -> dict[str, Any]:
    shared = frame.loc[frame[[
        "shared_device_accounts_30d", "shared_address_accounts_90d", "shared_payment_accounts_30d",
    ]].max(axis=1).ge(2)].copy()
    result = single_feature_diagnostic(shared)
    prevalence = float(shared[TARGET].mean()) if len(shared) else 0.0
    result.update({
        "rows": len(shared),
        "prevalence": prevalence,
        "maximum_univariate_ap_lift": result["maximum_univariate_pr_auc"] / prevalence if prevalence else None,
        "interpretation": "This conditions on already-shared identities; low lift is evidence against a simple reuse-shape simulator shortcut.",
    })
    return result


def topology_recall(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    working = frame.copy()
    working["_score"] = scores
    rows = []
    for topology, group in working.loc[working[TARGET].eq(1)].groupby("ring_topology", sort=True):
        metric = classification_metrics(group, group["_score"].to_numpy(float), threshold, include_ring_metrics=True)
        rows.append({
            "topology": topology,
            "requests": len(group),
            "request_recall": metric["recall"],
            "actual_rings": metric["ring_level"]["actual_rings"],
            "early_ring_recall": metric["ring_level"]["ring_recall"],
        })
    return {"by_topology": rows}


def false_positive_cost_sensitivity(frame: pd.DataFrame, scores: np.ndarray, threshold: float) -> list[dict[str, float]]:
    flagged = scores >= threshold
    truth = frame[TARGET].to_numpy(int)
    preventable = float(frame.loc[flagged & (truth == 1), "expected_loss_if_ring_inr"].sum())
    tp_review = float(frame.loc[flagged & (truth == 1), "review_cost_inr"].sum())
    rows = []
    for rate in (0.05, 0.075, 0.15):
        false_positive = float(
            (frame.loc[flagged & (truth == 0), "refund_amount_inr"] * rate
             + frame.loc[flagged & (truth == 0), "review_cost_inr"]).sum()
        )
        rows.append({
            "refund_value_rate": rate,
            "false_positive_cost_inr": false_positive,
            "net_preventable_value_inr": preventable - false_positive - tp_review,
        })
    return rows


def validate_policy_only(splits: dict[str, pd.DataFrame], config: TournamentConfig) -> dict[str, Any]:
    """Develop and validate the policy without scoring or fitting against final test."""
    winner, tournament_report = tournament(splits, config)
    model = fit_candidate(winner, candidate_builders(config.seed)[winner], splits["train"])
    raw_validation = predict_candidate(winner, model, splits["validation"])
    cutoff = splits["validation"]["event_timestamp"].median()
    calibration_mask = splits["validation"]["event_timestamp"] <= cutoff
    calibrator = fit_platt_calibrator(
        raw_validation[calibration_mask.to_numpy()],
        splits["validation"].loc[calibration_mask, TARGET],
        config.seed,
    )
    policy_frame = splits["validation"].loc[~calibration_mask].reset_index(drop=True)
    policy_scores = calibrate(calibrator, raw_validation[~calibration_mask.to_numpy()])
    threshold, _ = select_threshold(
        policy_frame, policy_scores, config.minimum_precision, config.minimum_validation_flags
    )
    return {
        "test_scored": False,
        "winner": winner,
        "tournament": tournament_report,
        "manual_review_threshold": threshold,
        "policy_validation": classification_metrics(policy_frame, policy_scores, threshold),
        "policy_validation_temporal_bootstrap_95pct_ci": bootstrap_intervals(
            policy_frame, policy_scores, threshold, config.bootstrap_samples, config.seed + 17
        ),
        "diagnostics": {
            "context_only": validation_diagnostic(splits["train"], splits["validation"], CONTEXT_FEATURE_COLUMNS),
            "graph_velocity_only": validation_diagnostic(splits["train"], splits["validation"], GRAPH_VELOCITY_FEATURE_COLUMNS),
            "device_payment_pair_rule": pair_reuse_diagnostic(splits["validation"]),
            "single_numeric_feature": single_feature_diagnostic(splits["validation"]),
            "within_shared_identity": within_shared_diagnostic(splits["validation"]),
        },
    }


def feature_summary(model_name: str, model: Any) -> list[dict[str, Any]]:
    if model_name == "catboost_calibrated":
        table = pd.DataFrame({"feature": MODEL_FEATURE_COLUMNS, "importance": model.get_feature_importance()})
        return table.sort_values("importance", ascending=False).head(20).to_dict(orient="records")
    if model_name == "logistic_l2":
        names = model.named_steps["preprocessor"].get_feature_names_out()
        values = model.named_steps["classifier"].coef_[0]
        table = pd.DataFrame({"feature": names, "coefficient": values})
        table["absolute_coefficient"] = table["coefficient"].abs()
        return table.sort_values("absolute_coefficient", ascending=False).head(20).to_dict(orient="records")
    return [{"feature": "graph_overlap_surprisal", "description": "Transparent baseline combines entity reuse, connected-account reach, refund velocity and multi-identifier overlap without privileging an exact identifier pair."}]


def tournament(splits: dict[str, pd.DataFrame], config: TournamentConfig) -> tuple[str, dict[str, Any]]:
    folds = rolling_folds(splits["train"], config.rolling_folds)
    report: dict[str, Any] = {"rolling_folds": [], "candidates": {}}
    for number, (fold_train, fold_validation) in enumerate(folds, start=1):
        report["rolling_folds"].append({"fold": number, "train_rows": len(fold_train), "validation_rows": len(fold_validation), "train_end": str(fold_train["event_timestamp"].max()), "validation_end": str(fold_validation["event_timestamp"].max())})
    for name, builder in candidate_builders(config.seed).items():
        rows = []
        for fold_train, fold_validation in folds:
            model = fit_candidate(name, builder, fold_train)
            scores = predict_candidate(name, model, fold_validation)
            metrics = raw_metrics(fold_validation, scores)
            metrics["recall_at_85pct_precision"] = constrained_recall(fold_validation, scores, config.minimum_precision, config.minimum_validation_flags)
            rows.append(metrics)
        report["candidates"][name] = {"folds": rows, "mean_pr_auc": float(np.mean([row["pr_auc"] for row in rows])), "mean_roc_auc": float(np.mean([row["roc_auc"] for row in rows])), "mean_brier_score": float(np.mean([row["brier_score"] for row in rows])), "mean_recall_at_precision_floor": float(np.mean([row["recall_at_85pct_precision"] for row in rows]))}
    winner = max(report["candidates"], key=lambda name: (report["candidates"][name]["mean_pr_auc"], report["candidates"][name]["mean_recall_at_precision_floor"], -report["candidates"][name]["mean_brier_score"]))
    report["winner"] = winner
    report["selection_rule"] = "Highest mean rolling-train PR-AUC, then recall subject to the 85% precision floor and minimum review volume, then lower Brier score. The final test split is excluded from this selection."
    return winner, report


def train_live_model(dataset_path: Path = DEFAULT_DATASET, model_dir: Path = DEFAULT_MODEL_DIR, report_dir: Path = DEFAULT_REPORT_DIR, config: TournamentConfig = TournamentConfig()) -> dict[str, Any]:
    splits = load_dataset(dataset_path)
    winner, tournament_report = tournament(splits, config)
    model = fit_candidate(winner, candidate_builders(config.seed)[winner], splits["train"])
    raw_validation = predict_candidate(winner, model, splits["validation"])
    cutoff = splits["validation"]["event_timestamp"].median()
    calibration_mask = splits["validation"]["event_timestamp"] <= cutoff
    calibrator = fit_platt_calibrator(raw_validation[calibration_mask.to_numpy()], splits["validation"].loc[calibration_mask, TARGET], config.seed)
    policy_frame = splits["validation"].loc[~calibration_mask].reset_index(drop=True)
    policy_scores = calibrate(calibrator, raw_validation[~calibration_mask.to_numpy()])
    threshold, policy_curve = select_threshold(policy_frame, policy_scores, config.minimum_precision, config.minimum_validation_flags)
    # The test is deliberately first touched here, after model and policy are fixed.
    raw_test = predict_candidate(winner, model, splits["test"])
    test_scores = calibrate(calibrator, raw_test)
    _, categorical = feature_types(splits["train"])
    medians = {name: float(splits["train"][name].median()) for name in MODEL_FEATURE_COLUMNS if name not in categorical}
    bundle = LiveModelBundle(winner, model, calibrator, list(MODEL_FEATURE_COLUMNS), categorical, medians, threshold, "marginshield-3.0.0-locked")
    diagnostics = {
        "context_only": diagnostic_model(splits["train"], splits["validation"], splits["test"], CONTEXT_FEATURE_COLUMNS),
        "graph_velocity_only": diagnostic_model(splits["train"], splits["validation"], splits["test"], GRAPH_VELOCITY_FEATURE_COLUMNS),
        "device_payment_pair_rule": {
            "validation": pair_reuse_diagnostic(splits["validation"]),
            "test": pair_reuse_diagnostic(splits["test"]),
        },
        "single_numeric_feature": {
            "validation": single_feature_diagnostic(splits["validation"]),
            "test": single_feature_diagnostic(splits["test"]),
        },
        "within_shared_identity": {
            "validation": within_shared_diagnostic(splits["validation"]),
            "test": within_shared_diagnostic(splits["test"]),
        },
    }
    report = {
        "dataset_version": "3.0.0-locked", "target": TARGET,
        "target_definition": "Coordinated multi-account refund-abuse ring event. No broad all-refund-abuse classifier is trained.",
        "feature_contract": MODEL_FEATURE_COLUMNS, "forbidden_features": TARGET_ONLY_COLUMNS,
        "tournament": tournament_report, "winner": winner,
        "calibration": {"method": "Platt scaling on the early validation window", "calibration_rows": int(calibration_mask.sum()), "policy_rows": int((~calibration_mask).sum()), "manual_review_threshold": threshold, "policy_selection": f"Maximize recall subject to >= {config.minimum_precision:.0%} validation precision and >= {config.minimum_validation_flags} flags; place the operating boundary at the validation-only log-odds midpoint to the next excluded score."},
        "policy_validation": classification_metrics(policy_frame, policy_scores, threshold),
        "policy_validation_temporal_bootstrap_95pct_ci": bootstrap_intervals(
            policy_frame, policy_scores, threshold, config.bootstrap_samples, config.seed + 17
        ),
        "policy_curve": policy_curve,
        "test": classification_metrics(splits["test"], test_scores, threshold),
        "test_temporal_bootstrap_95pct_ci": bootstrap_intervals(splits["test"], test_scores, threshold, config.bootstrap_samples, config.seed),
        "test_topology_recall": topology_recall(splits["test"], test_scores, threshold),
        "false_positive_cost_sensitivity": false_positive_cost_sensitivity(splits["test"], test_scores, threshold),
        "diagnostics": diagnostics, "feature_summary": feature_summary(winner, model),
        "synthetic_cost_assumptions": {
            "expected_loss_if_ring": "Refund amount plus 55% of the simulated gross margin on that refund.",
            "review_cost": "INR 65 per request, or INR 140 when the refund exceeds INR 5,000.",
            "false_positive_cost": "7.5% of refund value plus review cost for a legitimate request sent to review.",
            "net_preventable_value": "Preventable loss minus false-positive cost and review cost for correctly flagged ring requests.",
        },
        "limitations": [
            "Olist has no refund-fraud labels and no Olist source rows are used as fraud truth.",
            "All reported performance is synthetic benchmark performance, not real-world or production performance.",
            "The ring_id and all scenario metadata are evaluation-only and excluded from features and serving.",
            "Final-test ring topologies are structurally disjoint from train and validation and unfold over a slower event horizon, but remain synthetic.",
            "The device-payment pair rule is reported as an explicit shortcut diagnostic and is not the candidate-ring definition.",
            "The system recommends approve, verify evidence or manual review; it never auto-rejects a customer.",
            "The 85% validation precision constraint is applied to the point estimate; its day-block bootstrap interval is reported separately and may cross the floor.",
        ],
    }
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_dir / "live_refund_ring_model.joblib")
    predictions = splits["test"][["case_id", "event_timestamp", "merchant_id", TARGET]].copy()
    predictions["ring_probability"] = np.round(test_scores, 6)
    predictions["flagged"] = (test_scores >= threshold).astype(int)
    predictions.to_csv(model_dir / "live_ring_test_predictions.csv.gz", index=False, compression="gzip")
    (report_dir / "ring_model_report.json").write_text(json.dumps(_json_value(report), indent=2), encoding="utf-8")
    return report
