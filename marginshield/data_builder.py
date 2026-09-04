from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import math
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "processed"
DEFAULT_REPORTS = ROOT / "data" / "reports"

# This is intentionally a narrow contract. Context fields are retained only where
# they are plausibly available at refund-request time; no simulator metadata leaks in.
MODEL_FEATURE_COLUMNS = [
    "vertical",
    "payment_method",
    "refund_amount_inr",
    "refund_share",
    "account_age_days",
    "refund_velocity_24h",
    "refund_velocity_7d",
    "merchant_refund_volume_index_30d",
    "shared_device_accounts_30d",
    "shared_address_accounts_90d",
    "shared_identifier_types_30d",
    "device_payment_pair_accounts_30d",
    "linked_refund_burst_72h",
    "linked_merchants_30d",
    "graph_overlap_surprisal",
]

CONTEXT_FEATURE_COLUMNS = [
    "vertical", "payment_method", "refund_amount_inr", "refund_share", "account_age_days"
]
GRAPH_VELOCITY_FEATURE_COLUMNS = [
    column for column in MODEL_FEATURE_COLUMNS if column not in CONTEXT_FEATURE_COLUMNS
]

TARGET_ONLY_COLUMNS = [
    "ring_label",
    "ring_id",
    "scenario_type",
    "ring_topology",
    "loss_sequence_position",
    "simulator_internal_score",
]

ENTITY_WINDOWS = {
    "device_id": ("shared_device_accounts_30d", 30),
    "address_id": ("shared_address_accounts_90d", 90),
    "payment_token_id": ("shared_payment_accounts_30d", 30),
}

MANDATORY_COLUMNS = [
    "case_id", "event_timestamp", "split", "merchant_id", "customer_id",
    "device_id", "address_id", "payment_token_id", "ring_label", "is_synthetic",
]


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_value(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    return value


def write_gzip_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a byte-stable gzip CSV without clock or filename metadata."""
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, compresslevel=6, mtime=0) as compressed:
            with io.TextIOWrapper(compressed, encoding="utf-8", newline="") as text:
                frame.to_csv(text, index=False, lineterminator="\n")


def _hash_id(prefix: str, value: str) -> str:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=7).hexdigest().upper()
    return f"{prefix}-{digest}"


def _choice_without_replacement(rng: np.random.Generator, values: np.ndarray, count: int) -> np.ndarray:
    if count > len(values):
        raise ValueError("Scenario allocation exceeds available split rows")
    return rng.choice(values, size=count, replace=False)


def _split_bounds() -> dict[str, tuple[pd.Timestamp, pd.Timestamp]]:
    start = pd.Timestamp("2025-01-01", tz="UTC")
    return {
        "train": (start, start + pd.Timedelta(days=126)),
        "validation": (start + pd.Timedelta(days=126), start + pd.Timedelta(days=153)),
        "test": (start + pd.Timedelta(days=153), start + pd.Timedelta(days=180)),
    }


def _new_entity(prefix: str, namespace: str) -> str:
    return _hash_id(prefix, namespace)


def _base_cases(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create unlabeled refund requests before any scenario injection.

    The context distributions are Olist-inspired, but Olist does not contain
    refund-fraud labels and contributes no source rows to this benchmark.
    """
    start = pd.Timestamp("2025-01-01", tz="UTC")
    seconds = np.sort(rng.integers(0, 180 * 86_400, size=n))
    timestamps = start + pd.to_timedelta(seconds, unit="s")
    split = np.where(seconds < 126 * 86_400, "train", np.where(seconds < 153 * 86_400, "validation", "test"))

    verticals = np.array(["fashion", "beauty", "electronics", "home", "sports", "general"], dtype=object)
    vertical = rng.choice(verticals, size=n, p=[0.25, 0.11, 0.18, 0.18, 0.08, 0.20])
    payment_method = rng.choice(["upi", "card", "wallet", "netbanking", "cod"], size=n, p=[0.49, 0.27, 0.08, 0.06, 0.10])
    amount_scale = np.select(
        [vertical == "electronics", vertical == "fashion", vertical == "beauty"], [1.55, 0.84, 0.72], default=1.0
    )
    order_amount = np.clip(rng.lognormal(mean=7.55, sigma=0.72, size=n) * amount_scale, 249, 90_000)
    refund_share = np.clip(rng.beta(5.0, 1.65, size=n), 0.12, 1.0)
    refund_amount = np.rint(order_amount * refund_share).astype(int)
    merchant_count = max(140, n // 170)
    merchant_ids = np.array([f"MER-{value:05d}" for value in rng.integers(0, merchant_count, size=n)])
    customer_count = max(12_000, int(n * 0.74))
    customer_ids = np.array([f"CUS-{value:07d}" for value in rng.integers(0, customer_count, size=n)])
    cases = pd.DataFrame({
        "case_id": [f"MSR-{value:07d}" for value in range(1, n + 1)],
        "event_timestamp": timestamps,
        # The split is deliberately fixed before labels/scenarios are injected.
        "split": split,
        "merchant_id": merchant_ids,
        "customer_id": customer_ids,
        "vertical": vertical,
        "payment_method": payment_method,
        "order_amount_inr": np.rint(order_amount).astype(int),
        "refund_amount_inr": refund_amount,
        "refund_share": np.round(refund_share, 4),
        "account_age_days": np.round(np.clip(rng.gamma(2.5, 120, size=n), 1, 2_400), 1),
        "refund_reason": rng.choice(
            ["damaged", "not_as_described", "changed_mind", "late_delivery", "wrong_item", "duplicate_order"],
            size=n, p=[0.17, 0.21, 0.25, 0.12, 0.11, 0.14],
        ),
        "refund_method": rng.choice(
            ["original_rail", "instant_refund", "store_credit", "manual_bank_transfer"],
            size=n, p=[0.63, 0.22, 0.11, 0.04],
        ),
    })
    cases["device_id"] = [_new_entity("DEV", f"base-{index}") for index in range(n)]
    cases["address_id"] = [_new_entity("ADR", f"base-{index}") for index in range(n)]
    cases["payment_token_id"] = [_new_entity("PAY", f"base-{index}") for index in range(n)]
    cases["ring_label"] = 0
    cases["ring_id"] = ""
    cases["scenario_type"] = "independent_legitimate"
    cases["ring_topology"] = "none"
    cases["loss_sequence_position"] = -1
    return cases


def _scenario_rows(
    cases: pd.DataFrame, split: str, count: int, rng: np.random.Generator, available: np.ndarray | None = None
) -> list[np.ndarray]:
    if available is None:
        available = cases.index[cases["split"].eq(split)].to_numpy()
    selected = _choice_without_replacement(rng, available, count)
    rng.shuffle(selected)
    groups: list[np.ndarray] = []
    cursor = 0
    while cursor < len(selected):
        size = int(rng.integers(4, 8))
        group = selected[cursor:cursor + size]
        if len(group) >= 3:
            groups.append(group)
        cursor += size
    return groups


def _inject_cluster(
    cases: pd.DataFrame,
    rows: np.ndarray,
    split: str,
    cluster_id: str,
    kind: str,
    topology: str,
    rng: np.random.Generator,
) -> None:
    """Inject an event sequence. Labels arise from the mechanism, never a feature logit."""
    bounds = _split_bounds()[split]
    # Leave a margin so all requests stay inside their previously assigned split.
    start = bounds[0] + pd.Timedelta(days=float(rng.uniform(2.0, max(3.0, (bounds[1] - bounds[0]).days - 7))))
    if kind == "coordinated_ring":
        # Test rings are deliberately slower and less dense than train/validation rings.
        horizon_hours = float(rng.uniform(48, 120) if split == "test" else rng.uniform(10, 42))
        device = _new_entity("DEV", f"ring-device-{cluster_id}")
        payment = _new_entity("PAY", f"ring-payment-{cluster_id}")
        address = _new_entity("ADR", f"ring-address-{cluster_id}") if rng.random() < 0.35 else None
        for position, row in enumerate(rows):
            cases.loc[row, "customer_id"] = _new_entity("CUS", f"ring-{cluster_id}-{position}")
            cases.loc[row, "device_id"] = device
            cases.loc[row, "payment_token_id"] = payment
            if address:
                cases.loc[row, "address_id"] = address
            cases.loc[row, "event_timestamp"] = start + pd.Timedelta(hours=horizon_hours * position / max(len(rows) - 1, 1))
            cases.loc[row, "ring_label"] = 1
            cases.loc[row, "ring_id"] = cluster_id
            cases.loc[row, "scenario_type"] = "coordinated_refund_abuse"
            cases.loc[row, "ring_topology"] = topology
            cases.loc[row, "loss_sequence_position"] = position
            # Context is not made more suspicious than background cases.
            cases.loc[row, "refund_reason"] = rng.choice(["not_as_described", "wrong_item", "duplicate_order"])
    else:
        # Hard negatives intentionally overlap in graph structure and can burst.
        device = _new_entity("DEV", f"benign-device-{cluster_id}")
        address = _new_entity("ADR", f"benign-address-{cluster_id}")
        share_payment = kind == "household" and rng.random() < 0.22
        payment = _new_entity("PAY", f"benign-payment-{cluster_id}") if share_payment else None
        horizon_hours = float(rng.uniform(12, 88))
        for position, row in enumerate(rows):
            cases.loc[row, "customer_id"] = _new_entity("CUS", f"benign-{cluster_id}-{position}")
            cases.loc[row, "device_id"] = device
            cases.loc[row, "address_id"] = address
            if payment:
                cases.loc[row, "payment_token_id"] = payment
            cases.loc[row, "event_timestamp"] = start + pd.Timedelta(hours=horizon_hours * position / max(len(rows) - 1, 1))
            cases.loc[row, "scenario_type"] = f"benign_{kind}"
            cases.loc[row, "ring_topology"] = "shared_identity_burst"


def _inject_scenarios(cases: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    # Allocate clusters separately after chronological partitions already exist.
    # The test period contains slower event sequences, while retaining the same
    # device-plus-payment mechanism used in training.
    ring_rows = {"train": 1_050, "validation": 265, "test": 265}
    benign_rows = {"train": 2_400, "validation": 620, "test": 620}
    ring_counter = 1
    benign_counter = 1
    for split in ("train", "validation", "test"):
        for rows in _scenario_rows(cases, split, ring_rows[split], rng):
            topology = "slower_two_identifier" if split == "test" else "faster_two_identifier"
            _inject_cluster(cases, rows, split, f"RING-{ring_counter:05d}", "coordinated_ring", topology, rng)
            ring_counter += 1
        benign_available = cases.index[cases["split"].eq(split) & cases["ring_label"].eq(0)].to_numpy()
        benign_groups = _scenario_rows(cases, split, benign_rows[split], rng, benign_available)
        for number, rows in enumerate(benign_groups):
            kind = ("household", "office", "hostel")[number % 3]
            _inject_cluster(cases, rows, split, f"BENIGN-{benign_counter:05d}", kind, "shared_identity_burst", rng)
            benign_counter += 1
    # Floating timedelta construction can differ by one nanosecond across pandas
    # builds. Microsecond precision is more than sufficient for this benchmark and
    # makes the documented seed reproducible across supported environments.
    cases["event_timestamp"] = pd.to_datetime(cases["event_timestamp"], utc=True, format="mixed").dt.round("us")
    return cases.sort_values(["event_timestamp", "case_id"], kind="stable").reset_index(drop=True)


def _rolling_distinct_accounts(
    entity_ids: np.ndarray, customer_ids: np.ndarray, event_seconds: np.ndarray, window_days: int
) -> np.ndarray:
    window_seconds = window_days * 86_400
    events: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
    customers: dict[str, Counter[str]] = defaultdict(Counter)
    result = np.ones(len(entity_ids), dtype=int)
    for index, (entity_id, customer_id, timestamp) in enumerate(zip(entity_ids, customer_ids, event_seconds)):
        queue = events[str(entity_id)]
        counts = customers[str(entity_id)]
        while queue and queue[0][0] < int(timestamp) - window_seconds:
            _, expired = queue.popleft()
            counts[expired] -= 1
            if not counts[expired]:
                del counts[expired]
        queue.append((int(timestamp), str(customer_id)))
        counts[str(customer_id)] += 1
        result[index] = len(counts)
    return result


def apply_temporal_entity_counts(cases: pd.DataFrame) -> pd.DataFrame:
    updated = cases.copy()
    ordered = updated.sort_values(["event_timestamp", "case_id"], kind="stable").reset_index()
    event_seconds = pd.to_datetime(ordered["event_timestamp"], utc=True, format="mixed").astype("int64").to_numpy() // 1_000_000_000
    for entity_column, (feature_column, window_days) in ENTITY_WINDOWS.items():
        values = _rolling_distinct_accounts(
            ordered[entity_column].to_numpy(), ordered["customer_id"].to_numpy(), event_seconds, window_days
        )
        updated.loc[ordered["index"], feature_column] = values
    return updated


def apply_point_in_time_features(cases: pd.DataFrame) -> pd.DataFrame:
    """Build all graph/history features from events at or before the current request.

    The state is processed chronologically. Nothing in target-only metadata is read.
    """
    ordered = cases.sort_values(["event_timestamp", "case_id"], kind="stable").reset_index()
    seconds = pd.to_datetime(ordered["event_timestamp"], utc=True, format="mixed").astype("int64").to_numpy() // 1_000_000_000
    entity_events: dict[str, deque[tuple[int, str, str]]] = defaultdict(deque)
    entity_customers: dict[str, Counter[str]] = defaultdict(Counter)
    entity_merchants: dict[str, Counter[str]] = defaultdict(Counter)
    pair_events: dict[str, deque[tuple[int, str]]] = defaultdict(deque)
    pair_customers: dict[str, Counter[str]] = defaultdict(Counter)
    customer_events: dict[str, deque[int]] = defaultdict(deque)
    merchant_events: dict[str, deque[int]] = defaultdict(deque)
    output: dict[str, list[float]] = defaultdict(list)

    def expire_entity(key: str, now: int, window: int) -> None:
        queue = entity_events[key]
        while queue and queue[0][0] < now - window:
            _, customer, merchant = queue.popleft()
            entity_customers[key][customer] -= 1
            if not entity_customers[key][customer]:
                del entity_customers[key][customer]
            entity_merchants[key][merchant] -= 1
            if not entity_merchants[key][merchant]:
                del entity_merchants[key][merchant]

    for row, now in zip(ordered.itertuples(index=False), seconds):
        customer = str(row.customer_id)
        merchant = str(row.merchant_id)
        entity_specs = [(str(row.device_id), 30 * 86_400), (str(row.address_id), 90 * 86_400), (str(row.payment_token_id), 30 * 86_400)]
        entity_counts: list[int] = []
        merchant_union: set[str] = set()
        burst_counts: list[int] = []
        for entity, window in entity_specs:
            expire_entity(entity, int(now), window)
            current = len(entity_customers[entity] | Counter({customer: 1}))
            entity_counts.append(current)
            merchant_union.update(entity_merchants[entity])
            burst_counts.append(sum(1 for stamp, _, _ in entity_events[entity] if stamp >= int(now) - 72 * 3600))
        pair = f"{row.device_id}|{row.payment_token_id}"
        while pair_events[pair] and pair_events[pair][0][0] < int(now) - 30 * 86_400:
            _, expired = pair_events[pair].popleft()
            pair_customers[pair][expired] -= 1
            if not pair_customers[pair][expired]:
                del pair_customers[pair][expired]
        pair_count = len(pair_customers[pair] | Counter({customer: 1}))
        recent = customer_events[customer]
        while recent and recent[0] < int(now) - 7 * 86_400:
            recent.popleft()
        velocity_7d = len(recent) + 1
        velocity_24h = sum(stamp >= int(now) - 86_400 for stamp in recent) + 1
        merchant_recent = merchant_events[merchant]
        while merchant_recent and merchant_recent[0] < int(now) - 30 * 86_400:
            merchant_recent.popleft()
        # Baseline denominator is past requests only and does not depend on their outcome.
        merchant_rate = min(len(merchant_recent) / 300.0, 1.0)
        shared_types = sum(count >= 2 for count in entity_counts)
        burst = max(burst_counts, default=0) + 1
        # A transparent rarity proxy: pair reuse matters only alongside a broader local burst.
        surprisal = math.log1p(max(pair_count - 1, 0) * max(burst - 1, 0) * (1 + shared_types))
        output["refund_velocity_24h"].append(velocity_24h)
        output["refund_velocity_7d"].append(velocity_7d)
        output["merchant_refund_volume_index_30d"].append(merchant_rate)
        output["shared_device_accounts_30d"].append(entity_counts[0])
        output["shared_address_accounts_90d"].append(entity_counts[1])
        output["shared_payment_accounts_30d"].append(entity_counts[2])
        output["shared_identifier_types_30d"].append(shared_types)
        output["device_payment_pair_accounts_30d"].append(pair_count)
        output["linked_refund_burst_72h"].append(burst)
        output["linked_merchants_30d"].append(len(merchant_union | {merchant}))
        output["graph_overlap_surprisal"].append(surprisal)
        for entity, _ in entity_specs:
            entity_events[entity].append((int(now), customer, merchant))
            entity_customers[entity][customer] += 1
            entity_merchants[entity][merchant] += 1
        pair_events[pair].append((int(now), customer))
        pair_customers[pair][customer] += 1
        recent.append(int(now))
        merchant_recent.append(int(now))

    enriched = cases.copy()
    for column, values in output.items():
        enriched.loc[ordered["index"], column] = values
    return enriched


def build_entity_links(cases: pd.DataFrame) -> pd.DataFrame:
    links = []
    for column, entity_type in (("device_id", "device"), ("address_id", "address"), ("payment_token_id", "payment_token")):
        feature = ENTITY_WINDOWS[column][0]
        links.append(
            cases[["case_id", "event_timestamp", column, feature]].rename(columns={column: "entity_id", feature: "linked_accounts"}).assign(entity_type=entity_type)
        )
    return pd.concat(links, ignore_index=True)[["case_id", "event_timestamp", "entity_type", "entity_id", "linked_accounts"]]


def build_dataset(n: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    cases = _inject_scenarios(_base_cases(n, rng), rng)
    cases = apply_point_in_time_features(cases)
    margin_rates = {"fashion": 0.24, "beauty": 0.31, "electronics": 0.12, "home": 0.19, "sports": 0.21, "general": 0.18}
    margin = cases["vertical"].map(margin_rates).to_numpy(float) * cases["refund_amount_inr"].to_numpy(float)
    cases["expected_loss_if_ring_inr"] = np.rint(cases["refund_amount_inr"].to_numpy(float) + margin * 0.55).astype(int)
    cases["review_cost_inr"] = np.where(cases["refund_amount_inr"] > 5000, 140, 65).astype(int)
    cases["false_positive_cost_inr"] = np.rint(cases["refund_amount_inr"] * 0.075 + cases["review_cost_inr"]).astype(int)
    cases["simulator_internal_score"] = np.nan
    cases["is_synthetic"] = 1
    cases["generator_version"] = "1.1.1-audited"
    cases["calibration_note"] = "Olist-inspired commerce context only; Olist has no refund-fraud labels."
    links = build_entity_links(cases)
    reviews = cases.loc[cases["linked_refund_burst_72h"].ge(2), ["case_id", "event_timestamp"]].head(int(n * 0.12)).copy()
    reviews["review_queue"] = "synthetic_capacity_sample"
    return cases, links, reviews


def validate(cases: pd.DataFrame, links: pd.DataFrame, reviews: pd.DataFrame) -> dict[str, Any]:
    timestamps = pd.to_datetime(cases["event_timestamp"], utc=True)
    boundaries = {part: timestamps[cases["split"].eq(part)] for part in ("train", "validation", "test")}
    target_leakage = set(MODEL_FEATURE_COLUMNS) & set(TARGET_ONLY_COLUMNS)
    ring_spans = (
        cases.loc[cases["ring_label"].eq(1)]
        .assign(event_timestamp=lambda frame: pd.to_datetime(frame["event_timestamp"], utc=True, format="mixed"))
        .groupby(["split", "ring_id"])["event_timestamp"]
        .agg(lambda values: (values.max() - values.min()).total_seconds() / 3600)
    )
    checks = {
        "mandatory_columns_present": set(MANDATORY_COLUMNS).issubset(cases.columns),
        "unique_case_ids": bool(cases["case_id"].is_unique),
        "all_rows_marked_synthetic": bool(cases["is_synthetic"].eq(1).all()),
        "one_loss_class_target": set(cases["ring_label"].unique()).issubset({0, 1}),
        "chronological_split": bool(boundaries["train"].max() < boundaries["validation"].min() < boundaries["test"].min()),
        "no_target_columns_in_feature_contract": not bool(target_leakage),
        "three_entity_links_per_case": len(links) == len(cases) * 3,
        "ring_events_have_ids": bool(cases.loc[cases["ring_label"].eq(1), "ring_id"].ne("").all()),
        "benign_shared_identity_negatives": bool(cases["scenario_type"].str.startswith("benign_").any()),
        "test_rings_are_temporally_slower": bool(ring_spans.loc["test"].median() > ring_spans.loc["validation"].median()),
        "test_topology_label_is_honest": bool(cases.loc[cases["split"].eq("test") & cases["ring_label"].eq(1), "ring_topology"].eq("slower_two_identifier").all()),
        "feature_contract_excludes_latent_metadata": not bool(set(MODEL_FEATURE_COLUMNS) & {"ring_id", "scenario_type", "ring_topology", "loss_sequence_position"}),
    }
    return _json_value({
        "passed": all(checks.values()), "checks": checks,
        "rows": {"refund_cases": len(cases), "entity_links": len(links), "review_samples": len(reviews)},
        "target": {"ring_rate": float(cases["ring_label"].mean()), "by_split": cases.groupby("split")["ring_label"].mean().to_dict()},
        "scenarios": cases["scenario_type"].value_counts().to_dict(),
        "feature_contract": MODEL_FEATURE_COLUMNS, "target_only_columns": TARGET_ONLY_COLUMNS,
    })


def write_outputs(cases: pd.DataFrame, links: pd.DataFrame, reviews: pd.DataFrame, output_dir: Path, reports_dir: Path, seed: int) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_gzip_csv(cases, output_dir / "master_refund_cases.csv.gz")
    write_gzip_csv(links, output_dir / "entity_links.csv.gz")
    write_gzip_csv(reviews, output_dir / "analyst_reviews.csv.gz")
    dictionary = []
    for column in cases.columns:
        role = "model_feature" if column in MODEL_FEATURE_COLUMNS else "target_only" if column in TARGET_ONLY_COLUMNS else "metadata"
        dictionary.append({"column": column, "role": role, "model_input": column in MODEL_FEATURE_COLUMNS})
    pd.DataFrame(dictionary).to_csv(output_dir / "data_dictionary.csv", index=False)
    validation = validate(cases, links, reviews)
    (reports_dir / "validation_report.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")
    manifest = {
        "dataset_name": "MarginShield Coordinated Refund-Abuse Ring Benchmark",
        "version": "1.1.1-audited", "seed": seed, "target": "ring_label",
        "target_definition": "A refund request generated as part of a coordinated multi-account refund-abuse event sequence.",
        "all_output_rows_synthetic": True,
        "source_statement": "Olist has no refund-fraud labels. It informs only high-level commerce context; no Olist source row or label is used.",
        "limitations": [
            "This is a synthetic benchmark, not real-world or production performance evidence.",
            "True refund-abuse labels require merchant investigation outcomes, disputed refunds and chargeback evidence.",
            "The test rings unfold more slowly than train/validation rings, but retain the same two-identifier graph mechanism.",
        ],
        "validation_passed": validation["passed"],
    }
    (output_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return validation


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MarginShield's ring-native synthetic benchmark")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reports-dir", type=Path, default=DEFAULT_REPORTS)
    parser.add_argument("--rows", type=int, default=75_000)
    parser.add_argument("--seed", type=int, default=20260904)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.rows < 20_000:
        raise ValueError("Use at least 20,000 rows for stable temporal ring evaluation")
    cases, links, reviews = build_dataset(args.rows, args.seed)
    report = write_outputs(cases, links, reviews, args.output_dir, args.reports_dir, args.seed)
    print(json.dumps({"rows": len(cases), "ring_rate": report["target"]["ring_rate"], "validation_passed": report["passed"]}, indent=2))


if __name__ == "__main__":
    main()
