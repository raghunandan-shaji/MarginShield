from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter, defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from marginshield.feature_engine import apply_features_via_engine, datetime_seconds


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
    "shared_payment_accounts_30d",
    "shared_identifier_types_30d",
    "device_payment_pair_accounts_30d",
    "connected_accounts_max_window",
    "multi_identifier_neighbor_accounts",
    "identifier_reuse_balance",
    "linked_refund_burst_72h",
    "linked_merchants_30d",
    "linked_same_product_accounts_30d",
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
    "cluster_address_mode",
    "cluster_merchant_mode",
    "cluster_spacing_mode",
    "cluster_product_mode",
    "cluster_existing_customer_member",
]

ENTITY_WINDOWS = {
    "device_id": ("shared_device_accounts_30d", 30),
    "address_id": ("shared_address_accounts_90d", 90),
    "payment_token_id": ("shared_payment_accounts_30d", 30),
}

MANDATORY_COLUMNS = [
    "case_id", "event_timestamp", "split", "merchant_id", "customer_id",
    "device_id", "address_id", "payment_token_id", "product_id", "ring_label", "is_synthetic",
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
        "train": (start, start + pd.Timedelta(days=147)),
        "validation": (start + pd.Timedelta(days=147), start + pd.Timedelta(days=178)),
        "test": (start + pd.Timedelta(days=178), start + pd.Timedelta(days=210)),
    }


def _new_entity(prefix: str, namespace: str) -> str:
    return _hash_id(prefix, namespace)


def _base_cases(n: int, rng: np.random.Generator) -> pd.DataFrame:
    """Create unlabeled refund requests before any scenario injection.

    The context distributions are Olist-inspired, but Olist does not contain
    refund-fraud labels and contributes no source rows to this benchmark.
    """
    start = pd.Timestamp("2025-01-01", tz="UTC")
    seconds = np.sort(rng.integers(0, 210 * 86_400, size=n))
    timestamps = start + pd.to_timedelta(seconds, unit="s")
    split = np.where(seconds < 147 * 86_400, "train", np.where(seconds < 178 * 86_400, "validation", "test"))

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
    cases["product_id"] = [
        _new_entity("PRD", f"{item_vertical}-{int(item)}")
        for item_vertical, item in zip(vertical, rng.integers(0, 1_200, size=n))
    ]
    # Ordinary repeat customers retain identifiers; a minority rotate one identifier.
    # This creates organic account history without creating cross-account links.
    cases["device_id"] = [_new_entity("DEV", f"customer-device-{customer}") for customer in customer_ids]
    cases["address_id"] = [_new_entity("ADR", f"customer-address-{customer}") for customer in customer_ids]
    cases["payment_token_id"] = [_new_entity("PAY", f"customer-payment-{customer}") for customer in customer_ids]
    for column, prefix, share in (
        ("device_id", "DEV", 0.12), ("address_id", "ADR", 0.06), ("payment_token_id", "PAY", 0.10)
    ):
        rotated = rng.choice(n, size=int(n * share), replace=False)
        cases.loc[rotated, column] = [_new_entity(prefix, f"rotated-{column}-{index}") for index in rotated]
    # Incidental legitimate sharing exists outside explicit clusters. Shared identity
    # is therefore a noisy observation, never a proxy for scenario injection.
    for column, share in (("device_id", 0.025), ("address_id", 0.020), ("payment_token_id", 0.006)):
        borrowers = rng.choice(n, size=int(n * share), replace=False)
        lenders = rng.choice(n, size=len(borrowers), replace=True)
        cases.loc[borrowers, column] = cases.loc[lenders, column].to_numpy()
    cases["ring_label"] = 0
    cases["ring_id"] = ""
    cases["scenario_type"] = "independent_legitimate"
    cases["ring_topology"] = "none"
    cases["loss_sequence_position"] = -1
    cases["cluster_address_mode"] = "none"
    cases["cluster_merchant_mode"] = "none"
    cases["cluster_spacing_mode"] = "none"
    cases["cluster_product_mode"] = "none"
    cases["cluster_existing_customer_member"] = 0
    return cases


def _scenario_rows(
    cases: pd.DataFrame,
    split: str,
    count: int,
    rng: np.random.Generator,
    available: np.ndarray | None = None,
    min_size: int = 5,
    max_size: int = 9,
) -> list[np.ndarray]:
    if available is None:
        available = cases.index[cases["split"].eq(split)].to_numpy()
    selected = _choice_without_replacement(rng, available, count)
    rng.shuffle(selected)
    groups: list[np.ndarray] = []
    cursor = 0
    while cursor < len(selected):
        size = int(rng.integers(min_size, max_size + 1))
        group = selected[cursor:cursor + size]
        if len(group) >= min_size:
            groups.append(group)
        cursor += size
    return groups


def _inject_cluster(
    cases: pd.DataFrame,
    rows: np.ndarray,
    split: str,
    cluster_id: str,
    topology: str,
    is_ring: bool,
    rng: np.random.Generator,
    merchant_values: np.ndarray,
    merchant_vertical: dict[str, str],
) -> None:
    """Inject a graph-and-time event mechanism, never a final-feature formula."""
    bounds = _split_bounds()[split]
    count = len(rows)
    address_mode = str(rng.choice(["unique", "pairs", "hub"]))
    merchant_mode = str(rng.choice(["single", "pool", "spread"]))
    spacing_mode = str(rng.choice(["uniform", "bursty", "exponential"]))
    product_mode = str(rng.choice(
        ["same", "pairs", "spread"],
        p=[0.50, 0.30, 0.20] if is_ring else [0.25, 0.30, 0.45],
    ))
    existing_customer_share = float(rng.uniform(0.10, 0.60))
    if is_ring:
        horizon_hours = float(rng.uniform(96, 264) if split == "test" else rng.uniform(18, 144))
    else:
        horizon_hours = float(rng.uniform(18, 240))
    usable_seconds = int((bounds[1] - bounds[0]).total_seconds() - horizon_hours * 3600 - 2 * 86_400)
    start = bounds[0] + pd.Timedelta(seconds=int(rng.integers(86_400, max(86_401, usable_seconds))))
    if spacing_mode == "uniform":
        offsets = np.linspace(0.0, horizon_hours, count)
    elif spacing_mode == "bursty":
        offsets = np.sort(rng.beta(0.6, 0.6, count)) * horizon_hours
    else:
        raw = np.cumsum(rng.exponential(max(horizon_hours / count, 0.1), count))
        offsets = (raw - raw.min()) / max(raw.max() - raw.min(), 1e-9) * horizon_hours
    times = [start + pd.Timedelta(hours=float(offset)) for offset in offsets]

    devices = [_new_entity("DEV", f"{cluster_id}-device-{index}") for index in range(count)]
    payments = [_new_entity("PAY", f"{cluster_id}-payment-{index}") for index in range(count)]
    addresses = [_new_entity("ADR", f"{cluster_id}-address-{index}") for index in range(count)]

    def repeated(prefix: str, name: str) -> str:
        return _new_entity(prefix, f"{cluster_id}-{name}")

    if topology == "device_hub_rotating_payment":
        devices = [repeated("DEV", "device-hub")] * count
        payments = [repeated("PAY", f"payment-pair-{index // 2}") for index in range(count)]
        addresses = [repeated("ADR", f"address-pair-{index // 2}") for index in range(count)]
    elif topology == "payment_hub_rotating_device":
        payments = [repeated("PAY", "payment-hub")] * count
        devices = [repeated("DEV", f"device-pair-{index // 2}") for index in range(count)]
        addresses = [repeated("ADR", f"address-pair-{index // 2}") for index in range(count)]
    elif topology == "alternating_device_payment_chain":
        devices = [repeated("DEV", f"chain-device-{index // 2}") for index in range(count)]
        payments = [repeated("PAY", f"chain-payment-{(index + 1) // 2}") for index in range(count)]
    elif topology == "two_core_bridge":
        midpoint = count // 2
        devices[:midpoint + 1] = [repeated("DEV", "left-core")] * (midpoint + 1)
        payments[midpoint:] = [repeated("PAY", "right-core")] * (count - midpoint)
    elif topology == "sparse_address_payment_chain":
        addresses = [repeated("ADR", f"chain-address-{index // 2}") for index in range(count)]
        payments = [repeated("PAY", f"chain-payment-{(index + 1) // 2}") for index in range(count)]
    elif topology == "device_address_ladder":
        devices = [repeated("DEV", f"ladder-device-{index // 2}") for index in range(count)]
        addresses = [repeated("ADR", f"ladder-address-{(index + 1) // 2}") for index in range(count)]
    elif topology == "rotating_two_hub_bridge":
        midpoint = count // 2
        devices = [repeated("DEV", "left-device-hub" if index < midpoint else "right-device-hub") for index in range(count)]
        payments = [repeated("PAY", f"cross-payment-hub-{index % 2}") for index in range(count)]
    elif topology == "partial_pair_mesh":
        devices = [repeated("DEV", f"mesh-device-{index % 2}") for index in range(count)]
        payments = [repeated("PAY", f"mesh-payment-{(index + index // 3) % 3}") for index in range(count)]
        addresses = [repeated("ADR", f"mesh-address-{index // 2}") for index in range(count)]
    elif topology == "staggered_device_address_bridge":
        devices = [repeated("DEV", f"staggered-device-{index // 3}") for index in range(count)]
        addresses = [repeated("ADR", f"staggered-address-{(index + 1) // 3}") for index in range(count)]
        payments = [repeated("PAY", f"staggered-payment-{index % 2}") for index in range(count)]
    elif topology == "rotating_identifier_cycle":
        # Deliberately asymmetric cycle: unlike the development motifs, its three
        # entity layers do not share the same degree signature.
        devices = [repeated("DEV", f"cycle-device-{index % 2}") for index in range(count)]
        payments = [repeated("PAY", f"cycle-payment-{(index + 1) % 3}") for index in range(count)]
        addresses = [repeated("ADR", f"cycle-address-{(index + 2) % 4}") for index in range(count)]
    elif topology == "token_fan_address_pairs":
        payments = [repeated("PAY", f"fan-payment-{0 if index < (count * 2) // 3 else 1}") for index in range(count)]
        devices = [repeated("DEV", f"fan-device-{index // 2}") for index in range(count)]
        addresses = [repeated("ADR", f"fan-address-{index // 2}") for index in range(count)]
    elif topology == "three_core_sparse_bridge":
        devices = [repeated("DEV", f"three-core-device-{index % 3}") for index in range(count)]
        payments = [repeated("PAY", f"three-core-payment-{index // 3}") for index in range(count)]
        addresses = [repeated("ADR", f"three-core-address-{(index + 1) // 2}") for index in range(count)]
    elif topology == "family_wallet":
        devices = [repeated("DEV", f"family-device-{index}") for index in range(count)]
        payments = [repeated("PAY", "family-payment")] * count
    elif topology == "group_purchase":
        devices = [repeated("DEV", "group-device")] * count
        payments = [repeated("PAY", "group-payment")] * count
        addresses = [repeated("ADR", "group-address")] * count
    elif topology == "corporate_card":
        devices = [repeated("DEV", f"corporate-device-{index}") for index in range(count)]
        payments = [repeated("PAY", "corporate-payment")] * count
    elif topology == "hostel_kiosk":
        devices = [repeated("DEV", "hostel-kiosk")] * count
        addresses = [repeated("ADR", "hostel-address")] * count
    elif topology == "office_network":
        devices = [repeated("DEV", f"office-device-{index % 2}") for index in range(count)]
        addresses = [repeated("ADR", "office-address")] * count
    elif topology == "support_migration":
        devices = [repeated("DEV", "support-device")] * count
        payments = [repeated("PAY", f"support-payment-{index // 2}") for index in range(count)]
    else:
        raise ValueError(f"Unknown simulator topology: {topology}")

    address_defined = topology in {
        "sparse_address_payment_chain", "device_address_ladder", "partial_pair_mesh",
        "staggered_device_address_bridge", "rotating_identifier_cycle", "token_fan_address_pairs",
        "three_core_sparse_bridge",
        "hostel_kiosk", "office_network",
    }
    if topology in {"hostel_kiosk", "office_network"}:
        address_mode = "hub"
    elif topology in {"sparse_address_payment_chain", "device_address_ladder", "partial_pair_mesh"}:
        address_mode = "pairs"
    if not address_defined:
        if address_mode == "unique":
            addresses = [_new_entity("ADR", f"{cluster_id}-context-address-{index}") for index in range(count)]
        elif address_mode == "pairs":
            addresses = [repeated("ADR", f"context-address-pair-{index // 2}") for index in range(count)]
        else:
            addresses = [repeated("ADR", "context-address-hub")] * count

    selected_merchants = rng.choice(merchant_values, size=3, replace=False)
    if topology == "group_purchase":
        product_mode = "same"
    products = [str(cases.loc[row, "product_id"]) for row in rows]
    if product_mode == "same":
        products = [_new_entity("PRD", f"{cluster_id}-shared-product")] * count
    elif product_mode == "pairs":
        products = [_new_entity("PRD", f"{cluster_id}-product-pair-{position // 2}") for position in range(count)]
    used_customers: set[str] = set()
    for position, row in enumerate(rows):
        existing_customer = bool(rng.random() < existing_customer_share)
        customer_id = str(cases.loc[row, "customer_id"]) if existing_customer else _new_entity("CUS", f"{cluster_id}-customer-{position}")
        if customer_id in used_customers:
            customer_id = _new_entity("CUS", f"{cluster_id}-customer-{position}")
            existing_customer = False
        used_customers.add(customer_id)
        cases.loc[row, "customer_id"] = customer_id
        cases.loc[row, "device_id"] = devices[position]
        cases.loc[row, "payment_token_id"] = payments[position]
        cases.loc[row, "address_id"] = addresses[position]
        cases.loc[row, "product_id"] = products[position]
        cases.loc[row, "event_timestamp"] = times[position]
        if merchant_mode != "spread":
            merchant_index = 0 if merchant_mode == "single" else position % 2
            merchant_id = str(selected_merchants[merchant_index])
            cases.loc[row, "merchant_id"] = merchant_id
            cases.loc[row, "vertical"] = merchant_vertical[merchant_id]
        cases.loc[row, "ring_label"] = int(is_ring)
        cases.loc[row, "ring_id"] = cluster_id if is_ring else ""
        cases.loc[row, "scenario_type"] = "coordinated_refund_abuse" if is_ring else f"benign_{topology}"
        cases.loc[row, "ring_topology"] = topology
        cases.loc[row, "loss_sequence_position"] = position if is_ring else -1
        cases.loc[row, "cluster_address_mode"] = address_mode
        cases.loc[row, "cluster_merchant_mode"] = merchant_mode
        cases.loc[row, "cluster_spacing_mode"] = spacing_mode
        cases.loc[row, "cluster_product_mode"] = product_mode
        cases.loc[row, "cluster_existing_customer_member"] = int(existing_customer)

    # A shared token must use a coherent rail. Unique tokens keep their original
    # context distribution, including COD.
    token_positions: dict[str, list[int]] = defaultdict(list)
    for position, token in enumerate(payments):
        token_positions[token].append(position)
    for positions in token_positions.values():
        if len(positions) < 2:
            continue
        first_method = str(cases.loc[rows[positions[0]], "payment_method"])
        rail = first_method if first_method != "cod" else str(rng.choice(["upi", "card", "wallet", "netbanking"]))
        for position in positions:
            cases.loc[rows[position], "payment_method"] = rail


def _inject_scenarios(cases: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Inject split-local motifs from independent deterministic random streams."""
    ring_rows = {"train": 1_050, "validation": 265, "test": 265}
    benign_rows = {"train": 6_600, "validation": 1_650, "test": 1_650}
    development_topologies = (
        "device_hub_rotating_payment", "payment_hub_rotating_device",
        "alternating_device_payment_chain", "two_core_bridge",
    )
    # V4 final-test motifs are distinct from development and from the previously
    # examined V3 final test. This list is fixed before V4 model fitting.
    test_topologies = (
        "staggered_device_address_bridge", "rotating_identifier_cycle",
        "token_fan_address_pairs", "three_core_sparse_bridge",
    )
    benign_topologies = (
        "family_wallet", "group_purchase", "corporate_card",
        "hostel_kiosk", "office_network", "support_migration",
    )
    merchant_values = cases["merchant_id"].drop_duplicates().to_numpy()
    merchant_vertical = (
        cases.groupby("merchant_id")["vertical"]
        .agg(lambda values: values.value_counts().index[0])
        .to_dict()
    )
    ring_counter = 1
    benign_counter = 1
    for split_number, split in enumerate(("train", "validation", "test"), start=1):
        rng = np.random.default_rng(np.random.SeedSequence([seed, 10_000 + split_number]))
        ring_groups = _scenario_rows(cases, split, ring_rows[split], rng)
        ring_topologies = test_topologies if split == "test" else development_topologies
        for number, rows in enumerate(ring_groups):
            topology = ring_topologies[number % len(ring_topologies)]
            _inject_cluster(
                cases, rows, split, f"RING-{ring_counter:05d}", topology, True,
                rng, merchant_values, merchant_vertical,
            )
            ring_counter += 1
        benign_available = cases.index[cases["split"].eq(split) & cases["ring_label"].eq(0)].to_numpy()
        benign_groups = _scenario_rows(cases, split, benign_rows[split], rng, benign_available)
        for number, rows in enumerate(benign_groups):
            topology = benign_topologies[number % len(benign_topologies)]
            _inject_cluster(
                cases, rows, split, f"BENIGN-{benign_counter:05d}", topology, False,
                rng, merchant_values, merchant_vertical,
            )
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
    event_seconds = datetime_seconds(ordered["event_timestamp"]).to_numpy()
    for entity_column, (feature_column, window_days) in ENTITY_WINDOWS.items():
        values = _rolling_distinct_accounts(
            ordered[entity_column].to_numpy(), ordered["customer_id"].to_numpy(), event_seconds, window_days
        )
        updated.loc[ordered["index"], feature_column] = values
    return updated


def apply_point_in_time_features(cases: pd.DataFrame) -> pd.DataFrame:
    """Build model features through the same engine used by live scoring."""
    return apply_features_via_engine(cases)[0]


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
    cases = _inject_scenarios(_base_cases(n, rng), seed)
    cases = apply_point_in_time_features(cases)
    margin_rates = {"fashion": 0.24, "beauty": 0.31, "electronics": 0.12, "home": 0.19, "sports": 0.21, "general": 0.18}
    margin = cases["vertical"].map(margin_rates).to_numpy(float) * cases["refund_amount_inr"].to_numpy(float)
    cases["expected_loss_if_ring_inr"] = np.rint(cases["refund_amount_inr"].to_numpy(float) + margin * 0.55).astype(int)
    cases["review_cost_inr"] = np.where(cases["refund_amount_inr"] > 5000, 140, 65).astype(int)
    cases["false_positive_cost_inr"] = np.rint(cases["refund_amount_inr"] * 0.075 + cases["review_cost_inr"]).astype(int)
    cases["simulator_internal_score"] = np.nan
    cases["is_synthetic"] = 1
    cases["generator_version"] = "4.0.0-value-policy-locked"
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
    positive = cases.loc[cases["ring_label"].eq(1)]
    development_topologies = set(positive.loc[positive["split"].isin(["train", "validation"]), "ring_topology"])
    test_topologies = set(positive.loc[positive["split"].eq("test"), "ring_topology"])

    def structure_fingerprint(group: pd.DataFrame) -> tuple[tuple[int, ...], ...]:
        """Relation-aware degree signature of the generated account-entity graph."""
        signatures = []
        for column in ("device_id", "address_id", "payment_token_id"):
            degrees = group.groupby(column)["customer_id"].nunique()
            signatures.append(tuple(sorted((int(value) for value in degrees), reverse=True)))
        return tuple(signatures)

    development_fingerprints = {
        structure_fingerprint(group)
        for _, group in positive.loc[positive["split"].isin(["train", "validation"])].groupby("ring_id")
    }
    test_fingerprints = {
        structure_fingerprint(group)
        for _, group in positive.loc[positive["split"].eq("test")].groupby("ring_id")
    }
    validation = cases.loc[cases["split"].eq("validation")]
    validation_pair_reuse = validation["device_payment_pair_accounts_30d"].ge(2)
    validation_pair_positive = int((validation_pair_reuse & validation["ring_label"].eq(1)).sum())
    validation_pair_legitimate = int((validation_pair_reuse & validation["ring_label"].eq(0)).sum())
    pair_rule_precision = float(validation.loc[validation_pair_reuse, "ring_label"].mean()) if validation_pair_reuse.any() else 0.0
    hard_negative_topologies = {
        value.removeprefix("benign_")
        for value in cases.loc[cases["scenario_type"].str.startswith("benign_"), "scenario_type"].unique()
    }

    def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
        """Dependency-light exact AP used by the data-generation gate."""
        order = np.argsort(-scores, kind="stable")
        ordered_labels = labels[order]
        ordered_scores = scores[order]
        positives = int(ordered_labels.sum())
        if not positives:
            return 0.0
        cumulative = np.cumsum(ordered_labels)
        counts = np.arange(1, len(labels) + 1)
        boundaries = np.r_[ordered_scores[:-1] != ordered_scores[1:], True]
        recall = cumulative[boundaries] / positives
        precision = cumulative[boundaries] / counts[boundaries]
        return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))

    shared_mask = validation[[
        "shared_device_accounts_30d", "shared_address_accounts_90d", "shared_payment_accounts_30d",
    ]].max(axis=1).ge(2)
    shared_validation = validation.loc[shared_mask]
    shared_labels = shared_validation["ring_label"].to_numpy(int)
    shared_prevalence = float(shared_labels.mean()) if len(shared_labels) else 0.0
    univariate_shared = []
    for feature in MODEL_FEATURE_COLUMNS:
        if not pd.api.types.is_numeric_dtype(shared_validation[feature]) or not len(shared_validation):
            continue
        values = shared_validation[feature].to_numpy(float)
        high = average_precision(shared_labels, values)
        low = average_precision(shared_labels, -values)
        univariate_shared.append({
            "feature": feature,
            "best_pr_auc": max(high, low),
            "direction": "high" if high >= low else "low",
        })
    univariate_shared.sort(key=lambda row: row["best_pr_auc"], reverse=True)
    max_shared_ap = float(univariate_shared[0]["best_pr_auc"]) if univariate_shared else 0.0
    shared_ap_ratio = max_shared_ap / shared_prevalence if shared_prevalence else float("inf")
    independent = validation.loc[validation["scenario_type"].eq("independent_legitimate")]
    independent_sharing = int(independent[[
        "shared_device_accounts_30d", "shared_address_accounts_90d", "shared_payment_accounts_30d",
    ]].max(axis=1).ge(2).sum())
    any_shared_precision = float(shared_validation["ring_label"].mean()) if len(shared_validation) else 0.0

    injected_validation = validation.loc[validation["cluster_address_mode"].ne("none")]
    nuisance_coverage: dict[str, dict[str, dict[str, int]]] = {}
    nuisance_expected = {
        "cluster_address_mode": {"unique", "pairs", "hub"},
        "cluster_merchant_mode": {"single", "pool", "spread"},
        "cluster_spacing_mode": {"uniform", "bursty", "exponential"},
        "cluster_product_mode": {"same", "pairs", "spread"},
    }
    nuisance_parity = True
    for column, expected in nuisance_expected.items():
        nuisance_coverage[column] = {}
        for value in sorted(expected):
            counts = {
                "positive": int((injected_validation[column].eq(value) & injected_validation["ring_label"].eq(1)).sum()),
                "negative": int((injected_validation[column].eq(value) & injected_validation["ring_label"].eq(0)).sum()),
            }
            nuisance_coverage[column][value] = counts
            nuisance_parity &= counts["positive"] > 0 and counts["negative"] > 0

    positive_payment_methods = set(validation.loc[validation["ring_label"].eq(1), "payment_method"])
    observed_payment_methods = set(validation["payment_method"])
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
        "test_topology_names_are_disjoint": development_topologies.isdisjoint(test_topologies),
        "test_relation_graph_fingerprints_are_disjoint": development_fingerprints.isdisjoint(test_fingerprints),
        "four_development_and_test_ring_motifs": len(development_topologies) >= 4 and len(test_topologies) >= 4,
        "six_hard_negative_motifs": len(hard_negative_topologies) >= 6,
        "legitimate_pair_reuse_exceeds_positive_pair_reuse": validation_pair_legitimate >= validation_pair_positive,
        "validation_pair_rule_below_70pct_precision": pair_rule_precision < 0.70,
        "shared_identifier_rule_below_50pct_precision": any_shared_precision < 0.50,
        "independent_legitimate_rows_include_shared_identity": independent_sharing >= 25,
        "within_shared_univariate_ap_below_2x_prevalence": shared_ap_ratio < 2.0,
        "nuisance_modes_have_both_labels": bool(nuisance_parity),
        "positive_payment_methods_cover_validation_context": observed_payment_methods.issubset(positive_payment_methods),
        "feature_contract_excludes_latent_metadata": not bool(set(MODEL_FEATURE_COLUMNS) & {"ring_id", "scenario_type", "ring_topology", "loss_sequence_position"}),
    }
    return _json_value({
        "passed": all(checks.values()), "checks": checks,
        "rows": {"refund_cases": len(cases), "entity_links": len(links), "review_samples": len(reviews)},
        "target": {"ring_rate": float(cases["ring_label"].mean()), "by_split": cases.groupby("split")["ring_label"].mean().to_dict()},
        "scenarios": cases["scenario_type"].value_counts().to_dict(),
        "topologies": cases["ring_topology"].value_counts().to_dict(),
        "shortcut_gate": {
            "validation_pair_rule_precision": pair_rule_precision,
            "positive_pair_reuse_rows": validation_pair_positive,
            "legitimate_pair_reuse_rows": validation_pair_legitimate,
            "shared_identifier_rule_precision": any_shared_precision,
            "shared_subset_rows": len(shared_validation),
            "shared_subset_prevalence": shared_prevalence,
            "maximum_within_shared_univariate_pr_auc": max_shared_ap,
            "maximum_within_shared_univariate_ap_lift": shared_ap_ratio,
            "within_shared_univariate_features": univariate_shared,
            "independent_legitimate_shared_identity_rows": independent_sharing,
            "nuisance_mode_coverage": nuisance_coverage,
            "positive_payment_methods": sorted(positive_payment_methods),
        },
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
        "version": "4.0.0-value-policy-locked", "seed": seed, "target": "ring_label",
        "target_definition": "A refund request generated as part of a coordinated multi-account refund-abuse event sequence.",
        "all_output_rows_synthetic": True,
        "source_statement": "Olist has no refund-fraud labels. It informs only high-level commerce context; no Olist source row or label is used.",
        "limitations": [
            "This is a synthetic benchmark, not real-world or production performance evidence.",
            "True refund-abuse labels require merchant investigation outcomes, disputed refunds and chargeback evidence.",
            "The final test uses slower, structurally disjoint graph motifs, but remains synthetic.",
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
    parser.add_argument("--seed", type=int, default=20260905)
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
