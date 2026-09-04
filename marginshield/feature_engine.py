"""Incremental point-in-time features shared by offline training and live scoring."""

from __future__ import annotations

import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from typing import Any

import pandas as pd


DAY = 86_400
ENTITY_WINDOWS = {"device": 30 * DAY, "address": 90 * DAY, "payment_token": 30 * DAY}
PAIR_WINDOW = 30 * DAY
VELOCITY_WINDOW = 7 * DAY
MERCHANT_WINDOW = 30 * DAY
BURST_WINDOW = 72 * 3_600
MERCHANT_INDEX_DENOMINATOR = 300.0
CONTEXT_FEATURES = ("vertical", "payment_method", "refund_amount_inr", "refund_share", "account_age_days")


@dataclass(frozen=True)
class RawRefundEvent:
    case_id: str
    event_seconds: int
    customer_id: str
    merchant_id: str
    device_id: str
    address_id: str
    payment_token_id: str
    product_id: str
    vertical: str
    payment_method: str
    refund_amount_inr: float
    refund_share: float
    account_age_days: float


def datetime_seconds(values: pd.Series) -> pd.Series:
    """Convert datetimes to Unix seconds without assuming the pandas dtype unit."""
    parsed = pd.to_datetime(values, utc=True, format="mixed")
    return parsed.dt.as_unit("s").astype("int64")


def event_from_row(row: Any, event_seconds: int) -> RawRefundEvent:
    get = (lambda name: row[name]) if isinstance(row, dict) else (lambda name: getattr(row, name))
    return RawRefundEvent(
        case_id=str(get("case_id")),
        event_seconds=int(event_seconds),
        customer_id=str(get("customer_id")),
        merchant_id=str(get("merchant_id")),
        device_id=str(get("device_id")),
        address_id=str(get("address_id")),
        payment_token_id=str(get("payment_token_id")),
        product_id=str(get("product_id")),
        vertical=str(get("vertical")),
        payment_method=str(get("payment_method")),
        refund_amount_inr=float(get("refund_amount_inr")),
        refund_share=float(get("refund_share")),
        account_age_days=float(get("account_age_days")),
    )


class PointInTimeFeatureEngine:
    """Chronological entity state. Features are computed before the event is committed."""

    def __init__(self) -> None:
        self.entity_events: dict[str, deque[tuple[int, str, str, str, str]]] = defaultdict(deque)
        self.entity_customers: dict[str, Counter[str]] = defaultdict(Counter)
        self.entity_merchants: dict[str, Counter[str]] = defaultdict(Counter)
        self.pair_events: dict[tuple[str, str], deque[tuple[int, str]]] = defaultdict(deque)
        self.pair_customers: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
        self.customer_events: dict[str, deque[int]] = defaultdict(deque)
        self.merchant_events: dict[str, deque[int]] = defaultdict(deque)
        self.last_seconds: int | None = None
        self.events_seen = 0

    @staticmethod
    def _entity_key(entity_type: str, identifier: str) -> str:
        return f"{entity_type}:{identifier}"

    @staticmethod
    def _entity_specs(event: RawRefundEvent) -> list[tuple[str, str, int]]:
        return [
            ("device", str(event.device_id), ENTITY_WINDOWS["device"]),
            ("address", str(event.address_id), ENTITY_WINDOWS["address"]),
            ("payment_token", str(event.payment_token_id), ENTITY_WINDOWS["payment_token"]),
        ]

    def _guard_order(self, event: RawRefundEvent) -> None:
        if self.last_seconds is not None and event.event_seconds < self.last_seconds:
            raise ValueError(
                f"Out-of-order event {event.case_id}: {event.event_seconds} < {self.last_seconds}"
            )

    def _expire_entity(self, key: str, now: int, window: int) -> None:
        queue = self.entity_events[key]
        while queue and queue[0][0] < now - window:
            _, customer, merchant, _, _ = queue.popleft()
            self.entity_customers[key][customer] -= 1
            if not self.entity_customers[key][customer]:
                del self.entity_customers[key][customer]
            self.entity_merchants[key][merchant] -= 1
            if not self.entity_merchants[key][merchant]:
                del self.entity_merchants[key][merchant]

    def _expire_pair(self, pair: tuple[str, str], now: int) -> None:
        queue = self.pair_events[pair]
        while queue and queue[0][0] < now - PAIR_WINDOW:
            _, customer = queue.popleft()
            self.pair_customers[pair][customer] -= 1
            if not self.pair_customers[pair][customer]:
                del self.pair_customers[pair][customer]

    def _expire_customer(self, customer: str, now: int) -> None:
        queue = self.customer_events[customer]
        while queue and queue[0] < now - VELOCITY_WINDOW:
            queue.popleft()

    def _expire_merchant(self, merchant: str, now: int) -> None:
        queue = self.merchant_events[merchant]
        while queue and queue[0] < now - MERCHANT_WINDOW:
            queue.popleft()

    def features(self, event: RawRefundEvent) -> dict[str, Any]:
        self._guard_order(event)
        now = int(event.event_seconds)
        customer = str(event.customer_id)
        merchant = str(event.merchant_id)
        entity_counts: list[int] = []
        entity_customer_sets: list[set[str]] = []
        merchant_union: set[str] = set()
        burst_counts: list[int] = []
        same_product_customers: set[str] = {customer}

        for entity_type, identifier, window in self._entity_specs(event):
            key = self._entity_key(entity_type, identifier)
            self._expire_entity(key, now, window)
            entity_counts.append(len(self.entity_customers[key] | Counter({customer: 1})))
            entity_customer_sets.append(set(self.entity_customers[key]) | {customer})
            merchant_union.update(self.entity_merchants[key])
            burst_counts.append(sum(stamp >= now - BURST_WINDOW for stamp, _, _, _, _ in self.entity_events[key]))
            same_product_customers.update(
                prior_customer
                for stamp, prior_customer, _, product, _ in self.entity_events[key]
                if stamp >= now - PAIR_WINDOW and product == str(event.product_id)
            )

        pair = (str(event.device_id), str(event.payment_token_id))
        self._expire_pair(pair, now)
        pair_count = len(self.pair_customers[pair] | Counter({customer: 1}))

        self._expire_customer(customer, now)
        customer_history = self.customer_events[customer]
        velocity_7d = len(customer_history) + 1
        velocity_24h = sum(stamp >= now - DAY for stamp in customer_history) + 1

        self._expire_merchant(merchant, now)
        merchant_index = min(len(self.merchant_events[merchant]) / MERCHANT_INDEX_DENOMINATOR, 1.0)

        shared_types = sum(count >= 2 for count in entity_counts)
        burst = max(burst_counts, default=0) + 1
        connected_customers = set().union(*entity_customer_sets)
        overlap = Counter(
            linked
            for linked_set in entity_customer_sets
            for linked in linked_set
            if linked != customer
        )
        multi_identifier_neighbors = sum(count >= 2 for count in overlap.values())
        ordered_counts = sorted(entity_counts, reverse=True)
        reuse_balance = ordered_counts[1] / ordered_counts[0] if ordered_counts[0] > 1 else 0.0
        surprisal = math.log1p(
            max(len(connected_customers) - 1, 0) * max(burst - 1, 0)
            + multi_identifier_neighbors * (1 + shared_types)
        )
        return {
            "vertical": event.vertical,
            "payment_method": event.payment_method,
            "refund_amount_inr": event.refund_amount_inr,
            "refund_share": event.refund_share,
            "account_age_days": event.account_age_days,
            "refund_velocity_24h": velocity_24h,
            "refund_velocity_7d": velocity_7d,
            "merchant_refund_volume_index_30d": merchant_index,
            "shared_device_accounts_30d": entity_counts[0],
            "shared_address_accounts_90d": entity_counts[1],
            "shared_payment_accounts_30d": entity_counts[2],
            "shared_identifier_types_30d": shared_types,
            "device_payment_pair_accounts_30d": pair_count,
            "connected_accounts_max_window": len(connected_customers),
            "multi_identifier_neighbor_accounts": multi_identifier_neighbors,
            "identifier_reuse_balance": reuse_balance,
            "linked_refund_burst_72h": burst,
            "linked_merchants_30d": len(merchant_union | {merchant}),
            "linked_same_product_accounts_30d": len(same_product_customers),
            "graph_overlap_surprisal": surprisal,
        }

    def linked_state(self, event: RawRefundEvent) -> dict[str, dict[str, list[str]]]:
        linked: dict[str, dict[str, list[str]]] = {}
        for entity_type, identifier, _ in self._entity_specs(event):
            key = self._entity_key(entity_type, identifier)
            linked[entity_type] = {
                "customers": sorted(self.entity_customers[key]),
                "cases": sorted({case_id for _, _, _, _, case_id in self.entity_events[key]}),
            }
        return linked

    def commit(self, event: RawRefundEvent) -> None:
        self._guard_order(event)
        now = int(event.event_seconds)
        customer = str(event.customer_id)
        merchant = str(event.merchant_id)
        for entity_type, identifier, _ in self._entity_specs(event):
            key = self._entity_key(entity_type, identifier)
            self.entity_events[key].append((now, customer, merchant, str(event.product_id), str(event.case_id)))
            self.entity_customers[key][customer] += 1
            self.entity_merchants[key][merchant] += 1
        pair = (str(event.device_id), str(event.payment_token_id))
        self.pair_events[pair].append((now, customer))
        self.pair_customers[pair][customer] += 1
        self.customer_events[customer].append(now)
        self.merchant_events[merchant].append(now)
        self.last_seconds = now
        self.events_seen += 1

    def observe(self, event: RawRefundEvent) -> dict[str, Any]:
        computed = self.features(event)
        self.commit(event)
        return computed


def apply_features_via_engine(
    cases: pd.DataFrame, engine: PointInTimeFeatureEngine | None = None
) -> tuple[pd.DataFrame, PointInTimeFeatureEngine]:
    engine = engine or PointInTimeFeatureEngine()
    ordered = cases.sort_values(["event_timestamp", "case_id"], kind="stable").reset_index()
    seconds = datetime_seconds(ordered["event_timestamp"]).to_numpy()
    computed = [
        engine.observe(event_from_row(row, now))
        for row, now in zip(ordered.to_dict("records"), seconds)
    ]
    feature_frame = pd.DataFrame(computed, index=ordered["index"])
    enriched = cases.copy()
    for column in feature_frame.columns:
        if column not in CONTEXT_FEATURES:
            enriched.loc[feature_frame.index, column] = feature_frame[column]
    return enriched, engine
