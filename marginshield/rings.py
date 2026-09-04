from __future__ import annotations

import hashlib
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd

from marginshield.tournament import LiveModelBundle, score_live_bundle


def _short(value: str) -> str:
    return f"{str(value)[:4]}...{str(value)[-4:]}"


def _cluster_id(case_ids: list[str]) -> str:
    digest = hashlib.blake2b("|".join(sorted(case_ids)).encode("utf-8"), digest_size=5).hexdigest().upper()
    return f"RING-{digest}"


def build_ring_catalog(cases: pd.DataFrame, bundle: LiveModelBundle, window_days: int = 30) -> dict[str, Any]:
    """Build merchant-facing suspected rings without ring labels or simulator metadata."""
    frame = cases.copy()
    frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"], utc=True, format="mixed")
    if "split" in frame and frame["split"].eq("test").any():
        frame = frame.loc[frame["split"].eq("test")].copy()
    end_time = frame["event_timestamp"].max()
    frame = frame.loc[frame["event_timestamp"] >= end_time - pd.Timedelta(days=window_days)].copy()
    frame["ring_probability"] = score_live_bundle(bundle, frame)
    frame["pair_key"] = frame["device_id"].astype(str) + "|" + frame["payment_token_id"].astype(str)
    graph = nx.Graph()
    entity_metadata: dict[str, dict[str, Any]] = {}

    def add_entity(entity_type: str, identifier: str, accounts: int) -> str:
        node = f"entity:{entity_type}:{identifier}"
        if node not in entity_metadata:
            entity_metadata[node] = {
                "id": node,
                "label": f"{entity_type.replace('_', ' ').title()} · {_short(identifier)}",
                "kind": "entity",
                "entity_type": entity_type.replace("_", " "),
                "accounts": accounts,
            }
        else:
            entity_metadata[node]["accounts"] = max(entity_metadata[node]["accounts"], accounts)
        return node

    for pair_key, group in frame.groupby("pair_key", sort=False):
        accounts = int(group["customer_id"].nunique())
        if accounts < 2:
            continue
        has_model_support = bool((group["ring_probability"] >= bundle.threshold).any())
        if not has_model_support:
            continue
        device_id = str(group["device_id"].iloc[0])
        payment_id = str(group["payment_token_id"].iloc[0])
        device_node = add_entity("device", device_id, accounts)
        payment_node = add_entity("payment_token", payment_id, accounts)
        for case_id in group["case_id"]:
            case_node = f"case:{case_id}"
            graph.add_edge(case_node, device_node, relation="device")
            graph.add_edge(case_node, payment_node, relation="payment-token")
        for address_id, address_group in group.groupby("address_id", sort=False):
            address_accounts = int(address_group["customer_id"].nunique())
            if address_accounts < 2:
                continue
            address_node = add_entity("address", str(address_id), address_accounts)
            for case_id in address_group["case_id"]:
                graph.add_edge(f"case:{case_id}", address_node, relation="address")

    rings: list[dict[str, Any]] = []
    indexed = frame.set_index("case_id", drop=False)
    for component in nx.connected_components(graph):
        case_nodes = sorted(node for node in component if node.startswith("case:"))
        entity_nodes = sorted(node for node in component if node.startswith("entity:"))
        if len(case_nodes) < 2:
            continue
        case_ids = [node.removeprefix("case:") for node in case_nodes]
        cluster = indexed.loc[case_ids].sort_values("event_timestamp")
        entities = [entity_metadata[node] for node in entity_nodes]
        high = int((cluster["ring_probability"] >= bundle.threshold).sum())
        scores = cluster["ring_probability"].to_numpy(float)
        account_count = int(cluster["customer_id"].nunique())
        burst_hours = max((cluster["event_timestamp"].max() - cluster["event_timestamp"].min()).total_seconds() / 3600, 0.0)
        refund_exposure = float(cluster["refund_amount_inr"].sum())
        loss_values = cluster.get("expected_loss_if_ring_inr", cluster["refund_amount_inr"]).to_numpy(float)
        model_weighted_exposure = float(np.sum(scores * loss_values))
        latest_event = cluster["event_timestamp"].max()
        reasons = [
            f"{account_count} customer accounts connect through {len(entities)} shared identifiers, including both device and payment token.",
            f"{high} request{'s' if high != 1 else ''} meet the calibrated review policy.",
            f"INR {model_weighted_exposure:,.0f} model-weighted loss exposure across INR {refund_exposure:,.0f} in connected refunds.",
        ]
        nodes = [{
            "id": f"case:{row.case_id}", "label": row.case_id, "kind": "case",
            "risk_probability": round(float(row.ring_probability), 4),
            "refund_amount_inr": int(row.refund_amount_inr), "merchant_id": row.merchant_id,
        } for row in cluster.itertuples()] + entities
        edges = [{"source": source, "target": target, "relation": data["relation"]} for source, target, data in graph.subgraph(component).edges(data=True)]
        rings.append({
            "ring_id": _cluster_id(case_ids),
            "case_count": len(case_ids), "entity_count": len(entities),
            "high_risk_case_count": high, "mean_risk_probability": round(float(scores.mean()), 4),
            "max_linked_accounts": max(item["accounts"] for item in entities),
            "refund_exposure_inr": round(refund_exposure), "span_hours": round(burst_hours, 1),
            "model_weighted_exposure_inr": round(model_weighted_exposure),
            "latest_event_timestamp": latest_event.isoformat(),
            "reasons": reasons, "nodes": nodes, "edges": edges,
        })
    rings.sort(
        key=lambda ring: (
            ring["model_weighted_exposure_inr"], ring["mean_risk_probability"],
            ring["refund_exposure_inr"], ring["latest_event_timestamp"], ring["ring_id"],
        ),
        reverse=True,
    )
    for queue_rank, ring in enumerate(rings, start=1):
        ring["queue_rank"] = queue_rank
    return {
        "as_of": end_time.isoformat(), "window_days": window_days,
        "candidate_rings": rings[:30],
        "ranking_note": "P1 is reviewed first. Queue ranks are unique and ordered by model-weighted loss exposure.",
        "method": "Suspected clusters require a shared device-payment relationship across at least two accounts plus support from the calibrated coordinated-ring model. P1 is the highest queue rank; ranks are unique and ordered by model-weighted loss exposure. No ring labels, scenario types or future relative to the as-of timestamp are used.",
    }
