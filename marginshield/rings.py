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
    if "split" in frame and frame["split"].isin(["test", "live"]).any():
        frame = frame.loc[frame["split"].isin(["test", "live"])].copy()
    end_time = frame["event_timestamp"].max()
    frame = frame.loc[frame["event_timestamp"] >= end_time - pd.Timedelta(days=window_days)].copy()
    frame["ring_probability"] = score_live_bundle(bundle, frame)
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

    for column, entity_type, relation in (
        ("device_id", "device", "device"),
        ("address_id", "address", "address"),
        ("payment_token_id", "payment_token", "payment-token"),
    ):
        for identifier, group in frame.groupby(column, sort=False):
            accounts = int(group["customer_id"].nunique())
            if accounts < 2:
                continue
            entity_node = add_entity(entity_type, str(identifier), accounts)
            for case_id in group["case_id"]:
                graph.add_edge(f"case:{case_id}", entity_node, relation=relation)

    # Verification is the cheaper action and can begin below the manual-review
    # queue, so "any intervention" starts at the lower of the two boundaries.
    verify_threshold = float(getattr(bundle, "verify_evidence_threshold", None) or bundle.threshold)
    intervention_threshold = min(float(bundle.threshold), verify_threshold)

    def case_action(probability: float, shared_types: int, neighbours: int) -> str:
        structural = shared_types >= 2 or neighbours >= 1
        if probability >= verify_threshold and structural:
            return "verify_evidence"
        if probability >= bundle.threshold:
            return "manual_review"
        return "approve"

    rings: list[dict[str, Any]] = []
    indexed = frame.set_index("case_id", drop=False)
    for component in nx.connected_components(graph):
        case_nodes = sorted(node for node in component if node.startswith("case:"))
        entity_nodes = sorted(node for node in component if node.startswith("entity:"))
        if len(case_nodes) < 3:
            continue
        case_ids = [node.removeprefix("case:") for node in case_nodes]
        cluster = indexed.loc[case_ids].sort_values("event_timestamp")
        if cluster["customer_id"].nunique() < 3 or not (cluster["ring_probability"] >= intervention_threshold).any():
            continue
        entities = [entity_metadata[node] for node in entity_nodes]
        high = int((cluster["ring_probability"] >= intervention_threshold).sum())
        scores = cluster["ring_probability"].to_numpy(float)
        account_count = int(cluster["customer_id"].nunique())
        burst_hours = max((cluster["event_timestamp"].max() - cluster["event_timestamp"].min()).total_seconds() / 3600, 0.0)
        refund_exposure = float(cluster["refund_amount_inr"].sum())
        loss_values = cluster.get("expected_loss_if_ring_inr", cluster["refund_amount_inr"]).to_numpy(float)
        model_weighted_exposure = float(np.sum(scores * loss_values))
        latest_event = cluster["event_timestamp"].max()
        entity_types = ", ".join(sorted({item["entity_type"] for item in entities}))
        reasons = [
            f"{account_count} customer accounts connect through {len(entities)} submitted shared identifiers ({entity_types}).",
            f"{high} request{'s' if high != 1 else ''} meet the calibrated review policy.",
            f"INR {model_weighted_exposure:,.0f} model-weighted loss exposure across INR {refund_exposure:,.0f} in connected refunds.",
        ]
        nodes = []
        for row in cluster.itertuples():
            action = case_action(
                float(row.ring_probability),
                int(row.shared_identifier_types_30d),
                int(row.multi_identifier_neighbor_accounts),
            )
            nodes.append({
                "id": f"case:{row.case_id}", "label": row.case_id, "kind": "case",
                "risk_probability": round(float(row.ring_probability), 4), "action": action,
                "refund_amount_inr": int(row.refund_amount_inr), "merchant_id": row.merchant_id,
            })
        nodes += entities
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
        "method": "Suspected clusters are label-free connected components spanning at least three cases and three customer accounts through submitted device, address or payment-token links, with support from the calibrated coordinated-ring model. P1 is the highest queue rank; ranks are unique and ordered by model-weighted loss exposure. No ring labels, scenario types or future relative to the as-of timestamp are used.",
    }
