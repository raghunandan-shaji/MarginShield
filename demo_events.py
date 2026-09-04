"""Post a reproducible sequence of raw demo events to a running MarginShield API."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime, timedelta
from urllib.request import Request, urlopen


BASE_URL = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://127.0.0.1:8003"


def get(path: str) -> dict:
    with urlopen(f"{BASE_URL}{path}") as response:
        return json.load(response)


def post(path: str, payload: dict) -> dict:
    request = Request(
        f"{BASE_URL}{path}", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request) as response:
        return json.load(response)


def event(run: str, number: int, timestamp: datetime, **overrides: object) -> dict:
    payload = {
        "case_id": f"DEMO-{run}-{number:02d}", "event_timestamp": timestamp.isoformat(),
        "customer_id": f"DEMO-CUS-{run}-{number:02d}", "merchant_id": f"DEMO-MER-{number % 3}",
        "device_id": f"DEMO-DEV-{run}-{number:02d}", "address_id": f"DEMO-ADR-{run}-{number:02d}",
        "payment_token_id": f"DEMO-PAY-{run}-{number:02d}", "product_id": f"DEMO-PRD-{run}-{number:02d}",
        "vertical": "home", "payment_method": "upi", "refund_amount_inr": 3200,
        "refund_share": 0.9, "account_age_days": 180,
    }
    payload.update(overrides)
    return payload


def main() -> None:
    run = uuid.uuid4().hex[:6].upper()
    start = datetime.fromisoformat(get("/api/health")["latest_event_timestamp"]) + timedelta(seconds=1)
    events = [event(run, 0, start)]
    for number in range(1, 7):
        events.append(event(
            run, number, start + timedelta(hours=number * 6),
            device_id=f"DEMO-RING-DEVICE-{run}",
            address_id=f"DEMO-RING-ADDRESS-{run}-{number // 2}",
            product_id=f"DEMO-RING-PRODUCT-{run}",
        ))
    for payload in events:
        result = post("/api/events", payload)
        print(
            payload["case_id"],
            f"probability={result['ring_probability']:.3f}",
            f"action={result['action']}",
            f"linked_accounts={result['features']['connected_accounts_max_window']}",
        )
    print(f"Audit: {BASE_URL}/api/audit/{events[-1]['case_id']}")
    print(f"Rings: {BASE_URL}/rings.html")


if __name__ == "__main__":
    main()
