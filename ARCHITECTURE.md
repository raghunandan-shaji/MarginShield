# MarginShield Architecture

```mermaid
flowchart LR
  subgraph Offline["Offline development"]
    G["Event-sequence simulator"] --> E1["PointInTimeFeatureEngine"]
    E1 --> S["Chronological train / validation / test"]
    S --> T["Graph rule + Logistic + CatBoost tournament"]
    T --> C["Validation calibration and policy lock"]
    C --> X["One final test evaluation"]
    C --> B["Versioned LiveModelBundle"]
  end

  subgraph Online["Live decision path"]
    R["Raw refund event"] --> V["Typed Pydantic validation"]
    V --> E2["PointInTimeFeatureEngine"]
    E2 --> B
    B --> P["Approve / verify evidence / manual review"]
    P --> D["SQLite decision and analyst audit log"]
    E2 --> Q["Label-free 30-day relationship graph"]
    Q --> U["Casework and Rings UI"]
    D --> U
  end

  E1 -. "same Python implementation" .-> E2
```

## Trust Boundaries

- Raw events contain only information available before refund disposition.
- Features are computed before the current event is committed.
- The serialized bundle stores the exact feature names, medians, category handling,
  model, calibrator, threshold, target, and version.
- Ring grouping uses submitted identifiers and model scores, never evaluation labels.
- Simulator labels and topology metadata exist only for offline evaluation.
- Analyst outcomes are append-only events in the local prototype audit database.

## Runtime Order

`POST /api/events` validates the payload, rejects duplicates or out-of-order events,
computes point-in-time features, scores the frozen bundle, creates a structural
action, stores the complete decision record, then commits the event to feature
state. This ordering prevents the current request from becoming its own history.

The prototype replays the shipped historical dataset and prior live events at
startup. Production requires a durable feature store, event-time buffering,
authentication, encryption, privacy-safe identifiers, and concurrency controls
beyond this single-process lock.

