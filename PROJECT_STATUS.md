# MarginShield V4: Expert Handoff

Audit date: 2026-09-05

Active version: `4.0.0-value-policy-locked`

Status: submission-ready prototype; not production-ready

## Executive Verdict

MarginShield is a complete buildathon prototype for detecting **coordinated
refund-abuse rings at refund-request time**. Its strongest claim is methodological:
raw-event scoring, one shared offline/live feature engine, a narrow loss class,
hard legitimate negatives, validation-locked intervention policy, a structurally
different final test, graph investigation, explanations, and a durable audit trail.

It must not be described as state of the art or as validated on real fraud. All
performance is synthetic. V4 replaced the fixed precision floor with an explicit
value-and-capacity policy, so the final test now trades point precision for
materially more preventable loss recovered. Recall remains limited on the slowest
unseen topologies.

## Exact Problem And Action

A single refund request may look ordinary. A coordinated ring distributes requests
over multiple customer accounts while reusing or rotating devices, addresses,
payment tokens, products, and merchants. MarginShield computes only information
available at that request's timestamp and estimates whether it belongs to such a
ring.

The only target is `ring_label`: membership in a simulated coordinated multi-account
refund-abuse sequence. Generic return abuse, first-party fraud, chargebacks, and
single-account abuse are outside scope.

Policy:

1. Below the validation-locked probability threshold: `approve`.
2. At or above threshold: enter the one capacity-bounded intervention queue.
3. Within that queue, route to `verify_evidence` when at least two shared identifier
   types or one multi-identifier neighbour is observable; otherwise use `manual_review`.
4. There is no auto-reject action and no second verification workload.

## Architecture And Files

- Shared incremental features: `marginshield/feature_engine.py`
- Synthetic event generator and gates: `marginshield/data_builder.py`
- Tournament, calibration, policy, evaluation: `marginshield/tournament.py`
- Label-free candidate graphs: `marginshield/rings.py`
- Typed API, SQLite audit trail, dashboard payload: `server.py`
- UI: `index.html`, `rings.html`, `app.js`, `rings.js`, `chat.js`, `styles.css`
- Active model: `data/model/live_refund_ring_model.joblib`
- Active report: `data/reports/ring_model_report.json`
- Locked protocol: `SIMULATOR_V4_DESIGN.md`
- Superseded history: `data/model/v2.1/`, `data/reports/v2.1/`

SQLite `data/decisions.sqlite` is runtime state and is gitignored.

## What The Audit Changed

Claude's valid findings were reproduced and fixed:

- **Timestamp resolution:** pandas datetime integers could be microseconds rather
  than nanoseconds, silently expanding all windows by 1000x. Conversion now uses
  `.dt.as_unit("s")`; a 6-day/8-day regression test proves seven-day expiry.
- **Training/serving skew:** batch features and live features now use the exact same
  `PointInTimeFeatureEngine`; replaying all 75,000 rows reproduces the feature table.
- **Entity collision:** device, address, and token IDs are type-namespaced.
- **Simulator shortcut:** independent traffic now contains incidental sharing;
  legitimate mechanisms include sparse and dense reuse; address, merchant, timing,
  product, and existing-customer modes overlap across labels.
- **Non-live API:** `POST /api/events` now accepts raw events, computes features,
  scores, logs, and commits state in that order.
- **Unexplainable second threshold:** verification is now an evidence-based sub-route
  inside the same score queue, with no second probability boundary or hidden capacity.
- **Unvalidated input:** exact Pydantic schemas reject missing, extra, or impossible
  values with 422 responses.
- **No audit trail:** SQLite stores raw event, exact features, model version,
  threshold, probability, recommendation, evidence, prior links, and analyst actions.
- **Calibration leakage risk:** the active report records temporal rolling out-of-fold
  Platt calibration, with 17,526 training predictions not scored by their training model.
- **Weak uncertainty blocks:** confidence intervals now resample days rather than
  weeks.
- **Ring window mismatch:** evaluation and the Rings page both use a trailing 30-day
  candidate window.
- **Cost fragility:** results include false-positive cost sensitivity at 5%, 7.5%,
  and 15% of refund value.

One audit recommendation was rejected: moving the threshold because it improved the
already-scored v2.1 test would be test tuning. V3 uses a pre-declared validation-only
log-odds midpoint and a new seed/test period.

## Dataset

| Split | Rows | Positive requests |
|---|---:|---:|
| Train | 52,579 | 1,047 |
| Validation | 10,973 | 265 |
| Final test | 11,448 | 264 |

The split is assigned before label injection. Development rings use device/payment
hubs, alternating chains, and two-core bridges. Test rings use four slower and
relation-aware disjoint structures: device-address ladders, partial pair meshes,
rotating two-hub bridges, and sparse address-payment chains.

Hard negatives model family wallets, group purchases, corporate cards, hostels,
offices, and support migrations. Every row is synthetic. Product and merchant
context are present, but no source row from Olist is redistributed.

## Shortcut Gates

All generation gates pass. Key validation diagnostics:

| Diagnostic | Result |
|---|---:|
| Independent legitimate rows with shared identity | 376 |
| Precision of any-sharing rule | 11.02% |
| Shared subset prevalence | 10.95% |
| Best univariate AP within shared subset | 0.1426 |
| Best univariate AP lift over shared prevalence | 1.302x |
| Direct device-payment-pair rule precision | 13.68% |
| Context-only PR-AUC | 0.0385 |

The gate requires within-shared AP lift below 2.0. Final test is diagnostic only:
its corresponding lift is 1.781x. Passing these tests reduces simple artifacts; it
does not rule out multivariate simulator fingerprints.

## Model And Policy

Mean chronological training-fold metrics:

| Candidate | PR-AUC | ROC-AUC | Brier | Recall at review capacity |
|---|---:|---:|---:|---:|
| Graph rule | 0.1029 | 0.8572 | 0.0196 | 0.0675 |
| L2 logistic | 0.1927 | 0.8925 | 0.0193 | 0.1641 |
| Regularized CatBoost | 0.4600 | 0.9417 | **0.0145** | 0.3849 |
| Recency-weighted CatBoost | **0.4628** | **0.9429** | 0.0146 | **0.3887** |

The recency-weighted CatBoost is the locked winner. It uses depth 4, 360 trees,
learning rate 0.04, L2 regularization 30, and a 45-day sample-weight half-life.
Probabilities are calibrated with temporal rolling out-of-fold Platt scaling on
17,526 predictions. Per-case explanations are CatBoost SHAP contributions scaled
into calibrated log-odds. The locked review threshold is `0.2677768616`.

The largest global importances are linked merchants, linked 72-hour refund burst,
identifier reuse balance, graph overlap, shared device accounts, merchant vertical,
shared payment accounts, and linked same-product accounts. Importances are not causal.

## Locked Results

| Metric | Validation policy | Final synthetic test |
|---|---:|---:|
| Rows | 5,486 | 11,448 |
| PR-AUC | 0.5552 | 0.3010 |
| ROC-AUC | 0.9486 | 0.9142 |
| Brier score | 0.0140 | 0.0189 |
| Precision | 58.76% | 46.55% |
| Request recall | 47.90% | 20.45% |
| Flags | 97 | 116 |
| Flag rate | 1.77% | 1.01% |
| Ring-candidate precision | 47.50% | 41.25% |
| Early ring recall | 70.00% | 75.68% |
| Net synthetic preventable value | INR 107,883 | INR 94,055 |

Final confusion matrix: 11,122 TN, 62 FP, 210 FN, 54 TP.

The V4 final-test split was previously scored by the superseded bundle before the
4.1 model-layer refresh. No final-test labels were used to select the recency half-life,
model, calibration, or policy, but the refreshed results are comparative rather than
a pristine first-look holdout.

Day-block bootstrap 95% intervals:

- Final precision: 38.11%-54.63%.
- Final recall: 15.17%-25.62%.
- Final PR-AUC: 0.2598-0.3535.
- Final net synthetic value: INR 59,681-INR 127,575.

Precision is deliberately moderate rather than maximised. The interval is wide and
must be stated alongside the point estimate.

Early ring recall by unseen test topology:

| Topology | Early ring recall |
|---|---:|
| Token-fan address pairs | 88.89% |
| Three-core sparse bridge | 100.00% |
| Staggered device-address bridge | 50.00% |
| Rotating identifier cycle | 66.67% |

The system still misses the slowest rotating structures; claiming comprehensive ring
coverage would be false.

V4 removed the 85% precision floor. It was never derived from the cost model, and on
this validation window it is unreachable at the required flag count. Because a missed
ring forfeits roughly INR 2,100 while a false alert costs roughly INR 200, break-even
precision sits near 9%, so a high floor destroys value rather than protecting it.
The active policy maximises synthetic net preventable value inside a single 30-100
review capacity, and the report publishes `precision_floor_comparison` so the trade-off
stays auditable. Validation catches 57 of 119 positive requests and misses 62; the
final synthetic test catches 54 of 264 and misses 210, and reaches 75.7% early ring
recall. The product is a value-optimising triage layer, not a comprehensive detector.

The active model adds two deliberately small ML improvements. First, the tournament
evaluates a recency-weighted CatBoost candidate with a 45-day half-life, which won by
rolling mean PR-AUC (0.4628 versus 0.4600 for ordinary CatBoost). Second, Platt
calibration is fitted on 17,526 chronological out-of-fold predictions from the
training split, rather than on predictions from the final model over its own rows.
These improve model governance and adaptation modestly; they do not turn synthetic
benchmark results into production evidence.

## Frontend Data Contract

- Casework uses the top 520 scores from the final synthetic test.
- Portfolio uses the entire final synthetic test and reports locked-policy coverage.
- Policy Lab uses only the later validation policy window.
- Abuse Rings uses a trailing 30-day graph over final-test and replayed live/demo
  events; labels never form candidate components.
- Every probability is formatted as a percentage. Queue ranking and ring queue rank
  are distinct from probability and are never labelled as generic priority.
- `Ask MarginShield` receives sanitized report, policy, portfolio, and selected-item
  context from the server. It has a deterministic no-key fallback and an optional
  Gemini 2.5 Flash free-tier mode. Free-tier data-use terms make that mode unsuitable
  for real merchant data.

## Verification

- `31/31` unit and API tests pass in the approved writable project environment.
- Grounded-chat tests cover locked-report recall/precision answers and selected-case context.
- Static app-shell responses use `no-store`; both pages expose the same four-view navigation.
- Full-dataset offline/live feature replay passes.
- Target-only metadata is excluded from the exact model contract.
- Future entity mutations do not alter past features.
- Unseen categorical values score successfully.
- Duplicate and out-of-order live events are rejected.
- Runtime decisions and analyst actions are auditable.
- Ring queue ranks are unique and sorted by model-weighted exposure.
- Source, models, data, reports, and requirements are not exposed as static routes.

Artifact SHA-256:

- Dataset: `3224b11a1fb6a44a1b10104dc3571fe352e79c4007d6d394bfe165f2093b81e7`
- Entity links: `677d92c844593b9ce7afabc8b3d6ee84aa5ca2200498f3106d8dcb8712a780f9`
- Model: `dfa7e117dafd0b850af863de7be00b21b1cb83c1049f3dcf90c364e112006832`
- Report: `5591f569258038f9f7dcc7084fb34518529af66f7a5e2cb17c11e58efe73b807`

## Feasibility

**Buildathon submission: feasible and complete.** It runs locally without paid
infrastructure and demonstrates the requested detector plus honest precision,
recall, and false-positive cost.

**Merchant pilot: feasible with substantial work.** Assuming event-time identifiers
and adjudicated outcomes exist, a rough estimate is 6-10 weeks for ML, backend/data,
privacy, and risk-operations work. This is engineering judgment, not a Razorpay plan.

**Production: not feasible as-is.** Required work includes authentication,
privacy-safe identifier hashing, event-time buffering, durable feature storage,
merchant/time holdouts, delayed-label learning, prospective shadow evaluation,
drift/calibration monitoring, access controls, retention policy, and load testing.

Likely failure modes: missing or unstable identifiers, benign high-density sharing,
slow sparse rings, merchant-specific shifts, adversarial adaptation, delayed labels,
policy-induced selection bias, and review-capacity mismatch. NetworkX is sufficient
for this prototype; Neo4j is not required until graph persistence/query scale demands it.

## Submission Positioning

Lead with a request that looks normal alone, then show the prior links, probability,
named evidence, action, candidate ring, and immutable audit record. Follow it with a
legitimate shared-identity example and the shortcut audit. Do not lead with synthetic
precision or imply MarginShield replaces Razorpay's existing fraud systems.

The appropriate claim is: **a focused post-payment coordination layer that gives
refund-risk operations earlier, explainable ring evidence and a review-ready workflow.**
