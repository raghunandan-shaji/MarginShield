# MarginShield V3: Expert Handoff

Audit date: 2026-09-05

Active version: `3.0.0-locked`

Status: submission-ready prototype; not production-ready

## Executive Verdict

MarginShield is a complete buildathon prototype for detecting **coordinated
refund-abuse rings at refund-request time**. Its strongest claim is methodological:
raw-event scoring, one shared offline/live feature engine, a narrow loss class,
hard legitimate negatives, validation-locked intervention policy, a structurally
different final test, graph investigation, explanations, and a durable audit trail.

It must not be described as state of the art or as validated on real fraud. All
performance is synthetic. The final test has high point precision but low recall,
and the precision confidence interval crosses the 85% policy floor.

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
2. Above threshold with at least two shared identifier types or one account linked
   through multiple identifier types: `verify_evidence`.
3. Above threshold without that named structural evidence: `manual_review`.
4. There is no auto-reject action.

## Architecture And Files

- Shared incremental features: `marginshield/feature_engine.py`
- Synthetic event generator and gates: `marginshield/data_builder.py`
- Tournament, calibration, policy, evaluation: `marginshield/tournament.py`
- Label-free candidate graphs: `marginshield/rings.py`
- Typed API, SQLite audit trail, dashboard payload: `server.py`
- UI: `index.html`, `rings.html`, `app.js`, `rings.js`, `chat.js`, `styles.css`
- Active model: `data/model/live_refund_ring_model.joblib`
- Active report: `data/reports/ring_model_report.json`
- Locked protocol: `SIMULATOR_V3_DESIGN.md`
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
- **Unexplainable second threshold:** verify-evidence is now based on observable
  multi-identifier structure, not another probability cutoff.
- **Unvalidated input:** exact Pydantic schemas reject missing, extra, or impossible
  values with 422 responses.
- **No audit trail:** SQLite stores raw event, exact features, model version,
  threshold, probability, recommendation, evidence, prior links, and analyst actions.
- **Misleading calibration headline:** the report includes calibration for scores
  above 1%, where final-test ECE is 0.0551, not only the all-row ECE of 0.0046.
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
| Train | 52,364 | 1,047 |
| Validation | 11,275 | 261 |
| Final test | 11,361 | 262 |

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

| Candidate | PR-AUC | ROC-AUC | Brier | Recall at 85% precision |
|---|---:|---:|---:|---:|
| Graph rule | 0.1063 | 0.8608 | 0.0185 | 0.0000 |
| L2 logistic | 0.2375 | 0.9080 | 0.0173 | 0.0000 |
| Regularized CatBoost | **0.5349** | **0.9378** | **0.0130** | **0.1907** |

CatBoost is the locked winner. It uses depth 4, 360 trees, learning rate 0.04,
L2 regularization 30, and Platt calibration. Per-case explanations are CatBoost SHAP
contributions scaled into calibrated log-odds. The locked review threshold is
`0.5450029782`.

The largest global importances are linked merchants, linked 72-hour refund burst,
identifier reuse balance, graph overlap, shared device accounts, merchant vertical,
shared payment accounts, and linked same-product accounts. Importances are not causal.

## Locked Results

| Metric | Validation policy | Final synthetic test |
|---|---:|---:|
| Rows | 5,637 | 11,361 |
| PR-AUC | 0.5635 | 0.3795 |
| ROC-AUC | 0.9509 | 0.9164 |
| Brier score | 0.0143 | 0.0181 |
| Precision | 87.88% | 91.89% |
| Request recall | 22.83% | 12.98% |
| Flags | 33 | 37 |
| Flag rate | 0.59% | 0.33% |
| Ring-candidate precision | 78.57% | 85.00% |
| Early ring recall | 47.83% | 38.46% |
| Net synthetic preventable value | INR 69,018 | INR 80,609 |

Final confusion matrix: 11,096 TN, 3 FP, 228 FN, 34 TP.

Day-block bootstrap 95% intervals:

- Final precision: 83.33%-100%.
- Final recall: 9.91%-16.38%.
- Final PR-AUC: 0.3261-0.4366.
- Final net synthetic value: INR 53,546-INR 114,553.
- Validation precision: 79.23%-96.61%.

The point estimate clears the track's 85% bar; neither validation nor test interval
guarantees it. This uncertainty must be stated.

Early ring recall by unseen test topology:

| Topology | Early ring recall |
|---|---:|
| Device-address ladder | 80.00% |
| Partial pair mesh | 55.56% |
| Rotating two-hub bridge | 20.00% |
| Sparse address-payment chain | 0.00% |

The system is precise but conservative. It misses slow sparse structures; claiming
comprehensive ring coverage would be false.

The 85% precision floor was a pre-declared review-queue constraint, not evidence that
recall is unimportant. Validation catches 29 of 127 positive requests and misses 98;
the final synthetic test catches 34 of 262 and misses 228. Portfolio now makes these
coverage counts explicit, while Policy Lab shows that lower thresholds recover more
positives but fail the chosen precision floor. The product is therefore a selective
triage layer, not a comprehensive detector.

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

- `27/27` unit and API tests pass.
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

- Dataset: `de6bb654edac1e89446c36ca92c9236089c4683858b4b102f5093fc5fac1e414`
- Entity links: `efe116424412de1b90de50087aedf8f22eb46d9aa1bff8a44c4a8724d8b77b3c`
- Model: `084dd280d82186a0f26af1b40bc3d28f92cc02cc1b1d0687b2d390460de64130`
- Report: `3c775200e2b28d07308b6241c2d74299be006b86e48456d4e799b38dd3d62c0a`

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
