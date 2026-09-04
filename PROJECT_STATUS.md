# MarginShield: Audited Project Status

Audit date: 2026-09-04

## Executive verdict

MarginShield is worthy of a buildathon submission as an honest, working prototype.
It is more substantial than a notebook classifier because it joins point-in-time
graph features, a validation-locked decision policy, a live scoring API, per-case
explanations, and an analyst investigation workflow. Its algorithm is not novel and
its synthetic benchmark is not evidence of production performance.

The submission is strongest when framed as a product and evaluation methodology for
one narrow loss class. It should not be framed as a state-of-the-art Razorpay fraud
model, a production replacement, or proof that 88% precision will transfer to real
merchant traffic.

## Exact problem

Refund controls usually inspect one request or customer at a time. Coordinated abuse
can instead distribute refund requests across multiple customer accounts that reuse
devices, payment tokens, or addresses. A single request can look ordinary while the
time-evolving relationship pattern is not.

MarginShield scores each refund request using only information available at that
request's decision time. It then:

1. recommends **approve**, **manual review**, or **verify evidence**;
2. groups model-supported requests connected by shared entities;
3. ranks suspected rings by model-weighted conditional loss exposure; and
4. gives an analyst the graph, contributing signals, request history, and a review
   packet workflow. It never auto-rejects a customer.

The flagship target is only `ring_label`: membership in a simulated coordinated,
multi-account refund-abuse event sequence. It is not a broad refund, return, fraud,
chargeback, or account-risk classifier.

## What is active

- Dataset generator: `marginshield/data_builder.py` (version 1.1.1-audited)
- Model tournament and evaluation: `marginshield/tournament.py`
- Graph investigation catalog: `marginshield/rings.py`
- FastAPI service: `server.py`
- Analyst UI: `index.html`, `rings.html`, `app.js`, `rings.js`, `styles.css`
- Active dataset: `data/processed/master_refund_cases.csv.gz`
- Active model: `data/model/live_refund_ring_model.joblib`
- Active report: `data/reports/ring_model_report.json`

Old broad-refund artifacts and source-download experiments are retained under
`legacy/` and are not referenced by the active application.

## Data and leakage boundary

- 75,000 synthetic refund requests: 52,455 train, 11,234 validation, 11,311 test.
- 1,578 positive requests; overall simulated prevalence 2.104%.
- Explicit coordinated ring sequences and difficult legitimate household, office,
  and hostel clusters are injected only after chronological split boundaries exist.
- Test rings unfold more slowly than train/validation rings, but use the same shared
  device-plus-payment mechanism.
- Account history, velocities, merchant volume, entity reuse, pair reuse, bursts, and
  graph overlap are computed in event order from current and prior events only.
- `ring_id`, scenario type, topology, sequence position, target, and latent simulator
  metadata are excluded from the 15-feature serving contract.
- Olist supplies no rows and no refund-fraud labels. Public datasets informed the
  design only; every row and label used by the app is synthetic.

## Model and policy

The tournament compares a transparent graph-rule baseline, L2-regularized logistic
regression, and calibrated CatBoost over three rolling chronological training folds.
Logistic regression won by mean rolling-fold PR-AUC. Platt calibration is fitted on
the early validation half. The later validation half selects the manual-review
threshold by maximum recall subject to at least 85% precision and at least 30 flags.
The final test is excluded from model and threshold selection.

Locked manual-review threshold: 0.467238. Verify-evidence threshold: 0.603596.

| Metric | Validation policy window | Final synthetic test |
|---|---:|---:|
| Rows | 5,617 | 11,311 |
| PR-AUC | 0.8341 | 0.8138 |
| ROC-AUC | 0.9400 | 0.9191 |
| Brier score | 0.00489 | 0.00718 |
| Precision | 87.04% | 88.04% |
| Recall | 84.68% | 69.43% |
| Review volume | 108 | 209 |
| Flag rate | 1.92% | 1.85% |
| Early ring precision | 86.96% | 86.54% |
| Early ring recall | 100.00% | 95.65% |
| Synthetic net preventable value | INR 181,266 | INR 370,702 |

Test temporal-bootstrap 95% intervals:

- Precision: 81.05%-93.55%
- Recall: 64.71%-74.18%
- PR-AUC: 0.7125-0.8997
- Synthetic net preventable value: INR 179,068-INR 575,101

The point estimate clears 85% precision, but its confidence interval does not
guarantee 85%. The project must state both.

The validation precision is measured on the same later-validation slice used to
select the threshold under the 85% constraint. It proves policy-constraint
satisfaction, not independent generalization. The final synthetic test is the only
out-of-sample policy estimate.

Synthetic cost assumptions:

- Loss if ring = refund amount plus 55% of simulated gross margin.
- Review cost = INR 65, or INR 140 above a INR 5,000 refund.
- Legitimate-review cost = 7.5% of refund value plus review cost.
- Net value subtracts false-positive cost and review cost for true positives.

These assumptions are demonstration parameters, not Razorpay economics.

## Sanity-check results

- Dataset validation: all 12 contract checks pass.
- Automated suite: 18/18 `unittest` tests pass.
- Python and JavaScript syntax checks pass.
- Installed Python packages have no broken requirements.
- No null or non-finite values exist in active model features.
- Serialized model contract exactly matches the dataset feature contract.
- No target-only field enters the model or scoring API.
- Context-only test PR-AUC is 0.0261, close to random for the rare target; graph and
  velocity signals, not commerce context, drive separation.
- Final-test errors: 25 false positives, all simulated benign households; 81 ring
  requests are missed.
- Ring queue ranks are unique P1-P30 and sorted by descending model-weighted exposure.
- Case risk is rendered as calibrated probability/percentage, not a binary class.
- Policy Lab recomputes scenarios from validation only and identifies one locked
  operating point; it does not optimize on final-test labels.
- Casework and ring sidebars scroll independently on desktop.
- Desktop analyst workflows were visually exercised at 1280x720. Responsive CSS was
  inspected, but no separate physical-device browser run is claimed in this audit.
- Static routes expose only the intended frontend assets. Attempts to fetch the
  dataset, model, report, Python source, requirements, or this status file return 404.
- Dependency versions are pinned to the audited artifact-compatible environment and
  `httpx`, required by FastAPI's test client, is declared.
- Re-running the documented dataset command produces byte-identical dataset, entity
  link, and analyst-review gzip artifacts in the pinned environment.

## Known limitations and failure modes

1. **Synthetic-only truth.** Performance can collapse on real behavior, missing
   identifiers, delayed outcomes, or adversarial adaptation.
2. **Narrow simulator topology.** Test rings are slower, not structurally different.
   Several graph features exceed 0.995 correlation because the same relationship
   mechanism drives them. This is the largest validity risk.
   The direct `device_payment_pair_accounts_30d >= 2` rule alone achieves 86.90%
   precision and 82.64% recall on the final synthetic test. The flagship model's
   result therefore mostly demonstrates recovery of the generator's motif, not a
   sophisticated or generalizable fraud boundary.
3. **Household confusion.** All test false positives come from legitimate households
   sharing device and payment identifiers. Evidence verification is therefore safer
   than automatic refusal.
4. **Probability transfer.** Platt calibration is valid only for this simulator and
   prevalence. Production probabilities require representative outcomes and ongoing
   calibration monitoring.
5. **Economic transfer.** Loss and review-cost assumptions are synthetic.
6. **Operational gaps.** There is no authentication, persistent case state, data
   connector, analyst-feedback loop, drift monitor, audit log, or production graph
   store. NetworkX is used in memory; Neo4j is not required for this prototype.
7. **Extreme scores.** The narrow simulator creates many probabilities near 0 or 1.
   Real data should be expected to be noisier and less separable.

## Feasibility

### Current prototype

Status: feasible and complete for a local buildathon demo. It requires Python 3.10,
the listed requirements, roughly 75,000 generated rows, and no paid service.

### Credible merchant pilot

Status: feasible with material changes. A realistic estimate is 6-10 weeks for an ML
engineer, a data/backend engineer, and regular risk-operations input, assuming usable
point-in-time identifiers and adjudicated outcomes already exist. Required work:

- replace synthetic labels with merchant-confirmed ring investigations;
- define legal/privacy-safe device, payment-token, and address representations;
- replay historical requests with strict event-time joins;
- estimate real review friction and preventable loss;
- perform merchant/time holdouts, calibration, load tests, and analyst shadow mode;
- build case persistence, access control, feedback capture, and monitoring.

The estimate is engineering judgment, not a verified Razorpay delivery estimate.

### Production system

Status: not feasible as-is. Expect at least 3-6 months after data access for a limited
production deployment, with the schedule dominated by data quality, policy review,
integration, and prospective validation rather than classifier training.

Likely failure modes are identity sparsity, family/shared-office false positives,
merchant-specific behavior, label delay/bias, changing abuse tactics, leakage in
historical joins, and analyst capacity constraints.

## Submission positioning

This is likely to be noticed if the demo leads with the operational insight: one
ordinary-looking refund becomes suspicious only when its point-in-time device and
payment relationships reveal coordinated accounts. Show a legitimate household next
to a ring, the actual model contribution, the locked 85%-precision policy, and the
early-loss ring metric.

What is impressive:

- one sharply defined loss class rather than a generic fraud score;
- graph and temporal feature engineering with leakage tests;
- explicit hard negatives and honest false-positive analysis;
- validation-locked decision policy with review economics;
- complete path from generator to trained model, API, casework, and graph workflow.

What is not impressive on its own:

- logistic regression as an algorithm;
- the 88.04% synthetic test precision without its uncertainty and limitations;
- a dashboard without a clear action and evidence workflow.

Final verdict: submit it. It is a strong internship buildathon prototype, not the
best available refund-risk system. The single highest-value improvement before a
technical review is an out-of-distribution stress benchmark with structurally varied
rings and more legitimate multi-identifier sharing, locked without touching this
final test.

## Reproduction and verification

```bash
python3 build_dataset.py --rows 75000 --seed 20260904
python3 train_live_model.py \
  --dataset data/processed/master_refund_cases.csv.gz \
  --model-dir data/model \
  --report-dir data/reports \
  --minimum-precision 0.85 \
  --minimum-validation-flags 30
python3 -m unittest discover -s tests -v
python3 -m compileall -q .
node --check app.js
node --check rings.js
python3 -m uvicorn server:app --host 127.0.0.1 --port 8003
```

Audited artifact SHA-256 values:

- Dataset: `58d1543dc8d4eccd76bd4fe451218d59dceeb76eaafc3eba6a5a2bc383365451`
- Model: `7ef75fd7a7ee98f1f077eb9f091d292b911304696c2d52e61d80659d433fabe2`
- Report: `849ac71cb348d2dda37bf5d237f76473ef484a8bde6cc19cbfd40c08862b2bb9`

Audited environment: Python 3.10.5, pandas 2.3.2, NumPy 2.1.3,
scikit-learn 1.7.2, CatBoost 1.2.10, FastAPI 0.135.3, NetworkX 3.4.2,
and Uvicorn 0.43.0.

Reference methodology consulted during design:

- Amazon Fraud Dataset Benchmark: https://github.com/amazon-science/fraud-dataset-benchmark
- Fraud Detection Handbook: https://fraud-detection-handbook.github.io/fraud-detection-handbook/
- IBM AMLSim: https://github.com/IBM/AMLSim/
- FiFAR: https://springernature.figshare.com/articles/dataset/Financial_Fraud_Alert_Review_Dataset/28351172
