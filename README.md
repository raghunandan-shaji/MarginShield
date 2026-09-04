# MarginShield

MarginShield is a Python-first risk-operations prototype for one loss class:
**coordinated refund-abuse rings**. It looks for multiple customer accounts that
coordinate refunds through reused or rotating devices, addresses, payment tokens,
products, merchants, and timing patterns.

It is not a broad fraud classifier. It never auto-rejects a customer. A request is
either approved, sent to manual review, or sent for evidence verification when the
model threshold and named multi-identifier evidence both exist.

## Why This Is More Than A Notebook

- A raw refund event can be scored through `POST /api/events`.
- The same incremental point-in-time engine builds training and live features.
- Every live decision, feature vector, model version, explanation, and analyst action
  is persisted in SQLite.
- A label-free graph groups connected cases and ranks candidate rings by
  model-weighted conditional loss exposure.
- The synthetic generator contains difficult legitimate sharing: households,
  offices, hostels, corporate cards, group purchases, and support migrations.
- Generation fails when simple shared-identifier shortcuts exceed documented gates.

## Run

The audited environment is Python 3.10.5 with pinned dependencies.

```bash
python3 -m pip install -r requirements-app.txt
python3 build_dataset.py --rows 75000 --seed 20260905
python3 train_live_model.py \
  --dataset data/processed/master_refund_cases.csv.gz \
  --model-dir data/model \
  --report-dir data/reports \
  --minimum-precision 0.85 \
  --minimum-validation-flags 30
python3 -m unittest discover -s tests -v
python3 -m uvicorn server:app --host 127.0.0.1 --port 8003
```

Open `http://127.0.0.1:8003`; API documentation is at `/docs`.
With the server running, `python3 demo_events.py` posts a raw sequence and prints
the probability, action, and linked-account growth after each request.

## API

- `POST /api/events`: raw event -> point-in-time features -> model -> action -> audit log.
- `POST /api/score`: score an exact, typed precomputed feature contract.
- `POST /api/decisions/{case_id}/action`: persist an analyst disposition.
- `GET /api/audit/{case_id}`: retrieve the scored decision and action history.
- `GET /api/dashboard`: casework queue and validation-only policy analysis.
- `GET /api/rings`: trailing 30-day suspected-ring queue and graph summaries.
- `GET /api/model`: complete model and evaluation report.

## Locked V3 Result

The simulator and model policy were fixed in [`SIMULATOR_V3_DESIGN.md`](SIMULATOR_V3_DESIGN.md)
before the final test was scored. CatBoost won the chronological training tournament.
The early validation half fitted Platt calibration; the later half selected the
threshold by maximizing recall subject to at least 85% point precision and 30 flags.

| Metric | Validation policy window | Final synthetic test |
|---|---:|---:|
| PR-AUC | 0.5635 | 0.3795 |
| ROC-AUC | 0.9509 | 0.9164 |
| Brier score | 0.0143 | 0.0181 |
| Precision | 87.88% | 91.89% |
| Request recall | 22.83% | 12.98% |
| Review volume | 33 | 37 |
| Ring-candidate precision | 78.57% | 85.00% |
| Early ring recall | 47.83% | 38.46% |
| Synthetic net preventable value | INR 69,018 | INR 80,609 |

Final-test precision has a day-block bootstrap 95% interval of 83.33%-100%; the
point estimate clears 85%, but the interval does not guarantee it. Sparse
address-payment chains have 0% early ring recall. Those are explicit limitations,
not values to tune away after seeing test labels.

## Simulator Audit

The earlier v2.1 benchmark was superseded because identifier-reuse balance exposed a
synthetic shortcut. V3 adds organic independent sharing and overlapping cluster
context. On validation:

- 376 independent legitimate requests exhibit cross-account identifier sharing;
- a rule based on any sharing has 11.0% precision;
- the best single feature among already-shared cases has AP only 1.30 times that
  subset's prevalence, below the locked 2.0 gate;
- context-only PR-AUC is 0.0385;
- the direct device-payment-pair rule has 13.68% precision.

These checks make the benchmark harder and more honest. They do not establish that
the simulator matches Razorpay traffic.

## Non-Negotiable Limitation

Olist has **no refund-fraud labels**. No Olist row is used as fraud truth. All 75,000
records and labels are synthetic, so the reported numbers are simulator performance,
not real-world or production performance. A credible pilot needs event-time merchant
data, privacy-safe identifiers, and adjudicated refund-abuse outcomes.

See [`PROJECT_STATUS.md`](PROJECT_STATUS.md) for the expert handoff,
[`ARCHITECTURE.md`](ARCHITECTURE.md) for system boundaries, and
[`EXPLANATION_GUIDE.md`](EXPLANATION_GUIDE.md) for a ground-up explanation.
