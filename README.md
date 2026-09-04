# MarginShield

MarginShield is a Python-first, post-payment control for one loss class only:
**coordinated refund-abuse rings**.

It is not a broad fraud classifier and it never auto-rejects a customer. For each
refund request it recommends one of three actions: **approve**, **verify evidence**,
or **manual review**. The Rings page turns high-confidence linked requests into a
merchant-facing investigation graph.

## What It Demonstrates

- A time-aware synthetic benchmark with explicit coordinated account clusters.
- Legitimate shared-identity hard negatives: households, offices, and hostels,
  including refund bursts and multi-identifier overlap.
- Strictly point-in-time account, velocity, merchant, and graph features.
- An explainable graph-rule baseline, regularized logistic regression, and calibrated
  CatBoost tournament.
- A validation-locked action policy: maximize recall subject to at least 85% precision
  and at least 30 review flags.
- A FastAPI scoring endpoint and graph monitor ranked by the selected live model.

## Run

The requirements are pinned because scikit-learn does not guarantee that serialized
estimators load across library versions. The audited runtime is Python 3.10.5.

```bash
python3 -m pip install -r requirements-app.txt
python3 build_dataset.py --rows 75000 --seed 20260904
python3 train_live_model.py \
  --dataset data/processed/master_refund_cases.csv.gz \
  --model-dir data/model \
  --report-dir data/reports \
  --minimum-precision 0.85 \
  --minimum-validation-flags 30
python3 -m uvicorn server:app --host 127.0.0.1 --port 8003
```

Open `http://127.0.0.1:8003`. API documentation is at `/docs`.

## API

- `POST /api/score` scores one complete, pre-decision refund-request feature payload.
- `GET /api/dashboard` returns the live work queue and locked evaluation view.
- `GET /api/rings` returns suspected device-payment clusters ranked by the ring model.
- `GET /api/model` returns the full model report and limitations.

## Current Benchmark Result

The app-native tournament selected **regularized logistic regression**. On the
validation-policy window it achieved 87.0% precision and 84.7% recall across 108
flags. On the final synthetic test split it achieved 88.0% precision and 69.4%
recall; the temporal-bootstrap precision interval is 81.1%-93.5%.

These are benchmark results, not production claims. The final test rings unfold more
slowly than train/validation rings but retain the same device-plus-payment mechanism.
The report also includes ring-level metrics: the first valid flag must occur before
half the simulated ring loss.

The 87.0% validation precision is the constraint-selected operating point, not an
independent performance estimate. The 88.0% final-test precision is the out-of-sample
point estimate. A pair-reuse rule alone reaches 86.9% precision and 82.6% recall on
the synthetic test, which exposes how strongly the simulator encodes its core motif.

## Important Limitation

Olist does **not** have refund-fraud labels. MarginShield uses no Olist source row
as fraud truth. All records and labels in this project are synthetic; public source
knowledge informs high-level commerce context only. Production validation requires
merchant-confirmed abuse outcomes, denied-refund investigations, disputes, and
chargeback evidence.

Run checks with:

```bash
python3 -m unittest discover -s tests -v
```

The server exposes only the intended HTML, CSS, and JavaScript routes. Dataset,
model, report, source, requirements, and project-status files are not static assets.

The full audited status, feasibility assessment, artifact inventory, and submission
positioning are in [`PROJECT_STATUS.md`](PROJECT_STATUS.md).
