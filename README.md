# MarginShield

MarginShield is a Python-first risk-operations prototype for one loss class:
**coordinated refund-abuse rings**. It looks for multiple customer accounts that
coordinate refunds through reused or rotating devices, addresses, payment tokens,
products, merchants, and timing patterns.

It is not a broad fraud classifier. It never auto-rejects a customer. A request is
either approved, sent to manual review, or sent for evidence verification when the
model threshold and named multi-identifier evidence both exist.

## Problem Definition

Refund abuse is difficult to detect one request at a time. A coordinated group can
spread activity over multiple customer accounts so that each account looks ordinary,
while the group reuses or rotates devices, delivery addresses, payment tokens,
products, merchants, and timing patterns.

MarginShield answers one operational question at the instant a refund is requested:

> Does this request look like part of a coordinated, multi-account refund-abuse ring,
> and what defensive action is justified by the evidence available now?

The model target is only `ring_label`. Generic returns, chargebacks, single-account
abuse, merchant fraud, and broad payment fraud are deliberately outside the target.

## Decision Workflow

1. A raw refund event enters `POST /api/events`.
2. The incremental feature engine reads only events strictly before that request.
3. The calibrated CatBoost model estimates coordinated-ring probability.
4. The validation-locked policy maps the score to `approve` or `manual_review`.
5. An above-threshold request becomes `verify_evidence` only when observable graph
   structure also exists: at least two reused identifier types, or a neighbouring
   account connected through multiple identifier types.
6. The event, exact feature vector, model version, threshold, recommendation,
   explanation, and prior linked state are recorded in SQLite.
7. An analyst can accept the recommendation or send the case onward; the latest
   action is persisted and restored in the Casework UI.

There is no auto-reject path. The product manages risk by prioritising intervention
and evidence collection, not by making an irreversible customer decision.

## Product Surfaces

- **Casework:** independently scrollable queue, explicit sorting by ring risk or
  exposure, persisted analyst actions, model evidence, relationship context, and a
  decision trace.
- **Portfolio:** final-test refund exposure by merchant vertical together with
  request recall, early ring recall, review rate, and explicit caught/missed counts.
  This view prevents high precision from being mistaken for broad coverage.
- **Policy Lab:** validation-only threshold scenarios in ascending threshold order.
  The locked threshold maximises recall subject to at least 85% validation precision
  and at least 30 validation flags. Lower thresholds visibly recover more abuse but
  fail that declared precision floor; final-test labels never select the policy.
- **Abuse Rings:** label-free connected components built from trailing-window device,
  address, and payment-token links, ranked by model-weighted conditional exposure.
  The investigation queue may include replayed live/demo events as well as final-test
  events, so it is not presented as a held-out evaluation surface.

### What each screen is allowed to claim

| Screen | Data window | Purpose |
|---|---|---|
| Casework | Top 520 final synthetic-test scores | Demonstrate a review decision and its pre-decision evidence |
| Portfolio | Entire final synthetic test | Report exposure and locked-policy coverage honestly |
| Policy Lab | Later validation policy window only | Explain why the 54.5% threshold was locked |
| Abuse Rings | Trailing 30 days of final-test plus replayed live/demo events | Investigate label-free relationship components |

These boundaries are deliberate. Policy Lab must not use final-test labels, and the
mixed current Rings graph must not be quoted as held-out model performance.

## Grounded Assistant

The `Ask MarginShield` drawer is a plain-text risk assistant. Every request receives
a server-built context containing only the locked report, portfolio aggregates,
policy metadata, documented limitations, and the currently selected case or ring.
Its instruction forbids invented metrics, treats graph components as candidates
rather than confirmed fraud, and never recommends auto-rejection.

Without an API key it uses a deterministic local fallback for the core questions,
so the demo remains functional at zero cost and without network access. An optional
Gemini free-tier conversation layer can be enabled server-side:

```bash
export GEMINI_API_KEY="your-key"
export GEMINI_MODEL="gemini-2.5-flash"
python3 -m uvicorn server:app --host 127.0.0.1 --port 8003
```

The key is never sent to the browser. Google currently lists Gemini 2.5 Flash input
and output as free of charge on the free tier, but also states that free-tier content
may be used to improve its products. Use it only with this synthetic demo data; do
not submit real merchant or customer data under that policy. See the official
[model page](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash) and
[pricing/data-use table](https://ai.google.dev/gemini-api/docs/pricing).

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

The app loads the checked-in locked dataset, model bundle, and report at startup.
You do not need to rebuild or retrain to run the submitted prototype. Run the longer
dataset and training commands only when deliberately reproducing the benchmark.

## Free Deployment Options

### 1. Render Free Web Service - recommended first

Connect the GitHub repository as a Python web service and configure:

```text
Build command: python3 -m pip install -r requirements-app.txt
Start command: python3 -m uvicorn server:app --host 0.0.0.0 --port $PORT
Health check: /api/health
Environment: PYTHON_VERSION=3.10.5
```

Render documents a free 0.1 CPU / 512 MB web instance. The local submitted app uses
about 77 MB RSS after startup, so it is a plausible fit, although cloud startup must
still be tested. Free services sleep after 15 idle minutes and can take roughly a
minute to wake. Their filesystem is ephemeral, so SQLite analyst actions can be lost
on sleep, restart, or deploy. That is acceptable for a temporary judging demo, not
for a durable pilot. See [Render free services](https://render.com/docs/free) and
[Render compute plans](https://render.com/docs/compute-plans).

### Why GitHub Pages cannot host the complete app

GitHub Pages publishes static HTML, CSS, and JavaScript and does not support Python
or other server-side languages. It therefore cannot run FastAPI, load the CatBoost
bundle, score raw events, query SQLite, build ring graphs, persist analyst actions,
or protect a Gemini API key. This is an architectural limitation, not a repository
setting. See [What is GitHub Pages?](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
and GitHub's note that Pages does not support server-side languages such as Python in
[Creating a GitHub Pages site](https://docs.github.com/en/enterprise-cloud%40latest/pages/getting-started-with-github-pages/creating-a-github-pages-site).

A split deployment could put only the static files on Pages and host the API on
Render, but it adds CORS, two origins, and configuration without removing the need
for a Python host. For this submission, one Render web service is simpler and less
fragile.

### 2. Google Cloud Run - stronger runtime, billing account required

Cloud Run supports FastAPI source or container deployments and includes a monthly
free usage allowance. It requires an active billing account and charges usage beyond
the free limits, so it is not a zero-financial-risk option unless budgets and alerts
are configured. Container-local SQLite is also non-durable. See the
[FastAPI deployment guide](https://docs.cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-fastapi-service)
and [Cloud Run pricing](https://cloud.google.com/run/pricing).

### 3. Koyeb Free Instance - fallback only

Koyeb supports Git-driven FastAPI deployment, but its free instance is only 0.1 vCPU
and 512 MB RAM, scales to zero after an hour, and has ephemeral local storage. It may
run MarginShield, but cold startup and model loading need measurement before using it
as the judging URL. See [Koyeb instances](https://www.koyeb.com/docs/reference/instances)
and the [FastAPI guide](https://www.koyeb.com/docs/deploy/fastapi).

Hugging Face Docker Spaces are not listed as a free recommendation: current Hugging
Face documentation says creating a compute-backed Docker Space requires a paid plan.
Static hosting alone cannot run MarginShield's Python scoring service.

For any durable deployment, replace local SQLite with a managed database or mounted
persistent volume. Do not present a free ephemeral demo as a production architecture.

## API

- `POST /api/events`: raw event -> point-in-time features -> model -> action -> audit log.
- `POST /api/score`: score an exact, typed precomputed feature contract.
- `POST /api/decisions/{case_id}/action`: persist an analyst disposition.
- `POST /api/chat`: answer from sanitized report, portfolio, policy, and selected-item context.
- `GET /api/audit/{case_id}`: retrieve the scored decision and action history.
- `GET /api/dashboard`: casework queue and validation-only policy analysis.
- `GET /api/rings`: trailing 30-day suspected-ring queue and graph summaries.
- `GET /api/model`: complete model and evaluation report.

All score inputs use strict Pydantic contracts. Missing, extra, impossible, or naive
timestamp values are rejected. `/api/score` exists for typed precomputed features;
`/api/events` is the operational path because it computes point-in-time features and
commits state only after scoring succeeds.

## Repository Map

| Path | Responsibility |
|---|---|
| `marginshield/feature_engine.py` | Shared offline/live point-in-time graph, velocity, and history features |
| `marginshield/data_builder.py` | Synthetic event mechanisms, hard legitimate negatives, and simulator gates |
| `marginshield/tournament.py` | Baselines, CatBoost, calibration, threshold policy, metrics, and intervals |
| `marginshield/rings.py` | Label-free trailing-window connected components and investigation ranking |
| `server.py` | FastAPI contracts, live scoring, grounded assistant, dashboard payloads, and SQLite audit trail |
| `index.html`, `app.js`, `styles.css`, `chat.js` | Casework, portfolio, validation-policy UI, and assistant drawer |
| `rings.html`, `rings.js` | Abuse-ring investigation UI |
| `tests/` | Temporal leakage, feature parity, model policy, API, graph, and persistence tests |
| `data/model/` | Active locked model bundle and preserved superseded model |
| `data/reports/` | Active and superseded evaluation reports |
| `SIMULATOR_V3_DESIGN.md` | Protocol fixed before the final synthetic test was evaluated |

Runtime analyst state lives in `data/decisions.sqlite` and is gitignored. API tests
use a temporary SQLite database and cannot modify the demo's decision history.

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

Precision and recall are both important. The locked policy is intentionally a
high-confidence, low-coverage triage queue: in validation it flags 29 of 127 positive
requests and misses 98; in final test it flags 34 of 262 and misses 228. That is not
"good recall." It is the direct consequence of the pre-declared requirement to keep
validation precision at or above 85% with at least 30 flags. The lower-threshold
Policy Lab scenarios show the alternative: substantially higher recall, but too many
false alerts to satisfy that operating constraint. MarginShield should be presented
as a focused escalation layer, not a replacement for a broad fraud stack.

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
