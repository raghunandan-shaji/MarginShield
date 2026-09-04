# MarginShield Ring Benchmark

## Scope

One row is one refund request. The only training target is `ring_label`: a request
that belongs to a simulated coordinated, multi-account refund-abuse event sequence.
MarginShield does not train or present a generic "all refund abuse" label.

`ring_id`, `scenario_type`, `ring_topology`, `loss_sequence_position`, and all other
simulator metadata are evaluation-only. They are excluded from the model contract.

## Scenario Design

The generator first creates timestamps and fixes chronological train, validation, and
test partitions. It then injects within-partition event sequences:

- Coordinated rings share a device and payment token across multiple customer accounts
  and generate refund bursts.
- Train and validation rings unfold faster. Test rings are deliberately slower,
  while retaining the same two-identifier device-plus-payment mechanism.
- Households, offices, and hostels are legitimate negatives. They can share address
  and device identifiers, have bursts, and sometimes overlap on payment identifiers.

Labels arise from those event mechanisms, not from a linear formula over final model
features.

## Point-In-Time Boundary

Features are built while processing requests in event order. Each request can use its
own submitted identifiers and prior history only. It cannot use future entity links,
future refunds, post-decision outcomes, `ring_id`, a scenario label, or a latent
simulator score.

The model contract is defined in `marginshield/data_builder.py` and includes context,
velocity, merchant-normalized, shared-entity, device-payment-pair, and graph-overlap
features. It excludes all target-only columns by test.

## Evaluation

The tournament compares an explainable graph rule, regularized logistic regression,
and calibrated CatBoost using rolling folds inside the training period. Platt scaling
uses the early validation window. The later validation window selects the policy
threshold by maximum recall subject to at least 85% precision and at least 30 flags.

The final test is evaluated after the model and policy are locked. The report includes
PR-AUC, ROC-AUC, Brier score, precision, recall, flag rate, review volume,
false-positive cost, preventable loss, ring-level precision/recall, and temporal
bootstrap intervals. A ring is counted as detected only when its first valid flag is
before half of its simulated loss.

Several graph/velocity features are highly correlated in this simulator because its
ring mechanism is narrow. This is documented as a benchmark limitation, not hidden
as model novelty.

The report includes a direct shortcut diagnostic for
`device_payment_pair_accounts_30d >= 2`. Its final-test precision is 86.9% and recall
is 82.6%, demonstrating that much of the apparent separation is recovery of the
synthetic construction rule. This is not evidence of real-world abuse detection.

Generated timestamps are normalized to microsecond precision and gzip files omit
clock and filename metadata. Re-running the documented seed under the pinned
environment therefore reproduces byte-identical dataset artifacts.

## Limitation

Olist has no refund-fraud labels. It is not used as fraud truth and no source row is
redistributed. Every project row is synthetic. Therefore the benchmark measures a
documented simulator, not real-world or production performance.
