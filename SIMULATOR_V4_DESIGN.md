# MarginShield V4 Policy And Holdout Lock

This document fixes the V4 policy and final synthetic-test design before V4 model
training or final-test scoring.

## Why V4 Exists

V3 used a single 85% validation precision floor as its only intervention threshold.
That created a narrow high-confidence queue but left too much simulated coordinated
ring loss unreviewed. V4 treats the score boundary as a value-based review queue and
uses evidence verification as a structural sub-route inside that queue.

## Fixed Dataset Design

- Seed: `20260907`.
- Horizon and chronological boundaries: 210 days; train `[0, 147)`, validation
  `[147, 178)`, final test `[178, 210)`.
- Split assignment occurs before labels and scenario injection.
- Train and validation retain the four development motifs in V3.
- V4 final test uses four slower, relation-aware motifs not present in development
  or the prior V3 test: `staggered_device_address_bridge`,
  `rotating_identifier_cycle`, `token_fan_address_pairs`, and
  `three_core_sparse_bridge`.
- All point-in-time feature and shortcut-gate requirements from V3 remain in force.

## Fixed Model Selection

The candidate family remains graph-rule baseline, regularized logistic regression,
regularized CatBoost, and a recency-weighted regularized CatBoost. Rolling
chronological training folds select the winner by mean PR-AUC, then Brier score and
recall at the declared review capacity. The selected model is fitted on the full
training split. Platt scaling uses chronological out-of-fold predictions from the
training folds, so calibration rows are not scored by a model trained on those rows.

## Fixed Intervention Policy

The later validation half selects one score boundary using no final-test labels:

1. **Intervention queue:** maximise simulated net preventable value subject to a queue
   between 30 and 100 requests. The capacity represents roughly 6-7 reviewable
   requests per day over this validation window; it is a stated synthetic operating
   assumption, not a Razorpay fact.
2. **Evidence route:** a request already in that queue is routed to `verify_evidence`
   when it has at least two reused identifier types or a multi-identifier neighbour;
   otherwise it is routed to `manual_review`. This is an evidence-collection route,
   not another score boundary, capacity, or auto-rejection rule.

There is no auto-reject action. In the original V4 run, the final test was scored only
after the simulator, model, calibration, review capacity, and action contract were
locked. The later 4.1 model refresh uses the same fixed test split without using its
labels for selection, but is therefore a comparative refresh rather than a pristine
first-look holdout.

## Amendment 1 - 2026-09-05 - Historical Value-Based Policy

The original V4 proposal's 85% precision verification requirement was infeasible on
the validation policy window. Reaching 30 structural-evidence flags at 85% precision
was not supported by the available positives. A fixed floor was therefore retained
only as a published comparison, never as an unreported relaxed constraint.

The floor was removed because it was not derived from the cost model, and measurement
showed that it forfeited available value:

| Operating point on the validation policy window | Flags | Precision | Recall | Net value (INR) |
| --- | --- | --- | --- | --- |
| 85% precision comparison | 23 | 87.0% | 16.8% | 43,003 |
| Value-optimal within 30-100 capacity | 99 | 59.6% | 49.6% | 113,470 |
| Value-optimal unconstrained | 208 | 45.2% | 79.0% | 176,283 |

The asymmetry is the reason. A missed ring forfeits the full expected loss, on
average about INR 2,100, while a false alert costs roughly INR 200. Break-even
precision is therefore near 9%, and any round percentage above it destroys value.

This historical two-boundary implementation is superseded. It allowed verification
to begin below the review queue, which made the displayed manual-only metrics differ
from the actions served by the API and created a second hidden workload. The active
implementation uses the single-queue contract above.

The 85% floor is still computed and published in the model report as
`precision_floor_comparison`, so the value it would have forfeited stays auditable.

## Amendment 2 - 2026-09-05 - Temporal Calibration And Single Queue

The active model adds a recency-weighted CatBoost candidate with a 45-day half-life
and selects it only if it wins rolling chronological validation. The active winner
achieved mean PR-AUC 0.4628 versus 0.4600 for ordinary CatBoost. Calibration now uses
17,526 chronological out-of-fold predictions from the training split.

The active action boundary is `0.2677768616`. Verification uses that same boundary
and only changes the requested evidence workflow when structural evidence is already
visible. This keeps capacity, API actions, dashboard counts, and benchmark metrics
consistent.

## Interpretation Limits

All labels, loss estimates, review costs, capacities, and false-positive costs are
synthetic. Olist contains no refund-fraud labels. V4 estimates policy behaviour on a
simulator; it does not demonstrate production performance or establish a real
Razorpay review capacity.
