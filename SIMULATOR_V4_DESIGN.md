# MarginShield V4 Policy And Holdout Lock

This document fixes the V4 policy and final synthetic-test design before V4 model
training or final-test scoring.

## Why V4 Exists

V3 used a single 85% validation precision floor as its only intervention threshold.
That created a narrow high-confidence queue but left too much simulated coordinated
ring loss unreviewed. V4 treats manual review and evidence verification as separate
operational actions rather than pretending one probability cutoff serves both jobs.

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
and regularized CatBoost. Rolling chronological training folds select the winner by
mean PR-AUC, then calibration quality. Platt scaling uses the earlier half of the
validation split.

## Fixed Intervention Policy

The later validation half selects two thresholds using no final-test labels:

1. **Manual review:** maximise simulated net preventable value subject to a queue
   between 30 and 100 requests. The capacity represents roughly 6-7 reviewable
   requests per day over this validation window; it is a stated synthetic operating
   assumption, not a Razorpay fact.
2. **Verify evidence:** maximise recall subject to at least 85% validation precision
   and at least 30 flags. A request must also have at least two reused identifier
   types or a multi-identifier neighbour. This is an evidence-collection tier, not
   an auto-rejection rule.

There is no auto-reject action. The V4 final test is first scored only after the
simulator, model, calibration, review capacity, and both policy rules are locked.

## Amendment 1 - 2026-09-05 - Value-Based Tier Selection

The intervention policy above could not be locked as written. On the V4 validation
policy window the verification rule is infeasible: reaching 30 structural-evidence
flags at 85% precision needs roughly 26 true positives, and the ceiling is 14. The
maximum volume at 85% precision is 16 flags, so training aborted rather than
silently relaxing the gate.

Rather than lower the floor to whatever the data happened to allow, the floor was
removed from both tiers. It was never derived from the cost model, and measurement
shows it forfeits most of the available value:

| Operating point on the validation policy window | Flags | Precision | Recall | Net value (INR) |
| --- | --- | --- | --- | --- |
| 85% precision floor | 23 | 87.0% | 16.8% | 43,003 |
| Value-optimal within 30-100 capacity | 99 | 59.6% | 49.6% | 113,470 |
| Value-optimal unconstrained | 208 | 45.2% | 79.0% | 176,283 |

The asymmetry is the reason. A missed ring forfeits the full expected loss, on
average about INR 2,100, while a false alert costs roughly INR 200. Break-even
precision is therefore near 9%, and any round percentage above it destroys value.

Both tiers are now selected by maximising synthetic net preventable value under an
explicit review capacity, which is a real operational constraint rather than a
statistical convention:

1. **Manual review:** maximise net preventable value with a queue of 30-100
   validation flags. Unchanged from the original design.
2. **Verify evidence:** maximise net preventable value on the structural-evidence
   subset with 30-100 validation flags. A request must still show at least two
   reused identifier types or a multi-identifier neighbour.

Because verification is the cheaper, lighter-touch action and is gated on reused
identifiers, its economic boundary may sit below the manual-review queue. The
tiers are therefore separated by action and evidence, not by a nested score
ordering, and the previous constraint requiring the verification threshold to
exceed the manual-review threshold has been removed.

The 85% floor is still computed and published in the model report as
`precision_floor_comparison`, so the value it would have forfeited stays auditable.
There is still no auto-reject action, and the final test is still scored only once,
after the simulator, model, calibration, and both policy rules are locked.

## Interpretation Limits

All labels, loss estimates, review costs, capacities, and false-positive costs are
synthetic. Olist contains no refund-fraud labels. V4 estimates policy behaviour on a
simulator; it does not demonstrate production performance or establish a real
Razorpay review capacity.
