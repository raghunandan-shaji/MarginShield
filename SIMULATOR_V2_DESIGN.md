# MarginShield Simulator V2: Locked Design

Design lock date: 2026-09-04

This document fixes the simulator structure and acceptance gates before the v2 final
test is generated or scored. The final test may be evaluated once after the model,
calibration, and validation policy are locked.

## Objective

Generate a synthetic benchmark for one target only: coordinated multi-account refund
abuse. No single identifier or exact identifier pair may define the target. Positive
and negative sequences must overlap in graph density, velocity, amounts, merchant
context, and identifier reuse.

## Positive event mechanisms

Train and validation use four motifs:

1. `device_hub_rotating_payment`: one device coordinates accounts while payment
   tokens rotate, with limited local pair reuse.
2. `payment_hub_rotating_device`: one payment token coordinates accounts while
   devices rotate.
3. `alternating_device_payment_chain`: adjacent accounts overlap alternately through
   devices and payment tokens; no universal identifier exists.
4. `two_core_bridge`: two local groups are joined by a bridge account carrying one
   identifier from each side.

The final test uses structurally disjoint motifs and longer horizons:

1. `sparse_address_payment_chain`: alternating address and payment links.
2. `device_address_ladder`: adjacent device and address overlaps with rotating
   payments.
3. `rotating_two_hub_bridge`: two device hubs crossed by two alternating payment
   hubs, so neither relation reproduces the development bridge.
4. `partial_pair_mesh`: several small device/payment pools with no universal pair.

## Legitimate hard negatives

Every split contains all six mechanisms:

1. `family_wallet`: a household shares address, payment, and a small device pool.
2. `group_purchase`: several customer profiles legitimately share the same device,
   payment token, merchant, and refund burst.
3. `corporate_card`: employees share a corporate payment token and office address.
4. `hostel_kiosk`: residents share an address and assisted-order device.
5. `office_network`: employees share an address and small device pool.
6. `support_migration`: duplicate or migrated profiles temporarily share device and
   payment identifiers across merchants.

These are label-negative even when they trigger the previous pair-reuse rule.

## Point-in-time features

The model may use only request context and current/prior event state. General graph
features include device, payment and address account counts; union-connected accounts;
multi-identifier neighbor count; largest and second-largest entity degree; identifier
balance; pair reuse; linked refund velocity; linked merchant count; and a transparent
overlap rarity transform. Scenario names, ring IDs, topology, sequence position, and
all future events remain evaluation-only.

## Acceptance gates

- Chronological split boundaries exist before any scenario injection.
- Train/validation and test positive topology names are disjoint.
- Generated train/validation and test relation-degree fingerprints are disjoint.
- Test ring sequences have a longer median duration than validation sequences.
- At least four positive motifs and six hard-negative motifs are represented.
- Legitimate device-payment-pair reuse is at least as common as positive pair reuse
  on the validation split.
- The rule `device_payment_pair_accounts_30d >= 2` must remain below 70% precision on
  validation. This gate is locked before final-test evaluation.
- Context-only validation PR-AUC must remain below 0.08.
- The action threshold must reach at least 85% validation precision with at least 30
  flags. Otherwise the benchmark reports failure.
- Final-test precision, recall, PR-AUC, calibration, costs, ring metrics, topology
  slices, and bootstrap intervals are reported once without threshold retuning.

## Version 2.1 structural-gate amendment

Before accepting v2 as complete, a stricter relation-aware degree-signature audit
found that the original `rotating_two_hub_bridge` was graph-isomorphic to the
development `two_core_bridge` for some cluster sizes. The earlier gate compared
topology names rather than generated graph structure.

Version 2.1 changes only that test mechanism to the crossed four-hub construction
described above. The acceptance gate now compares actual per-relation entity-degree
fingerprints across development and test rings. This amendment was fixed before
generating or scoring the v2.1 final test. The v2 final score is deprecated; v2.1
receives a newly generated benchmark and one locked final evaluation.

## Research basis

- Grab describes device and other entity sharing as valuable but noisy graph edges,
  and explicitly warns that shared network identity need not imply a physical link:
  https://engineering.grab.com/graph-for-fraud-detection
- IBM AMLSim separates normal behavior models from multiple alert typologies rather
  than defining all abuse through one motif:
  https://github.com/IBM/AMLSim
- The Fraud Dataset Benchmark emphasizes standardized split and evaluation contracts:
  https://github.com/amazon-science/fraud-dataset-benchmark
- HoloScope combines graph topology with temporal spikes rather than treating either
  as a sufficient label rule: https://arxiv.org/abs/1705.02505
- Return-fraud research distinguishes organized techniques, account ageing, address
  manipulation, and multiple abuse scripts:
  https://research-repository.uwa.edu.au/en/publications/return-fraud-and-abuse-diagnosing-the-problem-targeting-the-respo/

All generated records remain synthetic. These sources motivate mechanisms and
evaluation discipline; they do not provide MarginShield labels or production claims.
