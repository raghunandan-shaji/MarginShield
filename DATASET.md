# MarginShield V4 Synthetic Benchmark

## Unit And Target

One row is one refund request. The target `ring_label=1` means that the request was
generated as part of a coordinated multi-account refund-abuse sequence. It does not
mean generic refund abuse or fraud.

Every row and label is synthetic. Olist has no refund-fraud labels and supplies no
fraud truth or redistributed source rows.

## Chronology

The 210-day horizon is partitioned before scenario injection:

- train: first 147 days;
- validation: next 31 days;
- final test: last 32 days.

The test period uses slower ring sequences and relation-aware topologies absent from
train and validation. Development after this locked evaluation requires a new
version and untouched test period.

## Mechanisms

Positive sequences vary device, address, payment-token, product, merchant, customer
history, and timing relationships. Difficult legitimate sequences include family
wallets, group purchases, corporate cards, hostel kiosks, office networks, and
support migrations. Ordinary independent traffic also includes incidental shared
identifiers.

Address, merchant, timing, product, and existing-customer modes overlap across
labels. Shared identity or same-product reuse is therefore insufficient by itself.

## Point-In-Time Contract

`marginshield/feature_engine.py` consumes events in chronological order. For each
request it computes features before committing that request to state. It includes:

- request context and refund economics;
- customer refund velocity;
- merchant trailing refund volume;
- trailing distinct accounts per submitted device, address, and payment token;
- device-payment pair reuse;
- direct connected-account reach and multi-identifier neighbours;
- linked 72-hour refund bursts and trailing merchant diversity;
- linked accounts refunding the same product; and
- graph overlap/reuse summaries.

The model cannot access future links, post-decision outcomes, `ring_id`, scenario
type, topology, sequence position, cluster-generation modes, or latent simulator
fields. Batch training and live API scoring call the same feature engine.

## Generation Gates

Generation fails on chronology, leakage, topology, hard-negative, category parity,
or shortcut violations. The key shortcut gate conditions on requests that already
share at least one identifier, then requires the best single numeric feature's AP to
remain below twice that subset's prevalence.

For active v4 validation data:

- shared subset: 2,027 rows;
- positive prevalence within subset: 10.95%;
- maximum univariate AP: 0.1426;
- AP lift: 1.302x;
- independent legitimate rows with sharing: 376;
- any-sharing rule precision: 11.02%.

These are artifact checks, not proof of real-world fidelity.

## Reproduction

```bash
python3 build_dataset.py --rows 75000 --seed 20260905
python3 -m unittest tests.test_dataset -v
```

The active data dictionary marks every column as model feature, target-only, or
metadata. The manifest and full gate output are in `data/processed/dataset_manifest.json`
and `data/reports/validation_report.json`.
