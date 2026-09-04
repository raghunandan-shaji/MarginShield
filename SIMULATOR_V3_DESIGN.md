# MarginShield Simulator v3 Lock

This document fixes the v3 benchmark design before model selection or final-test scoring.

## Target

The only positive class is a refund request generated as part of a coordinated, multi-account refund-abuse ring. Ordinary refund abuse, policy abuse by one account, chargebacks, and generic payment fraud are outside the target.

Olist has no refund-fraud labels. It is used only as loose inspiration for commerce context. Every benchmark row and every target label is synthetic, so benchmark performance is not evidence of production performance.

## Chronology

- Seed: `20260905`
- Total horizon: 210 days
- Train: days `[0, 147)`
- Validation: days `[147, 178)`
- Final test: days `[178, 210)`
- Boundaries are assigned before any scenario or label injection.
- All graph, velocity, customer-history, and merchant features are computed before committing the current event to state.
- The test split is scored only after the simulator, candidate models, calibration method, and threshold policy are locked on train and validation.

## Event Mechanisms

Ring and difficult-legitimate clusters are generated as timestamped event sequences. They vary independently along four nuisance axes:

- address reuse: unique, paired, or hub;
- merchant concentration: one merchant, a two-merchant pool, or spread;
- timing: uniform, bursty, or exponential spacing;
- customer history: a mixture of existing and newly created customer accounts.
- product coordination: same product, product pairs, or spread products, with all modes represented in both classes.

Shared payment tokens use a coherent payment rail. Unique tokens retain the ordinary payment-method distribution, including cash on delivery. Ordinary independent traffic also contains incidental cross-account identifier sharing. Product reuse is an observable coordination signal, but difficult legitimate group-purchase and household cases also reuse products so it cannot define the label by itself.

Train and validation use four ring motifs. Final test uses four slower, relation-aware motifs whose topology names and graph degree fingerprints are disjoint from development rings. Legitimate negatives include households, group purchases, corporate cards, hostels, offices, and support migrations.

## Leakage And Shortcut Gates

Generation fails unless all of the following hold:

- target-only fields are absent from the model feature contract;
- chronological boundaries and relation-aware topology separation hold;
- difficult legitimate shared-identity examples exist;
- at least 25 independent-legitimate validation rows exhibit cross-account sharing;
- a rule based on any shared identifier has below 50% precision on validation;
- the strongest single numeric feature inside the shared-identity validation subset has average precision below twice that subset's prevalence;
- every nuisance mode occurs in both positive and negative injected validation rows;
- positive validation rows cover every observed payment-method context;
- the exact offline feature table can be reproduced by the incremental live engine.

The gates reduce obvious simulator fingerprints; they cannot prove that synthetic data resembles Razorpay production traffic.

## Locked Modelling Policy

Candidates are an explainable graph rule, regularized logistic regression, and a regularized CatBoost classifier. Rolling chronological folds choose the model. Platt calibration uses the earlier half of validation. The later half selects one manual-review threshold by maximizing recall subject to at least 85% point precision and at least 30 flags.

The threshold lies at the midpoint in log-odds between the last included and first excluded validation score when such a gap exists. This is specified before final-test scoring and avoids representing a single observed score as a meaningful boundary. It does not use final-test labels.

No case is auto-rejected. Below threshold is approve. Above threshold with multi-identifier structural evidence is verify evidence; other above-threshold cases are manual review.
