# MarginShield: Ground-Up Explanation

## 1. The Business Loss

A merchant returns money when a customer asks for a refund. Some refunds are valid.
Some are abuse: for example, a person may claim an item never arrived while keeping
it. A coordinated ring makes this harder by spreading the activity across several
customer accounts. Each account can look normal alone.

MarginShield does not decide that a customer is guilty. It asks a narrower question:
**does this refund request look like one event in a coordinated multi-account ring?**

## 2. What The System Sees

At the instant a refund is requested, the merchant can submit the customer, merchant,
device, delivery-address, payment-token, and product identifiers plus refund amount
and account context. MarginShield looks backward only. It counts recent accounts and
refunds connected to those submitted identifiers.

An identifier is evidence, not proof. A household can share an address and card; an
office can share devices; a hostel can share an address. The benchmark deliberately
contains those legitimate cases.

## 3. Point-In-Time Features

Imagine the current request arrives at 10:00. The engine first asks questions such
as: how many different accounts used this device in the prior 30 days, how many
linked refunds appeared in the prior 72 hours, do the same accounts overlap through
several identifiers, and are linked accounts refunding the same product? Only after
scoring does it add the 10:00 request to history.

That detail prevents future leakage and prevents the request from counting itself as
past behaviour. The exact same Python engine builds the training table and serves
live requests, which removes a common training/serving mismatch.

## 4. Why There Are Three Models

The graph rule is a transparent baseline. Logistic regression tests whether a simple
weighted linear combination is enough. CatBoost can represent nonlinear interactions,
such as a refund burst being suspicious only when combined with rotating identifiers
and product reuse. Chronological training folds selected CatBoost; the final test was
not used to choose it.

CatBoost's output is calibrated with Platt scaling on the earlier validation half.
The later half chooses one review threshold. It maximizes recall while requiring at
least 85% point precision and 30 flags. SHAP contributions explain which observed
features raised or lowered each case's calibrated log-odds. They are model
explanations, not causal proof.

## 5. What The Actions Mean

- **Approve:** the probability is below the locked review threshold.
- **Manual review:** the probability is above threshold, but named graph evidence is thin.
- **Verify evidence:** the probability is above threshold and the account overlaps
  through at least two identifier types, so an analyst can request specific evidence.

There is deliberately no automatic rejection.

## 6. What A Ring Means

The Rings page builds a graph without using `ring_id` or labels. Case nodes connect
to device, address, or payment-token nodes shared by multiple accounts. A candidate
needs at least three cases, three accounts, and one above-threshold request. Queue
rank P1 is unique and means “review first”; it is ordered by probability-weighted
conditional loss, not an arbitrary score called priority.

For evaluation, a true ring counts as detected only if its first valid flag occurs
before half of that ring's simulated loss. This rewards early intervention rather
than eventual recognition.

## 7. How To Read The Result

Final synthetic precision is 91.9%: 34 of 37 flagged requests are positive in this
simulator. Recall is only 13.0%: 228 positive requests are missed. At ring level, 15
of 39 rings are detected early, and 85% of generated candidate clusters correspond
to a true simulated ring. The day-bootstrap precision interval is 83.3%-100%, so the
85% bar is not guaranteed under resampling.

Sparse address-payment chains have 0% early recall. MarginShield is therefore a
precise triage layer, not comprehensive fraud coverage.

## 8. Why V2.1 Was Rejected

V2.1 looked much better, but within already-shared identities one reuse-balance
feature separated positive and negative simulations too easily. V3 added independent
sharing and more diverse legitimate structures. Its lower metrics are more
defensible because the obvious shortcut no longer passes the audit.

## 9. What Is Real And What Is Not

Real: the point-in-time engine, API, trained model, calibration, graph construction,
SHAP evidence, action policy, audit database, tests, and UI.

Synthetic: every benchmark event, abuse label, loss estimate, and measured model
performance. Olist has no refund-fraud labels. Real deployment requires Razorpay or
merchant event data and adjudicated outcomes, followed by temporal and merchant
holdouts and prospective shadow testing.

