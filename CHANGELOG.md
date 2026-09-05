# Changelog

## 4.0.0-value-policy-locked - 2026-09-05

- Trained and locked the V4 model on the V4 dataset. The dataset had been rebuilt
  without a retrain, leaving a V3 model scoring V4 rows and a V3 report quoted as
  current performance.
- Removed the 85% precision floor from both intervention tiers. It was never derived
  from the cost model, and on the V4 validation window it is unreachable at the
  required flag count. Both boundaries now maximise synthetic net preventable value
  under an explicit 30-100 review capacity.
- Published `precision_floor_comparison` in the model report so the value a fixed
  floor would forfeit stays auditable rather than merely asserted.
- Recorded the change as Amendment 1 in `SIMULATOR_V4_DESIGN.md` rather than
  rewriting the pre-registered protocol.
- Removed the constraint requiring the verification threshold to exceed the
  manual-review threshold. Verification is the cheaper action and is gated on reused
  identifiers, so the tiers are separated by action and evidence, not by score order.
- Fixed ring-catalog actions, which used only the manual-review threshold and so
  mislabelled cases in the new verification band.
- Stamped the runtime SQLite database with its dataset and model generation. A
  rebuild now resets stale decisions automatically instead of refusing to start;
  previously any dataset rebuild left the server unable to boot.
- Replaced every hardcoded metric in the API tests with values derived from the
  locked report, and added a test asserting the report and dataset are the same
  generation.

## Interface and workflow corrections - 2026-09-05

- Split the frontend into explicit casework, final-test portfolio, validation policy,
  and mixed current-graph data contracts; removed the last stale held-out graph label.
- Added caught/missed request and early-ring coverage to Portfolio and exact confusion
  counts to every Policy Lab scenario.
- Restored distinct editorial deck copy for every product view.
- Added a plaintext grounded assistant with an optional Gemini free-tier backend and
  a deterministic local fallback.
- Disabled caching for app shell assets so the complete four-tab navigation is not
  masked by a stale browser copy.
- Added explicit queue sorting by ring probability or refund exposure in either direction.
- Replaced operational-signal text in queue rows with merchant verticals.
- Persisted analyst dispositions now restore into Casework and change button/status state.
- Replaced verbose policy badges with concise action states and corrected badge spacing.
- Removed the duplicated portfolio action list and added a compact colour legend.
- Ordered Policy Lab thresholds from low to high and documented the validation lock rule.
- Restored Casework, Portfolio, Policy Lab, and Abuse Rings navigation on both pages.
- Isolated API tests from the runtime SQLite database.

## 3.0.0-locked - 2026-09-05

- Fixed pandas timestamp-resolution dependence that could make windows 1000x too long.
- Replaced separate batch calculations with one offline/live incremental engine.
- Added organic shared identities and harder sparse legitimate cluster mechanisms.
- Added overlapping merchant, address, timing, customer-history, payment, and product context.
- Added a within-shared-identity shortcut gate and nuisance/category coverage gates.
- Added linked same-product coordination as a point-in-time signal.
- Re-locked a new seed, 210-day period, model, calibration, and threshold.
- Replaced the second score threshold with an observable structural evidence rule.
- Added typed raw-event scoring, SQLite decision history, and real analyst actions.
- Added day-block confidence intervals, tail calibration, topology recall, and cost sensitivity.
- Aligned ring evaluation and investigation to a trailing 30-day window.
- Preserved v2.1 model/report history rather than deleting an inconvenient result.

## 2.1.0-locked - superseded

Added structurally different test topologies and achieved strong synthetic metrics.
Subsequent audit found a simulator shortcut: low identifier-reuse balance among
already-shared identities separated the target too easily. Artifacts remain under
`data/model/v2.1/` and `data/reports/v2.1/` for auditability, but are not active.

## 2.0.0 - superseded

Introduced the coordinated-ring target and graph investigation UI. Earlier broad
refund-abuse experiments remain isolated from the active application.
