# MarginShield References

This is the provenance index for MarginShield. References are grouped by how they
were used. A source listed here is not automatically a training-label source:
MarginShield does not redistribute source rows, and Olist contains no
refund-abuse ground truth.

## Product And Challenge Context

- [Razorpay Buildathon](https://razorpay.com/buildathon/) - official challenge
  page and track brief used to frame MarginShield as an AI Risk Manager prototype.
- [Razorpay Security For Customers](https://razorpay.com/docs/security/customers/)
  - public context for Razorpay's existing payment, fraud, refund-pattern, and
  review workflows. MarginShield is positioned as a focused refund-ring
  investigation layer, not a replacement for Razorpay's fraud stack.

## Data Sources And Provenance

- [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce)
  - commerce structure, order-value distributions, product categories, delivery
  and review context. Used only to inform high-level context; it has no
  refund-abuse labels and no source rows are redistributed. The upstream license
  is listed as CC BY-NC-SA 4.0.
- [Fraud E-Commerce](https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce)
  - reference for account age, device reuse, identity velocity, and e-commerce
  fraud feature families. Used only for aggregate calibration. The upstream page
  does not declare a license, so no source rows are included.
- [Bank Account Fraud Dataset Suite](https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022)
  - reference for identity consistency, account/device features, and velocity.
  Used only for aggregate calibration under the stricter interpretation of the
  conflicting license metadata documented in `legacy/source_research`.
- [Financial Fraud Alert Review Dataset (FiFAR)](https://doi.org/10.6084/m9.figshare.28351172)
  - reference for alert prevalence, analyst disagreement, uncertainty, and
  defer-to-human behaviour. Used only for aggregate calibration and review-policy
  thinking. The project records CC BY 4.0 provenance. The legacy downloader uses
  this [Figshare archive endpoint](https://ndownloader.figshare.com/files/52147616).
- [ASOS GraphReturns](https://osf.io/c793h/) - design reference for customer-product
  graph relationships and return-versus-non-return task framing. It is not loaded
  by the active pipeline; upstream metadata was not verified sufficiently to make
  a license claim.
- [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data)
  - schema reference for transaction, identity, address, email, and device
  feature groups. It is not loaded by the active pipeline and remains subject to
  the competition rules.
- [Amazon Fraud Dataset Benchmark](https://github.com/amazon-science/fraud-dataset-benchmark)
  - provenance-aware benchmark and adapter design reference. Per-source licenses
  apply; no source rows are redistributed.
- [Fraud Detection Handbook](https://github.com/Fraud-Detection-Handbook/fraud-detection-handbook)
  - temporal fraud simulation, class imbalance, customer baselines, burst
  scenarios, feature transformation, and time-aware evaluation methodology.
- [IBM AMLSim](https://github.com/IBM/AMLSim) - coordinated graph-motif and linked
  entity scenario design reference. The upstream repository is Apache 2.0.

## Modelling And Evaluation

- [Saito and Rehmsmeier, "The Precision-Recall Plot Is More Informative than the
  ROC Plot When Evaluating Binary Classifiers on Imbalanced Datasets"](https://pmc.ncbi.nlm.nih.gov/articles/4349800/)
  - basis for treating PR-AUC, precision, and recall as primary operating metrics
  for a rare ring target rather than relying on ROC-AUC alone.
- [Prokhorenkova et al., "CatBoost: unbiased boosting with categorical features"](https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html)
  - basis for retaining CatBoost for nonlinear interactions and high-cardinality
  categorical context.
- [scikit-learn probability calibration guide](https://scikit-learn.org/stable/modules/calibration.html)
  - reference for separating model fitting from calibration and evaluating whether
  scores can be interpreted as probabilities. MarginShield uses chronological
  out-of-fold Platt calibration.
- [Liu, Hooi, and Faloutsos, "HoloScope: Topology-and-Spike Aware Fraud Detection"](https://arxiv.org/abs/1705.02505)
  - reference for combining graph topology with temporal spikes and for testing
  sparse or low-density coordinated groups.
- [Grab Engineering, "Graph for fraud detection"](https://engineering.grab.com/graph-for-fraud-detection)
  - industry reference for linked-entity graphs, noisy connections, graph
  explainability, and the operational difficulty of real-time graph updates.
- [Childs, Clare, and Townsley, "Return Fraud and Abuse: Diagnosing the Problem, Targeting the Response"](https://research-repository.uwa.edu.au/en/publications/return-fraud-and-abuse-diagnosing-the-problem-targeting-the-respo/)
  - domain reference for organised return abuse, identity artefacts, account
  ageing, and the importance of a consistent operational response. It informs the
  problem framing; it is not a source of labels for this benchmark.

## Assistant And API References

- [Gemini 2.5 Flash model documentation](https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash)
  - optional free-tier model used by the server-side grounded assistant.
- [Gemini API pricing and data-use table](https://ai.google.dev/gemini-api/docs/pricing)
  - reference for free-tier limits and the warning that free-tier content may be
  used to improve Google's products. Only synthetic demo data should be sent.
- [Gemini generate-content API](https://ai.google.dev/api/generate-content)
  - API contract used by the optional server-side assistant integration.
- The server calls Google's [Generative Language REST endpoint](https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent)
  only when the optional API key is configured; the key remains server-side.

## Deployment References

- [Render free services](https://render.com/docs/free) and [Render compute plans](https://render.com/docs/compute-plans)
  - recommended hosting option for the complete FastAPI prototype; free storage is
  ephemeral and services sleep when idle.
- [GitHub Pages overview](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages)
  and [creating a Pages site](https://docs.github.com/en/enterprise-cloud%40latest/pages/getting-started-with-github-pages/creating-a-github-pages-site)
  - reference for static hosting. Pages cannot run the Python API, model bundle,
  SQLite audit trail, or protected assistant key.
- [Google Cloud Run FastAPI guide](https://docs.cloud.google.com/run/docs/quickstarts/build-and-deploy/deploy-python-fastapi-service)
  and [Cloud Run pricing](https://cloud.google.com/run/pricing) - container
  deployment option with a free usage allowance but a billing account requirement.
- [Koyeb instance types](https://www.koyeb.com/docs/reference/instances) and
  [Koyeb FastAPI deployment](https://www.koyeb.com/docs/deploy/fastapi) - fallback
  hosting option with constrained free compute and ephemeral local storage.
- [Vercel pricing](https://vercel.com/pricing) and [Python runtime documentation](https://vercel.com/docs/functions/runtimes/python)
  - considered for deployment. It is suitable for a static frontend or small
  serverless functions, but is a poor fit for this single-process FastAPI service
  with a loaded CatBoost bundle and local SQLite state.

## UI And Interaction Inspiration

These sites informed visual exploration only. MarginShield does not claim to copy
their components, assets, or code, and none is an application dependency.

- [Skiper UI](https://skiper-ui.com/)
- [React Bits](https://reactbits.dev/get-started/index)
- [Aceternity UI](https://ui.aceternity.com/components)
- [Spline](https://spline.design/)
- [Godly Design: Griffin](https://godly.design/website/griffin/)

## Local Provenance Files

- [`legacy/source_research/dataset_sources.json`](legacy/source_research/dataset_sources.json)
  records dataset names, URLs, licenses, intended roles, and integration status.
- [`DATASET.md`](DATASET.md) defines the active synthetic target and feature groups.
- [`SIMULATOR_V4_DESIGN.md`](SIMULATOR_V4_DESIGN.md) records the locked simulator,
  evaluation, and policy protocol.
- [`ARCHITECTURE.md`](ARCHITECTURE.md) records the runtime trust boundaries and
  offline/live execution order.
