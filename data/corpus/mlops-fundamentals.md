# MLOps Fundamentals

MLOps is the set of practices for taking a model from a training script to
a reliably operated production system — versioning not just code but data
and models, automating training and deployment pipelines, and monitoring
a live model the way you'd monitor any other production service, plus
some things unique to ML (data drift, model staleness) that traditional
software monitoring doesn't cover.

## Experiment tracking and model registries

Training a model well requires iterating over hyperparameters, data
versions, and architectures — without tracking, it's easy to lose which
run produced which artifact under which conditions. Experiment tracking
tools (MLflow, Weights & Biases) log parameters, metrics, and artifacts
per run, so results are reproducible and comparable. A model registry sits
one layer up: it tracks named model versions through a lifecycle
(staging, production, archived), records which run produced each version,
and is the single source of truth a serving system queries for "what's
the current production model." Promoting a version from staging to
production should be a deliberate, auditable action, not a file overwrite.

## CI/CD for ML

Continuous integration for ML extends normal software CI (lint, unit
tests, build) with ML-specific checks: does the model still meet a minimum
accuracy threshold on a held-out set, did preprocessing code changes
break feature parity between training and serving, does the packaged
model artifact actually load and produce output of the expected shape.
Continuous deployment adds a promotion gate — often an accuracy-delta
check against the current production model — before a new version can
replace it, and ideally a rollback path that's just as automated as the
deploy path.

## Monitoring: the ML-specific part

Standard service monitoring (latency, error rate, resource usage) still
applies to a serving endpoint, but ML systems need two more things.
Data/feature drift monitoring compares the statistical distribution of
live input features against the training distribution — a two-sample
statistical test (Kolmogorov-Smirnov is common) per feature, flagging
features whose distribution has shifted meaningfully. Prediction
monitoring tracks the model's output distribution over time, which can
reveal problems (a stuck upstream feature, a broken preprocessing step)
even without ground-truth labels to check accuracy directly. Neither
substitutes for periodically re-evaluating against fresh labeled data
when it becomes available — drift detection tells you *when* to look
harder, not what the right answer is.

## Canary and shadow deployment

Rolling out a new model version to 100% of traffic immediately is risky
even after offline evaluation looks good, because offline metrics don't
always predict production behavior on the true live distribution. Canary
deployment routes a small percentage of live traffic to the new version
and compares outcomes before ramping up. Shadow deployment goes further:
the new version runs on every request alongside the current production
model, but its output is only logged and compared, never actually
returned to the user — zero user-facing risk, at the cost of running
inference twice per request.
