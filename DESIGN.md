# DESIGN.md — Handling 1,000 New Labelled Samples

## Short answer

**No.** The deployed model does not, and must not, automatically retrain itself on new
data. Retraining is a separate, offline, human-gated process, and the currently-deployed
model is never mutated in place. This document describes the process this service actually
implements the scaffolding for (model registry, evaluate endpoint, promotion/rollback with
an audit trail) and the parts of the loop that would be operated around it.

This is a deliberate choice, not a missing feature: in a clinical context, a model that
updates itself on incoming data is a patient-safety risk before it is an engineering
convenience. A clinician acting on a prediction needs to know that prediction came from a
model whose behavior was validated and signed off on — not one that silently shifted last
night because of a batch of noisy or biased recent labels.

## The process, end to end

### 1. New data arrives — it never touches the production model directly

The 1,000 new labelled samples land in a **staging/review location** — a `pending_review`
table or a quarantined CSV drop, not the training set and not anywhere the live service
reads from. They go through the same validation the API already enforces on every
prediction request (`app/schemas.py`'s range/type checks) before being considered usable:
malformed rows are rejected at this stage, the same way a bad `/predict` request is
rejected today.

The production model (`model_versions.status = 'production'`) is **frozen**. Nothing in
this pipeline writes to its artifact file or updates its weights. The only way its `status`
changes is through the promotion endpoint, described below, and that always creates a *new*
row rather than mutating the existing one.

### 2. Retraining happens offline, isolated from the live service

A retraining job — run manually, or on a schedule, or triggered by a monitoring alert (see
`/metrics/drift` in this service, which computes PSI-based feature drift against the
training baseline right now) — is a **separate process**, not a code path inside the
FastAPI service:

- It runs in its own environment (a notebook, a scheduled job, a training container),
  independent of the request-serving process. It reads the frozen production feature
  contract (`scaler.feature_names_in_` / `app/ml/schema.py::FEATURE_ORDER` in this
  codebase) and the combined old + new-reviewed data.
- It produces a **new artifact**: a new `.joblib`/`.keras` file (or a retrained scaler, if
  the feature distribution genuinely shifted — itself a decision requiring scrutiny, since
  a changed scaler changes what "the same input" means for every other version).
- The new artifact is registered as a **new `model_versions` row with `status='candidate'`**
  — exactly the mechanism this service already uses for the MLP alongside the seeded
  RandomForest. It is inert: nothing routes live traffic to it, and `GET /models/active`
  keeps returning the existing production version untouched.

### 3. Validation / challenger comparison before promotion is even considered

Before a human is even asked to approve anything, the candidate must clear an evaluation
gate — implemented here as `POST /models/{id}/evaluate`:

- It is scored against a **fixed, held-out evaluation set** that is *not* part of what it
  was trained on — in this service, a deterministic seeded slice of the reference dataset;
  in a full production loop, this would be a periodically-refreshed but still-held-out
  clinical validation set, ideally with some independently-adjudicated labels rather than
  only self-reported survey data (a limitation of the BRFSS-derived training data itself,
  noted in REPORT.md).
- Metrics (accuracy, precision, recall, F1, ROC-AUC — recall matters particularly here,
  since a false negative in a diabetes screening tool has different real-world cost than a
  false positive) are computed for the **candidate** and compared against the same metrics
  already stored on the **current production version** (`model_versions.metrics`). This is
  the champion/challenger comparison: two numbers side by side, on the same eval set, not a
  single number in isolation.
- The comparison is visible in the Model Registry view of the dashboard before anyone acts
  on it — both versions' metrics stored and displayed side by side.

A candidate that doesn't clear this bar simply stays a candidate. Nothing about receiving
new data forces a promotion; "we retrained" and "we should ship this" are different claims,
and the gate exists specifically to keep them different.

### 4. An explicit approval step gates promotion

Promotion is **not automatic**, ever — not on a schedule, not on a metric threshold being
crossed, not on a timer. `POST /models/{id}/promote`:

- Is restricted to the `ml_engineer` role (`require_ml_engineer` in `app/security.py`) —
  a clinician or viewer account cannot do this even accidentally.
- Requires a non-empty, human-written `reason` string — there is no "auto-promote" flag or
  unattended path.
- Refuses outright if the candidate has no recorded evaluation metrics (`422`) — you cannot
  promote something that was never compared against production.
- Verifies the candidate artifact actually loads before touching any status, so a
  promotion can't leave the service in a broken state.

In an operational deployment, this is the natural point to require a second sign-off (a
"maker/checker" pattern — one engineer prepares the candidate and evaluation, a different
person or a clinical/QA reviewer approves the promotion) rather than a single actor; the
role check here is the technical seam that pattern would attach to.

### 5. Versioned deployment — never an in-place overwrite

When a promotion is approved, the system does not overwrite the production artifact file or
mutate the existing `model_versions` row. It:

1. Demotes the current production version's row to `status='archived'` (its artifact file,
   metrics, and full history stay exactly as they were — nothing is deleted).
2. Promotes the candidate's row to `status='production'`.
3. Records **both** transitions in `promotion_events` — actor, timestamp, reason, and a
   metrics snapshot at that moment.

A DB-level partial unique index (`WHERE status = 'production'`) guarantees there is never a
moment with two production models or, via a bug, zero — the transition is atomic. Every
past version, and the full sequence of who-promoted-what-and-why, remains queryable
indefinitely (`GET /models/promotions/history`).

### 6. Rollback if the new version underperforms in production

Because promotion is versioned rather than destructive, rollback is the same operation run
in reverse — `POST /models/rollback` — not a restore-from-backup exercise:

- Demotes the current (underperforming) production version back to `archived`.
- Promotes the specified (or, by default, the most recently archived) prior production
  version back to `production` — the exact artifact that was previously serving traffic,
  unchanged.
- This, too, requires the `ml_engineer` role, a reason, and produces its own
  `promotion_events` entries, so a rollback is exactly as auditable as a promotion.

This is what makes rollback cheap and safe: it never depends on retraining anything or
reconstructing a lost artifact — the previous version's file and row were never touched.

### What would trigger a human to start this process

The 1,000 new samples don't retrain anything by themselves, but they should surface as a
**signal** for a human to decide whether to start the process above:

- `GET /metrics/drift` — this service's PSI-based comparison of the recent inference input
  distribution against the training baseline — flags if the new samples represent a
  population meaningfully different from what the production model was validated on.
- A simple count/cadence trigger ("1,000 new reviewed samples available") that opens a
  ticket or notification for an ML engineer to *decide* whether retraining is warranted,
  rather than a script that retrains and promotes unattended.

Both are inputs to a decision a person makes, not conditions that fire an automated
pipeline through to production traffic. That line — data can trigger a *review*, never a
*deployment* — is the core safety property this design is built to preserve.
