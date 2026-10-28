# Alert-Budgeted Clinical Early Warning

Draft manuscript and reference implementation for **Alert-Budgeted Clinical
Early Warning: Conformal Risk Control for Stateful Deterioration Alarm Policies**,
with *Engineering Applications of Artificial Intelligence* as the intended venue.
This is **not a submitted paper or a completed clinical reproduction**: real-data
comparative experiments, results, discussion and conclusion remain incomplete.

The implemented statistical procedure is a **patient-blocked finite-grid upper
confidence bound**, using Hoeffding or empirical-Bernstein concentration. It is
not the standard expected-risk conformal risk-control construction. Existing API
class names retain “Conformal” for compatibility, but the guarantee requires iid
patient blocks from the deployment population, a fixed scorer/policy family/grid,
and a support-wide loss bound fixed independently of calibration/test data.
Patient exchangeability alone does not justify this concentration proof.

## Contents and implemented components

```
main.tex                 single-column elsarticle draft
sections/                manuscript sections; method split into reviewable parts
refs.bib                 bibliography
code/eaai/policy/         causal stateful alarm and upper-confidence selectors
code/eaai/data/           official-format readers, masks, clocks and patient splits
code/eaai/scores/         prefix logistic/LightGBM and causal masked GRU references
code/eaai/eval/           diagnostic and inferential metrics
code/eaai/experiments/    synthetic E0, generic training/cache, budget-sweep drivers
code/tests/              data-free unit and safety-property regression tests
code/scripts/            command-line entry points
```

The risk estimand is the **unweighted mean of each patient's alarms per day**,
not a pooled ward total divided by pooled patient-days. Neither a per-patient
maximum nor a realised ward alarm cap follows from its population-mean bound.
Methods share a requested budget, not necessarily identical realised alarm rates.
The smallest certified threshold is deployed; utility-maximising selection among
feasible thresholds is not implemented.

## Quick start and synthetic diagnostic

Python **3.10+** is required. From `code/`:

```bash
python3 -m pip install -r requirements.txt
python3 -m pytest tests -q
python3 scripts/run_e0.py --patients 4000 --splits 200 --seed 0
python3 scripts/audit_official2012.py  # optional network QC, all 4000 set-A records
```

The GRU needs optional PyTorch; its tests skip when PyTorch is absent. The
reference GRU currently trains on CPU, not CUDA. External deep-model adapters
and a measured full-experiment GPU-hour budget have not been supplied.

E0 draws a synthetic heterogeneous cohort, evaluates a fixed threshold grid,
and repeatedly partitions that cohort into calibration and test patients:

| Output | Interpretation |
| --- | --- |
| `P(mean>B)` | held-out **finite-cohort** mean exceedance; not the theorem's population-risk event |
| `P(pt>B)` | individual patients above the average budget; no tail guarantee is claimed |
| `fallback_fraction` | fraction of splits with no finite feasible grid threshold |
| `rate/B` | empirical budget utilisation, including zero for fallback |

Repeated partitions reuse patients. Wilson intervals on those split indicators
are descriptive; they do not establish nominal coverage or prove the population
risk theorem. The deliberately invalid baseline changes only the confidence
radius's effective sample size, using **calibration-only** timestep counts.
It is not a reproduction of a named published timestep-conformal method.
Previous representative results were removed because they used an unsafe finite
fallback and an inflated timestep count from the whole cohort. Rerun the corrected
driver rather than reusing those numbers.

If no finite threshold is certified, the policy uses `threshold=+inf`, an exact
no-alarm fallback with risk zero and status `no_alarm_fallback`. Its grid index
is **-1, a sentinel, never the last loss-matrix column**. E0/E2 explicitly evaluate
zero fallback losses. E0 JSON uses `null` when the median threshold is infinite.
Raw loss values outside `[0,L_max]` are rejected instead of clipped. Optional
isotonic output is diagnostic only and never replaces the raw certificate.

## Data conventions and limits

Official sources: [2019](https://physionet.org/content/challenge-2019/1.0.0/) and
[2012](https://physionet.org/content/challenge-2012/1.0.0/).
Pass the root directory explicitly to the readers; no `EAAI_DATA_ROOT` environment
variable is consumed by the implementation.

- **2019:** put `.psv` files under `training_setA/` or `training_setB/` (20,336
  and 20,000 records). These are the two public hospital datasets; a public
  A→C hospital experiment is not supported. Supplied labels turn positive six
  hours before clinical onset. Default `label_lead_hours=0` retains the shipped
  label boundary; `=6` shifts it only when that boundary is identifiable.
  Initially positive records have `onset_left_censored=True` and onset sentinel
  zero: they cannot supply identifiable pre-onset targets or lead times. Shifted
  events beyond follow-up are excluded from lead-time cases, not silently capped.
- **2012:** put official `Time,Parameter,Value` records under `set-a/`, `set-b/`
  or `set-c/` and the corresponding `Outcomes-a/b/c.txt` at the root. Outcomes
  are joined by RecordID, never file order. Non-hourly measurements become
  available at `ceil(timestamp/60)`; the last valid within-hour measurement wins.
  The mortality input has decision times 0,...,48 h, **49 points, not a 49-hour
  deterioration follow-up**. All onset entries are -1 because no event onset is
  available. Mortality is returned separately, never included in model features.
  Admission features are Age/Gender/Height/ICUType; no hospital indicators are
  invented. `-1` is missing; other negative/nonfinite measurements are also
  missing with explicit `qc_invalid_measurements` counts. The official set-A
  network audit detects 24 such out-of-documentation-range measurements.
- Shared **names are not a complete cross-cohort semantic harmonisation**.
  Source units are retained; 2012 Lactate is documented in mmol/L whereas the
  2019 release documents mg/dL. Unit conversion, a shared static feature schema,
  and task-specific labels are required before direct external model transfer.
- Patient/record-disjoint splits prevent within-record leakage. The optional
  `purge` function is an outcome-dependent eligibility restriction and can cause
  selection bias; it is not a necessary leakage guard or proof against censoring.
  Public record identifiers cannot establish that repeated admissions belong to
  different people, so the independence assumption needs separate justification.
- Missing-value delta is finite elapsed time **before** incorporating the current
  observation, including at missing hours. Post-onset and padded training targets
  are NaN and excluded from the loss, rather than treated as negative examples.

## Baselines, cache compatibility and remaining work

The package provides prefix logistic regression (optional LightGBM) and a
**causal unidirectional** masked GRU. Perturbing future observations is tested
not to change past scores; patient prediction batches limit forward-pass memory.
Static missingness is median-imputed in the prefix learner and zero-imputed in
the GRU reference. Neither implementation is an external paper's complete model.

The clinical-score functions are **research-only partial instruments**. NEWS
omits consciousness and supplemental oxygen; SIRS lacks the immature-neutrophil
alternative; other omissions are listed in `MISSING_COMPONENTS`. Contiguous
extensions handle fractional chart values instead of silently assigning zero
to a gap between integer bands. Raw clinical scores are not probabilities: a
training-only probability adapter is required before ECE or net-benefit metrics.

Old unversioned caches are rejected. **Regenerate all earlier GRU/prediction
caches and downstream results**; changing the code cannot repair scores already
computed using future information. Cache format version 2 validates shapes,
lengths, onset metadata and finite valid scores.

Not implemented/completed: external architecture adapters, the complete multi-seed
clinical training grid, the official 2019 challenge utility scorer, full cross-site
and external-task experiments, observation-sparsification/subgroup study drivers,
and real-cohort results. MIMIC demo processing is not present. The E2 module is
a generic cached-score budget sweep, not a completed A→B/A→C transfer study.
No clinical superiority, submission status or measured 45-GPU-hour result is claimed.

Corrected synthetic outputs for seeds 0, 1 and 2 are under `code/audit_results/`.
They remain finite-cohort diagnostics, including cases where certification is
achieved by the operationally useless no-alarm fallback. See `AUDIT_REPORT.md`
for the second audit's defects, checks and limitations.
