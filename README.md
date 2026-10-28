# Alert-Budgeted Clinical Early Warning

Manuscript and reference implementation for **Alert-Budgeted Clinical
Early Warning: Patient-Blocked Risk Calibration for Stateful Deterioration Alarm Policies**,
with *Engineering Applications of Artificial Intelligence* as the intended venue.
This is **not a submitted paper or a clinical validation**. The real-data
comparative experiments and manuscript are complete for the documented scope;
clinical effectiveness, alert-fatigue reduction and deployment readiness are not
claimed.

The implemented statistical procedure is a **patient-blocked finite-grid upper
confidence bound**, using Hoeffding or empirical-Bernstein concentration. It is
not the standard expected-risk conformal risk-control construction. Existing API
class names retain “Conformal” for compatibility, but the guarantee requires iid
patient blocks from the deployment population, a fixed scorer/policy family/grid,
and a support-wide loss bound fixed independently of calibration/test data.
Patient exchangeability alone does not justify this concentration proof.

## Contents and implemented components

```
main.tex                 single-column elsarticle manuscript
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
python3 scripts/run_e2_reuse.py --help  # one loss matrix for all E2 partitions
python3 scripts/run_e2_target_calibration.py --help  # oracle target diagnostic
```

The GRU needs optional PyTorch; its tests skip when PyTorch is absent. The
reference GRU now accepts `--device cuda`; the reported 2019 runs used an
RTX 4090. It remains a reference masked GRU, not a GRU-D reproduction, and no
formal GPU-hour budget has been measured.

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

## Real-cohort status (2019 A/B)

All 40,336 public 2019 records were downloaded and checksum-verified. The
completed summaries are in `code/experiment_results/`; raw records and score
caches remain outside the repository.

- **E1 (score discrimination).** Prefix logistic, three seeds: A-test AUROC
  $0.7356\pm0.0133$, AUPRC $0.0343\pm0.0058$; B external AUROC
  $0.6761\pm0.0052$, AUPRC $0.0172\pm0.0002$. Masked GRU, three seeds:
  A-test $0.7434\pm0.0108$ / $0.0428\pm0.0048$; B external
  $0.6829\pm0.0042$ / $0.0183\pm0.0016$. A seed-0 patient bootstrap does not
  establish a significant GRU advantage on A (AUROC difference 0.0039, 95\%
  interval $[-0.0233,0.0316]$).
- **E2 (budgeted policies).** With A-only calibration and a 24-hour minimum
  observed stay, prefix logistic at 1 alarm/patient-day reaches A sensitivity
  0.585 (three-seed mean) but B sensitivity 0.315; the A-derived threshold is
  therefore not transportable without target calibration. At the tightest
  budget (0.25/day), mean A sensitivity is 0.185 and B sensitivity 0.081.
  The masked GRU reaches 0.684 at A and 0.375 at B at 1/day over three seeds,
  with a higher B alarm rate (0.328 versus 0.276); it is a secondary check, not
  a demonstrated improvement.
- **E2 target diagnostic.** An oracle target-site split of B (7,412 calibration
  and 7,412 test patients) raises B sensitivity at 1/day from 0.294 to 0.497
  but also raises the realised rate from 0.270 to 0.616. Target labels are used
  for threshold selection, so this is an upper bound, not a deployable label-free
  procedure. The GRU target diagnostic is seed 0 only.
- **Limits.** These are retrospective, record-disjoint experiments on one public
  challenge cohort with a fixed scorer grid. They are not clinical validation,
  do not establish alert fatigue reduction, and do not include label-free
  target adaptation, subgroup/sparsification analyses or the official 2019
  utility scorer.

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

## Baselines, cache compatibility and scope limits

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

Not implemented: label-free target adaptation, external architecture adapters,
the official 2019 challenge utility scorer, 2012 same-task transfer,
observation-sparsification and subgroup study drivers, and MIMIC processing.
The oracle target-site diagnostic in `experiment_results/target_summary.json`
is reported separately and is not a deployment method.
The E2 A→B experiment is complete for the reference scorers, but it is not an
A→C study and does not establish clinical superiority. No measured 45-GPU-hour
result, deployment claim or submission status is claimed.

Corrected synthetic outputs for seeds 0, 1 and 2 are under `code/audit_results/`.
They remain finite-cohort diagnostics, including cases where certification is
achieved by the operationally useless no-alarm fallback. See `AUDIT_REPORT.md`
for the second audit's defects, checks and limitations.
