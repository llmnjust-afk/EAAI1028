# Alert-Budgeted Clinical Early Warning

Manuscript and reference implementation for *Alert-Budgeted Clinical Early
Warning: Conformal Risk Control for Stateful Deterioration Alarm Policies*
(submitted to *Engineering Applications of Artificial Intelligence*).

Most early-warning systems are evaluated at a fixed decision threshold, which
means every method is compared at a different alarm load. This work instead
treats the alarm rate as a **budget** that must be certified, and calibrates a
stateful alarm policy against it at the patient level rather than the timestep
level.

## Contents

```
main.tex              manuscript (elsarticle, single column)
sections/             one file per section
refs.bib              bibliography
code/eaai/            reference implementation
code/tests/           66 unit tests, no data required
code/scripts/         command-line entry points
```

## The three pieces

1. **Stateful alarm policy** (`eaai/policy/alarm.py`). A threshold, a refractory
   window and an escalation tier. The refractory window is what makes alarms
   dependent within a patient, and it is exactly what the timestep-level
   calibration in prior work ignores.
2. **Blocked conformal risk control** (`eaai/policy/conformal.py`). Calibration
   units are patients, not timesteps. Proposition 1 in `sections/method.tex`
   bounds the probability that the deployed *population* alarm rate exceeds the
   budget; per-patient violation is reported as a separate diagnostic and is
   never claimed to follow from the proposition.
3. **Alarm-rate-matched evaluation** (`eaai/experiments/e2_budget.py`). Compares
   methods at equal alarms per patient-day and traces the lead-time/budget
   Pareto frontier.

## Quick start

```bash
cd code
python3 -m pip install -r requirements.txt
python3 -m pytest tests -q          # 66 passed
python3 scripts/run_e0.py           # certification benchmark, ~20 s on CPU
```

`run_e0.py` is the experiment that needs no clinical data and no GPU. It uses a
60-point threshold grid for speed; the reported quantities move only through
`log|Λ|`, so the grid resolution changes the slack by a few percent. It draws a
synthetic cohort with patient-level heterogeneity, calibrates under a fixed
budget, and reports two quantities that must not be conflated:

| quantity | meaning |
| --- | --- |
| `P(mean>B)` | how often the deployed population alarm rate exceeds the budget — what Proposition 1 bounds |
| `P(pt>B)` | how often an individual patient exceeds the budget — a diagnostic only |

Representative output at a budget of 1 alarm per patient-day, refractory window
4 h, 4000 patients, 200 random splits (`L_max = 5.54`, the 26-hour supremum):

```
procedure           n_cal  alpha  P(mean>B)           95% CI  P(pt>B)  rate/B   slack
blocked_bernstein     100   0.05      0.000  [0.000, 0.019]    0.016   0.082   1.108
blocked_bernstein     800   0.05      0.000  [0.000, 0.019]    0.285   0.734   0.243
blocked_hoeffding     800   0.05      0.000  [0.000, 0.019]    0.220   0.596   0.386
timestep_naive        100   0.05      0.430  [0.363, 0.499]    0.383   0.960   0.020
timestep_naive        800   0.05      0.235  [0.182, 0.298]    0.382   0.956   0.020
```

Three readings matter. Blocked calibration never exceeds the budget at any
calibration size, and it becomes less conservative as the calibration set grows,
which is the expected finite-sample trade-off. The first row also shows the cost
of that guarantee: at `n_cal = 100` the slack alone (1.108) exceeds the budget
of one alarm per day, so the certificate is vacuous and the policy suppresses
almost every alarm. A procedure can be valid and useless at the same time, which
is why `rate/B` is reported next to the violation frequency rather than instead
of it. The timestep shortcut uses the
*same* point estimator and differs only in treating the ~72 dependent steps per
patient as independent, so its guard band is 12 times narrower than the correct
one at equal calibration data; it then sits at 96% of budget with no margin and
breaks its own nominal level in 23% of deployments at `n_cal = 800` and 43% at
`n_cal = 100`. A second seed gives 9–35%, so the exact fraction is a function of
grid resolution and calibration size. That is the point: the shortcut certifies
nothing, so the violation rate is whatever the discretisation happens to give,
while the blocked procedure is valid by construction.

`P(pt>B)` is the diagnostic that must not be confused with the guarantee. It
rises with the calibration set because a sharper threshold concentrates alarms
on genuinely unstable patients, whose own rate then exceeds the average budget.
A bound on the mean says nothing about that tail, and the code reports it
separately for exactly that reason.

## Data

All cohorts are public and require no data use agreement.

| cohort | patients | role |
| --- | --- | --- |
| PhysioNet/CinC Challenge 2019, training set A | 20,336 | in-hospital training and calibration |
| PhysioNet/CinC Challenge 2019, training set B | 20,000 | cross-site test (hospitals B and C) |
| PhysioNet/CinC Challenge 2012 | 12,000 | external task transfer (mortality) |
| MIMIC-IV demo | 100 | pipeline smoke test only, no reported results |

Download the 2019 training sets and place each patient `.psv` under
`data/physionet2019/training_setA/` and `.../training_setB/`. Set
`EAAI_DATA_ROOT` to override the default location.

Two details that change results by several AUROC points if ignored:
`SepsisLabel` flips to 1 six hours **before** clinical onset, so the onset index
must be recovered as the first labelled step; and splits must be made by patient
with a purge gap of at least the label horizon, or trajectories from the same
admission leak across folds.

## Baselines

Off-the-shelf, not reimplemented: `pypots` (GRU-D, SAITS), `tsai`
(InceptionTime, ResNet, TCN), `lightgbm` for the tabular reference. Install them
from `requirements-optional.txt`. `eaai/scores/baselines.py` provides two
self-contained models that need nothing beyond core dependencies: a prefix
logistic regression over last-value and observation-rate features, and a masked
bidirectional GRU consuming values, mask and elapsed time.

Clinical scores (qSOFA, MEWS, NEWS, SIRS, partial SOFA) are implemented from
their source definitions in `eaai/scores/clinical.py`, because reviewers check
thresholds and partial-observation handling.

## Experiments

| id | what | training |
| --- | --- | --- |
| E0 | certification benchmark on synthetic cohort | none |
| E1 | in-hospital discrimination for 8 score functions x 5 seeds | 40 fits |
| E2 | cross-site transfer A to B and A to C | cached |
| E3 | blocked versus timestep calibration, 100 splits | none |
| E4 | budget sweep, lead-time/Pareto frontier | none |
| E5 | ablations: refractory, escalation, grid resolution, calibration size | none |
| E6 | observation sparsification | cached |
| E7 | subgroup and net-benefit analysis | cached |
| E8 | external task transfer to PhysioNet 2012 | 12 fits |

Scores are cached per `(cohort, model, seed)` by `eaai.scores.api.ScoreCache`,
so every experiment after E1 and E2 is post-hoc and needs no GPU. Total budget
is roughly 45 GPU-hours on a single 24 GB card.

## Status

Implemented and tested: policy, blocked conformal calibration, evaluation
metrics (AUROC/AUPRC with DeLong, ECE, calibration slope, lead time, net
benefit, bootstrap intervals, paired Wilcoxon), data harmonisation with an
ICULOS clock, patient-level splitting with purge, clinical scores, score cache,
synthetic cohort generator, E0 and E2 drivers, and the two reference baselines.

Not included: training loops for the external deep backbones (installed from
`pypots` and `tsai`; `eaai/experiments/e1_train.py` is backbone-agnostic and
accepts any estimator with `fit`/`predict`), and the real-cohort results, which
require downloading the data above.

## Citation

```bibtex
@article{eaai_alarm_budget,
  title  = {Alert-Budgeted Clinical Early Warning: Conformal Risk Control for
            Stateful Deterioration Alarm Policies},
  journal = {Engineering Applications of Artificial Intelligence},
  year   = {2026},
  note   = {Under review}
}
```
