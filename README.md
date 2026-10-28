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
code/tests/           56 unit tests, no data required
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
python3 -m pytest tests -q          # 56 passed
python3 scripts/run_e0.py           # certification benchmark, ~20 s on CPU
```

`run_e0.py` is the experiment that needs no clinical data and no GPU. It draws a
synthetic cohort with patient-level heterogeneity, calibrates under a fixed
budget, and reports two quantities that must not be conflated:

| quantity | meaning |
| --- | --- |
| `P(mean>B)` | how often the deployed population alarm rate exceeds the budget — what Proposition 1 bounds |
| `P(pt>B)` | how often an individual patient exceeds the budget — a diagnostic only |

Representative output at a budget of 1 alarm per patient-day, refractory window
4 h, `alpha = 0.05`:

```
procedure           n_cal  P(mean>B)  P(pt>B)  rate/B
blocked_bernstein     100     0.000    0.016   0.082
blocked_bernstein     800     0.000    0.292   0.750
blocked_hoeffding     800     0.000      --      --
timestep_naive        200     0.540    0.393   0.982
```

Two readings matter. Blocked calibration is valid at every calibration size and
becomes less conservative as the calibration set grows, which is the expected
finite-sample trade-off. The timestep-level shortcut that prior work uses
exceeds the budget in roughly half of deployments while looking well calibrated
on average, which is the failure this paper is about.

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
