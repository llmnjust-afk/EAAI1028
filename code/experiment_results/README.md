# Real-cohort experiment summaries

These JSON files are small, reviewable summaries copied from the Lab Compute
runs. Raw 2019 records and score caches live outside the repository under
`/data/lab/EAAI1028-data/` and are not redistributed here.

## Data and labels

- Source: PhysioNet/CinC Challenge 2019, public training sets A (20,336 records)
  and B (20,000 records).
- `label_lead_hours=6`, `horizon=6`: the shipped label boundary is shifted back
  to the candidate clinical onset, and the six preceding hours are positive.
- Left-censored records have no identifiable pre-onset target; post-onset and
  padded timesteps are excluded from discrimination and calibration metrics.
- Alarm-rate evaluation uses the full observed follow-up, not only the
  pre-onset window.

## E1: score discrimination (seed 0)

| Model | Partition | Patients | Positive patients | AUROC | AUPRC | ECE |
|---|---|---:|---:|---:|---:|---:|
| prefix logistic | A validation | 3,051 | 237 | 0.7653 | 0.0395 | 0.00083 |
| prefix logistic | A calibration | 2,033 | 177 | 0.7313 | 0.0375 | 0.00212 |
| prefix logistic | A test | 3,051 | 249 | 0.7405 | 0.0409 | 0.00128 |
| prefix logistic | B external | 20,000 | 919 | 0.6777 | 0.0173 | 0.00241 |
| masked GRU | A test | 3,051 | 249 | 0.7439 | 0.0443 | 0.00243 |
| masked GRU | B external | 20,000 | 919 | 0.6781 | 0.0201 | 0.00038 |

Across seeds 0--2, prefix logistic gives A-test AUROC $0.7356\pm0.0133$ and
AUPRC $0.0343\pm0.0058$; on B it gives $0.6761\pm0.0052$ and
$0.0172\pm0.0002$. The masked GRU gives A-test $0.7434\pm0.0108$ and
$0.0428\pm0.0048$; on B it gives $0.6829\pm0.0042$ and $0.0183\pm0.0016$.
The E2 GRU sweep is now reported for all three seeds; the target-site GRU
diagnostic remains seed 0 only.

Patient-level bootstrap (400 resamples, same patients for both models):
the GRU-minus-logistic A-test AUROC difference is 0.0039 with 95% interval
[-0.0233, 0.0316] and paired two-sided p=0.775; the AUPRC difference is 0.0034
with interval [-0.0055, 0.0138] and p=0.42. On B, the AUROC difference is
essentially zero and the AUPRC difference is 0.0027 with interval
[0.00016, 0.0059], p=0.05. The GRU is therefore not presented as a demonstrated
improvement; it is a secondary architecture check.

## E2: budgeted alarm policies (logistic, seed 0)

Minimum observed stay for the support bound: 24 h. This retains 15,837/20,336 A
records and 14,824/20,000 B records. Budgets are patient-average alarms per day.
Thresholds are calibrated on A calibration patients only; B uses the same
threshold and A-derived certificate, without target calibration.

| Budget | A alarm/day | A sensitivity | A median lead (h) | B alarm/day | B sensitivity | B median lead (h) |
|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 0.117 | 0.250 | 9.5 | 0.052 | 0.106 | 8.0 |
| 0.50 | 0.328 | 0.452 | 9.0 | 0.161 | 0.246 | 8.0 |
| 1.00 | 0.631 | 0.607 | 9.0 | 0.348 | 0.382 | 9.0 |
| 2.00 | 1.355 | 0.804 | 10.0 | 0.932 | 0.614 | 9.0 |
| 4.00 | 2.101 | 0.893 | 10.0 | 1.762 | 0.779 | 9.0 |
| 8.00 | 2.101 | 0.893 | 10.0 | 1.762 | 0.779 | 9.0 |

Eligible A test events: 168; B: 691. The 4- and 8-per-day rows coincide because
the threshold grid reaches its floor (0.01); the upper budgets are not binding.

Across the three logistic seeds, A sensitivity at budgets 0.25/0.5/1/2 is
$0.185\pm0.061$ / $0.397\pm0.067$ / $0.585\pm0.020$ / $0.774\pm0.032$; on B it
is $0.081\pm0.026$ / $0.179\pm0.063$ / $0.315\pm0.062$ / $0.557\pm0.051$.
The internal alarm rates are stable, but sensitivity varies materially at the
tightest budget and the cross-site loss is consistent.

Across the three-seed timestep comparator, budget 0.25 raises mean A alarm
rate from 0.108 to 0.180 and sensitivity from 0.185 to 0.283, with violation
0.113 vs 0.073; budget 0.50 raises rate from 0.312 to 0.425 and sensitivity from
0.397 to 0.470, with violation 0.196 vs 0.151. At 1/day the comparator remains
more active (0.744 vs 0.651), and at 2/day the grid collapses both procedures to
the same policy. These are range-specific diagnostics, not a coverage-rate claim.

The three-seed GRU at 1 alarm/day has A sensitivity 0.684 (SD 0.006) and B
sensitivity 0.375 (SD 0.078), versus 0.585 and 0.315 for logistic; its B alarm
rate is also higher (0.328 vs 0.276). The ordering changes with operating point,
so model choice is not a substitute for target-site calibration.

## Target-site calibration diagnostic

B is split 50/50 into patient-disjoint calibration and test halves. The
threshold is selected on B calibration labels and evaluated on B test labels;
this is an oracle upper bound, not a deployable label-free method.

| Budget | No-recal B sensitivity | Recal B sensitivity | No-recal B rate | Recal B rate |
|---:|---:|---:|---:|---:|
| 0.25 | 0.073 +/- 0.026 | 0.218 +/- 0.036 | 0.039 +/- 0.014 | 0.159 +/- 0.009 |
| 0.50 | 0.173 +/- 0.074 | 0.358 +/- 0.031 | 0.118 +/- 0.043 | 0.360 +/- 0.019 |
| 1.00 | 0.294 +/- 0.082 | 0.497 +/- 0.022 | 0.270 +/- 0.065 | 0.616 +/- 0.058 |
| 2.00 | 0.544 +/- 0.072 | 0.724 +/- 0.056 | 0.744 +/- 0.165 | 1.466 +/- 0.260 |

Means and sample SD over three logistic seeds. The target split retains 7,412
test patients and about 337 test events per seed. At 2/day the target selector
hits the threshold grid floor, so sensitivity saturates and the rate rises.

## Commands

```bash
python3 scripts/run_e1.py --data-root /data/lab/EAAI1028-data/physionet2019 \
  --subset A --model prefix_lr --seed 0 --external B \
  --cache-root /data/lab/EAAI1028-data/cache \
  --results-dir /data/lab/EAAI1028-data/results/e1

python3 scripts/run_e2.py --cache-root /data/lab/EAAI1028-data/cache \
  --model prefix_lr --seed 0 --cohort A_lead6 \
  --splits /data/lab/EAAI1028-data/results/e1/splits_A_s0.npz \
  --external-cohort B_lead6 --min-stay 24 --thresholds 200 \
  --budgets 0.25 0.5 1.0 2.0 4.0 8.0 --include-timestep \
  --out /data/lab/EAAI1028-data/results/e2

python3 scripts/run_e2_reuse.py --cache-root /data/lab/EAAI1028-data/cache \
  --model prefix_lr --seed 0 --cohort A_lead6 \
  --splits /data/lab/EAAI1028-data/results/e1/splits_A_s0.npz \
  --external-cohort B_lead6 --include-timestep \
  --out /data/lab/EAAI1028-data/results/e2

python3 scripts/run_e2_target_calibration.py \
  --cache-root /data/lab/EAAI1028-data/cache --model prefix_lr --seed 0 \
  --splits /data/lab/EAAI1028-data/results/e1/splits_A_s0.npz \
  --out /data/lab/EAAI1028-data/results/e2_target
```

The GRU run uses the same E1 driver with `--model gru --device cuda --epochs 20`.
It is a reference masked GRU, not a GRU-D reproduction.
