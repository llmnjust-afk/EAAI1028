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
The GRU E2 sweep was run only for seed 0, so its operational numbers are a
secondary check rather than a three-seed estimate.

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

The deliberately invalid timestep comparator at budget 0.25 selects a lower
threshold than the blocked procedure (0.0445 vs 0.0543), raises A alarm rate
from 0.117 to 0.183 and patient-violation fraction from 0.074 to 0.115. At
budget 0.50 it raises alarm rate from 0.328 to 0.450 and violation fraction from
0.161 to 0.214. At budgets at or above 1.0 the grid selects the same threshold
for both procedures on this split. These are single-split diagnostics, not a
coverage-rate claim.

The GRU at 1 alarm/day has A sensitivity 0.685 and B sensitivity 0.326, versus
0.607 and 0.382 for logistic. The ordering therefore changes across sites; model
choice is not a substitute for target-site calibration.

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
```

The GRU run uses the same E1 driver with `--model gru --device cuda --epochs 20`.
It is a reference masked GRU, not a GRU-D reproduction.
