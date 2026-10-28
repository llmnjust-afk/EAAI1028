# Second correctness audit — 28 October 2026

This audit follows repository commit `8e8378a`. It covers the supplied reference
code and consistency of the draft manuscript, not a claim to have completed all
planned clinical experiments or reproduced every external backbone.

## Confirmed defects and corrective changes

| Defect | Correction |
| --- | --- |
| Bidirectional GRU exposed earlier predictions to later observations | Unidirectional causal GRU; future-perturbation regression and prediction-batch equivalence check |
| PhysioNet 2012 assumed fictitious hourly wide records with embedded outcomes | Official long-form parsing, causal hourly availability, separate outcome join by RecordID |
| Real 2012 records contained out-of-documentation-range measurements | Negative/nonfinite outliers explicitly marked missing and counted in QC, not silently accepted |
| Mortality cohort invented static hospital variables or event onset | Actual admission descriptors; mortality stored separately; no deterioration onset or lead time invented |
| Missingness delta contained NaNs or reset during gaps | Finite elapsed time before the current observation, including missing hours |
| Post-onset/padded targets were treated as negatives | NaN targets excluded from learning and discrimination; left-censored onsets handled explicitly |
| Isotonic/clipped losses could replace the measured risk | Certificate uses raw empirical losses; out-of-bound losses rejected; isotonic output diagnostic only |
| Infeasible finite-grid fallback could still alarm | Exact `+inf` no-alarm policy, zero certificate, and -1 index sentinel; E0/E2 evaluate zero fallback losses |
| Alarm-loss support bound ignored escalation or truncated the stay search | Exact discrete supremum including escalation and independently declared minimum exposure |
| Naive comparator used steps from the entire cohort | Calibration-only timestep count, checked directly in the experiment-driver regression |
| Padded scores or unobserved events could enter operational metrics | Score sequences sliced to lengths; disjoint partitions validated; unobserved/admission events excluded from lead-time cases |
| Pareto ties retained dominated points | At equal alarm rates retain only the longest lead time |
| Metrics accepted malformed inputs or diluted signed-rank effect sizes with zero differences | Binary/probability/shape checks; zero differences excluded consistently; minimal DeLong class sizes validated |
| Clinical score intervals skipped fractional observations and omitted component descriptions | Contiguous interval extensions; oxygen/consciousness and immature-neutrophil omissions declared; partial research-only scores |
| String site labels failed, small rounded partitions could miscount | Deterministic site-position seeds and cumulative partition cuts |
| Old scores could be silently reused after causal fixes | Version-two validated caches reject obsolete unversioned outputs |
| Python requirement contradicted union syntax | Python 3.10+ metadata |
| Reference verification compared runtime tuples with JSON lists as unequal | Recursive sequence normalisation; numeric tolerances unchanged; regression verifies real numeric differences still fail |
| Manuscript proof had the wrong concentration tail and unsafe fallback | Bound `R-Rhat`, iid patient-block assumptions, fixed candidate family/support, and exact zero-risk fallback |
| Draft overstated deployment guarantees, available hospitals, completed backbones, submission and compute | Distinguish population/finite-cohort/patient-tail risk; two public 2019 sites; incomplete experiments stated; false submission/GPU-hour claims removed |

No monotonicity theorem is claimed for the alarm count. The selector does not
need monotonicity, and window sensitivity has a tested refractory-policy
counterexample to monotone improvement at lower thresholds. Hoeffding and
empirical-Bernstein are finite-grid upper-confidence methods, not the standard
expected-risk conformal construction; legacy class names remain for compatibility.

The manuscript's equations, algorithm and prose now match the implemented
smallest-certified-threshold convention. Utility-maximising selection among
feasible thresholds is not claimed to be implemented. Long formulas are split
to fit the text block, and the first-page footer identifies the file as an
unsubmitted draft with author information pending.

## Local verification

- `python3 -m pytest tests -q`: **124 passed** on Python 3.11.2 (31.85 seconds in the final local run).
- Tests include exhaustive small binary state-machine trajectories, discrete
  support supremum checks with escalation and long refractory durations, strict
  loss/grid validation, safe fallback, causal GRU predictions, delta/target
  masks, official-format outcome joins, cache invalidation, Pareto ties,
  zero-difference signed-rank effects and malformed-input guards.
- `python3 scripts/audit_official2012.py`: all **4,000** official set-A records
  parsed; tensor shape **(4000,49,10)**; **554** mortality outcomes; **24**
  out-of-documentation-range measurements marked missing with matching QC counts;
  all delta and observed-value entries finite. Source hashes are saved in
  `code/audit_results/official2012_qc.json`; downloads are temporary and deleted.
- `python3 scripts/verify_saved_e0.py --recompute`, followed by an independent
  `python3 scripts/verify_saved_e0.py` comparison: seeds **0,1,2**, each with
  4,000 synthetic patients, 200 repeated partitions and 27 settings. Outputs
  include parameters, fallback fractions and descriptive interval status.
  All three fresh runs match the saved references with relative tolerance
  `1e-9` and absolute tolerance `1e-10`; these tolerances were not loosened.
- At calibration size 800 and alpha 0.1, the invalid shortcut's finite-test-mean
  exceedance fractions are **0.06, 0.04, 0.13** for those seeds. Both blocked
  variants have zero finite-cohort exceedances in these saved runs, but at
  calibration size 100 and alpha 0.05 they always use the no-alarm fallback.
  These are empirical diagnostics, not proofs of population-risk validity or
  evidence of clinical effectiveness. Previous stronger benchmark claims are withdrawn.
- Local package versions: NumPy 2.4.6, pandas 3.0.6, SciPy 1.17.1,
  scikit-learn 1.9.1 and PyTorch 2.14.1+cpu. Optional LightGBM is not installed
  locally and its backend is not claimed to have been tested here.
- The corrected draft compiles with `latexmk` through the workspace compiler;
  final references resolve, and the repaired alarm equation has been visually checked.

## Independent Lab verification

The final Lab job completed with exit status zero after `compileall`, the full
test suite (**124 passed in 28.20 seconds**), fresh three-seed reference
verification, and the official set-A network QC. All seeds matched at the same
numeric tolerances as the local comparison. The official-data counts, tensor
shape and both SHA-256 source hashes match the local QC record above.

Lab versions were Python 3.10.12, NumPy 2.2.6, pandas 2.3.3, SciPy 1.15.3,
scikit-learn 1.7.2 and PyTorch 2.11.0+cu128. LightGBM was absent on Lab as well,
so its optional backend is not claimed tested. CUDA-enabled PyTorch being
installed does not imply a GPU training experiment: the supplied GRU reference
still executes on CPU. The completed checks concern reference correctness and
data ingestion, not clinical comparative effectiveness.

## Scope limits

Independent patient-block sampling is an assumption, not something established
by public record IDs. Direct 2019-to-2012 model transfer still needs explicit
unit conversion and compatible static schemas. Raw partial clinical scores need
a training-only probability adapter for probability metrics. External deep
backbone adapters, official 2019 utility scoring, the full clinical experimental
grid, clinical conclusions and measured GPU cost remain incomplete.

All old unversioned score caches and downstream clinical results must be
regenerated. Passing these checks does not prove absence of every possible
defect, and this research reference is not certified for clinical deployment.
