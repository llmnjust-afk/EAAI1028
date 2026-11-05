# Restore after Lab Compute restart

The repository contains the code, manuscript source, compiled PDFs, small
derived result files and the archived subject splits. Raw PhysioNet records and
score caches are intentionally not committed.

## 1. Reconnect to the existing Lab checkout

The last working checkout was `/data/lab/eaai`. If that volume survives the
restart, keep it and update in place:

```bash
cd /data/lab/eaai
git pull --ff-only
cd code
python3 -m pytest tests -q   # expected: 177 passed
```

If the volume is gone, clone afresh:

```bash
cd /data/lab
git clone https://github.com/llmnjust-afk/EAAI1028.git eaai
cd eaai/code
python3 -m pip install -r requirements.txt
python3 -m pytest tests -q   # expected: 177 passed
```

## 2. Check the existing data and cache

The previous machine held about **438 MB** of public records and a **57 MB**
score cache, outside git:

```text
/data/lab/eaai/data/physionet2019/training_setA/*.psv   # 20,336
/data/lab/eaai/data/physionet2019/training_setB/*.psv   # 20,000
/data/lab/eaai/data/physionet2012/                      # 2012 A/B/C
/data/lab/eaai/cache/
/data/lab/eaai/results/
```

Check before downloading anything:

```bash
find /data/lab/eaai/data/physionet2019/training_setA -name '*.psv' | wc -l
find /data/lab/eaai/data/physionet2019/training_setB -name '*.psv' | wc -l
du -sh /data/lab/eaai/cache
```

The three subject splits are archived in
`code/experiment_results/round2/splits/`, so the E2/U-CRC experiments do not
need to redownload the raw cohort merely to recover the split definition.

## 3. Rebuild the 2019 data if absent

```bash
cd /data/lab/eaai/code
python3 scripts/download_challenge2019.py \
  --out /data/lab/eaai/data/physionet2019 \
  --manifest /data/lab/eaai/data/physionet2019/manifest.json
```

The script verifies every object's size and MD5 ETag and is resumable. The 2012
cohort is not fetched by this script; restore it separately from
<https://physionet.org/content/challenge-2012/1.0.0/> into
`/data/lab/eaai/data/physionet2012/`.

## 4. Fast restart checks

From `/data/lab/eaai/code`, with the cache present:

```bash
python3 scripts/run_selectors.py \
  --cache-root ../cache --seed 0 --selection utility \
  --procedures blocked crc --budgets 0.25 0.5 1 2 \
  --out ../results/selectors/utility_crc_blocked_A_B_s0.json

python3 scripts/run_coverage.py \
  --cache-root ../cache --cohort A_lead6 --model prefix_lr --seed 0 \
  --out ../results/coverage/coverage_A_logistic_s0.json
```

The seed-level U-CRC source files are committed under
`results/selectors/utility_crc_blocked_A_B_s{0,1,2}.json`; regenerate them with
`--seed 1` and `--seed 2` only if the cache changes.

## 5. Manuscript build

```bash
cd /data/lab/eaai
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplementary.tex
```

Expected page counts are **17** for `main.pdf` and **10** for
`supplementary.pdf`. The compiled PDFs are committed, so LaTeX is not required
to inspect the current draft.

## 6. What is and is not archived

Archived in git:

- manuscript source and compiled PDFs;
- small aggregate JSON/CSV results in `results/` and
  `code/experiment_results/`;
- the three archived subject-split files in
  `code/experiment_results/round2/splits/`.

Not archived:

- raw PhysioNet records and score caches;
- any credential or token;
- temporary render directories and smoke-test outputs.

No external cohort beyond the public 2019 A/B and 2012 A/B/C data was run. The
2012 analysis is an outcome-shift stress test, not an independent same-task
sepsis cohort.
