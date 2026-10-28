# Restore after Lab Compute restart

The repository contains the code, manuscript source, compiled PDF and small
derived result files. Raw PhysioNet records and score caches are intentionally
not committed.

## 1. Clone and test

```bash
git clone https://github.com/llmnjust-afk/EAAI1028.git
cd EAAI1028/code
python3 -m pip install -r requirements.txt
python3 -m pytest tests -q   # expected: 135 passed
```

## 2. Restore the public data

Download the 2019 challenge data to a directory outside the repository, for
example `/data/lab/EAAI1028-data/physionet2019`, using
`code/scripts/download_challenge2019.py` or the PhysioNet archive. Verify the
published checksums before running experiments.

Expected layout:

```text
/data/lab/EAAI1028-data/physionet2019/
  training_setA/*.psv
  training_setB/*.psv
```

## 3. Rebuild caches and rerun

Use the commands in `code/experiment_results/README.md`. The derived outputs
from the last completed run are archived under
`code/experiment_results/round2/`, including the three subject-split files and
the LightGBM, target-budget, bootstrap and clinical-baseline JSONs. Those files
let a restarted session compare against the manuscript numbers before rebuilding
the 95 MB cache from raw records.

The manuscript PDF is committed as `main.pdf`; LaTeX is not required to inspect
the current submission draft. Raw records and caches remain outside git and
must be downloaded or regenerated after a Lab reset.
