"""Network smoke/QC test on all official PhysioNet 2012 set-A records.

Downloads remain in a temporary directory and are removed on exit. This checks
the data pipeline, not mortality discrimination or clinical effectiveness.
"""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from urllib.request import urlopen

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eaai.data.loaders import build_2012_tensors

BASE = "https://physionet.org/files/challenge-2012/1.0.0/"


def main():
    resources = {}
    for filename in ("set-a.tar.gz", "Outcomes-a.txt"):
        with urlopen(BASE + filename, timeout=120) as response:
            resources[filename] = response.read()
    hashes = {name: hashlib.sha256(content).hexdigest() for name, content in resources.items()}
    with tempfile.TemporaryDirectory(prefix="eaai-official2012-") as directory:
        root = Path(directory)
        folder = root / "set-a"
        folder.mkdir()
        with tarfile.open(fileobj=io.BytesIO(resources["set-a.tar.gz"]), mode="r:gz") as archive:
            for member in archive:
                if member.isfile() and Path(member.name).suffix == ".txt":
                    # Do not extract arbitrary archive paths or symlinks.
                    name = Path(member.name).name
                    if not Path(name).stem.isdigit():
                        raise ValueError(f"unexpected record name: {name}")
                    content = archive.extractfile(member)
                    if content is None:
                        raise ValueError("unreadable archive member")
                    (folder / name).write_bytes(content.read())
        (root / "Outcomes-a.txt").write_bytes(resources["Outcomes-a.txt"])
        bad_rows = []
        for path in sorted(folder.glob("*.txt")):
            raw = pd.read_csv(path)
            values = pd.to_numeric(raw["Value"], errors="raise")
            bad = (~np.isfinite(values)) | ((values < 0) & (values != -1))
            if bad.any():
                for row in raw.loc[bad].to_dict("records"):
                    bad_rows.append({"id": path.stem, **row})
        print(json.dumps({"out_of_documented_range": len(bad_rows),
                          "examples": bad_rows[:12]}, allow_nan=True), file=sys.stderr)
        batch = build_2012_tensors(root)
        assert len(batch["ids"]) == 4000
        assert (batch["lengths"] == 49).all()
        assert (batch["onset"] == -1).all()
        assert np.isfinite(batch["delta"]).all()
        assert np.isfinite(batch["values"][batch["mask"]]).all()
        assert np.isin(batch["mortality"], [0, 1]).all()
        assert sum(batch["qc_invalid_measurements"].values()) == len(bad_rows)
        index = batch["ids"].index("132539")
        hr = batch["variables"].index("HR")
        assert not batch["mask"][index, 0, hr]
        assert batch["values"][index, 1, hr] == 77.  # 00:37, later than HR=73 at 00:07.
        assert batch["mortality"][index] == 0.
        print(json.dumps({"patients": len(batch["ids"]), "shape": list(batch["values"].shape),
                          "mortality_events": int(batch["mortality"].sum()),
                          "invalid_measurements_marked_missing": len(bad_rows),
                          "finite_delta": True, "sha256": hashes}, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
