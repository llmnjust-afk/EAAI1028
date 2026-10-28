"""Download the public 2019 Challenge training sets A and B from the open bucket.

The PhysioNet web directory has no browsable index, so object keys are listed
from the public S3 endpoint. Every object is checked against its advertised
size and MD5 ETag. The script is resumable: existing valid files are skipped.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET

BUCKET = "https://physionet-open.s3.amazonaws.com"
PREFIX = "challenge-2019/1.0.0/training"
EXPECTED = {"A": 20336, "B": 20000}
NS = {"s3": "http://s3.amazonaws.com/doc/2006-03-01/"}


def list_keys(prefix: str) -> list[dict]:
    keys: list[dict] = []
    token = None
    while True:
        url = f"{BUCKET}/?list-type=2&prefix={prefix}"
        if token:
            url += "&continuation-token=" + urllib.parse.quote(token, safe="")
        with urllib.request.urlopen(url, timeout=60) as response:
            root = ET.fromstring(response.read())
        for item in root.findall("s3:Contents", NS):
            key = item.findtext("s3:Key", namespaces=NS)
            if not key.endswith(".psv"):
                continue
            keys.append({"key": key,
                         "size": int(item.findtext("s3:Size", namespaces=NS)),
                         "etag": item.findtext("s3:ETag", namespaces=NS).strip('"')})
        truncated = root.findtext("s3:IsTruncated", namespaces=NS) == "true"
        if not truncated:
            break
        token = root.findtext("s3:NextContinuationToken", namespaces=NS)
        if not token:
            raise RuntimeError("truncated S3 listing without continuation token")
    return keys


def md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(item: dict, root: Path) -> tuple[str, bool]:
    relative = item["key"].split("/training/", 1)[1]
    target = root / relative
    if target.exists() and target.stat().st_size == item["size"] and md5(target) == item["etag"]:
        return relative, False
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    url = f"{BUCKET}/{item['key']}"
    for attempt in range(3):
        try:
            urllib.request.urlretrieve(url, tmp)
            if tmp.stat().st_size != item["size"] or md5(tmp) != item["etag"]:
                raise ValueError(f"checksum mismatch for {item['key']}")
            tmp.replace(target)
            return relative, True
        except Exception:
            tmp.unlink(missing_ok=True)
            if attempt == 2:
                raise
    raise AssertionError("unreachable")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/physionet2019"))
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--manifest", type=Path, default=None)
    args = parser.parse_args()

    objects = []
    for subset in ("A", "B"):
        items = list_keys(f"{PREFIX}/training_set{subset}/")
        if len(items) != EXPECTED[subset]:
            raise RuntimeError(f"set {subset}: found {len(items)} records, expected {EXPECTED[subset]}")
        objects.extend(items)

    downloaded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(fetch, item, args.out) for item in objects]
        for done, future in enumerate(as_completed(futures), 1):
            _, created = future.result()
            downloaded += int(created)
            if done % 2000 == 0:
                print(f"{done}/{len(objects)} records, {downloaded} downloaded", flush=True)

    summary = {"source": "https://physionet.org/content/challenge-2019/1.0.0/",
               "sets": {s: EXPECTED[s] for s in ("A", "B")},
               "records": len(objects), "bytes": sum(o["size"] for o in objects),
               "downloaded": downloaded}
    manifest = args.manifest or args.out / "download_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
