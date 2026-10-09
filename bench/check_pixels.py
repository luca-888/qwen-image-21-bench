"""Compare decoded output pixels across runs (offline, serving, diagnostics) per workload.

PNG bytes differ between the offline runner and the serving path because of the encoder, so the
`image_sha256` in runs.jsonl cannot show equality across them. This hashes the decoded RGB pixels.

    python3 bench/check_pixels.py [results] [--json results/pixel_check.json]
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="?", default="results")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    groups: dict[str, dict[str, list[str]]] = collections.defaultdict(lambda: collections.defaultdict(list))
    for png in sorted(Path(args.results).glob("*/images/*.png")):
        workload = "edit" if "_edit_" in png.parent.parent.name else "t2i"
        pixels = np.asarray(Image.open(png).convert("RGB"))
        digest = hashlib.sha256(pixels.tobytes()).hexdigest()
        groups[workload][digest].append(str(png.relative_to(args.results)))

    out = {w: {"images": sum(len(v) for v in d.values()), "distinct_pixel_hashes": len(d), "by_hash": d}
           for w, d in groups.items()}
    for w, v in out.items():
        print(f"{w}: {v['images']} images, {v['distinct_pixel_hashes']} distinct pixel hash(es)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
