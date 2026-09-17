"""Assert ``datasets/common.py``'s crop mirror matches ``server/preprocessing.py``.

The mirror exists so the corpus can be built in a checkout that has no
``server/`` package. That convenience is only safe while the two agree
bit-for-bit: a crop that differs from the serving crop is an accuracy leak with
no visible symptom. Run this whenever either file changes.

    python datasets/verify_crop.py
    python datasets/verify_crop.py --server-root /path/to/other/checkout
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path

import numpy as np

import common

CASES: list[tuple[tuple[int, int], tuple[int, int, int, int]]] = [
    # (image h, w), (bbox x, y, w, h) -- centred, edge-clipped, corner,
    # larger-than-frame, non-square and degenerate boxes.
    ((480, 640), (270, 190, 100, 100)),
    ((480, 640), (0, 0, 120, 90)),
    ((480, 640), (600, 440, 40, 40)),
    ((200, 200), (10, 10, 180, 180)),
    ((200, 200), (0, 0, 200, 200)),
    ((100, 300), (140, 10, 80, 120)),
    ((300, 100), (10, 140, 120, 80)),
    ((64, 64), (30, 30, 2, 2)),
    ((1080, 1920), (900, 400, 250, 310)),
]

MARGINS = [0.0, -0.05, 0.05, 0.2, 0.4]


def load_server_preprocessing(server_root: Path):
    if not (server_root / "server" / "preprocessing.py").is_file():
        return None
    sys.path.insert(0, str(server_root))
    for name in [m for m in sys.modules if m == "server" or m.startswith("server.")]:
        del sys.modules[name]
    try:
        return importlib.import_module("server.preprocessing")
    except Exception as exc:
        print(f"could not import server.preprocessing from {server_root}: {exc}")
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-root", type=Path, default=common.REPO_ROOT)
    args = parser.parse_args()

    preprocessing = load_server_preprocessing(args.server_root)
    if preprocessing is None:
        print(
            f"SKIP: no server/preprocessing.py under {args.server_root}. "
            "Re-run from a checkout that has the server package."
        )
        return 0

    rng = np.random.default_rng(0)
    checked = 0
    for (height, width), bbox in CASES:
        image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
        for margin in MARGINS:
            expected = preprocessing.crop_face(image, bbox, margin)
            actual = common._mirror_crop_face(image, bbox, margin)
            if expected.shape != actual.shape or not np.array_equal(expected, actual):
                print(
                    f"MISMATCH image={height}x{width} bbox={bbox} margin={margin}: "
                    f"server {expected.shape} vs mirror {actual.shape}"
                )
                return 1
            checked += 1

    print(f"OK: mirror matches server/preprocessing.crop_face on {checked} case(s)")
    print(f"    margins tested: {MARGINS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
