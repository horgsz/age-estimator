"""Generate the parity fixtures from the repository's one public-domain face.

Deliberately derived rather than downloaded. The fixtures need to be committed
so the harness is reproducible, and nothing in `datasets/` may be redistributed
-- `datasets/manifest.csv` is gitignored for exactly that reason. The existing
test asset (`server/tests/assets/face.jpg`, an official White House portrait, a
US federal government work and therefore public domain) carries no such
restriction, so every fixture is a transform of it.

The transforms are chosen to hit the code paths that differ, not to look varied:

  full        the source at native size; a straightforward detection
  large       upscaled past MAX_DETECT_SIDE, so the detector's own downscale
              runs and the box has to be mapped back through it
  small       downscaled until the face is under 224px, so `resize_crop` takes
              INTER_LINEAR instead of INTER_AREA
  edge        the face pushed against the frame edge, so the crop square is
              slid back inside rather than centred
  clipped     the frame cropped tight around the face, so the square does NOT
              fit and the edge-replication padding path runs
  two-faces   the image beside a flipped copy of itself, so NMS has something
              to do and the per-face batch has more than one row

PNG, not JPEG, and that is not incidental: the browser and OpenCV use different
JPEG decoders, which disagree by a least-significant bit or two on some
coefficients. Comparing lossy-decoded pixels would put a floor under the tensor
agreement and confuse a real preprocessing bug with a decoder difference. PNG is
lossless, so both decoders must produce identical bytes, and any tensor
difference that remains is ours.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / "fixtures"
SOURCE = HERE.parent / "server" / "tests" / "assets" / "face.jpg"


def write(name: str, image: np.ndarray) -> str:
    path = FIXTURES / name
    ok = cv2.imwrite(str(path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise SystemExit(f"could not write {path}")
    return name


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    src = cv2.imread(str(SOURCE), cv2.IMREAD_COLOR)
    if src is None:
        raise SystemExit(f"could not read {SOURCE}")
    h, w = src.shape[:2]

    cases = []

    cases.append({"id": "full", "image": write("full.png", src), "model": "real"})

    # Past MAX_DETECT_SIDE (1024), so detection runs on a downscaled copy and
    # the boxes come back through the inverse scaling. Only just past it: the
    # fixture has to be committed, and a 3x upscale is a megabyte of PNG to
    # exercise a branch that a 2.2x upscale reaches just as well.
    large = cv2.resize(src, (int(w * 2.2), int(h * 2.2)), interpolation=cv2.INTER_LINEAR)
    cases.append({"id": "large", "image": write("large.png", large), "model": "real"})

    # Small enough that the face crop is under 224, flipping resize_crop from
    # INTER_AREA to INTER_LINEAR.
    small = cv2.resize(src, (w // 3, h // 3), interpolation=cv2.INTER_AREA)
    cases.append({"id": "small", "image": write("small.png", small), "model": "real"})

    # Face against the frame edge: the square crop still fits, so it is slid
    # back inside rather than padded.
    edge = src[0 : int(h * 0.62), int(w * 0.30) : w]
    cases.append({"id": "edge", "image": write("edge.png", edge), "model": "real"})

    # Face filling the frame: the square does NOT fit, so the crop is centred
    # and the deficit is filled by edge replication.
    clipped = src[int(h * 0.05) : int(h * 0.38), int(w * 0.30) : int(w * 0.72)]
    cases.append({"id": "clipped", "image": write("clipped.png", clipped), "model": "real"})

    # Two faces, so NMS has work to do and the batch has more than one row.
    pair = np.hstack([src, cv2.flip(src, 1)])
    cases.append({"id": "two-faces", "image": write("two-faces.png", pair), "model": "real"})

    # The same image through the other model, so the toggle is covered too.
    cases.append({"id": "full-apparent", "image": "full.png", "model": "apparent"})

    # THE STRICT CASE.
    #
    # A fixed box whose square side is exactly the model input, so `resize_crop`
    # resizes 224 -> 224. OpenCV's INTER_LINEAR at scale 1 is an exact identity,
    # so this case has NO resampling in it at all. Any tensor difference here is
    # unambiguously a cropping, channel-order or normalisation bug -- it cannot
    # be blamed on the resize, which is the excuse every other case leaves open.
    cases.append(
        {
            "id": "strict-no-resize",
            "image": "full.png",
            "model": "real",
            "boxes": [[80, 90, 224, 224]],
        }
    )

    # A non-default margin, so the geometry is exercised somewhere other than
    # the single value that happens to ship.
    cases.append(
        {"id": "margin-0.25", "image": "full.png", "model": "real", "crop_margin": 0.25}
    )

    (FIXTURES / "cases.json").write_text(json.dumps({"cases": cases}, indent=2) + "\n")
    print(f"wrote {FIXTURES / 'cases.json'} with {len(cases)} cases")


if __name__ == "__main__":
    main()
