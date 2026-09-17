# `datasets/` — a face-age corpus with real chronological ground truth

This directory assembles a training corpus for age estimation from four
sources. Its reason to exist is a labelling problem, so start there.

## Why this exists: UTKFace's labels are a model's output

The current model is trained on UTKFace alone. UTKFace's own documentation says
its ages "are estimated through the DEX algorithm and double checked by a human
annotator". DEX is a CNN trained on IMDB-WIKI, whose labels are scraped birth
dates differenced against photo dates. So a UTKFace label is:

```
IMDB-WIKI scraped date  ->  DEX network prediction  ->  human sanity check  ->  label
```

Two error layers from reality, and — because DEX was trained to match what
people *look*, not what records *say* — the target is **apparent** age. A model
fit to those labels learns to reproduce DEX, including its biases, which is the
most likely explanation for the current checkpoint's ~-7.5 year bias above 80.

The three added sources have genuine chronological ground truth: ages from
birth records, dated photographs, or transcribed metadata. `manifest.csv`
carries a `real_ground_truth` boolean so that distinction is never lost, and
every table in `reports/distribution.txt` is broken out by it.

## Sources

| source | images | subjects | ages | label provenance | real GT |
|---|---:|---:|---|---|---|
| `utkface` | 23,684 | 23,684 | 1–100 | DEX CNN estimate, human double-checked | **no** |
| `agedb` | 16,487 | 567 | 1–100 | manually transcribed from photo metadata | yes |
| `appa-real` | 7,591 | 7,591 | 1–100 | real age supplied with the image | yes |
| `fgnet` | 1,002 | 82 | 0–69 | dated personal photographs | yes |
| **total** | **48,764** | **31,924** | 0–100 | 25,080 real (51.4%) | |

Counts are after dropping three rows labelled age 101, which is outside the
model's 0–100 output bins.

### APPA-REAL

* `https://data.chalearnlap.cvc.uab.cat/AppaRealAge/appa-real-release.zip`
  (885 MB, sha256 `e1c25af4cd36269f69505ad9419fef639bdc92c8fff90280ebad67805553a307`)
* The `.es` domain cited by most papers, and by `yu4u/age-estimation-pytorch`,
  is **dead**. `chalearnlap.cvc.uab.cat` is the live host as of 2026-09-17.
* Ground truth CSVs (`gt_avg_{train,valid,test}.csv`) carry
  `file_name, num_ratings, apparent_age_avg, apparent_age_std, real_age`.
  **`real_age` is the training target.** `apparent_age_avg` and its standard
  deviation are carried into the manifest as `apparent_age` /
  `apparent_age_std` for reference and analysis only — using them would
  reintroduce exactly the apparent-age bias this corpus exists to escape.
* The release ships both originals and pre-cropped faces (`*_face.jpg`). **The
  originals are used.** The supplied crops apply a 40% margin, far outside the
  measured-safe band (see "Cropping" below); training on them would cost
  several years of MAE with no visible symptom.
* Official `train`/`valid`/`test` directories are used verbatim, mapped to
  `train`/`val`/`test`.
* **Licence:** research use, per the ChaLearn LAP terms on the download page.

### FG-NET

* `https://yanweifu.github.io/FG_NET_data/FGNET.zip`
  (46 MB, sha256 `228acddeba96e469f2ded9475ee87f4ab16398de27a2f3ec35a266a46ef2f7fc`)
* The `ibug.doc.ic.ac.uk` mirror returns HTTP 500; the URL above works.
* Ages come from **dated personal photographs** — the strongest provenance of
  the four, and the only source here that is not celebrity photography.
* Filenames encode identity and age: `078A11.JPG` is subject 78 at age 11. A
  trailing letter (`066A06a`, `066A06b`) distinguishes two scans of the same
  subject at the same age.
* 68-point landmarks ship in `FGNET/points`. They are not used here — the
  serving path has no landmark stage, and aligning training data by landmarks
  when inference cannot would be a train/serve mismatch.
* **Value:** 444 of 1,002 images (44.3%) are under 12. That is the corpus's
  densest real-ground-truth child coverage.
* **Caveats:** zero images above age 69, so it contributes nothing to the 70+
  problem. Scanned film, many images black and white, variable focus and
  resolution, not pre-aligned.
* **Licence:** free for research use.

### AgeDB — obtained from a third-party mirror; read this before relying on it

AgeDB is the **only remaining source with meaningful 70+ coverage** now that
MORPH-2's host is gone. It contributes 1,967 of the corpus's 2,209 real-ground-
truth images at 70+. Without it the corpus barely improves on UTKFace at the
top end, which is where the current model is worst.

**The official distribution cannot be scripted.** `https://ibug.doc.ic.ac.uk/resources/agedb/`
serves a Dropbox link to a password-protected zip, and the password must be
requested by email from `s.moschoglou@imperial.ac.uk`. No email was sent.

`download.sh` therefore pulls the HuggingFace mirror
[`marcelohaps/agedb`](https://huggingface.co/datasets/marcelohaps/agedb). What
was actually checked, and what was not:

*Checked, and consistent with the original:*

* 16,488 images — exactly the published count.
* 567 identities, ages 1–101 — the paper reports 568 subjects and 1–101. The
  one-identity discrepancy is unexplained and is most likely two spellings of
  one name being folded together, or one genuinely absent subject.
* Original filename scheme `{id}_{Identity}_{age}_{gender}.jpg` is intact, and
  **labels are parsed from those filenames**, not from the mirror's derived
  `metadata.csv`. That keeps one fewer re-derivation between the label and us.
* Sampled images are pre-cropped faces, mostly square, ~38–450 px, some
  greyscale — all consistent with AgeDB as described.

*Downloading it is fiddly.* Two traps, both handled in `download.sh`:

* 69 filenames contain a space (`16020_MorganFreeman _73_m.jpg`), which curl
  rejects unless the URL is percent-encoded. The manifest strips the resulting
  trailing space from the identity, or one celebrity would split across two
  subject groups and quietly break the subject-disjoint splits.
* HuggingFace rate-limits by IP and answers with a **189-byte `200 OK` text
  body**, not an HTTP error. On the first pass 3,394 of 16,488 "images" were
  that error page; they only surfaced later as unreadable files, and they
  inflated AgeDB's apparent detection failure rate to 20.7%. The downloader now
  validates the JPEG magic number and retries with backoff. Do not relax that
  to a file-size check — the smallest genuine image is 38×37 px, 951 bytes.

*Not checked, and not checkable without the original:*

* Whether any image was re-encoded, resized or dropped. Sampled files are
  small (median ~200 px, and a handful as low as 38 px), so several will be
  **upscaled** to the 224 px model input, which costs detail. Whether that is
  the mirror's doing or AgeDB's own framing cannot be determined from here.
* Whether any label was altered. Nothing suggests it, but nothing rules it out.

**Licence:** the mirror is tagged `license: other`, and its card says to check
the original terms. HuggingFace licence tags are re-declared by uploaders and
carry no authority. AgeDB's own terms are **non-commercial academic research
only**, and an uploader cannot relicense it. Treat this corpus as
non-commercial research use.

**Recommendation:** this is good enough to characterise the corpus and to
experiment with. Before anything is published or shipped, request the real
password by email and re-verify the mirror against the official archive.
`build_manifest.py` re-derives everything from filenames, so swapping in the
official extract is a matter of repointing `--agedb-root`.

### UTKFace

Not downloaded here — it belongs to `ml/` and is expected at `<repo>/data/UTKFace`,
set up per `ml/README.md`. **Licence:** non-commercial research use.

## Rejected sources, and why

| dataset | why not |
|---|---|
| IMDB-WIKI / IMDB-Clean | Circular. DEX was trained on IMDB-WIKI and DEX labelled UTKFace, so training on it reintroduces the exact bias this corpus exists to escape. |
| Adience | Age *buckets*, not exact ages. |
| MORPH-2 | Host is dead; all UNCW pages 404. |
| CACD | Metadata 404s, and 16–62 only. |
| AFAD | Self-reported ages, 15–40 only. |

## Identity-aware splitting — the important part

UTKFace never needed this: every image is a different person, so a random 5%
holdout is 1,185 independent faces.

FG-NET and AgeDB are **longitudinal**. FG-NET has 82 subjects with ~12 images
each; AgeDB has 567 with ~29 each. Split those at random per image and the same
face lands in both train and test. The model can then score well by recognising
*who* it is looking at and recalling that person's ages, and the test number
stops measuring age estimation at all — while looking fine.

So:

* Both are split with `GroupShuffleSplit` on `subject_id`, 80/10/10, seed 42.
* The holdout is split again **by group**, so val and test are subject-disjoint
  from each other too. Otherwise val would leak into model selection for test.
* `verify_no_subject_overlap()` asserts, per source and globally, that no
  `subject_id` appears in two splits. The build **fails** rather than writing a
  quietly broken manifest.

The 80/10/10 differs deliberately from UTKFace's 90/5/5. Grouped sources spend
their holdout in whole subjects: at 5%, FG-NET's test set would be four people,
and the score would describe those four faces rather than the model.

**UTKFace's split is preserved exactly.** `build_manifest.py` reads
`ml/splits/{train,val,test}.csv` and carries those assignments through
unchanged, so the retrained model stays comparable to the current baseline and
nothing previously held out leaks into training.

## Cropping — must match serving exactly

`server/preprocessing.py` is the single source of truth for crop geometry:

```
YuNet bbox -> square, side = max(w, h) * (1 + 2 * margin)
           -> slid back inside the frame if it fits, edge-padded if it does not
           -> resize 224 -> ImageNet normalise
```

`CROP_MARGIN = 0.0`. That is a measured value, not a guess: two independent
sweeps — `ml/`'s, which re-frames ground-truth crops, and the server's
end-to-end harness, which runs real YuNet detection — both minimised at 0.0 and
agreed within 0.07 years at every margin tested. Erring wide is the dangerous
direction; the curve is strongly asymmetric. **Do not pick a different margin
without re-running those sweeps.**

`common.py` imports `server.preprocessing` when the server package is present
and otherwise uses a mirror of `crop_face`, so the corpus can be built in a
checkout that only has `datasets/`. `verify_crop.py` asserts the two produce
byte-identical output across nine framing cases (centred, edge-clipped, corner,
larger-than-frame, non-square, degenerate) and five margins:

```bash
python datasets/verify_crop.py
```

Detection notes:

* UTKFace and AgeDB ship pre-cropped faces, so detection **re-frames** them.
  APPA-REAL and FG-NET originals are full scenes, so detection genuinely
  **locates** the face there, and failure rates are correspondingly higher.
* When a photo contains several faces the largest detection is taken; the
  labelled subject dominates the frame in portrait-style photography.
* Scanned film, grain and B&W depress YuNet's confidence, which hits FG-NET
  hardest. Rather than discard the corpus's main source of child faces, a
  second pass retries at score threshold 0.25 (against the serving default of
  0.6) and `detect_pass` records which pass succeeded, so the weaker detections
  stay auditable. Rows that fail both keep `face_detected=False` and no crop.

Measured failure rates — 27 failures across 48,764 images (0.06%):

| source | images | detected | failed | fail % | rescue pass |
|---|---:|---:|---:|---:|---:|
| `agedb` | 16,487 | 16,467 | 20 | 0.12% | 40 |
| `appa-real` | 7,591 | 7,591 | 0 | 0.00% | 34 |
| `fgnet` | 1,002 | 1,002 | 0 | 0.00% | 4 |
| `utkface` | 23,684 | 23,677 | 7 | 0.03% | 5 |

FG-NET was the expected problem — scanned film, grain, black and white — and
detected at 100%, with only 4 images needing the rescue threshold. The
expensive lesson was elsewhere: AgeDB first measured at a 20.7% failure rate,
which turned out to be 3,394 rate-limit error pages saved as `.jpg`, not a
detection problem at all. Failure rates are a data-integrity signal before they
are a detector signal.

## Manifest schema

`manifest.csv`, one row per image:

| column | meaning |
|---|---|
| `path` | repo-root-relative path to the **original** image |
| `age` | training target: real chronological age where available, DEX estimate for UTKFace |
| `source` | `utkface`, `appa-real`, `fgnet`, `agedb` |
| `split` | `train`, `val`, `test` |
| `real_ground_truth` | `True` for APPA-REAL / FG-NET / AgeDB, `False` for UTKFace |
| `apparent_age` | crowd-rated apparent age (APPA-REAL only); **never a training target** |
| `apparent_age_std` | spread of those ratings (APPA-REAL only) |
| `subject_id` | identity group; the unit the grouped splits are taken over |
| `crop_path` | 224×224 crop written by `crop_faces.py`, or empty on detection failure |
| `face_detected` | whether a face was found |
| `detect_pass` | `primary` (0.6), `rescue` (0.25), or `none` |

`subject_id` is synthetic-but-unique for UTKFace and APPA-REAL, which have no
identity annotation, so the overlap assertion is meaningful across all sources.

## Usage

```bash
pip install -r datasets/requirements.txt

./datasets/download.sh              # ~1.1 GB into datasets/raw/
python datasets/build_manifest.py   # -> manifest.csv, asserts split disjointness
python datasets/crop_faces.py       # -> crops/, adds crop_path/face_detected
python datasets/report_distribution.py  # -> reports/
```

`datasets/raw/` and `datasets/crops/` are gitignored. **No dataset image is
ever committed** — several of these licences forbid redistribution outright.

## What this directory does not do

It does not train anything. Whether to train on all 48,764 images or only on
the 25,080 with real ground truth is a modelling decision, not a data one, and
it is left open deliberately. The numbers that bear on it:

| | full corpus | real GT only |
|---|---:|---:|
| images | 48,764 | 25,080 |
| train images | 39,364 | 18,049 |
| under-12 images | 4,567 | 1,284 |
| 70+ images | 3,560 | 2,209 |
| 80+ images | 1,348 | 696 |
| DEX-derived labels | 23,684 | 0 |

The tension is that UTKFace is simultaneously the corpus's biggest liability
and its biggest source of children. It contributes 3,062 of the 4,144 images
aged 0–9, against 1,082 from all three real-ground-truth sources combined — and
a DEX estimate of a small child's age is exactly the kind of label DEX is
worst at. Dropping UTKFace removes every second-hand label and costs 54% of the
corpus; keeping it means the retrained model is still partly fitting another
network's output.

At the top end there is no tension: 70+ goes from 699 DEX-labelled UTKFace
images to 2,209 real-ground-truth ones, almost entirely from AgeDB. That is a
clear win whichever way the UTKFace question is settled.

`reports/distribution.txt` and `reports/age_distribution.png` have the full
breakdown.
