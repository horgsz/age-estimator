# Model licensing and data provenance

The checkpoints in `checkpoints/` are derived from datasets that are licensed
for **non-commercial research use only**. The code in this repository is the
author's own; the weights are not unencumbered.

**This repository is public, and the GitHub Pages build publishes both
checkpoints as downloadable ONNX files.** That is a deliberate change from when
this file was written, and it is a redistribution: anyone visiting
<https://horgsz.github.io/age-estimator/> fetches
`age_model_realgt.onnx` (6.2 MB) from the site, and both files are readable
directly out of `checkpoints/` in this repository.

Nothing below has been re-checked against that fact. The terms on this page
permit **non-commercial research use only**, which publishing a research demo
plausibly falls within, but "plausibly" is doing real work in that sentence:

* AgeDB is 66% of the real-age corpus and its terms are non-commercial. It was
  obtained from an unverified third-party mirror (see below), so the exact terms
  the weights inherit have not been confirmed against the official
  distribution.
* UTKFace is non-commercial research only.
* A derived model is generally treated as carrying its training data's
  restrictions; that is an assumption, not a licence grant anyone has given.

If this project is ever more than a demo — or if the repository picks up a
licence file implying broader rights than the weights carry — resolve the
AgeDB provenance question first. Do not treat "it was already public" as having
settled it.

## What each model was trained on

### `age_model.pt` / `age_model.onnx` — "apparent age"

| | |
|---|---|
| sha256 (first 12) | `56894c480044` |
| Corpus | UTKFace, 23,684 images |
| Licence | Non-commercial research only |
| Source | https://susanqq.github.io/UTKFace/ |

**Label provenance matters here.** UTKFace's own page states the ages "are
estimated through the DEX algorithm and double checked by a human annotator."
DEX is a model trained on IMDB-WIKI's scraped birth dates, so these labels are
a neural network's output rather than verified chronological ages. This model
therefore predicts *apparent* age — how old a face looks — and it is measurably
good at that (regression slope 0.935 against crowd-judged apparent age, versus
0.817 against real age).

### `age_model_realgt.pt` / `age_model_realgt.onnx` — "real age" (default)

| | |
|---|---|
| sha256 (first 12) | `fb629f49987a` |
| Corpus | 25,080 images with real chronological ages |

| Source | Images | Licence | Provenance |
|---|---|---|---|
| AgeDB | 16,487 | Non-commercial research only | Ages transcribed by hand from captions |
| APPA-REAL | 7,591 | Research use | Real ages from image-owner metadata |
| FG-NET | 1,002 | Free to use | Dated personal photographs |

**AgeDB caveat.** AgeDB is 66% of this corpus, and it was obtained from a
third-party HuggingFace mirror (`marcelohaps/agedb`), not from the official
Imperial College distribution, which is a password-protected archive released
by email. Labels were parsed from the original filenames rather than trusting
the mirror's derived columns, but the mirror has **not** been verified against
the official archive. The mirror is tagged `license: other`; an uploader cannot
relicense a dataset, so AgeDB's original non-commercial terms still apply.

Practically: "trained on real ground truth" substantially means "trained on
AgeDB", and that dependency is unaudited.

## If you ever want to publish or commercialise this

1. Verify the AgeDB mirror against the official archive
   (https://ibug.doc.ic.ac.uk/resources/agedb/ — password by email).
2. Retrain on data you can license. None of the corpora above permit
   commercial use.
3. Consider consent-based commercial datasets instead; several have appeared
   recently driven by age-verification regulation.

## Accuracy, honestly stated

Measured on a held-out, identity-disjoint test split (n=3,818) of the real-GT
corpus. No subject appears in both training and test data.

| model | MAE vs real age | CS@5 | slope |
|---|---|---|---|
| `age_model.pt` (apparent) | 9.13 | 42.2% | 0.744 |
| `age_model_realgt.pt` (real) | 6.39 | 56.2% | 0.808 |

Both numbers are error against *real chronological age*. That is not the target
`age_model.pt` optimises, so it is the honest number for that model rather than
a flattering one.

Do not compare these to the in-corpus UTKFace figure of 5.55 that appears in
older notes: that measured agreement with DEX-estimated labels, not error
against real age, and the smaller number is the weaker result.

**Neither model is suitable for age verification or age gating.** About 30% of
under-18s are displayed as 18 or over by the real-age model, and about 40% by
the apparent-age model.
