# Model licensing and data provenance

The checkpoints in `checkpoints/` are derived from datasets that are licensed
for **non-commercial research use only**. The code in this repository is the
author's own; the weights are not unencumbered.

**This repository is private, and the weights should not be redistributed
publicly or used commercially without checking the terms below.**

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
