# Age estimation — data & training pipeline

Trains a DEX-style age regressor on [UTKFace](https://susanqq.github.io/UTKFace/)
(aligned & cropped) and publishes a checkpoint plus an ONNX export for the
serving layer.

Scope: this directory owns the data pipeline, the model, and the artifact
contract in `checkpoints/`. It does not touch `server/` or `web/`.

## Setup

```bash
python3 -m venv .venv                          # from the repo root
.venv/bin/python -m pip install -r ml/requirements.txt
```

## Data

```bash
mkdir -p data
curl -L -o data/UTKFace.tar.gz \
  https://huggingface.co/datasets/py97/UTKFace-Cropped/resolve/main/UTKFace.tar.gz
tar -xzf data/UTKFace.tar.gz -C data
```

`data/` is gitignored. Filenames are `{age}_{gender}_{race}_{datetime}.jpg`;
three files in this tarball are missing a field and are skipped with a warning.

```bash
.venv/bin/python ml/data.py      # manifest + age histogram + splits
```

This prints the age distribution and writes `ml/splits/{train,val,test}.csv`
(90/5/5, stratified by 5-year age bin, seed 42). Those CSVs are committed so
the split is reproducible and other components can reference the exact test set.
Ages outside 1..101 are dropped; 23,686 of 23,708 images survive.

### Label provenance — read before quoting the MAE

UTKFace age labels are **not verified chronological ages**. From the official
dataset page (susanqq.github.io/UTKFace):

> "The ground truth of age, gender and race are estimated through the DEX
> algorithm and double checked by a human annotator."

So the labels are *model output with human review*, not birth records. Every
metric in this README is therefore **agreement with DEX-plus-annotator, not
error against true age**. Consequences worth stating plainly:

- There is an **irreducible label-noise floor** under every number here. A ~4-5
  year MAE is close enough to plausible annotation error that some of the
  residual is the labels. Some apparent errors are very likely the model being
  *right* about a mislabelled face. Do not read further improvement in this
  range as straightforwardly real.
- The model is partly **distilling an earlier estimator**, which flatters
  in-corpus evaluation and caps what the score can mean.
- Cross-dataset comparisons (and published UTKFace leaderboard numbers) are
  **not** comparable to these unless they use this exact split.

The thin tails have both the fewest samples *and* the least reliable labels, so
treat the 70+ rows as indicative only. Nothing here should be presented to an
end user as a measured age.

#### The architectural irony

DEX — *Deep EXpectation*, Rothe et al. — **is the 101-bin softmax-expectation
method this model uses.** We built a DEX-style network and trained it on
DEX-generated labels. We are substantially distilling DEX with a layer of human
correction on top, which sets a real ceiling: the model is rewarded for
reproducing DEX's behaviour, including DEX's own failure modes.

That is not just a curiosity. It has a concrete, measurable consequence for the
young-age bias — see [Decoding](#decoding-the-mean-is-the-wrong-statistic),
where it turns out to explain the part of the bias that better decoding cannot
remove.

## Train

```bash
cd ml && ../.venv/bin/python train.py                 # ~30 epochs on MPS
cd ml && ../.venv/bin/python train.py --limit 2000 --epochs 1   # smoke test
```

MobileNetV3-Small-100 (ImageNet pretrained) with the classifier replaced by a
101-way head over ages 0..100. CrossEntropy with `label_smoothing=0.1`, AdamW
(lr 3e-4, wd 1e-4), cosine schedule with a 2-epoch warmup, batch 64. The device
defaults to MPS. AMP is probed once at startup and silently falls back to fp32
if the autocast forward/backward produces non-finite values.

Augmentation is `RandomResizedCrop(224, scale=(0.8, 1.0))` → `RandomZoomOut`
(`--zoom-out`, default `0.5`) → horizontal flip → ±15° rotation → colour jitter.
Best-val-MAE weights are written to `checkpoints/age_model.pt`.

> The backbone weights are mirrored to `checkpoints/pretrained/` with `curl`.
> Hugging Face's `us.aws.cdn.hf.co` edge serves an incomplete certificate chain
> that Python's `ssl` module rejects, so `timm`'s built-in downloader fails on
> this machine; `curl` completes the chain from the system trust store.

## Evaluate

```bash
cd ml && ../.venv/bin/python eval.py
```

Reports overall MAE, CS@5 and RMSE on the held-out test split, plus a per-decade
MAE breakdown with support counts, and writes `ml/reports/scatter.png` and
`ml/reports/metrics.json`. It also stamps the measured test MAE into
`meta["test_mae"]` in the checkpoint (during training that field holds the best
val MAE as a placeholder).

## Export

```bash
cd ml && ../.venv/bin/python export_onnx.py
```

Writes `checkpoints/age_model.onnx` and asserts parity with PyTorch within
1e-4 on real test images.

## Artifact contract

`checkpoints/age_model.pt` is a `torch.save` dict:

| key | contents |
| --- | --- |
| `state_dict` | **bare `timm` state dict** — 244 tensors, no `backbone.` prefix |
| `meta` | `{"backbone": "mobilenetv3_small_100", "num_bins": 101, "input_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "test_mae": <float>}` |

`checkpoints/age_model.onnx` is opset 17 with a dynamic batch axis:

| tensor | name | shape | dtype |
| --- | --- | --- | --- |
| input | `input` | `[N, 3, 224, 224]` NCHW | float32 |
| output | `logits` | `[N, 101]` | float32 |

The `state_dict` loads directly into a bare `timm` model, with no key surgery:

```python
ckpt = torch.load("checkpoints/age_model.pt", map_location="cpu")
model = timm.create_model(ckpt["meta"]["backbone"],
                          num_classes=ckpt["meta"]["num_bins"], pretrained=False)
model.load_state_dict(ckpt["state_dict"])      # strict=True
```

> Checkpoints written before 2026-09-16 prefixed every key with `backbone.`,
> leaking this repo's wrapper module into the artifact and forcing consumers to
> strip it. That is fixed at source in `save_checkpoint`; `load_checkpoint`
> still accepts either layout. `ml/reports/artifact_manifest.json` records the
> layout, content hashes, and provenance of the published files so a consumer
> can detect a re-publish instead of discovering it through disagreeing numbers.

### Second artifact: `age_model_realgt.pt`

The real-ground-truth retrain publishes a **separate pair of files** and does not
replace the originals:

| file | corpus | labels | `meta.test_mae` | decode |
|---|---|---|---:|---|
| `age_model.pt` / `.onnx` | UTKFace | DEX-estimated (apparent) | 5.5472 | expectation-era; median recommended |
| `age_model_realgt.pt` / `.onnx` | AgeDB+APPA-REAL+FG-NET | real chronological | 6.393 | `median` (recorded in `meta`) |

**The two `test_mae` values are not comparable** — different corpora, different
label semantics. 6.393 is not "worse than" 5.5472; they answer different
questions. See the real-GT section for the like-for-like comparison.

The new artifact adds one key to `meta`:

```python
meta["decode"]  # "median" -- the decode test_mae was measured under
```

This was added to the *new* artifact rather than retrofitted to the shipped one,
so no existing consumer's schema changes. It exists because a bare `test_mae` is
ambiguous: the same weights score 6.393 or 9.347 depending on a decision that was
previously recorded nowhere in the file. `build_meta(..., decode=...)` is
optional and omitted for the original artifact, whose contract predates it.

Each artifact has its own manifest sidecar — `reports/artifact_manifest.json` and
`reports/artifact_manifest_realgt.json` — so publishing one never invalidates the
other's recorded hash.

### Consuming the output

The head is a classifier, not a scalar regressor. Preprocess by resizing the
square crop **directly to 224x224** (no separate centre crop -- a 256-resize
plus 224-crop would silently discard 23% of the frame and put the input below
the training augmentation's scale floor), scale to `[0, 1]`, then ImageNet
normalize with the `mean`/`std` from `meta`.

**Decode with the median, not the mean.** The obvious DEX decode is the soft
expectation `sum_i p_i * i`, but it is badly mean-reverting here and the median
of the same distribution is strictly better on every axis:

| decode | MAE | CS@5 | bias | 0-9 MAE | 0-9 bias |
| --- | --- | --- | --- | --- | --- |
| soft expectation | 5.547 | 55.5% | +2.13 | 4.89 | +4.82 |
| **median** | **4.841** | **68.4%** | **+0.35** | **1.58** | **+0.77** |
| mode (argmax) | 5.237 | 65.8% | -0.17 | 1.55 | +0.60 |

```python
probs = softmax(logits, axis=1)                  # [N, 101]
cdf = probs.cumsum(axis=1)
age = (cdf < 0.5).sum(axis=1)                    # median bin
iqr = (cdf < 0.75).sum(axis=1) - (cdf < 0.25).sum(axis=1)   # uncertainty
```

`AgeEstimator.median()` in `ml/model.py` does exactly this for the PyTorch path;
`AgeEstimator.expectation()` retains the mean/std decode for comparison.

Use quantiles rather than the standard deviation as the confidence signal when
decoding with the median -- both describe the same distribution, but quantiles
are robust to the same tail mass the median is.

This matters more than consistency. **The label-smoothing pedestal inflates
`sigma` exactly as it biases the mean**, so the uncertainty channel was
miscalibrated too: the server session measured mean `sigma` 12.7 years giving
**93.3% coverage on a nominally-68% interval** -- the confidence bar read
falsely wide on every face. Switching to 0.16/0.84 CDF quantiles gives 11.8
years mean width and **75% coverage against 68% nominal**. One decode change
fixes the point estimate and the interval together, because one mechanism was
corrupting both.

The general form of this, worth carrying to any binned-classification
regressor: label smoothing corrupts **every moment** of the decoded
distribution, not just the first. The mean was caught because it produced a
visibly wrong number; the variance was wrong by a larger *relative* margin and
went unnoticed for longer, because an uncertainty channel that degrades
*toward* caution looks like appropriate humility. A too-wide confidence bar is
the failure mode least likely to be reported by anyone using the system.

## Results

Test split (1,185 held-out images), `checkpoints/age_model.pt`:

> **What these numbers measure.** UTKFace ages are DEX-algorithm estimates with
> human review, not verified chronological ages, so every figure below is
> *agreement with DEX-plus-annotator*, not error against true age. Part of the
> residual is label noise, and some counted errors are likely correct
> predictions on mislabelled faces. See
> [Label provenance](#label-provenance--read-before-quoting-the-mae).
>
> **This is now measured, not just asserted.** Against documented chronological
> ages on APPA-REAL the same checkpoint scores **8.52**, not 5.55. See
> [External validation](#external-validation--measured-against-real-ages-not-dex-labels).
> Quote that number, not this one, for any claim about real-world age accuracy.

| metric | value |
| --- | --- |
| MAE | **5.55 years** |
| CS@5 | **55.5%** |
| RMSE | 7.46 years |
| mean bias | +2.13 years |

Per-decade breakdown — the interesting part, because UTKFace is heavily skewed
toward ages 20-35:

Same caveat applies per row, and more sharply: label reliability is worst
exactly where support is thinnest.

| decade | support | MAE | CS@5 | bias |
| --- | --- | --- | --- | --- |
| 0-9 | 154 | 4.89 | 67.5% | +4.82 |
| 10-19 | 76 | 5.91 | 52.6% | +4.65 |
| 20-29 | 368 | 5.07 | 56.5% | +4.57 |
| 30-39 | 228 | 4.92 | 61.8% | +2.01 |
| 40-49 | 112 | 6.08 | 46.4% | +0.92 |
| 50-59 | 116 | 6.35 | 48.3% | -0.02 |
| 60-69 | 65 | 5.46 | 53.8% | -3.27 |
| 70-79 | 34 | 7.82 | 38.2% | -7.29 |
| 80+ | 32 | 10.80 | 28.1% | -10.80 |

The bias column is monotonically decreasing, which is textbook regression toward
the mean: the model is pulled toward the data-rich 20-40 band and compresses
both tails. The 80+ decade is off by -10.8 years on 32 test images, and for the
young decades the bias is nearly equal to the MAE, meaning the model almost
never under-predicts a child. Treat predictions below ~15 and above ~70 as
weakly supported.

**Most of that bias is a decoding artefact, not a learned one** -- see below.
The table above uses the soft-expectation decode for continuity with the
original spec; decoding the same checkpoint with the median cuts the 0-9 MAE
from 4.89 to 1.58 and overall MAE from 5.547 to 4.841.

### Conditioning: error by *true* age vs by *predicted* age

The per-decade table above bins by **true** age, which is the correct frame for
evaluating a model. A UI cannot do that — it only knows the number it is about
to display, so it must threshold on the **predicted** value. Those are different
curves, and the server session measured the difference end to end:

| shown | n | MAE | bias |
| --- | --- | --- | --- |
| 0-5 | 25 | 3.09 | +1.54 |
| 5-10 | 103 | 3.98 | +3.55 |
| 20-30 | 286 | 4.08 | +2.30 |
| 40-50 | 121 | **7.60** | +2.83 |
| 65-70 | 34 | **8.19** | -0.20 |
| 75+ | 22 | 5.83 | -3.48 |

It inverts the story. By true age, 80+ is the worst bin (MAE 10.80). By
predicted age, outputs **under 12 are the most accurate region the model has**,
and the worst band is **40-70**. The cause is the same mean-reversion described
below: the model rarely commits to an extreme value, so when it does emit one it
is usually right — while the middle of the range absorbs everything it is unsure
about. A genuine 85-year-old is displayed as roughly 68.

Two consequences worth carrying:

- **`E[|err| | predicted]` is the curve to quote to a user**, and it is less
  flattering in the middle than `E[|err| | true]`. Any user-facing error bar or
  caveat threshold should be derived from it.
- **It is decode-dependent.** The table above uses the soft-expectation decode.
  Because that decode is the main source of the compression (next section), a
  median decode widens the output range and shifts these bins — the 80+ bias
  improves from -10.80 to -7.47, so predictions that were unreachable at the top
  of the range become reachable. Recompute this table under whichever decode
  ships rather than porting these numbers across.

#### The tail bias does not survive re-conditioning

Under the shipped median decode, the old-age bias inverts depending on which
variable you condition on — measured end to end by the server session:

| rule | n | bias |
| --- | --- | --- |
| true age >= 80 | 32 | **-7.81** |
| **shown** >= 80 | 21 | **+1.10** |
| **shown** >= 65 | 80 | +1.04 |

Both numbers are correct and they are not in conflict; they answer different
questions. `eval.py`'s per-decade table conditions on **true** age, which is the
right frame for judging a model. A UI can only condition on what it displays.
Under median an 85-year-old now shows as ~77.7 (was ~67.7), so the residual
compression no longer pushes anyone past a high threshold — which means a
">= 80 is unreliable" caveat never fires on the population it exists for, and
would fire only on people the model got roughly right.

**So do not port `eval.py`'s tail-bias figure into product logic.** The
actionable signal under median is mid-adulthood precision: shown 40-75 is 27% of
cases at MAE 7.27, against 3.83 elsewhere. Ages shown under 12 are now the
*best* region (MAE 1.76), not the worst.

### Decoding: the mean is the wrong statistic

`decode_compare.py` collapses the 101-bin distribution to an age seven
different ways on the *same* checkpoint -- no retraining:

| decode | MAE | CS@5 | bias | 0-9 MAE | 0-9 bias | 80+ MAE |
| --- | --- | --- | --- | --- | --- | --- |
| soft expectation | 5.547 | 55.5% | +2.13 | 4.89 | +4.82 | 10.80 |
| **median** | **4.841** | **68.4%** | +0.35 | 1.58 | +0.77 | 8.09 |
| mode (argmax) | 5.237 | 65.8% | -0.17 | **1.55** | **+0.60** | **8.09** |
| pedestal-corrected mean | 4.955 | 63.7% | +0.92 | 2.39 | +1.88 | 9.20 |

The ranking reproduces on the val split (median 4.885 / 68.2%, expectation
5.594 / 55.6%), so this is a real effect and not a decode chosen against the
test set.

Two things put mass in the tails of the predicted distribution, and the mean
averages over all of it:

1. **Label smoothing.** `label_smoothing=0.1` trains the model to place a
   uniform pedestal of `0.1/101` on *every* bin. That pedestal's own
   expectation is 50, so it pulls each prediction toward the middle by roughly
   `E ~= 0.9 * age + 5`. At age 5 that predicts a +4.5 year bias; the measured
   0-9 bias is +4.82. Subtracting the pedestal back out ("pedestal-corrected"
   above) recovers most of the overall gap, which confirms the mechanism.
2. **Boundary skew.** Probability mass cannot extend below bin 0, so for a
   young face the distribution is genuinely right-skewed and the mean sits
   above the peak regardless of smoothing.

The median is robust to both, which is why it beats the pedestal correction
despite being the cruder fix. The mode is marginally better still at the two
extremes but throws away sub-bin resolution and loses 0.4 years in the
data-rich 30-60 band.

This is a free win: it is a change of decode, not of weights. The artifact
emits raw `logits`, so **the contract is unaffected** and the choice belongs to
whatever consumes it.

#### Is the young-age bias ours, or the labels'?

Two explanations compete for the +4.82 year bias at ages 0-9. Either it is a
**decoding artefact** (the mean of a 101-bin distribution is mean-reverting
near the age-0 boundary), or it is **in the labels** and the model learned it
faithfully — which DEX-generated labels would plausibly produce, since DEX
exhibits exactly this mean-reversion. These have different fixes, and the
decode comparison separates them: if the bias survives a decode that does *not*
average over the support, it was never a decoding artefact.

Bias by decade, expectation vs. mode (positive = predicts too old):

| decade | n | expectation | mode | removed by mode |
| --- | --- | --- | --- | --- |
| 0-9 | 154 | +4.82 | **+0.60** | 88% |
| 10-19 | 76 | +4.65 | +1.03 | 78% |
| 20-29 | 368 | +4.57 | +1.29 | 72% |
| 30-39 | 228 | +2.01 | -1.15 | — |
| 60-69 | 65 | -3.27 | -2.89 | 12% |
| 70-79 | 34 | -7.29 | -5.26 | 28% |
| 80+ | 32 | -10.80 | -6.41 | 41% |

**Verdict: at the young end it is overwhelmingly ours, and it is free to fix.**
Mode removes 88% of the 0-9 bias (+4.82 → +0.60) and median removes 84%
(→ +0.77). The two decodes are *not* biased high by a similar amount, which is
the outcome that would have indicted the labels. The label-smoothing mechanism
above independently predicts the magnitude (+4.5 at age 5 vs +4.82 measured),
so both the discriminator and the mechanism agree.

**But the residual is real, and the old end is a different story.** Mode leaves
+0.60 at 0-9 and still -6.41 at 80+, removing only 41% there. No decoding
change touches that. The most likely explanation is the architectural irony
above: if DEX-generated labels already carry DEX's mean-reversion, then our
soft-expectation decode was applying a *second* layer of it on top. Switching
decode strips our layer; the layer baked into the training targets is
unreachable, because as far as the loss is concerned it is the truth.

That is a limitation of the dataset, not of the model, and it cannot be
resolved from inside UTKFace — measuring it requires labels of independent
provenance (see Future work). Stated plainly: **the residual young-age bias is
small and the large old-age bias is probably partly inherited, and this
evaluation is structurally incapable of proving otherwise.**

Training peaked at epoch 14/30 (val MAE 5.594) and then overfit. Best-val
checkpointing keeps the epoch-14 weights. A regularized variant
(`--wd 0.05 --mixup 0.2`) was tried and was **worse** (val MAE 6.07): mixing two
faces produces an image with no well-defined age, which fights the ordinal
soft-expectation head. The `--mixup` flag remains available but defaults to off.

### Zoom-out augmentation

`RandomResizedCrop(scale=(0.8, 1.0))` only ever crops *in*, so it buys tolerance
to framings *tighter* than UTKFace and none at all to wider ones. Since a live
detector's boxes scatter around the ideal framing, the model had near-zero
headroom on the wide side. `RandomZoomOut` (`--zoom-out`, default `0.5`)
reflect-pads the image and shrinks it back, simulating margins out to ~0.15.

**This augmentation was initially argued against, and that argument was wrong.**
The reasoning — that `RandomResizedCrop` only crops inward, so widening its
scale floor would trade away accuracy at the framing actually served — was
correct about `RandomResizedCrop` but did not follow through to the conclusion
that a *different* augmentation could add wide-side tolerance at no cost. The
measurement reversed it: the augmentation is not merely free, it wins on the
clean test set *and* flattens the wide-side cliff:

| | baseline | + zoom-out |
| --- | --- | --- |
| test MAE | 5.747 | **5.547** |
| CS@5 | 52.2% | **55.5%** |
| MAE @ margin 0.10 | 6.214 | **5.949** |
| MAE @ margin 0.15 | 7.386 | **6.134** |

## Crop-margin sensitivity

The serving path crops `side = max(w, h) * (1 + 2 * CROP_MARGIN)` around a YuNet
detection. UTKFace is already tightly cropped, so training framing corresponds
to a specific margin, and inference must reproduce it.

`measure_crop_margin.py` over 300 random images (300/300 detected, 0% failure)
gives a median implied margin of **0.0135**, IQR [0.0012, 0.0268]. Two caveats
qualify that number, and both push it *down*:

1. **Truncated crops.** 48% of samples produce a YuNet box that runs past the
   image edge — UTKFace crops cut off part of the face, which compresses the
   apparent box. Excluding those, the median rises to **0.0213** (n=156).
2. **Scale/context invariance does not hold.** The measurement runs YuNet on
   already-tight 200x200 crops, but at inference YuNet sees a full frame.
   Re-running detection on images pasted into a larger synthetic canvas gives a
   systematically *larger* box relative to the face (median per-image delta
   -0.038 to -0.076), implying a full-frame margin between **-0.05 and 0.0**.
   The spread across padding styles (replicate / gray / blur) is as large as the
   effect, so the magnitude is **not identified** — only the direction is.

`margin_sweep.py` measures what framing error actually costs. Negative margins
crop tighter than UTKFace:

| CROP_MARGIN | baseline MAE | shipped (zoom-out) MAE |
| --- | --- | --- |
| -0.08 | 5.738 | 5.724 |
| -0.05 | 5.642 | 5.548 |
| -0.025 | 5.651 | 5.501 |
| **0.00** | 5.715 | **5.495** |
| 0.0135 | 5.747 | 5.547 |
| 0.0213 | 5.786 | 5.584 |
| 0.05 | 5.896 | 5.706 |
| 0.10 | 6.214 | 5.949 |
| 0.15 | 7.386 | 6.134 |
| 0.20 | 9.01 | 6.264 |
| 0.30 | 11.55 | 6.987 |
| 0.40 | 13.60 | 8.637 |

> The "baseline" column is the **pre-augmentation** model and is retained only
> for comparison. Figures quoted elsewhere as 5.747 MAE, 52.2% CS@5, or the
> 13.60 worst-case at margin 0.40 all refer to that superseded model. The
> shipped artifact is the zoom-out model: **5.547 MAE, 55.5% CS@5, 8.637 at
> margin 0.40**. Both columns use replicate padding for the wide rows.

The curve is strongly **asymmetric**: cropping tighter than training costs
essentially nothing over the whole range tested, while cropping wider degrades
sharply. Erring wide is the dangerous direction.

### How much of the wide-side penalty is a padding artefact?

Wide framings have to invent surroundings that a UTKFace crop does not contain,
and the choice of filler changes the answer a lot. `--pad-modes` brackets it:

| CROP_MARGIN | replicate | texture | noise | gray |
| --- | --- | --- | --- | --- |
| 0.0135 | 5.547 | 5.547 | 5.547 | 5.547 |
| 0.10 | 5.949 | 6.404 | 6.049 | 6.357 |
| 0.15 | 6.134 | 7.043 | 6.131 | 6.818 |
| 0.20 | 6.264 | 7.717 | 6.438 | 6.982 |
| 0.30 | 6.987 | 9.192 | 8.462 | 8.228 |
| 0.40 | 8.637 | 10.830 | 10.874 | 10.397 |

**Replicate padding materially understates the penalty.** Against real
photographic texture the cost at margin 0.15 is +1.50 years rather than +0.59,
and at 0.30 it is +3.65 rather than +1.44. Replicate smears edge pixels outward,
which is photometrically consistent with the face and unusually benign;
`texture` composites onto upscaled patches of other UTKFace photos, which is
real image statistics but introduces a hard seam that genuine wide framing would
not have. **Truth is bracketed between the two**, so the earlier replicate-only
table was the optimistic end of the range, and the practical safe band is
narrower than the `[0.0, 0.10]` it implied — nearer `[0.0, 0.05]`.

Note `noise` tracks `replicate` closely until 0.30: high-frequency noise is
plainly non-face and appears to be largely ignored, whereas real texture is
in-distribution for the backbone and can actively mislead it. That is the reason
to trust `texture` as the pessimistic bound rather than `noise`.

This does not change the recommendation — 0.0 is still right, and is now more
clearly right, since the penalty for erring wide is larger than first measured.
It does raise the value of the zoom-out augmentation correspondingly.

**Recommendation: `CROP_MARGIN = 0.0`.** It is the empirical optimum for the
shipped model, and it sits inside the flat region across the entire plausible
range of the true constant (-0.05 to +0.02), so the unresolved full-frame
uncertainty above costs at most ~0.35 years. The zoom-out augmentation is what
makes this safe: it converted a sharp, poorly-identified constant into a flat,
forgiving one. Anything in [-0.05, +0.05] is within 0.21 years.

Settling the constant properly would need real full-frame photographs with known
tight crops; synthetic canvases cannot do it, and UTKFace cannot supply them.

**Update: APPA-REAL supplied them.** See the margin sweep in the external
validation section below, which re-runs this curve on real full-scene
photographs. The optimum there is about -0.05 rather than 0.0 — so full-frame
detections are indeed slightly *wider* relative to the face, the direction this
section predicted but could not measure. The penalty for shipping 0.0 is ~0.42
years on that corpus, inside the tolerance claimed above.

## External validation — measured against real ages, not DEX labels

> **These are out-of-corpus numbers and are not comparable to the in-corpus
> figures above.** Everything above is agreement with UTKFace's DEX-generated
> labels. Everything in this section is error against *documented chronological
> age* on corpora the model never saw. The two answer different questions and
> the numbers must not be quoted interchangeably. The artifact was not modified
> to produce these, and `meta["test_mae"]` still holds the in-corpus figure.

Reproduce with:

```bash
python ml/external_eval.py --datasets appa fgnet      # measure
python ml/external_analysis.py                        # decompose
```

Both corpora are non-commercial research use only and stay gitignored under
`data/`. APPA-REAL ships pre-cropped faces at a 40% margin; we deliberately do
not use them, because 40% is far outside our verified safe band and would have
measured the crop bug rather than the model. YuNet runs on the original images
with the serving crop geometry at margin 0.0, decoding with the shipped median.

### Headline

| corpus | target | n | MAE | CS@5 | bias |
| --- | --- | --- | --- | --- | --- |
| APPA-REAL | `real_age` (chronological) | 7,534 | **8.52** | 46.2% | +2.81 |
| APPA-REAL | `apparent_age_avg` (crowd) | 7,534 | **7.53** | 49.4% | +3.05 |
| FG-NET | real age | 998 | 7.01 | 57.1% | +5.76 |
| FG-NET | real age, colour + unpadded only | 619 | 6.01 | 63.2% | +4.53 |

Detection failure was 0.75% on APPA-REAL and 0.40% on FG-NET, so nothing here
is survivorship over a weak detector.

**The in-corpus 4.76 does not survive contact with chronological age.** Against
real ages the model is at 8.52. That gap is the honest answer to "how much of
our number was label agreement", and it is large. What follows is the
decomposition of *why*, because the parts behave very differently.

### The model predicts apparent age, and the slope proves it better than the MAE

Scoring against `apparent_age_avg` instead of `real_age` improves MAE by only
about one year (8.52 → 7.53), which on its own looks like weak evidence. The
regression slopes are far more decisive:

| target | slope | R² |
| --- | --- | --- |
| `real_age` | 0.817 | 0.617 |
| `apparent_age_avg` | **0.935** | **0.671** |

A slope of 1.0 means the target is tracked with no scale compression. Against
apparent age we are at 0.935 — nearly calibrated. Against chronological age we
are at 0.817, i.e. systematically compressed toward the middle. **The model is a
well-calibrated predictor of how old a face looks, and a compressed predictor of
how old the person is.** That is precisely what training on DEX-generated labels
should produce, and it is the cleanest confirmation we have that the UTKFace
label-provenance caveat is not merely theoretical.

Our error also correlates **+0.445** with the human apparent-vs-real gap on the
same faces (19.8% shared variance): when humans misjudge a face, we tend to
misjudge it in the same direction. The residual is not arbitrary noise.

### The elderly result: at 70-79 we are level with a single human

This reframes the worst-looking number in the in-corpus report. Per decade,
conditioned on true age, against chronological age:

| band | n | our MAE vs real | our MAE vs apparent | our bias | 34-rater crowd | **single rater** |
| --- | --- | --- | --- | --- | --- | --- |
| 0-9 | 661 | 4.79 | 4.40 | +4.31 | 1.17 | 1.52 |
| 10-19 | 1202 | 8.47 | 6.84 | +7.42 | 3.82 | 4.50 |
| 20-29 | 2056 | 7.83 | 6.81 | +4.17 | 3.40 | 4.65 |
| 30-39 | 1552 | 9.55 | 8.68 | +1.99 | 4.21 | 5.80 |
| 40-49 | 935 | 10.30 | 9.57 | +0.56 | 5.36 | 6.97 |
| 50-59 | 608 | 8.08 | 7.91 | +0.44 | 5.39 | 6.76 |
| 60-69 | 278 | 9.36 | 8.71 | −2.90 | 6.07 | 7.39 |
| 70-79 | 130 | **10.08** | 8.07 | −7.12 | 8.43 | **10.01** |
| 80+ | 112 | 13.01 | 8.44 | −11.85 | 9.75 | **10.02** |

The widely-quoted human figure for this dataset is 4.12 years, but that is the
mean of ~34 raters, which averages rater noise away. One model should be
compared against *one* rater. APPA-REAL publishes per-image inter-rater standard
deviation (mean 4.34 years), so a single rater's expected error can be recovered
by re-injecting that spread: **5.32 years overall**, not 4.12.

On that fair comparison:

- At **70-79 we are at 10.08 against a single human's 10.01** — statistically
  indistinguishable.
- At **80+ we are at 13.01 against 10.02** — worse, but the same order, not a
  different regime.
- Our bias at 70-79 is −7.12 where the *human crowd's* own bias is −7.99. We
  underestimate the elderly slightly **less** than the human consensus does.

So the old-age bias is mostly not a defect in our model. Faces at that age
genuinely read younger than their chronological age to human observers, the
labels we trained on encode that perception, and we reproduce it. Scored against
what humans actually perceive, our 80+ MAE falls from 13.01 to **8.44** and the
bias from −11.85 to −4.96.

This does not make the residual acceptable for a product that claims to estimate
age — it means the remaining headroom is much smaller than the in-corpus table
implied, and that closing it needs better targets, not just a better model.

The one band where we are clearly and unarguably worse than humans is **10-19**
(8.47 against a single rater's 4.50, bias +7.42). Teenagers are where our real
weakness is, not the elderly. That is a genuinely new finding — it is invisible
in-corpus, where 10-19 looked mid-table.

### FG-NET: separating image quality from age reasoning

FG-NET is 44.3% under age 12 (444 of 1,002 images across 82 subjects), roughly
ten times APPA-REAL's child density, which is why it is here.

**FG-NET says nothing whatsoever about the elderly end.** Parsing all 1,002
filenames gives an age range of **0 to 69 with zero images above 69**. Any
old-age conclusion drawn from FG-NET would be vacuous.

It is also scanned film — much of it black and white, variable quality, not
pre-aligned — so some error is image quality rather than age reasoning. Those
separate cleanly. Comparing grayscale against colour *within a single age band*
(under 20), which breaks the confound that older photographs are both more
likely to be B&W and of younger subjects:

| subset (age < 20) | n | MAE | bias |
| --- | --- | --- | --- |
| colour | 545 | 5.44 | +4.98 |
| grayscale | 161 | **9.53** | **+9.39** |

Grayscale alone costs ~4.4 years of bias at matched age. A further effect comes
from framing: 21.9% of FG-NET crops needed padding because the square crop ran
off the edge of an already-tight scan, and those score 8.80 against 6.51 for the
rest.

On the cleanest subset (colour, no padding) the under-12 result is:

| under-12 subset | n | MAE | CS@5 | bias |
| --- | --- | --- | --- | --- |
| all | 440 | 5.22 | 77.3% | +4.88 |
| colour only | 346 | 3.94 | 84.4% | +3.54 |
| colour + unpadded | 288 | **3.62** | **85.8%** | **+3.17** |

So on clean images of young children the model is at **3.62 MAE against real
chronological age** — its strongest external result, and consistent with the
server's finding that displayed-age-under-12 is the model's best region. Roughly
a third of the raw FG-NET child error was film stock, not face reasoning.

A residual **+3.17 bias on children against real age** remains. Note this is the
opposite sign to what a mean-reverting decode would produce at the young
boundary, and the median decode already removed that mechanism — so this is not
the pedestal effect returning. It is consistent with UTKFace's DEX labels
systematically reading children as older, which is the label-bias branch of the
earlier discriminator, now with an independent corpus behind it.

### What this changes, and what it does not

- **The margin constant is confirmed, and its one untested assumption is now
  tested.** On real full-scene photographs the optimum is ~-0.05 rather than
  0.0, costing ~0.42 years at the shipped setting. Worth revisiting; not worth
  an emergency change.
- **The old-age bias is mostly perceptual and partly inherited**, not a training
  defect. Our 70-79 error matches a single human's.
- **Teenagers (10-19) are the real weak spot**, and that was invisible in-corpus.
- **A third of our headline accuracy was label agreement.** 4.76 in-corpus
  against 8.52 vs chronological age is the size of that effect.
- **None of this is fixed by a decode change.** These are target and training
  distribution problems.

## Real-ground-truth retrain — a third measurement regime

> **Three regimes now exist in this document and they are not interchangeable.**
> The [Results](#results) section is UTKFace test, DEX-estimated labels.
> [External validation](#external-validation--measured-against-real-ages-not-dex-labels)
> is real ages, model never trained on them. **This section** is a *different
> model* trained on real ages and scored on a real-age test split. A number
> lifted from one regime into another is wrong even when the arithmetic is
> right. The artifacts are separate files for the same reason.

### The corpus

`datasets/manifest.csv`, built by a sibling session, filtered to
`real_ground_truth == True`:

| source | images | subjects | notes |
|---|---:|---:|---|
| AgeDB | 16,487 | 567 | manually verified, ~29 images/subject |
| APPA-REAL | 7,591 | 7,591 | one image per subject |
| FG-NET | 1,002 | 82 | longitudinal, ~12 images/subject |
| **total** | **25,080** | **8,240** | larger than the 23,684 UTKFace corpus |

20 rows have `face_detected == False` and no crop; dropped. Crops were produced
at `CROP_MARGIN = 0.0`, matching serving. 70+ support is 2,206 (8.8%) against
UTKFace's thin tail, and teens 10-19 are 2,050.

**Splits are identity-aware and that is load-bearing, not hygiene.** AgeDB and
FG-NET are longitudinal — the same person appears at many ages. A random split
puts one photo of a person in train and another in test, so the model can score
well by recognising the individual rather than reading the face. **Leakage
presents as a *better* number, not as an error**, which is why
`assert_no_subject_leakage()` in `realgt_data.py` is an assertion rather than a
warning: a silent invariant that only ever makes results look good is one nobody
investigates. Re-verified independently here — zero subject overlap across all
three split pairs.

**Provenance caveat:** AgeDB came from the HuggingFace mirror
`marcelohaps/agedb`, not the official password-gated source. Labels were parsed
from original filenames rather than the mirror's derived columns. Re-encoding
cannot be ruled out and median source resolution is ~200px, so some images are
upscaled to 224. See `datasets/README.md`.

### The label-smoothing experiment, and a mechanism confirmed

Earlier this pipeline found that a **median** decode beat soft-expectation by a
wide margin (4.84 vs 5.55 MAE) on the shipped weights, and attributed it to the
uniform pedestal that `label_smoothing=0.1` trains into every bin. That pedestal
has expectation exactly 50, so the decoded mean is pulled toward it:
`E ≈ 0.9·age + 5`.

That was a mechanism inferred from one model. The retrain made it **falsifiable**:
if the pedestal is the cause, removing it should shrink the expectation-vs-median
gap, and if the gap survives the mechanism was wrong.

| variant | smoothing | E−median MAE gap (test) | (val) | decode that wins |
|---|---|---:|---:|---|
| shipped (UTKFace) | 0.1 | +0.220 | — | median |
| realgt CE | 0.1 | +0.297 | +0.327 | median |
| realgt CE | **0.0** | **−0.122** | **−0.129** | **expectation** |
| realgt DLDL (Gaussian σ=2.5) | — | −0.003 | −0.016 | tie |

**The sign flips.** And the expectation-decode bias tracks the pedestal
monotonically, which is the quantitative form of the same claim:

| variant | expectation bias |
|---|---:|
| shipped (ls=0.1, UTKFace) | +3.835 |
| realgt CE (ls=0.1) | +1.400 |
| realgt CE (ls=0.0) | −0.379 |
| realgt DLDL | **+0.002** |

**The median decode was compensating for the loss function, not for the task.**
It was the right call to ship — it recovered real accuracy from weights we
already had — but it was a workaround for a training choice, and it is worth
being clear about which of those two things a fix is.

**DLDL** replaces the uniform pedestal with a Gaussian soft target (σ=2.5)
centred on the true age. Uniform smoothing tells the model that 3 and 90 are
equally plausible alternatives for a 5-year-old, which is absurd for an ordinal
target; a Gaussian encodes that neighbouring ages are near-misses. The target is
**renormalised after truncation at the support edges** — without that, ages near
0 and 100 lose their out-of-range tail and get silently down-weighted, i.e.
exactly the extremes this retrain exists to fix.

### Which decode ships on the new artifact, and why it is *not* the old reason

Under DLDL the two decodes tie on MAE (6.390 expectation vs 6.393 median — noise)
but median is **+4.1pp on CS@5** (56.2% vs 52.1%), and that reproduces on val.
So median ships again, on CS@5 grounds, *not* as a pedestal workaround.

The generalisable warning: **a decode choice is a property of the trained
distribution, not of the problem.** Carrying one across a retrain without
re-measuring would have silently cost accuracy here. This is why
`meta["decode"]` now exists on the new artifact — an artifact carrying a bare
`test_mae` is ambiguous, since the same weights score 6.39 or 9.13 depending on
a decision recorded nowhere in the file.

### Like-for-like: old vs new

**The 8.52 figure from external validation is not a valid baseline any more.**
It was measured over all 7,534 detected APPA-REAL images, none of which the old
model had seen. The new model trains on APPA-REAL's train split. Comparing the
two would score a held-out model against a partially-seen one. Both models are
therefore scored on the **test split only**, which is held out for both.

Real-GT test split, n=3,818, 2,044 subjects, each model at its best decode:

| model | decode | MAE | CS@5 | bias | slope |
|---|---|---:|---:|---:|---:|
| shipped (UTKFace/DEX) | median | 9.127 | 42.2% | +2.485 | 0.744 |
| realgt CE ls=0.1 | median | 6.850 | 52.6% | +0.079 | 0.798 |
| realgt CE ls=0.0 | expectation | 6.767 | 49.7% | −0.379 | 0.794 |
| **realgt DLDL** | **median** | **6.393** | **56.2%** | **−0.171** | **0.808** |

Per source, test split only, median decode:

| source | n | shipped | realgt CE | DLDL |
|---|---:|---:|---:|---:|
| APPA-REAL | 1,978 | 9.317 | 6.882 | **6.404** |
| AgeDB | 1,728 | 8.993 | 6.986 | **6.578** |
| FG-NET | 112 | 7.839 | 4.205 | **3.330** |

FG-NET's 3.33 (CS@5 85.7%) is the largest relative gain, but **n=112 — do not
over-read it.** FG-NET is also 44% under-12, the region where every model here
does best, so the sample is favourable as well as small.

### Slope: the compression is better, and still real

Regression slope of predicted on true age is more diagnostic than MAE for this
failure mode, because a model can lower MAE simply by predicting the mean more
often. The old model scored **0.817 against real age** and **0.935 against
apparent age** — a well-calibrated predictor of how old a face *looks*, and a
compressed predictor of how old someone *is*.

Training on real ages moves the real-age slope **0.744 → 0.808** on this split.
That is a real improvement and it is **not** a return to 1.0. Compression
survives the removal of DEX labels entirely, so it was never purely a label
artefact.

### The two called-out bands

Per-decade, test split, median decode:

| band | n | shipped MAE | shipped bias | DLDL MAE | DLDL bias |
|---|---:|---:|---:|---:|---:|
| 0-9 | 245 | 5.19 | +4.73 | **3.80** | +3.13 |
| **10-19** | 344 | 11.22 | +10.40 | **6.99** | **+6.07** |
| 20-29 | 797 | 8.53 | +6.56 | 5.71 | +3.52 |
| 30-39 | 826 | 9.00 | +3.81 | 5.28 | −0.07 |
| 40-49 | 598 | 10.08 | +2.08 | 6.49 | −2.24 |
| 50-59 | 421 | 8.05 | −0.82 | 7.67 | −2.71 |
| 60-69 | 273 | 8.39 | −4.56 | 7.89 | −4.71 |
| **70-79** | 195 | 11.58 | −9.19 | **8.96** | **−7.48** |
| **80+** | 119 | 12.76 | −12.49 | **9.55** | **−8.69** |

**Teens (10-19): improved substantially but still the worst bias in the table.**
MAE 11.22 → 6.99, bias +10.40 → +6.07. External validation had found teens to be
the genuine weak spot (old model bias +7.42 against a single human rater's 4.50),
and training on real ages cut the bias by 42%. A +6 year bias on teenagers is
still the largest signed error anywhere in the range, and it is in the direction
that matters most for any age-gating use: **teenagers read as adults.**

**70+: moved, which settles a question.** Bias −12.49 → −8.69 at 80+. The
previous finding was that our elderly bias *matched a single human rater's* error
on the same images, which left it ambiguous — irreducible perceptual difficulty,
or learned from labels? **Training on real chronological ages moved it by 3.8
years without changing the architecture, so part of it was demonstrably learned
from the labels.** That part was never a property of faces.

But the majority of it did not move. A −8.69 residual at 80+ after training on
real ages, with 2,206 images of 70+ faces in the corpus, is not explained by
label provenance or by data scarcity. The honest statement is that this
experiment **partitioned** the elderly bias into a label-induced component (now
removed) and a residual (not explained here) — it did not eliminate it, and it
does not identify what the residual is.

#### Why the architecture was deliberately not improved

A reasonable reader will ask why a retrain was not also an opportunity to try a
better backbone. It was held fixed on purpose — same ImageNet-pretrained
`mobilenetv3_small_100`, same 101-bin head — so that **the only thing that
changed was the data**.

That constraint is what makes the 3.8-year movement mean anything. Had the
architecture moved too, the elderly bias would still have improved by the same
3.8 years and the number would have been *uninterpretable*: no amount of care in
measuring it could have attributed it between labels and capacity after the fact.
The partition is a property of the experiment's design, not of how carefully the
result was measured, and no subsequent analysis could have recovered it.

The same applies to the decode comparison. Holding architecture, head, optimiser,
schedule and seed fixed across the three loss variants is what licenses reading
the sign flip as caused by the smoothing pedestal rather than by anything else
that happened to differ.

The cost is real — this says nothing about whether a larger backbone would do
better, and it probably would. That question is still open precisely because
answering it here would have closed a more valuable one.

### A destroyed checkpoint, and what it cost

Partway through publishing I wrote the chosen artifact to
`checkpoints/age_model_realgt.pt` — which was **also** the path the CE variant
had trained to. The CE weights were overwritten and unrecoverable. Nothing
errored: the file was still a perfectly valid checkpoint, just no longer the one
the reports referenced. It was caught only by hashing the state dicts and
noticing two supposedly-different variants were byte-identical.

The CE variant was retrained to a distinct path. It reproduced closely but not
exactly (val MAE 6.911 vs the original 7.079 — MPS is not bit-deterministic
across runs), so **every CE number in this section comes from the reproducible
checkpoint, not the destroyed one.** The conclusions were unaffected: the decode
sign flip and the bias ladder both reproduce. Had they not, there would have
been no way to tell which run was the anomaly.

`save_checkpoint()` now refuses to overwrite a path in `PUBLISHED_PATHS`.
The general lesson is the one this repo keeps relearning in different costumes:
**a path that is both an experiment output and a published artifact will
eventually be written by the wrong one, and the failure is silent** because the
file remains valid. The earlier version of this was republishing
`age_model.pt` under a live consumer; this was the same mistake pointed at
myself. Publish paths and experiment paths must be disjoint.

### A checkpoint that advertised a score it had never earned

`train.py` stamps the best *validation* MAE into `meta["test_mae"]`, on the
assumption that `eval.py` overwrites it with a real held-out measurement
afterwards. For the shipped artifact it did. For the three real-GT variants it
never ran, so all three sat on disk advertising a **val** number in a field
named `test_mae`. The caveat existed, but only as a comment in `train.py` —
invisible to anyone reading the file.

Every one of them flattered itself, because val was the selection set:

| variant | advertised (val) | measured (test) | error |
|---|---:|---:|---:|
| realgt CE | 6.911 | 6.850 | −0.06 |
| realgt ls=0.0 | 6.464 | 6.767 | **+0.30** |
| realgt DLDL | 6.263 | 6.393 | +0.13 |

This is the same failure the artifact-manifest work was meant to stop, one level
further in: **a file inheriting an accuracy figure it did not earn.** The fix is
the same principle — provenance must travel *with* the artifact, not alongside it
in source. `train.py` now writes `test_mae_source` and `val_mae` into `meta`, so
a provisional number announces itself.

The three variants have been re-stamped with their measured test figures, plus a
`decode` key and a `role` field marking them as experiment intermediates rather
than published artifacts. `ls0` was the one that mattered: it is the **only**
checkpoint here where expectation beats median (6.767 vs 6.889), so a consumer
defaulting to median would have silently measured the worse decode and concluded
that removing label smoothing underperforms — inverting this section's central
result. Caught by the server session, which asked why only one of four
checkpoints carried the `decode` key.

### Conditioning, again — and the number neither table shows

Every per-decade table above bins by **true** age. That is correct for model
evaluation and wrong for a UI, which can only threshold on what the model
outputs. Both views, median decode, real-GT test split:

| band | by TRUE age (n / MAE / bias) | by DISPLAYED age (n / MAE / bias) |
|---|---|---|
| 0-9 | 245 / 3.80 / +3.13 | 224 / 2.20 / −0.38 |
| 10-19 | 344 / 6.99 / **+6.07** | 238 / 4.84 / **−2.12** |
| 40-49 | 598 / 6.49 / −2.24 | 593 / 7.24 / +1.18 |
| 70-79 | 195 / 8.96 / **−7.48** | 166 / 7.89 / −0.31 |
| 80+ | 119 / 9.55 / **−8.69** | 59 / 6.20 / **+1.02** |

**The signs invert.** Teens carry a +6.07 bias by true age and −2.12 by
displayed age; 80+ carries −8.69 and +1.02. Both are correct. They answer
different questions, and a caveat keyed to the wrong one fires on the wrong
people — which has now happened three times in this project.

One genuine improvement worth recording: under the real-GT model the
displayed-age bias is **near zero in every band** (−0.71 to +1.18 across the UI
bands), where the shipped model ran to **+7.31** at displayed 40-54. Conditioned
on what it shows, the new model is close to unbiased; only precision degrades
with age (MAE 2.72 at under-13 rising to 7.64 at 65+).

**But neither table shows the number that matters for a gate.** At a hard
threshold the two conditionings give wildly different error rates, because
adults outnumber minors in this corpus:

| threshold | true minors shown as adult | shown-adult actually minor |
|---:|---:|---:|
| 13 | 18.1% | 1.6% |
| 16 | 25.6% | 2.8% |
| **18** | **30.2%** | **4.2%** |
| 21 | 34.3% | 6.6% |

**30% of true under-18s are displayed as 18 or over** — by the *better* model;
the shipped one is 40.3%. Read from the observable side the same gate looks 96%
correct. That gap is pure base rate, and it is the strongest argument in this
repo against using these predictions for anything gate-like. `predicted_age_bands.py`
computes both directions precisely so neither can be quoted alone.

### What this retrain cost us, measurably

Training on APPA-REAL's train split **consumed part of the only instrument that
could explain the residual.** APPA-REAL is the one corpus here carrying both
`real_age` and `apparent_age_avg`, which is what makes it able to separate "the
model is wrong" from "the face looks that age". The model has now seen 5,613 of
those rows, so any future apparent-vs-real decomposition is restricted to the
1,978-row test split rather than the full 7,591.

That was the right trade — the corpus was needed and held-out evaluation is
preserved — but it was not free, and it is worth recording as a cost rather than
discovering later as a limitation. The instrument got smaller as a direct result
of the experiment that made us want to use it.

This matters because the attractive next move is to assume the unexplained
residual (slope stuck at 0.808, 80+ bias stuck at −8.69) is irreducible
perceptual difficulty. **Nothing here shows that.** Two independent measures
agree that roughly a third of the compression was learned from labels and the
rest was not; neither identifies what the rest is. AgeDB and FG-NET carry real
ages only, so they cannot answer it. FG-NET additionally has **zero images above
age 69**, so it is silent on precisely the band in question.

### σ controls interval *width* — a prediction, mostly refuted

This section previously argued that DLDL caused a coverage regression. **It was
wrong, and the regression it explained did not exist.** The original claim is
replaced rather than deleted, because the way it failed is more instructive than
the claim was.

**What I claimed.** The server session reported 68%-nominal interval coverage of
60.8% under the real-GT model, down from 75% under the shipped one. I argued the
cause was structural to DLDL: it trains every face toward a Gaussian of fixed
σ=2.5, KL penalises sharpening past that target, so decoded width is anchored to
a hyperparameter rather than expressing per-face uncertainty. The 0.16/0.84
quantiles of `N(μ, 2.5)` span 5.0 years, which against MAE 6.39 must undercover.

I stated two disconfirming conditions in advance: widths should be *less
dispersed* under DLDL, and width should correlate *less well* with `|error|`.

**What the measurement showed.** The server re-ran the shipped checkpoint on the
**same real-GT split, same harness, same crop** (MAE 9.110 vs 9.127 — control
passes), which is the comparison that had never been made:

| model | MAE | coverage | mean width | sd width | r(width, \|err\|) |
|---|---:|---:|---:|---:|---:|
| shipped | 9.110 | **61.7%** | 20.61 | 12.92 | +0.309 |
| realgt DLDL | 6.343 | **60.8%** | 11.62 | 4.22 | +0.364 |

- **Dispersion: confirmed.** sd 12.92 → 4.22, IQR 14 → 4, p5–p95 span 42 → 12
  years. Widths really are clustered under DLDL, predictable from the loss alone.
- **Predictiveness: refuted**, by the condition I set. r went +0.309 → **+0.364**
  — stronger, not weaker. Clustering cost no ranking power.
- **The σ=2.5 anchor: refuted on magnitude.** ±1σ is 5.0 years; observed mean
  width is 11.62, and only 3.1% of faces sit at or below 5.0. KL permits far more
  hedging than the target σ, so the mass is not where the mechanism required.

**And the regression was not real.** The 75% figure came from the *UTKFace*
split; the 60.8% from the *real-GT* split. Two variables changed at once. On a
common corpus the two models differ by **0.9pp** (61.7% vs 60.8%) — the shipped
model undercovers by the same ~7pp. The drop was the corpus, not the model.

**What DLDL actually did to the interval:** equal coverage at **44% narrower**
width, with width slightly *more* correlated with error. That is a strictly
better interval, and it had been filed as a degradation.

The ~7pp shortfall against the nominal 68% is real but belongs to **both**
models, making it the one derived number in this system that is *not* per-model —
after every other number turned out to be. Quantiles remain un-retuned; that
reasoning is firmer now that there is nothing model-specific to chase.

**Consequence for the queued σ sweep**, sharpened by the result: report **width
and coverage as separate columns**, not coverage alone. σ demonstrably controls
width (20.6 → 11.6 is a large real effect) while moving coverage by 0.9pp — the
two decoupled. A sweep watching only coverage would read a halved interval as
"no effect" and miss the improvement entirely. The double duty is real; the
second duty is width, not calibration.

**Why this section is worth keeping.** The mechanism was plausible, internally
consistent, derived from a property of the loss I had verified numerically — and
it was constructed to explain a comparison that was never valid. A structural
explanation for an artifact is harder to dislodge than a wrong number, because it
*predicts* the artifact: wrong numbers get contradicted by data, wrong mechanisms
get corroborated by it. The falsification conditions are what made it cheap to
retire; without them it would have been an unfalsifiable story that happened to
fit. **State the disconfirming evidence in advance, and check the baseline before
explaining the delta** — that discipline matters more than any of the specific
column recommendations above, and it is the one thing to carry into the σ sweep.

A corollary about where numbers live. This figure existed only in prose — it had
no home in code, so it had no declared scope, and the first consumer to use it
supplied one. That is how a corpus-level property became a per-model regression.
It now lives in the server's `INTERVAL_CALIBRATION` at module scope, served as a
sibling of the per-model table rather than a field inside it, so its shape
carries its scope. An unhoused number will acquire whatever scope its first
reader assumes.

### What this does not show

- **Not a better product model, necessarily.** If the goal is to predict how old
  someone *looks*, the shipped UTKFace model is better at that by construction.
  This model is better at chronological age. Those are different products, and
  the 5.55-vs-6.39 comparison people will reach for is between two different
  questions on two different corpora.
- **σ=2.5 was not tuned, and it does more than one job.** It was the midpoint of
  a suggested range and the first value tried, so the DLDL result may improve or
  may be partly luck. σ also controls the **width of the predicted distribution**
  (measured: mean width 20.6 → 11.6 vs the shipped model) without materially
  moving coverage — so a sweep must report width and coverage separately. See
  the section above.
- **One seed per variant.** The CE-vs-DLDL gap (6.850 vs 6.393) is large enough
  to be believable; the expectation-vs-median gaps near zero are within what a
  seed change could plausibly move. The *sign flip* across smoothing levels
  reproduces on val and is the robust claim, not any individual decimal.
- **AgeDB dominates at 66% of the corpus**, so "real ground truth" here largely
  means "AgeDB", with its mirror-provenance caveats.

## Future work

**Done: APPA-REAL and FG-NET.** What was previously the top item here has been
carried out; see the external validation section above. The short version is
that it confirmed the caveat rather than dispelling it — the model is a
well-calibrated predictor of *apparent* age (slope 0.935) and a compressed
predictor of chronological age (slope 0.817), and roughly a third of the
headline accuracy was label agreement.

**Still outstanding: AgeDB.** It is manually collected and verified, with wide
age range and good 70+ support where both UTKFace and FG-NET are thinnest. Our
80+ estimate currently rests on 112 APPA-REAL images and nothing else, which is
the weakest-supported claim in this README.

A sharper version of the same point, raised by the server session: because
UTKFace labels are themselves softmax-expectation estimates, and expectation
decoding is precisely what produced the compression we removed, **the in-corpus
score may flatter a median decode differently than it flattered the mean**. The
4.84 is agreement-with-DEX under a decode DEX did not use. Neither of us can
say a priori whether that flatters or penalises median — only that the *size* of
the improvement is measured against a yardstick carrying a related bias.

One partial independent check does exist, though it is weaker than a real-age
corpus. The server session's end-to-end 4.762 was obtained with **YuNet framing
each crop**, not UTKFace's own crop geometry — so the result does not depend on
the framing pipeline that generated the labels. That rules out the narrowest
version of the worry (a pure label-generation-geometry artefact) while leaving
the broader one (label *values* produced by expectation decoding) untouched.
APPA-REAL remains the only thing that settles it.

Now that a real-age baseline exists, two follow-ups are unblocked and were
previously gated on it:

- **A class-balanced or age-weighted loss for the thin tails.** This was
  deliberately not attempted before, because in-corpus validation could not
  distinguish "helped" from "fit the labels' bias more closely". It now can.
  The external data also sharpens the target: **10-19 is our real weak spot**
  (MAE 8.47 against a single rater's 4.50), not the elderly, so any reweighting
  should be aimed there rather than at 80+ as the in-corpus table suggested.
- **Training against apparent-age labels directly.** APPA-REAL ships them, and
  the slope analysis shows that is effectively the function we already compute.
  Making it explicit would let the uncertainty interval mean something
  well-defined, rather than straddling two different targets.

### Known-stale measurements

- The **crop-margin sweep above was measured under the soft-expectation
  decode.** Margin 0.0 has been re-verified under median end to end (4.762), but
  the full curve has not been re-swept. The shape is expected to hold and the
  levels to drop; until that is redone, treat the absolute MAE values in those
  tables as expectation-decode figures and the *shape* as the load-bearing part.
- The per-decade and per-decile tables in `reports/metrics.json` are likewise
  expectation-decoded, since `eval.py` still defaults to that decode for
  continuity with the original spec.

## Files

| file | purpose |
| --- | --- |
| `data.py` | manifest, stratified splits, augmentations, `UTKFaceDataset` |
| `model.py` | `AgeEstimator`, checkpoint save/load, artifact metadata |
| `train.py` | training loop |
| `eval.py` | test-set metrics, per-decade breakdown, scatter plot |
| `export_onnx.py` | ONNX export + parity verification |
| `measure_crop_margin.py` | YuNet margin measurement (serving/training framing) |
| `margin_sweep.py` | MAE vs. crop-margin sensitivity |
| `decode_compare.py` | expectation vs mode vs median decode comparison |
| `external_eval.py` | APPA-REAL / FG-NET evaluation against real ages (read-only) |
| `external_analysis.py` | decomposes external error into model, perception and label terms |
| `publish_manifest.py` | sha256 + provenance sidecar (one per published artifact) |
| `realgt_data.py` | real-GT corpus loader + the no-subject-leakage assertion |
| `realgt_compare.py` | like-for-like old-vs-new on held-out splits, all decodes |
| `predicted_age_bands.py` | error by *predicted* age + both threshold error rates |
| `splits/` | committed train/val/test CSVs |
| `reports/` | metrics, training history, scatter plot, external validation |
