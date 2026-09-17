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
| `publish_manifest.py` | sha256 + provenance sidecar for the published artifact |
| `splits/` | committed train/val/test CSVs |
| `reports/` | metrics, training history, scatter plot, external validation |
