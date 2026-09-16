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

UTKFace age labels are **dataset-provided annotations of unverified
provenance**, not birth records. They are known to contain errors, and parts of
the corpus were labelled with the help of automated age estimation rather than
by documented ground truth. Three consequences worth stating plainly:

- There is an **irreducible label-noise floor** under every number in this
  README. A ~4-5 year MAE is close enough to plausible annotation error that
  some of the residual is the labels, not the model. Do not read further
  improvement in this range as straightforwardly real.
- Because some labels may descend from model estimates, a model trained here
  can be partly fitting an **earlier estimator's biases**, which flatters
  in-corpus evaluation.
- Cross-dataset comparisons (and published UTKFace leaderboard numbers) are
  **not** comparable to these unless they use this exact split.

The per-decade table is the honest view: the thin tails have both few samples
*and* the least reliable labels, so treat 70+ figures as indicative only.
Nothing here should be presented to an end user as a measured age.

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

Use the IQR rather than the standard deviation as the confidence signal when
decoding with the median -- both describe the same distribution, but the IQR is
robust to the same tail mass the median is.

## Results

Test split (1,185 held-out images), `checkpoints/age_model.pt`:

| metric | value |
| --- | --- |
| MAE | **5.55 years** |
| CS@5 | **55.5%** |
| RMSE | 7.46 years |
| mean bias | +2.13 years |

Per-decade breakdown — the interesting part, because UTKFace is heavily skewed
toward ages 20-35:

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
| `splits/` | committed train/val/test CSVs |
| `reports/` | metrics, training history, scatter plot |
