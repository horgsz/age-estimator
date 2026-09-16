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
| `state_dict` | model state dict |
| `meta` | `{"backbone": "mobilenetv3_small_100", "num_bins": 101, "input_size": 224, "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225], "test_mae": <float>}` |

`checkpoints/age_model.onnx` is opset 17 with a dynamic batch axis:

| tensor | name | shape | dtype |
| --- | --- | --- | --- |
| input | `input` | `[N, 3, 224, 224]` NCHW | float32 |
| output | `logits` | `[N, 101]` | float32 |

### Consuming the output

The head is a classifier, not a scalar regressor. Preprocess with resize to 256,
center crop 224, scale to `[0, 1]`, then ImageNet normalize with the `mean`/`std`
from `meta`. Recover the age as the soft expectation over bin indices, and read
the distribution's standard deviation as an uncertainty estimate:

```python
probs = softmax(logits, axis=1)          # [N, 101]
bins = np.arange(101)
age = (probs * bins).sum(axis=1)
sigma = np.sqrt((probs * (bins - age[:, None]) ** 2).sum(axis=1))
```

`AgeEstimator.predict()` in `ml/model.py` does exactly this for the PyTorch path.

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

It is not merely free — it wins on the clean test set *and* flattens the
wide-side cliff:

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
| 0.20 | — | 6.264 |

The curve is strongly **asymmetric**: cropping tighter than training costs
essentially nothing over the whole range tested, while cropping wider degrades
sharply. Erring wide is the dangerous direction.

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
