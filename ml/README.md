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
| MAE | **5.75 years** |
| CS@5 | **52.2%** |
| RMSE | 7.67 years |
| mean bias | +2.35 years |

Per-decade breakdown — the interesting part, because UTKFace is heavily skewed
toward ages 20-35:

| decade | support | MAE | CS@5 | bias |
| --- | --- | --- | --- | --- |
| 0-9 | 154 | 5.64 | 53.2% | +5.64 |
| 10-19 | 76 | 5.66 | 53.9% | +4.96 |
| 20-29 | 368 | 5.32 | 54.6% | +4.83 |
| 30-39 | 228 | 4.86 | 62.3% | +2.07 |
| 40-49 | 112 | 5.86 | 42.9% | +0.01 |
| 50-59 | 116 | 7.03 | 43.1% | -0.11 |
| 60-69 | 65 | 6.00 | 52.3% | -2.19 |
| 70-79 | 34 | 8.18 | 29.4% | -7.47 |
| 80+ | 32 | 9.56 | 34.4% | -9.37 |

The bias column is monotonically decreasing, which is textbook regression toward
the mean: the model is pulled toward the data-rich 20-40 band and compresses
both tails. Note that for ages 0-9 the MAE and the bias are identical (+5.64),
meaning the model *never* under-predicts a child. The 80+ decade is off by
-9.4 years on 32 test images. Treat predictions below ~15 and above ~70 as
weakly supported.

Training peaked at epoch 14/30 (val MAE 5.709) and then overfit, with train loss
falling to 1.76 while val MAE drifted back to ~6.1. Best-val checkpointing keeps
the epoch-14 weights. A regularized variant (`--wd 0.05 --mixup 0.2`) was tried
and was **worse** (val MAE 6.07): mixing two faces produces an image with no
well-defined age, which fights the ordinal soft-expectation head. The `--mixup`
flag remains available but defaults to off.

## Crop-margin sensitivity

The serving path crops `side = max(w, h) * (1 + 2 * CROP_MARGIN)` around a YuNet
detection. UTKFace is already tightly cropped, so training framing corresponds
to a specific margin, and inference must reproduce it.

`measure_crop_margin.py` over 300 random images (300/300 detected, 0% failure)
gives a median implied margin of **0.0135**, IQR [0.0012, 0.0268].

`margin_sweep.py` measures what framing error actually costs:

| CROP_MARGIN | MAE | CS@5 |
| --- | --- | --- |
| 0.00 | 5.72 | 54.6% |
| **0.0135** | **5.75** | **52.2%** |
| 0.05 | 5.90 | 52.7% |
| 0.10 | 6.21 | 51.8% |
| 0.15 | 7.39 | 46.6% |
| 0.20 | 9.01 | 40.9% |
| 0.30 | 11.55 | 30.4% |
| 0.40 | 13.60 | 25.1% |

Margins up to ~0.10 cost less than half a year. Beyond that it degrades sharply:
a 0.4 margin more than doubles the error to 13.6 years. Keep `CROP_MARGIN` at
**0.0135 (±0.05)**. Wider framings are simulated with replicate padding, so
those rows are indicative rather than exact, but the trend is unambiguous.

Note that `RandomResizedCrop(scale=(0.8, 1.0))` only ever crops *in*, so it buys
tolerance to framings tighter than UTKFace and none to wider ones. The margin
constant has to be right; augmentation will not paper over it.

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
