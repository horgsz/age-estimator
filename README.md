# age-estimator

Age estimation from a face photo, with a webcam UI.

**Live, entirely client-side: <https://horgsz.github.io/age-estimator/>**

Two ways to run it, sharing one interface and one set of weights:

| | runs where | inference | use |
| --- | --- | --- | --- |
| [`server/`](server/README.md) + [`web/`](web/README.md) | your machine | FastAPI + PyTorch | development, evaluation, the crop-margin sweep |
| the static build | the visitor's browser | onnxruntime-web (WASM) | GitHub Pages, where no Python can run |

```sh
make dev            # API + web UI, the local workflow
make build-static   # the client-side bundle
make parity         # prove the two paths agree
make test           # server test suite
make eval           # end-to-end MAE of the deployed inference path
```

The browser build is a **second implementation of preprocessing that already
exists in Python**. That is a liability, not a feature: a silent mismatch
between them degrades every prediction while leaving every test green — which
has happened here once already, with a crop margin that cost 2.4x accuracy past
a healthy-looking evaluation. [`parity/`](parity/README.md) is the check that
stops it recurring, and it requires the two paths' 224x224 input tensors to be
*identical*, not merely close.

## Read before quoting any number

* [`MODEL_LICENSING.md`](MODEL_LICENSING.md) — what the weights were trained on,
  what that permits, and what is still unverified.
* **Neither model is usable for age verification.** About 30% of under-18s are
  displayed as 18 or over by the real-age model, and about 40% by the
  apparent-age model.
* The two models answer *different questions* — how old someone **is** versus
  how old they **look** — so their error figures are not a ranking. Every
  derived figure in the UI is keyed to the checkpoint it was measured on.

## Layout

```
server/      FastAPI app: YuNet detection, the crop, the PyTorch predictor
web/         the UI, and the browser inference path (src/browser/)
parity/      the harness that compares the two paths, and its results
ml/          training and offline evaluation (read-only from here)
checkpoints/ the two served models, .pt and .onnx (read-only from here)
```
