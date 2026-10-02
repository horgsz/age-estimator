#!/usr/bin/env bash
# Train the large, server-only model: CLIP ViT-B/16 (86M params) on
# AgeDB + APPA-REAL + FG-NET + IMDB-Clean + CACD, focused on ages 14-60.
#
# Too large for the browser build; it is served by the FastAPI server
# (`AGE_MODEL_PATH_REAL=checkpoints/age_model_server.pt make dev`).
#
# Prerequisites: datasets/manifest_v2.csv built and cropped
#   (cd datasets && ../.venv/bin/python build_manifest.py --skip-missing --out manifest_v2.csv
#    && ../.venv/bin/python crop_faces.py --manifest manifest_v2.csv)
#
# Knobs: EPOCHS=8 SAMPLES=150000 LR=5e-5 WD=0.05 WAIT_PID=<pid to wait for>
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
MANIFEST="$ROOT/datasets/manifest_v2.csv"
OUT="$ROOT/checkpoints/age_model_server.pt"

if [ -n "${WAIT_PID:-}" ]; then
  echo "waiting for PID $WAIT_PID (the GPU is busy) ..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
fi

printf '\n=== 4/6 train (CLIP ViT-B/16, age-focus 14-60, DLDL) ===\n'
(cd ml && "$PY" train.py \
  --corpus realgt --loss dldl --age-focus \
  --manifest "$MANIFEST" --datasets-root "$ROOT" \
  --backbone vit_base_patch16_clip_224 \
  --epochs "${EPOCHS:-8}" --samples-per-epoch "${SAMPLES:-150000}" \
  --lr "${LR:-5e-5}" --wd "${WD:-0.05}" --warmup-epochs 1 \
  --checkpoint "$OUT")

printf '\n=== 5/6 held-out test: served model vs server model ===\n'
(cd ml && "$PY" realgt_compare.py --datasets-root "$ROOT" --manifest "$MANIFEST" \
  --checkpoints "served=$ROOT/checkpoints/age_model_realgt.pt" "server=$OUT" \
  --out "$ROOT/ml/reports/server_model_comparison_test.json")

echo
echo "Done. Compare ml/reports/server_model_comparison_test.json."
echo "Serve it locally: AGE_MODEL_PATH_REAL=checkpoints/age_model_server.pt make dev"
