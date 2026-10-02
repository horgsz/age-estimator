#!/usr/bin/env bash
# End-to-end local retrain of the served real-age model, focused on ages 14-60.
#
#   download -> manifest -> YuNet crops -> fine-tune -> held-out test -> ONNX
#
# Every step is idempotent, so an interrupted run can simply be restarted. The
# candidate is written beside the served model, never over it:
#   checkpoints/age_model_candidate.{pt,onnx}
#
# Knobs (environment variables):
#   PY=.venv/bin/python   interpreter with ml/ + datasets/ requirements
#   EPOCHS=12             training epochs
#   LR=1e-4               peak learning rate (3e-4 is the from-scratch default)
#   INIT=checkpoints/age_model_realgt.pt   start weights; INIT=imagenet to retrain from scratch
#   SKIP_DOWNLOAD=1       skip datasets/download.sh
#   SKIP_IMDB=1           leave IMDB-Clean out (download and training)
#   EXTRA_TRAIN_ARGS=...  passed through to ml/train.py
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
EPOCHS="${EPOCHS:-12}"
LR="${LR:-1e-4}"
INIT="${INIT:-checkpoints/age_model_realgt.pt}"
CANDIDATE="$ROOT/checkpoints/age_model_candidate.pt"

step() { printf '\n=== %s ===\n' "$*"; }

if [ "${SKIP_DOWNLOAD:-0}" != 1 ]; then
  step "1/6 download"
  SKIP_IMDB="${SKIP_IMDB:-0}" bash datasets/download.sh
fi

step "2/6 manifest (identity-disjoint splits)"
(cd datasets && "$PY" build_manifest.py --skip-missing)

step "3/6 YuNet crops at the serving margin (existing crops are reused)"
(cd datasets && "$PY" crop_faces.py)

SOURCES=(agedb appa-real fgnet)
[ "${SKIP_IMDB:-0}" = 1 ] || SOURCES+=(imdb-clean)
(cd ml && "$PY" realgt_data.py --datasets-root "$ROOT" --sources "${SOURCES[@]}")

step "4/6 train (age-focus 14-60, DLDL)"
INIT_ARGS=()
[ "$INIT" = imagenet ] || INIT_ARGS=(--init-from "$ROOT/$INIT")
(cd ml && "$PY" train.py \
  --corpus realgt --loss dldl --age-focus \
  --sources "${SOURCES[@]}" \
  --datasets-root "$ROOT" \
  --checkpoint "$CANDIDATE" \
  --epochs "$EPOCHS" --lr "$LR" --warmup-epochs 1 \
  ${INIT_ARGS[@]+"${INIT_ARGS[@]}"} ${EXTRA_TRAIN_ARGS:-})

step "5/6 held-out test: served model vs candidate"
(cd ml && "$PY" realgt_compare.py --datasets-root "$ROOT" \
  --checkpoints "served=$ROOT/checkpoints/age_model_realgt.pt" "candidate=$CANDIDATE" \
  --out "$ROOT/ml/reports/candidate_comparison_test.json")

step "6/6 ONNX export"
(cd ml && "$PY" export_onnx.py --checkpoint "$CANDIDATE" \
  --onnx "$ROOT/checkpoints/age_model_candidate.onnx")

echo
echo "Done. Compare the 14-60 rows in ml/reports/candidate_comparison_test.json."
echo "Promoting the candidate to the served model is a separate step: its digest,"
echo "measured accuracy and the static registry all have to be updated."
