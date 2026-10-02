#!/usr/bin/env bash
# Publish the large server-only model as a Hugging Face Space and point the
# GitHub Pages site at it.
#
#   one-time:  .venv/bin/hf auth login        (a token with write access)
#   then:      scripts/deploy_hf_space.sh [space-name]
#
# Free CPU Spaces sleep after ~48h without traffic; the first request after
# that takes about a minute while it wakes.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="${PY:-$ROOT/.venv/bin/python}"
SPACE_NAME="${1:-age-estimator-api}"
MODEL="$ROOT/checkpoints/age_model_server.pt"
YUNET="$ROOT/server/models/face_detection_yunet_2023mar.onnx"

[ -f "$MODEL" ] || { echo "missing $MODEL; train it with scripts/train_server_model.sh" >&2; exit 1; }
[ -f "$YUNET" ] || { echo "missing $YUNET; run the server once (make api) to fetch it" >&2; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp deploy/hf-space/{Dockerfile,requirements.txt,README.md} "$STAGE/"
mkdir -p "$STAGE/server/models" "$STAGE/checkpoints"
cp server/*.py "$STAGE/server/"
cp -R server/tools "$STAGE/server/" 2>/dev/null || true
cp "$YUNET" "$STAGE/server/models/"
cp "$MODEL" "$STAGE/checkpoints/"

SPACE_ID="$("$PY" - "$SPACE_NAME" "$STAGE" <<'PY'
import sys
from huggingface_hub import HfApi
name, stage = sys.argv[1], sys.argv[2]
api = HfApi()
space = f"{api.whoami()['name']}/{name}"
api.create_repo(space, repo_type="space", space_sdk="docker", exist_ok=True)
api.upload_folder(repo_id=space, repo_type="space", folder_path=stage,
                  commit_message="Deploy age estimator API")
print(space)
PY
)"
URL="https://$(echo "$SPACE_ID" | tr '/' '-' | tr '[:upper:]' '[:lower:]').hf.space"
echo "Space: https://huggingface.co/spaces/$SPACE_ID"
echo "API:   $URL"

echo "waiting for $URL/health (first build takes a few minutes) ..."
for _ in $(seq 1 60); do
  if curl -fsS "$URL/health" >/dev/null 2>&1; then echo "  up"; break; fi
  sleep 20
done
curl -fsS "$URL/health" >/dev/null || { echo "API not healthy yet; check the Space build logs" >&2; exit 1; }

# Point the Pages build at it and redeploy. Uses the repo owner's gh login.
GH="env -u GH_TOKEN gh"
$GH variable set AGE_API_BASE -R horgsz/age-estimator --body "$URL"
$GH workflow run pages.yml -R horgsz/age-estimator --ref master
echo "Pages redeploy started; the site will default to the hosted model."
echo "Undo: gh variable delete AGE_API_BASE -R horgsz/age-estimator && rerun pages.yml"
