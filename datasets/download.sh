#!/usr/bin/env bash
# Download and extract every obtainable source into datasets/raw/.
#
# curl, not Python: on a machine behind a TLS-intercepting proxy the system
# keychain holds the root CA and Python's bundled certifi does not. Python
# downloads fail with CERTIFICATE_VERIFY_FAILED there; curl works.
#
# UTKFace is NOT downloaded here. It is ml/'s dataset and is expected at
# <repo>/data/UTKFace, as set up by ml/README.md.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW="$HERE/raw"
DL="$RAW/_dl"
mkdir -p "$DL"

fetch() { # url dest
  if [ -s "$2" ]; then echo "  already have $(basename "$2")"; return; fi
  curl -fSL --retry 3 --max-time 3600 -o "$2.part" "$1"
  mv "$2.part" "$2"
}

# --- APPA-REAL -------------------------------------------------------------
# The .es domain cited by most papers (and by yu4u/age-estimation-pytorch) is
# dead; chalearnlap.cvc.uab.cat is the live host as of 2026-09-17.
echo "APPA-REAL (885 MB) ..."
fetch "https://data.chalearnlap.cvc.uab.cat/AppaRealAge/appa-real-release.zip" \
      "$DL/appa-real-release.zip"
[ -d "$RAW/appa-real-release" ] || unzip -q "$DL/appa-real-release.zip" -d "$RAW"

# --- FG-NET ----------------------------------------------------------------
# The ibug.doc.ic.ac.uk mirror returns HTTP 500; yanweifu's copy works.
echo "FG-NET (46 MB) ..."
fetch "https://yanweifu.github.io/FG_NET_data/FGNET.zip" "$DL/FGNET.zip"
[ -d "$RAW/fgnet/FGNET" ] || unzip -q "$DL/FGNET.zip" -d "$RAW/fgnet"

# --- AgeDB -----------------------------------------------------------------
# The official distribution is a password-protected Dropbox zip whose password
# must be requested by email from the maintainers, so it cannot be scripted.
# This pulls a third-party HuggingFace mirror instead. READ datasets/README.md
# before relying on it -- its provenance is plausible but unverified, and the
# uploader's licence tag does not override AgeDB's own non-commercial terms.
echo "AgeDB (~150 MB, third-party mirror -- see README) ..."
REPO="marcelohaps/agedb"
BASE="https://huggingface.co/datasets/$REPO/resolve/main"
mkdir -p "$RAW/agedb"
curl -fsSL -o "$RAW/agedb/metadata.csv" "$BASE/train/metadata.csv"

LIST="$DL/agedb_files.txt"
if [ ! -s "$LIST" ]; then
  : > "$LIST"
  curl -fsSL "https://huggingface.co/api/datasets/$REPO/tree/main/train/images" \
    | python3 -c "import sys,json;[print(x['path']) for x in json.load(sys.stdin) if x['type']=='directory']" \
    | while read -r shard; do
        curl -fsSL "https://huggingface.co/api/datasets/$REPO/tree/main/$shard?limit=1000" \
          | python3 -c "import sys,json;[print(x['path']) for x in json.load(sys.stdin) if x['type']=='file']" >> "$LIST"
      done
fi

# Idempotent: re-running only fetches files that are missing or zero-length.
RAW="$RAW" BASE="$BASE" xargs -P 16 -I{} sh -c '
  p="{}"; o="$RAW/agedb/${p#train/}"
  mkdir -p "$(dirname "$o")"
  [ -s "$o" ] || curl -fsSL --retry 2 --max-time 60 -o "$o" "$BASE/$p"
' < "$LIST"

echo
echo "Counts:"
echo "  appa-real originals: $(find "$RAW/appa-real-release" -maxdepth 2 -name '*.jpg' ! -name '*_face.jpg' 2>/dev/null | wc -l | tr -d ' ')"
echo "  fgnet images:        $(find "$RAW/fgnet/FGNET/images" -iname '*.jpg' 2>/dev/null | wc -l | tr -d ' ')"
echo "  agedb images:        $(find "$RAW/agedb/images" -type f 2>/dev/null | wc -l | tr -d ' ')"
