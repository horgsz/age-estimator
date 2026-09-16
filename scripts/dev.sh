#!/usr/bin/env bash
#
# Run the API and the web dev server together.
#
#   ./scripts/dev.sh
#
# Ctrl-C stops both. Output from each is prefixed so the interleaved logs stay
# readable.
#
# Written for the bash 3.2 that ships with macOS: no `wait -n`, no `sed -u`.
# `set -m` puts each background pipeline in its own process group, so killing
# the negated pid takes down the whole pipeline (uvicorn/vite *and* the awk
# prefixer) rather than orphaning the real server.

set -euo pipefail
set -m

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="${VENV:-$ROOT/.venv}"
PYTHON="$VENV/bin/python"
API_HOST="${API_HOST:-127.0.0.1}"
API_PORT="${API_PORT:-8000}"
WEB_HOST="${WEB_HOST:-127.0.0.1}"
WEB_PORT="${WEB_PORT:-5173}"

if [ ! -x "$PYTHON" ]; then
  echo "error: no virtualenv at $VENV" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r server/requirements.txt" >&2
  exit 1
fi

if [ ! -d "$ROOT/web/node_modules" ]; then
  echo "==> installing web dependencies"
  (cd web && npm install)
fi

GROUPS_TO_KILL=""

cleanup() {
  trap - INT TERM EXIT
  for pgid in $GROUPS_TO_KILL; do
    kill -TERM "-$pgid" 2>/dev/null || true
  done
  wait 2>/dev/null || true
}
trap cleanup INT TERM EXIT

prefix() { awk -v tag="$1" '{ print tag " " $0; fflush() }'; }

echo "==> api  http://$API_HOST:$API_PORT"
"$PYTHON" -m uvicorn server.app:app \
  --host "$API_HOST" --port "$API_PORT" --reload 2>&1 | prefix "[api]" &
api_pid=$!
GROUPS_TO_KILL="$GROUPS_TO_KILL $api_pid"

echo "==> web  http://$WEB_HOST:$WEB_PORT"
(cd web && npm run dev -- --host "$WEB_HOST" --port "$WEB_PORT") 2>&1 | prefix "[web]" &
web_pid=$!
GROUPS_TO_KILL="$GROUPS_TO_KILL $web_pid"

# Poll instead of `wait -n`, which bash 3.2 does not have.
while kill -0 "$api_pid" 2>/dev/null && kill -0 "$web_pid" 2>/dev/null; do
  sleep 1
done

echo "==> a process exited; shutting down"
