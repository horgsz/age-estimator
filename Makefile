# Serving layer + web UI for the age estimator.
#
#   make setup   one-time install (python venv + npm)
#   make dev     run the API and the web UI together
#   make test    run the server test suite
#   make build   type-check and bundle the web UI

PYTHON      ?= python3
VENV        ?= .venv
VENV_PY     := $(VENV)/bin/python
API_HOST    ?= 127.0.0.1
API_PORT    ?= 8000

.PHONY: setup venv web-deps dev api web test build preview fmt-check clean

setup: venv web-deps

venv: $(VENV_PY)

$(VENV_PY):
	$(PYTHON) -m venv $(VENV)
	$(VENV)/bin/pip install --upgrade pip
	$(VENV)/bin/pip install -r server/requirements.txt

web-deps: web/node_modules

web/node_modules: web/package.json
	cd web && npm install
	@touch web/node_modules

## Run API + Vite together (Ctrl-C stops both).
dev: setup
	./scripts/dev.sh

## API only.
api: venv
	$(VENV_PY) -m uvicorn server.app:app --host $(API_HOST) --port $(API_PORT) --reload

## Web dev server only.
web: web-deps
	cd web && npm run dev

test: venv
	$(VENV_PY) -m pytest

build: web-deps
	cd web && npm run build

preview: build
	cd web && npm run preview

clean:
	rm -rf web/dist web/node_modules
	find server -name '__pycache__' -type d -prune -exec rm -rf {} +
