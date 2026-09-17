# Serving layer + web UI for the age estimator.
#
#   make setup         one-time install (python venv + npm)
#   make dev           run the API and the web UI together
#   make test          run the server test suite
#   make build         type-check and bundle the web UI (server-backed)
#   make build-static  bundle the fully client-side build for GitHub Pages
#   make parity        prove the browser path matches the server path
#   make eval          end-to-end MAE of the deployed inference path

PYTHON      ?= python3
VENV        ?= .venv
VENV_PY     := $(VENV)/bin/python
API_HOST    ?= 127.0.0.1
API_PORT    ?= 8000

.PHONY: setup venv web-deps dev api web test build build-static preview preview-static \
        registry parity parity-deps fixtures eval fmt-check clean

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

## The GitHub Pages build: no server, inference runs in the browser.
##
## VITE_BASE must match where the site is served from. It is a project site, so
## the deployed prefix is /age-estimator/; the default of / is right for a local
## `make preview-static`.
VITE_BASE ?= /

build-static: web-deps registry
	cd web && VITE_BASE=$(VITE_BASE) npm run build:static

preview-static: build-static
	cd web && npm run preview

## Regenerate the static model registry from server/'s own tables.
##
## Committed, because generating it verifies each ONNX export against the
## checkpoint its accuracy figures were measured on, which needs torch.
## server/tests/test_static_registry.py fails if the committed copy drifts.
registry: venv
	$(VENV_PY) -m server.tools.export_static_registry \
		--out web/public/models/models.json

fixtures: venv
	$(VENV_PY) parity/make_fixtures.py

parity-deps:
	cd parity && npm install && npx playwright install chromium

## Run the same images through the server path and the browser path and report
## every difference. See parity/README.md -- and do not widen a tolerance to
## make this pass.
parity: build-static
	$(VENV_PY) parity/run_python.py \
		--cases parity/fixtures/cases.json --out parity/python.json
	node parity/run_browser.mjs --dist web/dist --base $(VITE_BASE) \
		--cases parity/fixtures/cases.json --out parity/browser.json
	$(VENV_PY) parity/compare.py \
		--python parity/python.json --browser parity/browser.json --strict

preview: build
	cd web && npm run preview

## End-to-end MAE of the deployed path (YuNet -> server crop -> model).
##
##   make eval                              # ml/splits/test.csv, active margin
##   make eval EVAL_MARGINS="0 0.0135 0.1"  # sweep margins
##   make eval EVAL_CSV=... EVAL_ARGS="--limit 500"
EVAL_CSV     ?= ml/splits/test.csv
EVAL_MARGINS ?=
EVAL_ARGS    ?=

eval: venv
	$(VENV_PY) -m server.tools.eval_end_to_end --csv $(EVAL_CSV) \
		$(if $(EVAL_MARGINS),--margins $(EVAL_MARGINS),) $(EVAL_ARGS)

clean:
	rm -rf web/dist web/node_modules parity/node_modules
	rm -f parity/python.json parity/browser.json
	find web/public -mindepth 1 ! -name .gitignore ! -name models \
		! -name models.json -prune -exec rm -rf {} +
	find server -name '__pycache__' -type d -prune -exec rm -rf {} +
