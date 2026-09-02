# Canonical invocations for every routine operation on this repo.
# Run `make help` to discover targets. Docs, CI, and agents call these targets
# rather than the underlying commands, so there is one place a change lands.

UV      ?= uv
PYTEST  ?= $(UV) run pytest
RUFF    ?= $(UV) run ruff
PYRIGHT ?= $(UV) run pyright

PKG     ?= geo_graphs

TILE_LAT  ?= 36.1699
TILE_LON  ?= -115.1398
TILE_SIZE ?= 1024
TILE_RES  ?= 1.0
TESTS   ?= tests
SRC     ?= $(PKG) $(TESTS)

.DEFAULT_GOAL := help
.PHONY: help sync format lint typecheck test-offline test-network test check clean roundtrip

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | sort \
	  | awk -F':.*?## ' '{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

sync:  ## Install all dependency groups into .venv
	$(UV) sync --all-groups

format:  ## Rewrite source in place: ruff format + safe lint fixes
	$(RUFF) format $(SRC)
	$(RUFF) check --fix $(SRC)

lint:  ## Check formatting and lint rules without writing
	$(RUFF) format --check $(SRC)
	$(RUFF) check $(SRC)

typecheck:  ## Static type check the package
	$(PYRIGHT) $(PKG)

test-offline:  ## Tests that need no network (the pyproject default)
	$(PYTEST)

test-network:  ## Only the tests that reach external services
	$(PYTEST) -m network

test:  ## Full suite, offline and network alike
	$(PYTEST) -m ""

check: lint typecheck test-offline  ## Fast pre-commit loop: lint + typecheck + offline tests

clean:  ## Delete caches and build products (destructive; touches no source)
	rm -rf .pytest_cache .ruff_cache build dist ./*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

roundtrip:  ## Score the OSM->mask->graph round trip on one tile (hits the network)
	$(UV) run python -m $(PKG).roundtrip \
	  --lat $(TILE_LAT) --lon $(TILE_LON) --size $(TILE_SIZE) \
	  --resolution $(TILE_RES) $(if $(OUT),--out $(OUT),)
