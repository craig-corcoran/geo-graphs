# Canonical invocations for every routine operation on this repo.
# Run `make help` to discover targets. Docs, CI, and agents call these targets
# rather than the underlying commands, so there is one place a change lands.

UV      ?= uv
PYTEST  ?= $(UV) run pytest
RUFF    ?= $(UV) run ruff
PYRIGHT ?= $(UV) run pyright

PKG      ?= geo_graphs
EXAMPLES ?= examples

TILE_LAT  ?= 36.1699
TILE_LON  ?= -115.1398
TILE_SIZE ?= 1024
TILE_RES  ?= 1.0
TRAIN_ARGS ?=
TESTS   ?= tests
SRC     ?= $(PKG) $(TESTS)

# SpaceNet Roads. The bucket is public over plain HTTPS: no credentials, no
# AWS CLI. Only RGB-PanSharpen is extracted -- it is 24% of an AOI by size, and
# the 8-band products are 64% we have no model for.
DATA_DIR          ?= data
SPACENET_BUCKET   ?= https://spacenet-dataset.s3.amazonaws.com/spacenet/SN3_roads/tarballs
SPACENET_TARBALL  ?= SN3_roads_train_AOI_2_Vegas.tar.gz
SPACENET_PRODUCT  ?= PS-RGB
SPACENET_LABELS   ?= geojson_roads

.DEFAULT_GOAL := help
.PHONY: help sync format lint typecheck test-offline test-network test check clean roundtrip examples train fetch-spacenet extract-spacenet spacenet-usage

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

train:  ## Train the segmentation model and score it per stage (hits the network)
	$(UV) run python -m $(PKG).train $(TRAIN_ARGS)

examples:  ## Execute the walkthrough notebook to prove it still runs (hits the network)
	$(UV) run jupyter execute $(EXAMPLES)/*.ipynb

fetch-spacenet:  ## Download a SpaceNet tarball into DATA_DIR (large; resumable)
	@mkdir -p $(DATA_DIR)
	curl -L -C - --retry 5 --retry-delay 5 \
	  -o $(DATA_DIR)/$(SPACENET_TARBALL) \
	  $(SPACENET_BUCKET)/$(SPACENET_TARBALL)

extract-spacenet:  ## Unpack only SPACENET_PRODUCT and the labels (writes to DATA_DIR)
	tar xzf $(DATA_DIR)/$(SPACENET_TARBALL) -C $(DATA_DIR) \
	  '*/$(SPACENET_PRODUCT)/*' '*/$(SPACENET_LABELS)/*'

spacenet-usage:  ## Report disk used by downloads and extractions
	@du -sh $(DATA_DIR)/* 2>/dev/null || echo "  nothing in $(DATA_DIR)"
	@df -h $(DATA_DIR) | tail -1
