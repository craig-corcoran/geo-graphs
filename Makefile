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
COVERAGE_OUT ?= outputs
SWEEP_ARGS ?=
CENSUS_ARGS ?=
ORACLE_ARGS ?=
CROSSCHECK_ARGS ?=
SHOWCASE_ARGS ?=
EXPLAINER_ARGS ?=
SCRIPTS    ?= scripts
TESTS   ?= tests
SRC     ?= $(PKG) $(TESTS)

# SpaceNet Roads. The bucket is public over plain HTTPS: no credentials, no
# AWS CLI. Only RGB-PanSharpen is extracted -- it is 24% of an AOI by size, and
# the 8-band products are 64% we have no model for.
DATA_DIR          ?= data
OSM_EXTRACT       ?= $(DATA_DIR)/nevada-latest.osm.pbf
GEOFABRIK         ?= https://download.geofabrik.de/north-america/us
SPACENET_BUCKET   ?= https://spacenet-dataset.s3.amazonaws.com/spacenet/SN3_roads/tarballs
SPACENET_TARBALL  ?= SN3_roads_train_AOI_2_Vegas.tar.gz
SPACENET_PRODUCT  ?= PS-RGB
SPACENET_LABELS   ?= geojson_roads

.DEFAULT_GOAL := help
.PHONY: help sync format lint typecheck test-offline test-network test check clean roundtrip examples train coverage-endpoints threshold-sweep census link-oracle osm-crosscheck fetch-osm-extract showcase apls-explainer fetch-spacenet extract-spacenet spacenet-usage

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

coverage-endpoints:  ## Train the same fused config at full and at zero lidar coverage (hits the network; writes COVERAGE_OUT)
	$(UV) run python -m $(PKG).train --lidar --coverage full \
	  --out $(COVERAGE_OUT)/coverage_full.json $(TRAIN_ARGS)
	$(UV) run python -m $(PKG).train --lidar --coverage none \
	  --out $(COVERAGE_OUT)/coverage_none.json $(TRAIN_ARGS)

threshold-sweep:  ## Sweep predict_mask's threshold against APLS on a frozen checkpoint (no training)
	$(UV) run python $(SCRIPTS)/threshold_sweep.py $(SWEEP_ARGS)

census:  ## Count dead-end endpoints and gap-closing candidates on a frozen checkpoint (writes outputs/endpoint_census.json)
	$(UV) run python $(SCRIPTS)/endpoint_census.py $(CENSUS_ARGS)

link-oracle:  ## Score the oracle gap-closer on the reporting holdout (no training; writes outputs/link_oracle.json)
	$(UV) run python $(SCRIPTS)/link_oracle.py $(ORACLE_ARGS)

osm-crosscheck:  ## Cross-check SpaceNet labels and stub purity against OSM (reads OSM_EXTRACT; writes outputs/osm_crosscheck.json)
	$(UV) run python $(SCRIPTS)/osm_crosscheck.py \
	  --way-source pbf --extract $(OSM_EXTRACT) $(CROSSCHECK_ARGS)

fetch-osm-extract:  ## Download the Geofabrik OSM extract into DATA_DIR and verify its MD5 (large; resumable)
	@mkdir -p $(DATA_DIR)
	curl -L -C - --retry 5 -o $(OSM_EXTRACT) $(GEOFABRIK)/$(notdir $(OSM_EXTRACT))
	curl -L --retry 5 -o $(OSM_EXTRACT).md5 $(GEOFABRIK)/$(notdir $(OSM_EXTRACT)).md5
	cd $(dir $(OSM_EXTRACT)) && md5 -q $(notdir $(OSM_EXTRACT)) \
	  | diff - <(cut -d' ' -f1 $(notdir $(OSM_EXTRACT)).md5)

examples:  ## Execute the walkthrough notebook to prove it still runs (hits the network)
	$(UV) run jupyter execute $(EXAMPLES)/*.ipynb

showcase:  ## Rebuild the showcase page from the checkpoint (writes outputs/showcase.html)
	$(UV) run python $(SCRIPTS)/build_site_data.py $(SHOWCASE_ARGS)

apls-explainer:  ## Rebuild the APLS explainer page (writes outputs/apls_explainer.html)
	$(UV) run python $(SCRIPTS)/build_apls_explainer.py $(EXPLAINER_ARGS)

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
