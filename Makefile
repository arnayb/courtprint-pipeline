# CourtPrint: common commands. Run from the repo root.
export PYTHONPATH := $(CURDIR)
export LAKE_ROOT ?= $(CURDIR)/data/lake
export WAREHOUSE_PATH ?= $(CURDIR)/data/warehouse.duckdb
export DAGSTER_HOME ?= $(CURDIR)/.dagster
DBT = cd warehouse && dbt
DAGSTER = dagster asset materialize -m orchestration.definitions --select '*'

.PHONY: install demo pipeline extract dbt ml dashboard ui test clean

install:            ## install Python dependencies
	pip install -r requirements.txt

$(DAGSTER_HOME):
	mkdir -p $(DAGSTER_HOME)

warehouse/target/manifest.json:
	$(DBT) parse --profiles-dir .

demo: $(DAGSTER_HOME) warehouse/target/manifest.json   ## full pipeline on synthetic data (offline)
	COURTPRINT_SOURCE=demo $(DAGSTER)

pipeline: $(DAGSTER_HOME) warehouse/target/manifest.json  ## full pipeline on real nba_api data
	COURTPRINT_SOURCE=nba_api $(DAGSTER)

extract:            ## just land real data in bronze (no Dagster)
	python -m ingestion.extract_nba

dbt:                ## just rebuild silver + gold and run data tests
	$(DBT) build --profiles-dir .

ml:                 ## just refit the model and refresh the serving layer
	python -m ml.archetypes

dashboard:          ## launch the Streamlit dashboard
	streamlit run dashboard/app.py

ui: $(DAGSTER_HOME)  ## launch the Dagster UI (lineage, runs, schedules)
	dagster dev -m orchestration.definitions

test:               ## unit tests
	pytest -q

clean:              ## wipe lake, warehouse, artifacts (keeps code)
	rm -rf data/lake data/warehouse.duckdb* artifacts warehouse/target warehouse/logs
