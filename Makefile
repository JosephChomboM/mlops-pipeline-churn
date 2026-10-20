VENV := .venv
BIN := $(VENV)/bin

.PHONY: setup lint format test simulate init status serve predict mlflow notebooks docker-build docker-up clean

setup:            ## venv + package in editable mode with dev extras
	python3 -m venv $(VENV) && $(BIN)/pip install -q -e ".[dev]"

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

test:
	$(BIN)/pytest

simulate:         ## init + six simulated months, writes reports/
	$(BIN)/churn simulate

init:             ## frozen split + first champion
	$(BIN)/churn init

status:
	$(BIN)/churn status

serve:            ## API on :8000, docs on /docs
	$(BIN)/uvicorn churn.serving.api.app:app --reload

predict:
	curl -s -X POST localhost:8000/predict -H "Content-Type: application/json" -d @tests/fixtures/customer.json

mlflow:           ## MLflow UI on :5000
	$(BIN)/mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db --port 5000

notebooks:        ## re-run every notebook in place (needs `make simulate` first)
	$(BIN)/jupyter nbconvert --to notebook --execute --inplace notebooks/*.ipynb

docker-build:
	docker build -f docker/api.Dockerfile -t churn-api .
	docker build -f docker/pipeline.Dockerfile -t churn-pipeline .

docker-up:
	docker compose up --build api mlflow

clean:
	rm -rf mlruns artifacts data/splits data/simulation data/predictions logs .pytest_cache .ruff_cache
