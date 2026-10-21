"""Project settings: paths plus everything in configs/config.yaml."""

import os
from pathlib import Path

import yaml


def _find_root() -> Path:
    if "CHURN_HOME" in os.environ:
        return Path(os.environ["CHURN_HOME"])
    for path in (Path.cwd(), *Path.cwd().parents):
        if (path / "configs" / "config.yaml").exists():
            return path
    return Path.cwd()


ROOT = _find_root()
_cfg = yaml.safe_load((ROOT / "configs" / "config.yaml").read_text())

# Paths
DATA_DIR = ROOT / "data"
RAW_CSV = DATA_DIR / "raw" / "telco_churn.csv"
SPLITS_DIR = DATA_DIR / "splits"
SIM_DIR = DATA_DIR / "simulation"
PORTFOLIO_DIR = SIM_DIR / "portfolio"
LABEL_STORE = SIM_DIR / "label_store.csv"
PREDICTIONS_DIR = DATA_DIR / "predictions"
REPORTS_DIR = ROOT / "reports"
ARTIFACTS_DIR = ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
CHAMPION_DIR = Path(os.getenv("CHAMPION_DIR", ARTIFACTS_DIR / "champion"))
MLRUNS_DIR = ROOT / "mlruns"

TARGET = "churn"
ID_COL = "customer_id"

# Data
RAW_URL = _cfg["data"]["raw_url"]
RANDOM_STATE = _cfg["data"]["random_state"]
HOLDOUT_FRACTION = _cfg["data"]["holdout_fraction"]
PRODUCTION_FRACTION = _cfg["data"]["production_fraction"]
TEST_FRACTION = _cfg["data"]["test_fraction"]

# MLflow (SQLite: the file store is in maintenance mode and the registry needs a DB)
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLRUNS_DIR / 'mlflow.db'}")
EXPERIMENT_NAME = _cfg["mlflow"]["experiment"]
REGISTERED_MODEL_NAME = _cfg["mlflow"]["registered_model"]

# Quality gates
MIN_GINI = _cfg["gates"]["min_gini"]
MIN_RECALL = _cfg["gates"]["min_recall"]
MAX_TRAIN_TEST_GINI_GAP = _cfg["gates"]["max_train_test_gini_gap"]
MAX_ECE = _cfg["gates"]["max_ece"]
MAX_SEGMENT_CALIBRATION_Z = _cfg["gates"]["max_segment_calibration_z"]

CALIBRATION_FOLDS = _cfg["calibration"]["folds"]
PROMOTION_GINI_TOLERANCE = _cfg["promotion"]["gini_tolerance"]
PROMOTION_MIN_WIN_PROBABILITY = _cfg["promotion"]["min_win_probability"]
PROMOTION_BOOTSTRAP_SAMPLES = _cfg["promotion"]["bootstrap_samples"]

# Business
RETENTION_OFFER_COST = _cfg["business"]["retention_offer_cost"]
OFFER_SUCCESS_RATE = _cfg["business"]["offer_success_rate"]
RETAINED_MONTHS = _cfg["business"]["retained_months"]

# Monitoring
LABEL_MATURITY_DAYS = _cfg["monitoring"]["label_maturity_days"]
PSI_WARN = _cfg["monitoring"]["psi_warn"]
PSI_RETRAIN = _cfg["monitoring"]["psi_retrain"]
PSI_BINS = _cfg["monitoring"]["psi_bins"]

# Segments
SEGMENTS = _cfg["segments"]["columns"]
TENURE_BANDS = _cfg["segments"]["tenure_bands"]
TENURE_LABELS = _cfg["segments"]["tenure_labels"]
MIN_SEGMENT_ROWS = _cfg["segments"]["min_rows"]

SHAP_SAMPLE = _cfg["interpretability"]["shap_sample"]
STRESS_STD = _cfg["interpretability"]["stress_std"]

COHORT_SIZE = _cfg["simulation"]["cohort_size"]
SCHEDULE = [tuple(step) for step in _cfg["simulation"]["schedule"]]

NUMERIC_FEATURES = _cfg["features"]["numeric"]
CATEGORICAL_FEATURES = _cfg["features"]["categorical"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
