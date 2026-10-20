"""Download, clean and split the IBM Telco dataset.

The split is made once and persisted so every run and every model version is
compared on exactly the same rows:

    development (50%) -> train / test      fit, gates, decision rule
    holdout     (20%) -> frozen            champion vs challenger only
    production  (30%) -> production_pool   monthly cohorts in the simulation
"""

import hashlib
import json
import ssl
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from churn import config

COLUMN_MAP = {
    "customerID": "customer_id",
    "gender": "gender",
    "SeniorCitizen": "senior_citizen",
    "Partner": "partner",
    "Dependents": "dependents",
    "tenure": "tenure",
    "PhoneService": "phone_service",
    "MultipleLines": "multiple_lines",
    "InternetService": "internet_service",
    "OnlineSecurity": "online_security",
    "OnlineBackup": "online_backup",
    "DeviceProtection": "device_protection",
    "TechSupport": "tech_support",
    "StreamingTV": "streaming_tv",
    "StreamingMovies": "streaming_movies",
    "Contract": "contract",
    "PaperlessBilling": "paperless_billing",
    "PaymentMethod": "payment_method",
    "MonthlyCharges": "monthly_charges",
    "TotalCharges": "total_charges",
    "Churn": "churn",
}


def _ssl_context() -> ssl.SSLContext:
    # python.org builds on macOS ship without a CA bundle; certifi fixes that when available
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


def download_raw() -> None:
    if config.RAW_CSV.exists():
        return
    config.RAW_CSV.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {config.RAW_URL}")
    with urllib.request.urlopen(config.RAW_URL, context=_ssl_context(), timeout=60) as response:
        config.RAW_CSV.write_bytes(response.read())


def clean(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.rename(columns=COLUMN_MAP)[list(COLUMN_MAP.values())].copy()
    df = df.drop_duplicates(subset=[config.ID_COL])
    # TotalCharges is a string with " " for customers with tenure 0; left as NaN for the imputer
    df["total_charges"] = pd.to_numeric(df["total_charges"].str.strip(), errors="coerce")
    df["senior_citizen"] = df["senior_citizen"].map({0: "No", 1: "Yes"})
    df[config.TARGET] = (df[config.TARGET] == "Yes").astype(int)
    return df.reset_index(drop=True)


def load_clean() -> pd.DataFrame:
    download_raw()
    return clean(pd.read_csv(config.RAW_CSV))


def dataframe_hash(df: pd.DataFrame) -> str:
    """Content fingerprint, logged with every run as a lightweight data version."""
    digest = hashlib.sha256(pd.util.hash_pandas_object(df, index=False).values.tobytes())
    return digest.hexdigest()[:12]


def build_splits() -> dict[str, pd.DataFrame]:
    df = load_clean()
    seed = config.RANDOM_STATE

    rest, production = train_test_split(
        df, test_size=config.PRODUCTION_FRACTION, stratify=df[config.TARGET], random_state=seed
    )
    holdout_share = config.HOLDOUT_FRACTION / (1 - config.PRODUCTION_FRACTION)
    dev, holdout = train_test_split(rest, test_size=holdout_share, stratify=rest[config.TARGET], random_state=seed)
    train, test = train_test_split(dev, test_size=config.TEST_FRACTION, stratify=dev[config.TARGET], random_state=seed)

    splits = {"train": train, "test": test, "holdout": holdout, "production_pool": production}
    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, part in splits.items():
        part.to_csv(config.SPLITS_DIR / f"{name}.csv", index=False)
        manifest[name] = {
            "rows": len(part),
            "churn_rate": round(float(part[config.TARGET].mean()), 4),
            "hash": dataframe_hash(part),
        }
    (config.SPLITS_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return splits


def load_split(name: str) -> pd.DataFrame:
    path = config.SPLITS_DIR / f"{name}.csv"
    if not path.exists():
        build_splits()
    return pd.read_csv(path)


def X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    return df[config.FEATURES], df[config.TARGET]
