"""
preprocessing.py
----------------
Data cleaning, encoding, scaling, RFM feature engineering, and
train / validation / test splitting for the churn prediction pipeline.

Usage (standalone):
    python src/preprocessing.py

Produces:
    data/X_train.csv, data/X_val.csv, data/X_test.csv
    data/y_train.csv, data/y_val.csv, data/y_test.csv
    models/preprocessor.pkl
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

NUMERIC_FEATURES = [
    "session_count",
    "avg_session_duration",
    "last_login_days_ago",
    "total_playtime_hours",
    "win_rate",
    "total_spend_usd",
    "friend_count",
    "achievement_count",
    "consecutive_losses",
]
CATEGORICAL_FEATURES = ["game_mode_primary"]
TARGET = "churn_label"
ID_COL = "player_id"


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_data(path: Path = DATA_DIR / "players.csv") -> pd.DataFrame:
    """
    Load raw player data from CSV.

    Parameters
    ----------
    path : Path to the CSV file.

    Returns
    -------
    pd.DataFrame
    """
    log.info("Loading data from %s …", path)
    df = pd.read_csv(path)
    log.info("Loaded %d rows × %d cols", *df.shape)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate player_id rows, keeping the first occurrence.

    Parameters
    ----------
    df : Raw dataframe.

    Returns
    -------
    Deduplicated dataframe.
    """
    before = len(df)
    df = df.drop_duplicates(subset=[ID_COL], keep="first")
    removed = before - len(df)
    if removed:
        log.warning("Removed %d duplicate rows.", removed)
    return df


def cap_outliers(df: pd.DataFrame, cols: list[str], lower: float = 0.01, upper: float = 0.99) -> pd.DataFrame:
    """
    Winsorise numeric columns at the given quantile boundaries.

    Parameters
    ----------
    df    : Input dataframe.
    cols  : Columns to cap.
    lower : Lower quantile threshold.
    upper : Upper quantile threshold.

    Returns
    -------
    Dataframe with capped values.
    """
    df = df.copy()
    for col in cols:
        lo = df[col].quantile(lower)
        hi = df[col].quantile(upper)
        if lo == hi:
            log.warning("  %s: zero-variance after quantile clipping — skipping.", col)
            continue
        clipped = df[col].clip(lower=lo, upper=hi)
        n_clipped = (df[col] != clipped).sum()
        df[col] = clipped
        log.debug("  %s: clipped %d values to [%.2f, %.2f]", col, n_clipped, lo, hi)
    log.info("Outlier capping applied to %d columns.", len(cols))
    return df


def engineer_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append Recency / Frequency / Monetary (RFM) features.

    - rfm_recency  : inverse of last_login_days_ago (higher = more recent)
    - rfm_frequency: normalised session_count
    - rfm_monetary : total_spend_usd (already in the schema)
    - rfm_score    : composite RFM percentile score (0–100)

    Parameters
    ----------
    df : Dataframe with raw player columns.

    Returns
    -------
    Dataframe with three additional RFM columns.
    """
    df = df.copy()
    max_days = df["last_login_days_ago"].max() + 1
    df["rfm_recency"] = max_days - df["last_login_days_ago"]
    df["rfm_frequency"] = df["session_count"]
    df["rfm_monetary"] = np.log1p(df["total_spend_usd"].fillna(0))

    # Percentile rank each dimension, then average
    for col in ["rfm_recency", "rfm_frequency", "rfm_monetary"]:
        df[f"{col}_pct"] = df[col].rank(pct=True) * 100

    df["rfm_score"] = (
        (df["rfm_recency_pct"] + df["rfm_frequency_pct"] + df["rfm_monetary_pct"]) / 3.0
    ).round(2)

    df.drop(columns=["rfm_recency_pct", "rfm_frequency_pct", "rfm_monetary_pct"], inplace=True)
    log.info("RFM features added: rfm_recency, rfm_frequency, rfm_monetary, rfm_score")
    return df


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Build a scikit-learn ColumnTransformer that:
    - Imputes numeric NaNs with the median.
    - Scales numeric features with StandardScaler.
    - Imputes categorical NaNs with the mode.
    - One-hot encodes categorical features.

    Parameters
    ----------
    numeric_features    : List of numeric column names.
    categorical_features: List of categorical column names.

    Returns
    -------
    Unfitted ColumnTransformer.
    """
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    categorical_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", _SafeOneHotEncoder()),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features),
    ], remainder="drop")
    return preprocessor


from sklearn.base import BaseEstimator, TransformerMixin

class _SafeOneHotEncoder(BaseEstimator, TransformerMixin):
    """
    Minimal One-Hot encoder that handles unseen categories gracefully
    and produces a dense array with named columns.
    """
    from sklearn.preprocessing import OneHotEncoder as _OHE

    def __init__(self):
        self._enc = None

    def fit(self, X, y=None):
        from sklearn.preprocessing import OneHotEncoder
        self._enc = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        self._enc.fit(X)
        self.is_fitted_ = True
        return self

    def transform(self, X):
        return self._enc.transform(X)

    def fit_transform(self, X, y=None):
        self.fit(X)
        return self.transform(X)

    def get_feature_names_out(self, input_features=None):
        return self._enc.get_feature_names_out(input_features)


def split_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str = TARGET,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple:
    """
    Stratified 70 / 15 / 15 train-validation-test split.

    Parameters
    ----------
    df          : Full dataframe after feature engineering.
    feature_cols: Columns to use as features.
    target_col  : Target column name.
    train_frac  : Fraction for training set.
    val_frac    : Fraction for validation set (remainder is test).
    seed        : Random state for reproducibility.

    Returns
    -------
    (X_train, X_val, X_test, y_train, y_val, y_test)
    """
    X = df[feature_cols]
    y = df[target_col]

    test_frac = 1.0 - train_frac - val_frac

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_frac, stratify=y, random_state=seed
    )
    val_relative = val_frac / (train_frac + val_frac)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_relative, stratify=y_train_val, random_state=seed
    )
    log.info(
        "Split → train: %d | val: %d | test: %d | total: %d",
        len(X_train), len(X_val), len(X_test), len(X_train) + len(X_val) + len(X_test),
    )
    log.info(
        "Churn rate → train: %.2f%% | val: %.2f%% | test: %.2f%%",
        y_train.mean() * 100, y_val.mean() * 100, y_test.mean() * 100,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_preprocessing_pipeline(
    input_path: Path = DATA_DIR / "players.csv",
    output_dir: Path = DATA_DIR,
    model_dir: Path = MODEL_DIR,
) -> dict:
    """
    Full preprocessing pipeline: load → clean → RFM → split → fit-transform.

    Parameters
    ----------
    input_path : CSV with raw player data.
    output_dir : Where to save split CSVs.
    model_dir  : Where to save the fitted preprocessor.

    Returns
    -------
    dict with keys: X_train, X_val, X_test, y_train, y_val, y_test,
                    preprocessor, feature_names, rfm_df
    """
    df = load_data(input_path)
    df = remove_duplicates(df)
    df = cap_outliers(df, NUMERIC_FEATURES)
    df = engineer_rfm(df)

    rfm_features = ["rfm_recency", "rfm_frequency", "rfm_monetary", "rfm_score"]
    all_numeric = NUMERIC_FEATURES + rfm_features
    all_features = all_numeric + CATEGORICAL_FEATURES

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df, all_features)

    preprocessor = build_preprocessor(all_numeric, CATEGORICAL_FEATURES)
    X_train_t = preprocessor.fit_transform(X_train)
    X_val_t = preprocessor.transform(X_val)
    X_test_t = preprocessor.transform(X_test)

    # Feature names after transformation
    cat_enc = preprocessor.named_transformers_["cat"]["encode"]
    cat_feature_names = cat_enc.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    feature_names = all_numeric + cat_feature_names

    # Save arrays to disk
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    for name, arr, idx in [
        ("X_train", X_train_t, X_train.index),
        ("X_val", X_val_t, X_val.index),
        ("X_test", X_test_t, X_test.index),
    ]:
        pd.DataFrame(arr, columns=feature_names, index=idx).to_csv(
            output_dir / f"{name}.csv"
        )
    for name, ser in [("y_train", y_train), ("y_val", y_val), ("y_test", y_test)]:
        ser.to_csv(output_dir / f"{name}.csv", header=True)

    preprocessor_path = model_dir / "preprocessor.pkl"
    joblib.dump(preprocessor, preprocessor_path)
    log.info("Preprocessor saved → %s", preprocessor_path)

    # Also save the RFM-enriched full dataframe for downstream use
    df.to_csv(DATA_DIR / "players_enriched.csv", index=False)
    log.info("Enriched dataset saved → %s", DATA_DIR / "players_enriched.csv")

    return {
        "X_train": X_train_t,
        "X_val": X_val_t,
        "X_test": X_test_t,
        "y_train": y_train,
        "y_val": y_val,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "rfm_df": df,
    }


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    result = run_preprocessing_pipeline()
    log.info("Preprocessing complete. Feature count: %d", len(result["feature_names"]))
    log.info("Features: %s", result["feature_names"])
