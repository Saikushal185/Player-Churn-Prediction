"""
train.py
--------
Train, tune, and evaluate four classification models for player churn:
  1. Logistic Regression  (baseline)
  2. Random Forest
  3. XGBoost
  4. LightGBM

Improvements over the base prompt:
  - SMOTE oversampling to handle class imbalance.
  - Optuna hyperparameter optimisation (faster than GridSearchCV).
  - StratifiedKFold (5-fold) cross-validation.
  - Precision-Recall curve with optimal threshold selection.
  - Calibration plot (reliability diagram).
  - All evaluation artefacts saved to reports/.

Usage:
    python src/train.py

Reads:  data/X_train.csv, data/y_train.csv, data/X_val.csv, data/y_val.csv,
        data/X_test.csv, data/y_test.csv
Saves:  models/best_model.pkl
        models/model_comparison.csv
        reports/figures/roc_curves.png
        reports/figures/pr_curves.png
        reports/figures/confusion_matrix.png
        reports/figures/calibration_plot.png
"""

import logging
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibrationDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

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
FIG_DIR = ROOT / "reports" / "figures"

N_OPTUNA_TRIALS = 30   # Increase for better tuning; reduce for speed
OPTUNA_TIMEOUT = 120   # Max seconds per model tuning run (wall-clock guard)
CV_FOLDS = 5


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_splits(data_dir: Path = DATA_DIR) -> tuple:
    """
    Load pre-processed train / validation / test splits.

    Returns
    -------
    (X_train, y_train, X_val, y_val, X_test, y_test) as numpy arrays.
    """
    log.info("Loading data splits …")
    X_train = pd.read_csv(data_dir / "X_train.csv", index_col=0).values
    y_train = pd.read_csv(data_dir / "y_train.csv", index_col=0).squeeze().values
    X_val   = pd.read_csv(data_dir / "X_val.csv",   index_col=0).values
    y_val   = pd.read_csv(data_dir / "y_val.csv",   index_col=0).squeeze().values
    X_test  = pd.read_csv(data_dir / "X_test.csv",  index_col=0).values
    y_test  = pd.read_csv(data_dir / "y_test.csv",  index_col=0).squeeze().values
    log.info("Shapes → train: %s | val: %s | test: %s", X_train.shape, X_val.shape, X_test.shape)
    return X_train, y_train, X_val, y_val, X_test, y_test


# ---------------------------------------------------------------------------
# SMOTE oversampling
# ---------------------------------------------------------------------------

def apply_smote(X: np.ndarray, y: np.ndarray, seed: int = 42) -> tuple:
    """
    Apply SMOTE to balance the minority class in the training set.

    Parameters
    ----------
    X    : Feature matrix.
    y    : Binary labels.
    seed : Random state.

    Returns
    -------
    (X_resampled, y_resampled)
    """
    log.info("Applying SMOTE … class distribution before: %s", dict(zip(*np.unique(y, return_counts=True))))
    sm = SMOTE(random_state=seed)
    X_res, y_res = sm.fit_resample(X, y)
    log.info("After SMOTE: %s", dict(zip(*np.unique(y_res, return_counts=True))))
    return X_res, y_res


# ---------------------------------------------------------------------------
# Model factories and Optuna objectives
# ---------------------------------------------------------------------------

def _objective_rf(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 4, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
        "class_weight": "balanced",
        "n_jobs": -1,
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _objective_xgb(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "scale_pos_weight": trial.suggest_float("scale_pos_weight", 1.0, 5.0),
        "eval_metric": "logloss",
        "use_label_encoder": False,
        "random_state": 42,
        "n_jobs": -1,
    }
    model = XGBClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


def _objective_lgbm(trial, X, y, cv):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "max_depth": trial.suggest_int("max_depth", 3, 12),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 20, 150),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "class_weight": "balanced",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": -1,
    }
    model = LGBMClassifier(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()


# ---------------------------------------------------------------------------
# Optimal threshold selection
# ---------------------------------------------------------------------------

def find_optimal_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """
    Find the classification threshold that maximises the F1-score on the
    validation set.  This is critical for imbalanced churn datasets where
    the default 0.5 threshold is rarely optimal.

    Parameters
    ----------
    y_true  : Ground-truth labels.
    y_proba : Predicted probabilities for class 1.

    Returns
    -------
    Optimal threshold (float).
    """
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores[:-1])
    optimal = float(thresholds[best_idx])
    log.info("Optimal threshold: %.4f  (F1=%.4f)", optimal, f1_scores[best_idx])
    return optimal


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------

def train_all_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = N_OPTUNA_TRIALS,
) -> dict:
    """
    Train and tune all four models.  Returns a dict of fitted models and their
    validation metrics.

    Parameters
    ----------
    X_train  : Training features (post-SMOTE).
    y_train  : Training labels (post-SMOTE).
    X_val    : Validation features (original, no SMOTE).
    y_val    : Validation labels.
    n_trials : Number of Optuna trials per model.

    Returns
    -------
    dict[model_name] = {"model": ..., "val_auc": ..., "threshold": ...}
    """
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=42)
    results = {}

    # ------------------------------------------------------------------
    # 1. Logistic Regression (no tuning needed — fast baseline)
    # ------------------------------------------------------------------
    log.info("Training Logistic Regression …")
    lr = LogisticRegression(
        C=1.0, max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=42
    )
    lr.fit(X_train, y_train)
    lr_proba = lr.predict_proba(X_val)[:, 1]
    lr_auc = roc_auc_score(y_val, lr_proba)
    lr_thresh = find_optimal_threshold(y_val, lr_proba)
    results["LogisticRegression"] = {"model": lr, "val_auc": lr_auc, "threshold": lr_thresh,
                                      "proba_val": lr_proba}
    log.info("Logistic Regression  →  val AUC: %.4f", lr_auc)

    # ------------------------------------------------------------------
    # 2. Random Forest (Optuna)
    # ------------------------------------------------------------------
    log.info("Tuning Random Forest with Optuna (%d trials) …", n_trials)
    rf_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    rf_study.optimize(lambda t: _objective_rf(t, X_train, y_train, cv), n_trials=n_trials, timeout=OPTUNA_TIMEOUT)
    best_rf_params = rf_study.best_params
    best_rf_params.update({"class_weight": "balanced", "n_jobs": -1, "random_state": 42})
    rf = RandomForestClassifier(**best_rf_params)
    rf.fit(X_train, y_train)
    rf_proba = rf.predict_proba(X_val)[:, 1]
    rf_auc = roc_auc_score(y_val, rf_proba)
    rf_thresh = find_optimal_threshold(y_val, rf_proba)
    results["RandomForest"] = {"model": rf, "val_auc": rf_auc, "threshold": rf_thresh,
                                "proba_val": rf_proba}
    log.info("Random Forest  →  val AUC: %.4f | best params: %s", rf_auc, best_rf_params)

    # ------------------------------------------------------------------
    # 3. XGBoost (Optuna)
    # ------------------------------------------------------------------
    log.info("Tuning XGBoost with Optuna (%d trials) …", n_trials)
    xgb_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    xgb_study.optimize(lambda t: _objective_xgb(t, X_train, y_train, cv), n_trials=n_trials, timeout=OPTUNA_TIMEOUT)
    best_xgb_params = xgb_study.best_params
    best_xgb_params.update({"eval_metric": "logloss", "use_label_encoder": False,
                              "random_state": 42, "n_jobs": -1})
    xgb = XGBClassifier(**best_xgb_params)
    xgb.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    xgb_proba = xgb.predict_proba(X_val)[:, 1]
    xgb_auc = roc_auc_score(y_val, xgb_proba)
    xgb_thresh = find_optimal_threshold(y_val, xgb_proba)
    results["XGBoost"] = {"model": xgb, "val_auc": xgb_auc, "threshold": xgb_thresh,
                           "proba_val": xgb_proba}
    log.info("XGBoost  →  val AUC: %.4f | best params: %s", xgb_auc, best_xgb_params)

    # ------------------------------------------------------------------
    # 4. LightGBM (Optuna)
    # ------------------------------------------------------------------
    log.info("Tuning LightGBM with Optuna (%d trials) …", n_trials)
    lgbm_study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    lgbm_study.optimize(lambda t: _objective_lgbm(t, X_train, y_train, cv), n_trials=n_trials, timeout=OPTUNA_TIMEOUT)
    best_lgbm_params = lgbm_study.best_params
    best_lgbm_params.update({"class_weight": "balanced", "random_state": 42,
                               "n_jobs": -1, "verbosity": -1})
    lgbm = LGBMClassifier(**best_lgbm_params)
    lgbm.fit(X_train, y_train)
    lgbm_proba = lgbm.predict_proba(X_val)[:, 1]
    lgbm_auc = roc_auc_score(y_val, lgbm_proba)
    lgbm_thresh = find_optimal_threshold(y_val, lgbm_proba)
    results["LightGBM"] = {"model": lgbm, "val_auc": lgbm_auc, "threshold": lgbm_thresh,
                            "proba_val": lgbm_proba}
    log.info("LightGBM  →  val AUC: %.4f | best params: %s", lgbm_auc, best_lgbm_params)

    return results


def evaluate_on_test(
    model,
    threshold: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
) -> dict:
    """
    Evaluate a fitted model on the held-out test set.

    Parameters
    ----------
    model      : Fitted sklearn-compatible classifier.
    threshold  : Decision threshold from validation set.
    X_test     : Test features.
    y_test     : Test labels.
    model_name : Name string for logging.

    Returns
    -------
    dict with AUC, Average Precision, F1, Precision, Recall metrics.
    """
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)

    test_auc = roc_auc_score(y_test, proba)
    test_ap = average_precision_score(y_test, proba)
    report = classification_report(y_test, preds, output_dict=True)

    metrics = {
        "model": model_name,
        "test_auc": round(test_auc, 4),
        "avg_precision": round(test_ap, 4),
        "f1_churn": round(report["1"]["f1-score"], 4),
        "precision_churn": round(report["1"]["precision"], 4),
        "recall_churn": round(report["1"]["recall"], 4),
        "f1_macro": round(report["macro avg"]["f1-score"], 4),
        "threshold": round(threshold, 4),
    }
    log.info(
        "%s  →  AUC: %.4f | F1: %.4f | Recall: %.4f | Precision: %.4f",
        model_name, test_auc, metrics["f1_churn"],
        metrics["recall_churn"], metrics["precision_churn"],
    )
    log.info("\n%s", classification_report(y_test, preds, target_names=["Active", "Churned"]))
    return metrics


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_roc_curves(results: dict, y_val: np.ndarray, fig_dir: Path = FIG_DIR) -> None:
    """
    Plot ROC curves for all models on the validation set.

    Parameters
    ----------
    results : Output of train_all_models().
    y_val   : Validation labels.
    fig_dir : Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "forestgreen", "tomato", "darkorange"]
    for (name, res), color in zip(results.items(), colors):
        fpr, tpr, _ = roc_curve(y_val, res["proba_val"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — Model Comparison")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved ROC curves → %s", fig_dir / "roc_curves.png")


def plot_pr_curves(results: dict, y_val: np.ndarray, fig_dir: Path = FIG_DIR) -> None:
    """
    Plot Precision-Recall curves with optimal thresholds marked.

    Parameters
    ----------
    results : Output of train_all_models().
    y_val   : Validation labels.
    fig_dir : Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "forestgreen", "tomato", "darkorange"]
    for (name, res), color in zip(results.items(), colors):
        prec, rec, threshs = precision_recall_curve(y_val, res["proba_val"])
        ap = average_precision_score(y_val, res["proba_val"])
        ax.plot(rec, prec, color=color, lw=2, label=f"{name} (AP={ap:.3f})")
        # Mark optimal threshold point
        f1s = 2 * prec * rec / (prec + rec + 1e-8)
        best_idx = np.argmax(f1s[:-1])
        ax.scatter(rec[best_idx], prec[best_idx], color=color, s=80, zorder=5)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves — Model Comparison")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    fig.savefig(fig_dir / "pr_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved PR curves → %s", fig_dir / "pr_curves.png")


def plot_confusion_matrix(
    model,
    threshold: float,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str,
    fig_dir: Path = FIG_DIR,
) -> None:
    """
    Plot normalised confusion matrix for the best model on the test set.

    Parameters
    ----------
    model      : Best fitted model.
    threshold  : Optimal decision threshold.
    X_test     : Test features.
    y_test     : Test labels.
    model_name : Name string for the plot title.
    fig_dir    : Directory to save the figure.
    """
    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= threshold).astype(int)
    cm = confusion_matrix(y_test, preds, normalize="true")

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=["Active", "Churned"])
    disp.plot(ax=ax, colorbar=True, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name} (threshold={threshold:.2f})")
    fig.savefig(fig_dir / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved confusion matrix → %s", fig_dir / "confusion_matrix.png")


def plot_calibration(
    results: dict,
    y_val: np.ndarray,
    X_val: np.ndarray,
    fig_dir: Path = FIG_DIR,
) -> None:
    """
    Plot reliability (calibration) diagram for all models.
    Well-calibrated models hug the diagonal; useful for trust in probability scores.

    Parameters
    ----------
    results : Output of train_all_models().
    y_val   : Validation labels.
    X_val   : Validation features.
    fig_dir : Directory to save the figure.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["steelblue", "forestgreen", "tomato", "darkorange"]
    for (name, res), color in zip(results.items(), colors):
        CalibrationDisplay.from_predictions(
            y_val, res["proba_val"], n_bins=10, ax=ax,
            name=name, color=color,
        )
    ax.set_title("Calibration Plot (Reliability Diagram)")
    ax.legend(loc="upper left")
    fig.savefig(fig_dir / "calibration_plot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved calibration plot → %s", fig_dir / "calibration_plot.png")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    """Full training pipeline: load → SMOTE → train → evaluate → save."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    X_train, y_train, X_val, y_val, X_test, y_test = load_splits()

    # Apply SMOTE only to training data
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)

    results = train_all_models(X_train_sm, y_train_sm, X_val, y_val)

    # Pick best model by val AUC
    best_name = max(results, key=lambda k: results[k]["val_auc"])
    best = results[best_name]
    log.info("Best model: %s  (val AUC=%.4f)", best_name, best["val_auc"])

    # Full test evaluation
    all_metrics = []
    for name, res in results.items():
        m = evaluate_on_test(res["model"], res["threshold"], X_test, y_test, name)
        all_metrics.append(m)

    comparison_df = pd.DataFrame(all_metrics).sort_values("test_auc", ascending=False)
    comparison_df.to_csv(MODEL_DIR / "model_comparison.csv", index=False)
    log.info("Model comparison:\n%s", comparison_df.to_string(index=False))

    # Save best model + threshold
    joblib.dump(
        {"model": best["model"], "threshold": best["threshold"], "name": best_name},
        MODEL_DIR / "best_model.pkl",
    )
    log.info("Best model saved → %s", MODEL_DIR / "best_model.pkl")

    # Save all models for later use in the dashboard
    for name, res in results.items():
        joblib.dump(res["model"], MODEL_DIR / f"{name.lower()}_model.pkl")

    # Plots
    plot_roc_curves(results, y_val)
    plot_pr_curves(results, y_val)
    plot_confusion_matrix(best["model"], best["threshold"], X_test, y_test, best_name)
    plot_calibration(results, y_val, X_val)

    log.info("Training complete.")


if __name__ == "__main__":
    main()
