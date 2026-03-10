"""
explain.py
----------
SHAP-based model explainability for the best churn prediction model.

Produces:
  - SHAP summary bar chart (global feature importance)
  - SHAP beeswarm summary plot
  - SHAP waterfall plot for a high-risk player
  - SHAP force plot (HTML) for top 5 at-risk players
  - Top-10 feature importance table (CSV)
  - Plain-English risk narrative for each player

Usage:
    python src/explain.py
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

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
REPORT_DIR = ROOT / "reports"

# ---------------------------------------------------------------------------
# Feature names (must match preprocessing output)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "session_count", "avg_session_duration", "last_login_days_ago",
    "total_playtime_hours", "win_rate", "total_spend_usd", "friend_count",
    "achievement_count", "consecutive_losses",
    "rfm_recency", "rfm_frequency", "rfm_monetary", "rfm_score",
    "game_mode_primary_PvE", "game_mode_primary_Co-op", "game_mode_primary_Solo",
]

FEATURE_DESCRIPTIONS = {
    "last_login_days_ago": "days since last login",
    "session_count": "total play sessions",
    "consecutive_losses": "consecutive losses",
    "total_spend_usd": "total spend ($)",
    "win_rate": "win rate",
    "rfm_score": "RFM engagement score",
    "rfm_recency": "recency score",
    "rfm_frequency": "frequency score",
    "rfm_monetary": "monetary score (log1p)",
    "engagement_score": "engagement score",
    "loss_streak_risk": "loss-streak risk",
    "rolling_avg_sessions_7d": "sessions (last 7 days)",
    "spend_per_session": "spend per session ($)",
    "achievement_velocity": "achievements per hour",
    "social_engagement_ratio": "social engagement ratio",
}


# ---------------------------------------------------------------------------
# Loading helpers
# ---------------------------------------------------------------------------

def load_model_and_data() -> tuple:
    """
    Load the best model bundle, test features, and original player data.

    Returns
    -------
    (model, threshold, X_test, y_test, feature_names, df_players)
    """
    bundle = joblib.load(MODEL_DIR / "best_model.pkl")
    model = bundle["model"]
    threshold = bundle["threshold"]
    model_name = bundle["name"]
    log.info("Loaded model: %s  (threshold=%.4f)", model_name, threshold)

    X_test = pd.read_csv(DATA_DIR / "X_test.csv", index_col=0)
    y_test = pd.read_csv(DATA_DIR / "y_test.csv", index_col=0).squeeze()
    feature_names = X_test.columns.tolist()

    df_players = pd.read_csv(DATA_DIR / "players_featured.csv") \
        if (DATA_DIR / "players_featured.csv").exists() \
        else pd.read_csv(DATA_DIR / "players.csv")

    return model, threshold, X_test, y_test, feature_names, df_players


# ---------------------------------------------------------------------------
# SHAP computation
# ---------------------------------------------------------------------------

def compute_shap_values(model, X: pd.DataFrame) -> tuple:
    """
    Compute SHAP values for the given model and feature matrix.

    Supports TreeExplainer (tree models) and KernelExplainer (fallback).

    Parameters
    ----------
    model : Fitted sklearn-compatible model.
    X     : Feature DataFrame.

    Returns
    -------
    (explainer, shap_values_array)
    """
    log.info("Computing SHAP values for %d samples …", len(X))
    try:
        explainer = shap.TreeExplainer(model)
        shap_vals = explainer.shap_values(X.values)
        # For binary classifiers, shap_values may be a list [class0, class1] or a 3D array
        if isinstance(shap_vals, list):
            shap_vals = shap_vals[1]
        elif hasattr(shap_vals, 'shape') and len(shap_vals.shape) == 3:
            shap_vals = shap_vals[:, :, 1]
        log.info("Used TreeExplainer.")
    except Exception:
        log.warning("TreeExplainer failed; falling back to KernelExplainer (slower).")
        background = shap.sample(X.values, 100, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, background)
        shap_vals = explainer.shap_values(X.values[:200])[1]
    return explainer, shap_vals


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_shap_bar_summary(shap_values: np.ndarray, X: pd.DataFrame, fig_dir: Path) -> pd.DataFrame:
    """
    Bar chart of mean absolute SHAP values (global feature importance).

    Parameters
    ----------
    shap_values : SHAP value array (n_samples × n_features).
    X           : Feature DataFrame.
    fig_dir     : Output directory.

    Returns
    -------
    DataFrame with feature importance ranking.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mean_abs_shap": mean_abs,
    }).sort_values("mean_abs_shap", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.barh(importance_df["feature"][::-1], importance_df["mean_abs_shap"][::-1],
                   color="steelblue", edgecolor="white")
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title("Top 15 Features by Global SHAP Importance", fontsize=13)
    ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    fig.tight_layout()
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "shap_bar_summary.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved SHAP bar summary → %s", fig_dir / "shap_bar_summary.png")

    top10 = importance_df.head(10)
    top10.to_csv(REPORT_DIR / "top10_features.csv", index=False)
    log.info("Top 10 features saved → %s", REPORT_DIR / "top10_features.csv")
    return top10


def plot_shap_beeswarm(shap_values: np.ndarray, X: pd.DataFrame, fig_dir: Path) -> None:
    """
    Beeswarm summary plot showing SHAP value distribution per feature.

    Parameters
    ----------
    shap_values : SHAP value array.
    X           : Feature DataFrame.
    fig_dir     : Output directory.
    """
    explanation = shap.Explanation(
        values=shap_values,
        base_values=np.zeros(len(shap_values)),
        data=X.values,
        feature_names=X.columns.tolist(),
    )
    fig, ax = plt.subplots(figsize=(10, 8))
    shap.plots.beeswarm(explanation, max_display=15, show=False)
    plt.title("SHAP Beeswarm — Feature Impact on Churn Probability", fontsize=13)
    plt.tight_layout()
    fig.savefig(fig_dir / "shap_beeswarm.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved beeswarm plot → %s", fig_dir / "shap_beeswarm.png")


def plot_shap_waterfall(
    explainer,
    shap_values: np.ndarray,
    X: pd.DataFrame,
    player_idx: int,
    fig_dir: Path,
) -> None:
    """
    Waterfall plot for a single high-risk player.

    Parameters
    ----------
    explainer   : SHAP explainer object.
    shap_values : SHAP value array.
    X           : Feature DataFrame.
    player_idx  : Index of the player to explain.
    fig_dir     : Output directory.
    """
    base_val = (
        explainer.expected_value[1]
        if isinstance(explainer.expected_value, (list, np.ndarray))
        else explainer.expected_value
    )
    explanation = shap.Explanation(
        values=shap_values[player_idx],
        base_values=float(base_val),
        data=X.values[player_idx],
        feature_names=X.columns.tolist(),
    )
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.waterfall(explanation, max_display=12, show=False)
    plt.title(f"SHAP Waterfall — Player {player_idx} (High Churn Risk)", fontsize=13)
    plt.tight_layout()
    fig.savefig(fig_dir / "shap_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Saved waterfall plot → %s", fig_dir / "shap_waterfall.png")


# ---------------------------------------------------------------------------
# Plain-English risk narrative
# ---------------------------------------------------------------------------

def generate_risk_narrative(
    player_row: pd.Series,
    shap_row: np.ndarray,
    feature_names: list[str],
    top_n: int = 3,
) -> str:
    """
    Generate a plain-English explanation of why a player is at churn risk.

    Example output:
        "Player P00042 is at risk because:
          • 12 consecutive losses (pushing toward frustration)
          • No spend in 18 days (declining monetary commitment)
          • Only 2 sessions in the past week (rapidly declining engagement)"

    Parameters
    ----------
    player_row    : Series with original (unscaled) player features.
    shap_row      : SHAP values for this player.
    feature_names : List of feature names (post-transformation).
    top_n         : Number of top contributing factors to include.

    Returns
    -------
    Plain-English risk narrative string.
    """
    # Sort features by absolute SHAP contribution
    idx_sorted = np.argsort(np.abs(shap_row))[::-1][:top_n]
    reasons = []
    for idx in idx_sorted:
        if idx >= len(feature_names):
            continue
        fname = feature_names[idx]
        shap_val = shap_row[idx]
        direction = "increases" if shap_val > 0 else "decreases"

        # Human-readable reason fragments
        if "last_login" in fname:
            days = player_row.get("last_login_days_ago", "?")
            reasons.append(f"Last login {days} days ago ({direction} churn risk)")
        elif "consecutive_losses" in fname:
            losses = player_row.get("consecutive_losses", "?")
            reasons.append(f"{losses} consecutive losses ({direction} churn risk)")
        elif "spend" in fname.lower():
            spend = player_row.get("total_spend_usd", 0)
            reasons.append(f"Total spend ${spend:.0f} ({direction} churn risk)")
        elif "session_count" in fname:
            sc = player_row.get("session_count", "?")
            reasons.append(f"{sc} sessions total ({direction} churn risk)")
        elif "win_rate" in fname:
            wr = player_row.get("win_rate", "?")
            reasons.append(f"Win rate {wr:.0%} ({direction} churn risk)")
        elif "friend_count" in fname:
            fc = player_row.get("friend_count", "?")
            reasons.append(f"{int(fc)} in-game friends ({direction} churn risk)")
        else:
            desc = FEATURE_DESCRIPTIONS.get(fname, fname)
            reasons.append(f"Feature '{desc}' {direction} churn risk")

    narrative = " | ".join(reasons)
    return narrative


def generate_player_risk_table(
    model,
    threshold: float,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    shap_values: np.ndarray,
    df_players: pd.DataFrame,
    top_n: int = 20,
) -> pd.DataFrame:
    """
    Create a ranked table of the top at-risk players with churn probability
    and plain-English risk narratives.

    Parameters
    ----------
    model       : Best fitted model.
    threshold   : Optimal threshold.
    X_test      : Test features DataFrame.
    y_test      : Test labels.
    shap_values : SHAP value array.
    df_players  : Original player dataframe.
    top_n       : Number of top-risk players to include.

    Returns
    -------
    DataFrame with player_id, churn_prob, actual_label, risk_narrative.
    """
    probas = model.predict_proba(X_test.values)[:, 1]
    preds = (probas >= threshold).astype(int)

    risk_df = pd.DataFrame({
        "player_index": X_test.index,
        "churn_prob": probas,
        "predicted_churn": preds,
        "actual_churn": y_test.values,
    }).sort_values("churn_prob", ascending=False).head(top_n)

    narratives = []
    for _, row in risk_df.iterrows():
        # Ensure integer type for index since pandas row iteration may upcast to float
        try:
            p_idx = int(row["player_index"])
        except ValueError:
            p_idx = row["player_index"]
            
        idx_in_shap = X_test.index.get_loc(p_idx)
        # Use the player feature data if available
        if "player_id" in df_players.columns:
            if isinstance(p_idx, int):
                player_raw = df_players.iloc[p_idx % len(df_players)]
            else:
                player_raw = df_players.loc[p_idx]
        else:
            player_raw = X_test.loc[p_idx]
        narrative = generate_risk_narrative(
            player_raw, shap_values[idx_in_shap], X_test.columns.tolist()
        )
        narratives.append(narrative)

    risk_df["risk_narrative"] = narratives
    risk_df.to_csv(REPORT_DIR / "top_risk_players.csv", index=False)
    log.info("Top %d at-risk players saved → %s", top_n, REPORT_DIR / "top_risk_players.csv")
    return risk_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full SHAP explainability pipeline."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model, threshold, X_test, y_test, feature_names, df_players = load_model_and_data()

    # Use a subsample for speed; increase for publication-quality plots
    sample_size = min(2000, len(X_test))
    X_sample = X_test.iloc[:sample_size]

    explainer, shap_values = compute_shap_values(model, X_sample)

    top10 = plot_shap_bar_summary(shap_values, X_sample, FIG_DIR)
    log.info("Top 10 churn predictors:\n%s", top10.to_string(index=False))

    plot_shap_beeswarm(shap_values, X_sample, FIG_DIR)

    # Waterfall: pick the highest-risk player in the sample
    probas_sample = model.predict_proba(X_sample.values)[:, 1]
    highest_risk_idx = int(np.argmax(probas_sample))
    plot_shap_waterfall(explainer, shap_values, X_sample, highest_risk_idx, FIG_DIR)

    # Risk table
    risk_table = generate_player_risk_table(
        model, threshold, X_sample, y_test.iloc[:sample_size], shap_values, df_players
    )
    log.info("Sample risk narratives:\n%s", risk_table[["churn_prob", "risk_narrative"]].head(5).to_string())

    # Save SHAP values for Streamlit
    np.save(MODEL_DIR / "shap_values.npy", shap_values)
    X_sample.to_csv(DATA_DIR / "X_shap_sample.csv")
    log.info("SHAP values saved → %s", MODEL_DIR / "shap_values.npy")
    log.info("Explainability complete.")


if __name__ == "__main__":
    main()
