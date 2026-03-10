"""
features.py
-----------
Advanced feature engineering for the Player Churn Prediction pipeline.

Each feature is accompanied by a business justification explaining WHY it
matters for predicting churn.

Usage (standalone):
    python src/features.py

Reads:  data/players_enriched.csv   (output of preprocessing.py)
Writes: data/players_featured.csv
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

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
# Constants
# ---------------------------------------------------------------------------
MAX_RECENCY_DAYS = 180       # cap for recency normalisation
MAX_SESSIONS = 500           # cap for session frequency normalisation
MAX_SESSION_DURATION = 240   # cap for session duration normalisation (minutes)
MAX_FRIENDS = 50             # cap for friend count normalisation
LOSS_STREAK_THRESHOLD = 4    # consecutive losses above which churn risk spikes
LOSS_STREAK_SCALE = 2.0      # sigmoid sharpness for loss streak risk

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"

# ---------------------------------------------------------------------------
# Feature engineering functions
# ---------------------------------------------------------------------------


def compute_spend_per_session(df: pd.DataFrame) -> pd.Series:
    """
    Average USD spent per play session.

    Business justification
    ----------------------
    Players who spend money per session are financially committed — monetised
    players are far less likely to churn because they have a sunk-cost anchor.
    A declining spend_per_session flags disengagement before churn occurs.
    """
    return (df["total_spend_usd"].fillna(0) / df["session_count"].clip(lower=1)).round(4)


def compute_rolling_avg_sessions_7d(df: pd.DataFrame) -> pd.Series:
    """
    Estimated sessions over the past 7 days (approximated from totals).

    Business justification
    ----------------------
    Short-term session frequency is the strongest leading indicator of
    imminent churn. A player who was highly active but has dropped to near
    zero sessions in the last week is in the 'at-risk' window.

    Approximation:
        session_count / max(last_login_days_ago, 7) * 7
    (Real projects use time-series data; this is a realistic approximation.)
    """
    recency_days = df["last_login_days_ago"].clip(lower=1)
    daily_rate = df["session_count"] / recency_days.clip(lower=7)
    return (daily_rate * 7).round(2)


def compute_win_rate_trend(df: pd.DataFrame) -> pd.Series:
    """
    Synthetic win-rate trend signal (positive = improving, negative = declining).

    Business justification
    ----------------------
    A player on a losing streak is much more likely to quit. We encode
    'losing momentum' by combining win_rate with consecutive_losses.
    Negative values signal compounding frustration.

    Formula: win_rate - (consecutive_losses / 15) * 0.5
    """
    consecutive_norm = df["consecutive_losses"] / 15.0
    return (df["win_rate"].fillna(0.5) - consecutive_norm * 0.5).round(4)


def compute_engagement_score(df: pd.DataFrame) -> pd.Series:
    """
    Composite 0–100 engagement score derived from four behavioural signals.

    Business justification
    ----------------------
    No single metric fully captures engagement. This weighted composite gives
    a single actionable number that encodes:
      - How recently the player played (40 % weight)
      - How many sessions they've had (25 % weight)
      - How long each session is (20 % weight)
      - Social activity via friend count (15 % weight)

    Scores below 30 → high-churn risk cohort for targeted re-engagement.
    """
    # Each component normalised to [0, 1]
    max_days = 180
    recency_score = 1.0 - (df["last_login_days_ago"].clip(0, max_days) / max_days)
    freq_score = (df["session_count"].clip(0, 500) / 500)
    duration_score = (df["avg_session_duration"].fillna(45).clip(5, 240) / 240)
    social_score = (df["friend_count"].fillna(0).clip(0, 50) / 50)

    composite = (
        0.40 * recency_score
        + 0.25 * freq_score
        + 0.20 * duration_score
        + 0.15 * social_score
    ) * 100
    return composite.round(2)


def compute_loss_streak_risk(df: pd.DataFrame) -> pd.Series:
    """
    Binary/probabilistic risk flag for loss-streak-induced churn.

    Business justification
    ----------------------
    Research ('Tilting' phenomenon in gaming) shows that players who suffer
    5+ consecutive losses are 3× more likely to quit within 7 days.
    This feature creates a smooth risk score that feeds directly into the model
    and also drives the 'At-Risk' player segment.

    Formula: sigmoid((consecutive_losses - 4) / 2)
    """
    x = (df["consecutive_losses"] - 4) / 2.0
    return (1 / (1 + np.exp(-x))).round(4)


def compute_monetisation_tier(df: pd.DataFrame) -> pd.Series:
    """
    Categorical spend tier: 'Whale', 'Dolphin', 'Minnow', 'Free'.

    Business justification
    ----------------------
    Monetisation tier determines the financial impact of each churn event and
    should drive differentiated retention budgets (e.g. VIP support for Whales).
    Encoded as ordinal integer for modelling (Free=0 … Whale=3).
    """
    spend = df["total_spend_usd"].fillna(0)
    tier = pd.cut(
        spend,
        bins=[-0.01, 0, 20, 100, np.inf],
        labels=[0, 1, 2, 3],   # Free, Minnow, Dolphin, Whale
    ).astype(int)
    return tier


def compute_achievement_velocity(df: pd.DataFrame) -> pd.Series:
    """
    Achievements unlocked per hour of playtime.

    Business justification
    ----------------------
    Achievement velocity measures 'mastery progression'. Players who are
    progressing through game content at a healthy rate are invested in the
    game loop and are unlikely to churn. A sharp drop in velocity signals
    content exhaustion or frustration.
    """
    playtime = df["total_playtime_hours"].clip(lower=0.5)
    return (df["achievement_count"] / playtime).round(4)


def compute_social_engagement_ratio(df: pd.DataFrame) -> pd.Series:
    """
    Ratio of friend count to session count (social density).

    Business justification
    ----------------------
    Social connections are one of the most powerful retention hooks.
    Players with high social-to-session ratios are embedded in a community,
    making churn socially costly. Used to identify 'community leaders'
    who can be leveraged for referral programmes.
    """
    return (df["friend_count"].fillna(0) / df["session_count"].clip(lower=1)).round(4)


def compute_spend_recency_interaction(df: pd.DataFrame) -> pd.Series:
    """
    Interaction term: total_spend_usd × (1 / (1 + last_login_days_ago)).

    Business justification
    ----------------------
    A player who spent recently signals active monetary commitment. This
    interaction term captures BOTH spending level AND how recent that spend
    was — a high-spending player who hasn't logged in for 25 days is far
    more valuable to re-engage than a free-player with similar recency.
    """
    recency_decay = 1.0 / (1.0 + df["last_login_days_ago"])
    return (df["total_spend_usd"].fillna(0) * recency_decay).round(4)


# ---------------------------------------------------------------------------
# Main engineering pipeline
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all feature engineering transformations to the enriched dataframe.

    Parameters
    ----------
    df : Dataframe produced by preprocessing.engineer_rfm()

    Returns
    -------
    Dataframe with all original columns plus new engineered features.
    """
    df = df.copy()

    log.info("Engineering features …")
    df["spend_per_session"] = compute_spend_per_session(df)
    df["rolling_avg_sessions_7d"] = compute_rolling_avg_sessions_7d(df)
    df["win_rate_trend"] = compute_win_rate_trend(df)
    df["engagement_score"] = compute_engagement_score(df)
    df["loss_streak_risk"] = compute_loss_streak_risk(df)
    df["monetisation_tier"] = compute_monetisation_tier(df)
    df["achievement_velocity"] = compute_achievement_velocity(df)
    df["social_engagement_ratio"] = compute_social_engagement_ratio(df)
    df["spend_recency_interaction"] = compute_spend_recency_interaction(df)

    new_features = [
        "spend_per_session",
        "rolling_avg_sessions_7d",
        "win_rate_trend",
        "engagement_score",
        "loss_streak_risk",
        "monetisation_tier",
        "achievement_velocity",
        "social_engagement_ratio",
        "spend_recency_interaction",
    ]
    log.info("Added %d engineered features: %s", len(new_features), new_features)
    return df


def summarise_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Print a summary statistics table for the engineered features.

    Parameters
    ----------
    df : Dataframe after engineer_features()

    Returns
    -------
    Summary DataFrame (also logged).
    """
    eng_cols = [
        "spend_per_session", "rolling_avg_sessions_7d", "win_rate_trend",
        "engagement_score", "loss_streak_risk", "monetisation_tier",
        "achievement_velocity", "social_engagement_ratio", "spend_recency_interaction",
    ]
    summary = df[eng_cols].describe().T
    log.info("Engineered feature summary:\n%s", summary.to_string())
    return summary


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

def main() -> None:
    """Load enriched data, engineer features, and save result."""
    input_path = DATA_DIR / "players_enriched.csv"
    if not input_path.exists():
        log.error(
            "%s not found. Run src/preprocessing.py first.", input_path
        )
        raise FileNotFoundError(input_path)

    df = pd.read_csv(input_path)
    df = engineer_features(df)
    summarise_features(df)

    output_path = DATA_DIR / "players_featured.csv"
    df.to_csv(output_path, index=False)
    log.info("Feature dataset saved → %s  (%d rows × %d cols)", output_path, *df.shape)


if __name__ == "__main__":
    main()
