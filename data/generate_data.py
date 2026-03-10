"""
generate_data.py
----------------
Generates a realistic synthetic dataset of 50,000 player records for the
Player Churn Prediction project. Distributions are calibrated to mimic
real free-to-play game behaviour (power-law spending, bimodal login recency, etc.)

Run:
    python data/generate_data.py
Output:
    data/players.csv
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path

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
N_PLAYERS = 50_000
RANDOM_SEED = 42
OUTPUT_PATH = Path(__file__).parent / "players.csv"
CHURN_THRESHOLD_DAYS = 30   # players inactive > 30 days are "churned"


def _clip(arr: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip(arr, lo, hi)


def generate_players(n: int = N_PLAYERS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate a synthetic player dataset with realistic correlations.

    Behavioural assumptions
    -----------------------
    - ~35 % of players are churned (inactive > 30 days).
    - Spending follows a heavy-tailed distribution: most players spend $0,
      a small 'whale' segment (~5 %) spends heavily.
    - Session frequency and playtime are positively correlated.
    - Win-rate is roughly Normal(0.50, 0.12) — most players near 50 %.
    - Consecutive losses correlate negatively with win-rate and positively
      with churn probability.

    Parameters
    ----------
    n    : Number of player records to generate.
    seed : NumPy random seed for reproducibility.

    Returns
    -------
    pd.DataFrame with columns matching the project schema.
    """
    rng = np.random.default_rng(seed)

    log.info("Generating %d synthetic player records …", n)

    # -----------------------------------------------------------------------
    # 1. player_id
    # -----------------------------------------------------------------------
    player_ids = [f"P{str(i).zfill(6)}" for i in range(1, n + 1)]

    # -----------------------------------------------------------------------
    # 2. Churn flag first (drives correlated features below)
    # -----------------------------------------------------------------------
    churn_prob_base = 0.35
    churn_label = rng.binomial(1, churn_prob_base, n)

    # -----------------------------------------------------------------------
    # 3. last_login_days_ago  (bimodal: active ↔ churned)
    # -----------------------------------------------------------------------
    active_days = rng.integers(0, 30, n)            # 0–29 days
    churned_days = rng.integers(31, 180, n)         # 31–180 days
    last_login_days_ago = np.where(churn_label == 1, churned_days, active_days)

    # -----------------------------------------------------------------------
    # 4. session_count  (correlated with activity; churned players have fewer)
    # -----------------------------------------------------------------------
    base_sessions = rng.exponential(scale=40, size=n)       # right-skewed
    session_count = np.where(
        churn_label == 1,
        base_sessions * rng.uniform(0.3, 0.7, n),           # churned: fewer
        base_sessions * rng.uniform(0.8, 1.5, n),           # active: more
    )
    session_count = _clip(session_count.astype(int), 1, 500)

    # -----------------------------------------------------------------------
    # 5. avg_session_duration (minutes)
    # -----------------------------------------------------------------------
    avg_session_duration = rng.normal(loc=45, scale=20, size=n)
    avg_session_duration = _clip(avg_session_duration, 5, 240).round(1)

    # -----------------------------------------------------------------------
    # 6. total_playtime_hours  (derived from sessions × duration)
    # -----------------------------------------------------------------------
    noise = rng.normal(1.0, 0.15, n)
    total_playtime_hours = (session_count * avg_session_duration / 60.0) * noise
    total_playtime_hours = _clip(total_playtime_hours, 0.5, 5000).round(2)

    # -----------------------------------------------------------------------
    # 7. win_rate  (churned players slightly lower)
    # -----------------------------------------------------------------------
    win_rate = rng.normal(loc=0.50, scale=0.12, size=n)
    win_rate = np.where(churn_label == 1, win_rate - rng.uniform(0, 0.08, n), win_rate)
    win_rate = _clip(win_rate, 0.01, 0.99).round(3)

    # -----------------------------------------------------------------------
    # 8. total_spend_usd  (heavy-tailed; ~40 % zero-spenders)
    # -----------------------------------------------------------------------
    spend_zero_mask = rng.random(n) < 0.40
    spend_whale_mask = rng.random(n) < 0.05
    spend_normal = rng.exponential(scale=25, size=n)
    spend_whale = rng.uniform(200, 2000, size=n)
    total_spend_usd = np.where(spend_zero_mask, 0.0,
                      np.where(spend_whale_mask, spend_whale, spend_normal))
    # Churned players tend to have spent less recently
    total_spend_usd = np.where(churn_label == 1, total_spend_usd * 0.6, total_spend_usd)
    total_spend_usd = _clip(total_spend_usd, 0, 5000).round(2)

    # -----------------------------------------------------------------------
    # 9. friend_count  (social players churn less)
    # -----------------------------------------------------------------------
    friend_count_base = rng.negative_binomial(n=3, p=0.3, size=n)
    friend_count = np.where(churn_label == 1,
                            friend_count_base // 2,
                            friend_count_base)
    friend_count = _clip(friend_count, 0, 200).astype(int)

    # -----------------------------------------------------------------------
    # 10. game_mode_primary
    # -----------------------------------------------------------------------
    modes = ["PvP", "PvE", "Co-op", "Solo"]
    mode_probs = [0.40, 0.25, 0.20, 0.15]
    game_mode_primary = rng.choice(modes, size=n, p=mode_probs)

    # -----------------------------------------------------------------------
    # 11. achievement_count  (correlated with total playtime)
    # -----------------------------------------------------------------------
    achievement_count = (total_playtime_hours * rng.uniform(0.3, 0.8, n)).astype(int)
    achievement_count = _clip(achievement_count, 0, 500)

    # -----------------------------------------------------------------------
    # 12. consecutive_losses  (correlated with churn & low win-rate)
    # -----------------------------------------------------------------------
    base_losses = rng.integers(0, 6, n)
    extra_losses = rng.integers(0, 5, n)
    consecutive_losses = np.where(churn_label == 1, base_losses + extra_losses, base_losses)
    consecutive_losses = _clip(consecutive_losses, 0, 15)

    # -----------------------------------------------------------------------
    # 13. Re-derive churn_label from last_login_days_ago (ground truth)
    #     This ensures the label is ALWAYS consistent with the feature.
    # -----------------------------------------------------------------------
    churn_label = (last_login_days_ago > CHURN_THRESHOLD_DAYS).astype(int)

    # -----------------------------------------------------------------------
    # 14. Introduce ~2 % realistic nulls (simulates data quality issues)
    # -----------------------------------------------------------------------
    null_cols = ["avg_session_duration", "win_rate", "total_spend_usd", "friend_count"]
    df = pd.DataFrame({
        "player_id": player_ids,
        "session_count": session_count,
        "avg_session_duration": avg_session_duration,
        "last_login_days_ago": last_login_days_ago,
        "total_playtime_hours": total_playtime_hours,
        "win_rate": win_rate,
        "total_spend_usd": total_spend_usd,
        "friend_count": friend_count,
        "game_mode_primary": game_mode_primary,
        "achievement_count": achievement_count,
        "consecutive_losses": consecutive_losses,
        "churn_label": churn_label,
    })

    for col in null_cols:
        null_idx = rng.choice(n, size=int(n * 0.02), replace=False)
        df.loc[null_idx, col] = np.nan

    n_churned = df["churn_label"].sum()
    log.info("Churn rate: %.1f %% (%d churned / %d total)", df["churn_label"].mean() * 100, n_churned, len(df))
    log.info("Null counts:\n%s", df.isnull().sum()[df.isnull().sum() > 0])
    return df


def main() -> None:
    """Entry point: generate dataset and save to CSV."""
    df = generate_players()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    log.info("Saved %d rows → %s", len(df), OUTPUT_PATH)
    log.info("Schema:\n%s", df.dtypes)
    log.info("Sample:\n%s", df.head(3).to_string())


if __name__ == "__main__":
    main()
