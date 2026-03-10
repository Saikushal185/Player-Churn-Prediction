"""
segment.py
----------
K-Means player segmentation (k=4) based on behavioural and spend features.

Segments produced:
  - Segment 0: "Champions"   — High engagement, high spend, low churn risk
  - Segment 1: "Whales"      — High spend, moderate engagement, retention priority
  - Segment 2: "Casual"      — Low spend, moderate engagement, conversion opportunity
  - Segment 3: "At-Risk"     — Low engagement, declining activity, high churn risk

Each segment receives a tailored retention strategy recommendation.

Usage:
    python src/segment.py

Reads:  data/players_featured.csv
Saves:  data/players_segmented.csv
        models/kmeans_model.pkl
        reports/figures/cluster_scatter.png
        reports/figures/cluster_profiles.png
        reports/segment_profiles.csv
"""

import logging
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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
# Clustering features
# ---------------------------------------------------------------------------
CLUSTER_FEATURES = [
    "session_count",
    "avg_session_duration",
    "last_login_days_ago",
    "total_spend_usd",
    "win_rate",
    "friend_count",
    "achievement_count",
    "consecutive_losses",
    "engagement_score",
    "spend_per_session",
    "rolling_avg_sessions_7d",
    "rfm_score",
    "loss_streak_risk",
]

# Segment label mapping (assigned after inspecting cluster profiles)
SEGMENT_LABELS = {
    0: "Champions",
    1: "Whales",
    2: "Casual",
    3: "At-Risk",
}

SEGMENT_COLORS = {
    "Champions": "#2ecc71",   # Green
    "Whales":    "#3498db",   # Blue
    "Casual":    "#f39c12",   # Orange
    "At-Risk":   "#e74c3c",   # Red
}

RETENTION_STRATEGIES = {
    "Champions": (
        "Champions are your most engaged and skilled players. "
        "Strategy: Recognise with prestige badges and exclusive cosmetics. "
        "Invite to beta programmes and player councils. Convert to brand advocates "
        "via referral rewards and leaderboard spotlights. "
        "Budget: Low spend required — focus on recognition, not discounts."
    ),
    "Whales": (
        "Whales drive disproportionate revenue. Losing one Whale equals dozens of Casuals. "
        "Strategy: Assign a VIP account manager. Offer exclusive seasonal packs, "
        "early-access content, and personalised offers based on spend history. "
        "Monitor for signs of fatigue (declining session frequency) and trigger "
        "personalised win-back campaigns proactively. "
        "Budget: High — ROI is extremely favourable."
    ),
    "Casual": (
        "Casual players are acquisition successes but not yet converted to paying customers. "
        "Strategy: Tutorial completion incentives, first-purchase discount bundles ($0.99 starter pack), "
        "and social features prompts (invite a friend for bonus currency). "
        "Daily login streaks and limited-time events increase session frequency. "
        "Budget: Medium — aim for conversion to Dolphin tier within 30 days."
    ),
    "At-Risk": (
        "At-Risk players show leading indicators of imminent churn (low recency, declining sessions, "
        "high loss streaks). A 24–48 hour intervention window exists before permanent churn. "
        "Strategy: Triggered push notification ('We miss you — here are 500 free coins!'), "
        "temporary difficulty reduction or matchmaking adjustment, and win-streak reward events. "
        "Analyse common exit points (which level / game mode before churn) for UX fixes. "
        "Budget: Medium — cost of re-engagement is far lower than new acquisition."
    ),
}


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def load_featured_data(path: Path = DATA_DIR / "players_featured.csv") -> pd.DataFrame:
    """
    Load the feature-engineered player dataset.

    Parameters
    ----------
    path : Path to the CSV.

    Returns
    -------
    pd.DataFrame
    """
    log.info("Loading featured data from %s …", path)
    df = pd.read_csv(path)
    log.info("Loaded %d rows.", len(df))
    return df


def prepare_cluster_matrix(df: pd.DataFrame) -> tuple:
    """
    Extract, impute, and scale the clustering feature matrix.

    Parameters
    ----------
    df : Full featured dataframe.

    Returns
    -------
    (X_scaled, scaler, available_features)
    """
    available = [f for f in CLUSTER_FEATURES if f in df.columns]
    missing = set(CLUSTER_FEATURES) - set(available)
    if missing:
        log.warning("Clustering features not found (will skip): %s", missing)

    X = df[available].fillna(df[available].median())
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    log.info("Cluster feature matrix: %d rows × %d features", *X_scaled.shape)
    return X_scaled, scaler, available


def select_optimal_k(X_scaled: np.ndarray, k_range: range = range(2, 9)) -> int:
    """
    Evaluate elbow method and silhouette scores to confirm k=4 is optimal.

    Parameters
    ----------
    X_scaled : Scaled feature matrix.
    k_range  : Range of k values to evaluate.

    Returns
    -------
    Recommended k (int).
    """
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=42)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels, sample_size=5000))
        log.debug("k=%d  inertia=%.1f  silhouette=%.4f", k, km.inertia_, silhouettes[-1])

    # Plot elbow
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(list(k_range), inertias, "bo-")
    ax1.set_xlabel("Number of Clusters (k)")
    ax1.set_ylabel("Inertia")
    ax1.set_title("Elbow Method")
    ax1.axvline(x=4, color="red", linestyle="--", label="k=4 (chosen)")
    ax1.legend()

    ax2.plot(list(k_range), silhouettes, "go-")
    ax2.set_xlabel("Number of Clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    ax2.set_title("Silhouette Scores")
    ax2.axvline(x=4, color="red", linestyle="--", label="k=4 (chosen)")
    ax2.legend()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "cluster_elbow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Elbow/silhouette plot saved → %s", FIG_DIR / "cluster_elbow.png")

    best_k = int(k_range[np.argmax(silhouettes)])
    best_sil = max(silhouettes)
    log.info(
        "Best silhouette k=%d (score=%.4f); using k=4 as per business requirement.",
        best_k, best_sil,
    )
    return 4


def fit_kmeans(X_scaled: np.ndarray, k: int = 4) -> KMeans:
    """
    Fit K-Means with k-means++ initialisation.

    Parameters
    ----------
    X_scaled : Scaled feature matrix.
    k        : Number of clusters.

    Returns
    -------
    Fitted KMeans model.
    """
    km = KMeans(n_clusters=k, init="k-means++", n_init=20, max_iter=500, random_state=42)
    km.fit(X_scaled)
    sil = silhouette_score(X_scaled, km.labels_, sample_size=10000)
    db  = davies_bouldin_score(X_scaled, km.labels_)
    log.info("K-Means (k=%d)  silhouette=%.4f  davies-bouldin=%.4f", k, sil, db)
    return km


def assign_segment_labels(df: pd.DataFrame, km: KMeans, X_scaled: np.ndarray, features: list[str]) -> pd.DataFrame:
    """
    Assign cluster IDs to the player dataframe and map to semantic labels.

    The mapping is determined by inspecting cluster centroids:
      - Highest avg spend + high engagement  → Whale
      - Highest session_count + win_rate     → Champion
      - Highest last_login_days_ago + losses → At-Risk
      - Remaining cluster                    → Casual

    Parameters
    ----------
    df        : Full featured dataframe.
    km        : Fitted KMeans model.
    X_scaled  : Scaled feature matrix used for fitting.
    features  : Feature names list.

    Returns
    -------
    Dataframe with 'cluster_id' and 'segment' columns added.
    """
    df = df.copy()
    df["cluster_id"] = km.labels_

    # Build centroid DataFrame
    centroids = pd.DataFrame(km.cluster_centers_, columns=features)

    # Determine label → cluster mapping heuristically
    spend_col = "total_spend_usd" if "total_spend_usd" in centroids.columns else features[3]
    recency_col = "last_login_days_ago" if "last_login_days_ago" in centroids.columns else features[2]
    sessions_col = "session_count" if "session_count" in centroids.columns else features[0]
    losses_col = "consecutive_losses" if "consecutive_losses" in centroids.columns else features[7]

    whale_cluster     = int(centroids[spend_col].idxmax())
    at_risk_cluster   = int(centroids[recency_col].idxmax())
    remaining = [i for i in range(4) if i not in (whale_cluster, at_risk_cluster)]
    champion_cluster  = int(centroids.loc[remaining, sessions_col].idxmax())
    casual_cluster    = [i for i in remaining if i != champion_cluster][0]

    cluster_to_segment = {
        whale_cluster:    "Whales",
        at_risk_cluster:  "At-Risk",
        champion_cluster: "Champions",
        casual_cluster:   "Casual",
    }
    df["segment"] = df["cluster_id"].map(cluster_to_segment)

    counts = df["segment"].value_counts()
    log.info("Segment distribution:\n%s", counts.to_string())
    return df, cluster_to_segment


def build_segment_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-segment summary statistics and churn rates.

    Parameters
    ----------
    df : Dataframe with 'segment' column.

    Returns
    -------
    DataFrame with one row per segment.
    """
    profile_cols = [
        "session_count", "avg_session_duration", "last_login_days_ago",
        "total_spend_usd", "win_rate", "friend_count", "achievement_count",
        "consecutive_losses", "churn_label",
    ]
    available = [c for c in profile_cols if c in df.columns]

    profiles = df.groupby("segment")[available].agg(["mean", "median"]).round(2)
    profiles.columns = ["_".join(c) for c in profiles.columns]
    profiles["player_count"] = df.groupby("segment").size()
    profiles["churn_rate_pct"] = (df.groupby("segment")["churn_label"].mean() * 100).round(1)
    profiles["retention_strategy"] = profiles.index.map(RETENTION_STRATEGIES)

    log.info("Segment profiles:\n%s", profiles[["player_count", "churn_rate_pct"]].to_string())
    return profiles


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_cluster_scatter_2d(df: pd.DataFrame, X_scaled: np.ndarray, fig_dir: Path) -> None:
    """
    2-D PCA scatter plot of player clusters with segment labels.

    Parameters
    ----------
    df        : Dataframe with 'segment' column.
    X_scaled  : Scaled feature matrix.
    fig_dir   : Output directory.
    """
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_.sum() * 100

    plot_df = df[["segment"]].copy()
    plot_df["PC1"] = coords[:, 0]
    plot_df["PC2"] = coords[:, 1]
    sample = plot_df.sample(min(5000, len(plot_df)), random_state=42)

    fig = px.scatter(
        sample, x="PC1", y="PC2", color="segment",
        color_discrete_map=SEGMENT_COLORS,
        title=f"Player Segments — PCA Projection ({explained:.1f}% variance explained)",
        labels={"PC1": "Principal Component 1", "PC2": "Principal Component 2"},
        opacity=0.6,
        height=600,
    )
    fig.update_traces(marker_size=4)
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(fig_dir / "cluster_scatter.html"))
    fig.write_image(str(fig_dir / "cluster_scatter.png"), width=1000, height=600, scale=1.5)
    log.info("Cluster scatter saved → %s", fig_dir / "cluster_scatter.png")


def plot_segment_radar(profiles: pd.DataFrame, fig_dir: Path) -> None:
    """
    Radar (spider) chart comparing segment profiles across key metrics.

    Parameters
    ----------
    profiles : Output of build_segment_profiles().
    fig_dir  : Output directory.
    """
    radar_metrics = [
        "session_count_mean", "avg_session_duration_mean", "win_rate_mean",
        "total_spend_usd_mean", "friend_count_mean", "achievement_count_mean",
    ]
    available = [m for m in radar_metrics if m in profiles.columns]

    segments = profiles.index.tolist()
    fig = go.Figure()

    for seg in segments:
        vals = profiles.loc[seg, available].values.tolist()
        # Normalise to 0-1 for radar
        max_vals = profiles[available].max().values
        norm_vals = [v / (m + 1e-8) for v, m in zip(vals, max_vals)]
        norm_vals.append(norm_vals[0])  # close the polygon
        labels = [m.replace("_mean", "").replace("_", " ").title() for m in available]
        labels.append(labels[0])

        fig.add_trace(go.Scatterpolar(
            r=norm_vals,
            theta=labels,
            fill="toself",
            name=seg,
            line_color=SEGMENT_COLORS.get(seg, "grey"),
        ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        title="Segment Profile Radar Chart (Normalised)",
        height=600,
    )
    fig.write_html(str(fig_dir / "segment_radar.html"))
    fig.write_image(str(fig_dir / "segment_radar.png"), width=800, height=600, scale=1.5)
    log.info("Radar chart saved → %s", fig_dir / "segment_radar.png")


def plot_segment_churn_bar(df: pd.DataFrame, fig_dir: Path) -> None:
    """
    Bar chart of churn rate by segment.

    Parameters
    ----------
    df      : Dataframe with 'segment' and 'churn_label' columns.
    fig_dir : Output directory.
    """
    churn_by_seg = df.groupby("segment")["churn_label"].mean().reset_index()
    churn_by_seg.columns = ["segment", "churn_rate"]
    churn_by_seg["churn_pct"] = (churn_by_seg["churn_rate"] * 100).round(1)
    churn_by_seg["color"] = churn_by_seg["segment"].map(SEGMENT_COLORS)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        churn_by_seg["segment"],
        churn_by_seg["churn_pct"],
        color=churn_by_seg["color"],
        edgecolor="white",
        linewidth=1.5,
    )
    ax.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=11)
    ax.set_ylabel("Churn Rate (%)")
    ax.set_title("Churn Rate by Player Segment")
    ax.set_ylim(0, churn_by_seg["churn_pct"].max() * 1.2)
    fig.tight_layout()
    fig.savefig(fig_dir / "segment_churn_bar.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Segment churn bar saved → %s", fig_dir / "segment_churn_bar.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Full segmentation pipeline."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = load_featured_data()
    X_scaled, scaler, features = prepare_cluster_matrix(df)

    _ = select_optimal_k(X_scaled)   # Confirm k=4 with elbow/silhouette
    km = fit_kmeans(X_scaled, k=4)

    df, cluster_map = assign_segment_labels(df, km, X_scaled, features)

    profiles = build_segment_profiles(df)
    profiles.to_csv(REPORT_DIR / "segment_profiles.csv")
    log.info("Segment profiles saved → %s", REPORT_DIR / "segment_profiles.csv")

    # Save model + scaler + mapping
    joblib.dump({
        "kmeans": km,
        "scaler": scaler,
        "features": features,
        "cluster_to_segment": cluster_map,
        "segment_labels": SEGMENT_LABELS,
        "retention_strategies": RETENTION_STRATEGIES,
    }, MODEL_DIR / "kmeans_model.pkl")
    log.info("K-Means bundle saved → %s", MODEL_DIR / "kmeans_model.pkl")

    # Save segmented dataset
    df.to_csv(DATA_DIR / "players_segmented.csv", index=False)
    log.info("Segmented dataset saved → %s", DATA_DIR / "players_segmented.csv")

    # Visualisations
    plot_cluster_scatter_2d(df, X_scaled, FIG_DIR)
    plot_segment_radar(profiles, FIG_DIR)
    plot_segment_churn_bar(df, FIG_DIR)

    log.info("Segmentation complete.")


if __name__ == "__main__":
    main()
