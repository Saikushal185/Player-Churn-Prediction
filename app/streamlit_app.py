"""
streamlit_app.py
----------------
Four-tab Streamlit dashboard for the Player Churn Prediction project.

Tabs:
  1. Dashboard  — KPI cards, churn rate gauge, top at-risk players
  2. Predict    — Real-time churn probability for a single player (+ SHAP waterfall)
  3. Segments   — Interactive PCA scatter, radar chart, retention strategies
  4. Insights   — SHAP feature importance, beeswarm, model comparison

Improvements vs base prompt:
  - Sidebar global filters (game mode, churn risk tier, spend tier)
  - Animated Plotly gauge chart for churn probability
  - What-If simulator with sliders
  - CSV export for top-risk player table
  - st.session_state for filter persistence across tabs

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import logging
import sys
from io import BytesIO
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from preprocessing import _SafeOneHotEncoder
import __main__
setattr(__main__, '_SafeOneHotEncoder', _SafeOneHotEncoder)

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figures"
REPORT_DIR = ROOT / "reports"

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Player Churn Dashboard",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Report a bug": "https://github.com/Saikushal185/Player-Churn-Prediction/issues",
        "About": "Player Churn Prediction — End-to-End ML Pipeline",
    },
)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.metric-card {
    background: linear-gradient(135deg, #1e1e2e, #2a2a3e);
    border: 1px solid #444;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
    color: white;
    box-shadow: 0 2px 8px rgba(0,0,0,0.3);
    transition: transform 0.15s ease;
}
.metric-card:hover { transform: translateY(-2px); }
.metric-value { font-size: 2rem; font-weight: bold; margin: 0.3rem 0; }
.metric-label { font-size: 0.85rem; color: #aaa; }
.churn-high   { color: #e74c3c; }
.churn-medium { color: #f39c12; }
.churn-low    { color: #2ecc71; }
.sidebar-header { font-size: 1.1rem; font-weight: bold; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Cached data loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_players() -> pd.DataFrame:
    """Load the segmented player dataset (falls back to featured or raw)."""
    for fname in ("players_segmented.csv", "players_featured.csv", "players.csv"):
        p = DATA_DIR / fname
        if p.exists():
            return pd.read_csv(p)
    st.error("No player data found. Run the pipeline first (see README).")
    st.stop()


@st.cache_resource(show_spinner=False)
def load_model_bundle() -> dict | None:
    """Load the best model bundle."""
    p = MODEL_DIR / "best_model.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource(show_spinner=False)
def load_kmeans_bundle() -> dict | None:
    """Load the K-Means segmentation bundle."""
    p = MODEL_DIR / "kmeans_model.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_resource(show_spinner=False)
def load_preprocessor():
    """Load the fitted ColumnTransformer."""
    p = MODEL_DIR / "preprocessor.pkl"
    if not p.exists():
        return None
    return joblib.load(p)


@st.cache_data(show_spinner=False)
def load_model_comparison() -> pd.DataFrame | None:
    p = MODEL_DIR / "model_comparison.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def load_top_risk() -> pd.DataFrame | None:
    p = REPORT_DIR / "top_risk_players.csv"
    if not p.exists():
        return None
    return pd.read_csv(p)


# ---------------------------------------------------------------------------
# Helper: predict for a single player dict
# ---------------------------------------------------------------------------

def predict_single(player_dict: dict, model_bundle: dict, preprocessor) -> tuple[float, str]:
    """
    Predict churn probability for a single player.

    Parameters
    ----------
    player_dict  : Raw player features as a dict.
    model_bundle : {"model": ..., "threshold": ..., "name": ...}
    preprocessor : Fitted ColumnTransformer.

    Returns
    -------
    (churn_probability, risk_tier)
    """
    from src.features import engineer_features
    from src.preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES

    df_single = pd.DataFrame([player_dict])

    # RFM
    max_days = 180
    df_single["rfm_recency"] = max_days - df_single["last_login_days_ago"]
    df_single["rfm_frequency"] = df_single["session_count"]
    df_single["rfm_monetary"] = df_single["total_spend_usd"].fillna(0)
    df_single["rfm_score"] = 50.0  # default percentile for single record

    df_single = engineer_features(df_single)

    rfm_features = ["rfm_recency", "rfm_frequency", "rfm_monetary", "rfm_score"]
    all_numeric = NUMERIC_FEATURES + rfm_features

    feature_df = df_single[all_numeric + CATEGORICAL_FEATURES]
    X = preprocessor.transform(feature_df)

    model = model_bundle["model"]
    threshold = model_bundle["threshold"]
    proba = float(model.predict_proba(X)[0, 1])

    if proba >= 0.75:
        tier = "High Risk"
    elif proba >= 0.50:
        tier = "Medium Risk"
    else:
        tier = "Low Risk"

    return proba, tier


# ---------------------------------------------------------------------------
# Gauge chart
# ---------------------------------------------------------------------------

def make_gauge(proba: float) -> go.Figure:
    """
    Create an animated Plotly gauge chart for the churn probability.

    Parameters
    ----------
    proba : Churn probability in [0, 1].

    Returns
    -------
    Plotly Figure.
    """
    pct = proba * 100
    if pct < 30:
        bar_color = "#2ecc71"
    elif pct < 65:
        bar_color = "#f39c12"
    else:
        bar_color = "#e74c3c"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=pct,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "Churn Probability (%)", "font": {"size": 18}},
        delta={"reference": 50, "increasing": {"color": "#e74c3c"}, "decreasing": {"color": "#2ecc71"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "white"},
            "bar": {"color": bar_color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 2,
            "bordercolor": "#333",
            "steps": [
                {"range": [0, 30],  "color": "#d5f5e3"},
                {"range": [30, 65], "color": "#fef9e7"},
                {"range": [65, 100], "color": "#fadbd8"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 4},
                "thickness": 0.75,
                "value": 50,
            },
        },
    ))
    fig.update_layout(
        height=300,
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "white"},
        margin=dict(l=30, r=30, t=50, b=0),
    )
    return fig


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def render_sidebar(df: pd.DataFrame) -> pd.DataFrame:
    """
    Render global sidebar filters and return the filtered dataframe.
    Filter state is stored in st.session_state for persistence.

    Parameters
    ----------
    df : Full player dataframe.

    Returns
    -------
    Filtered dataframe.
    """
    st.sidebar.title("🎮 Churn Dashboard")
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Global Filters**")

    # Game mode
    modes = sorted(df["game_mode_primary"].dropna().unique().tolist())
    default_modes = st.session_state.get("filter_modes", modes)
    selected_modes = st.sidebar.multiselect(
        "Game Mode", modes, default=default_modes, key="filter_modes"
    )

    # Churn risk tier
    if "churn_risk_level" in df.columns:
        risk_levels = sorted(df["churn_risk_level"].dropna().unique().tolist())
    else:
        # Derive from churn_label
        df["churn_risk_level"] = df.get("churn_label", 0).map({0: "Active", 1: "Churned"})
        risk_levels = ["Active", "Churned"]
    default_risk = st.session_state.get("filter_risk", risk_levels)
    selected_risk = st.sidebar.multiselect(
        "Churn Status", risk_levels, default=default_risk, key="filter_risk"
    )

    # Spend tier
    if "monetisation_tier" in df.columns:
        tier_map = {0: "Free", 1: "Minnow", 2: "Dolphin", 3: "Whale"}
        df["spend_tier_label"] = df["monetisation_tier"].map(tier_map)
    else:
        df["spend_tier_label"] = pd.cut(
            df["total_spend_usd"].fillna(0),
            bins=[-1, 0, 20, 100, 1e9],
            labels=["Free", "Minnow", "Dolphin", "Whale"],
        )
    spend_tiers = ["Free", "Minnow", "Dolphin", "Whale"]
    default_tiers = st.session_state.get("filter_spend", spend_tiers)
    selected_tiers = st.sidebar.multiselect(
        "Spend Tier", spend_tiers, default=default_tiers, key="filter_spend"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(f"**Filtered players:** {len(df):,}")

    # Apply filters
    mask = (
        df["game_mode_primary"].isin(selected_modes)
        & df["churn_risk_level"].isin(selected_risk)
        & df["spend_tier_label"].isin(selected_tiers)
    )
    filtered = df[mask]
    return filtered


# ---------------------------------------------------------------------------
# Tab 1: Dashboard
# ---------------------------------------------------------------------------

def tab_dashboard(df: pd.DataFrame) -> None:
    """Render the main KPI dashboard tab."""
    st.header("📊 Player Retention Dashboard")

    # KPI metrics
    total = len(df)
    churned = int(df["churn_label"].sum()) if "churn_label" in df.columns else 0
    churn_rate = churned / total * 100 if total > 0 else 0
    avg_spend = df["total_spend_usd"].fillna(0).mean()
    avg_sessions = df["session_count"].mean()

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Players",      f"{total:,}")
    c2.metric("Churned Players",    f"{churned:,}",  delta=f"{churn_rate:.1f}% churn rate", delta_color="inverse")
    c3.metric("Churn Rate",         f"{churn_rate:.1f}%")
    c4.metric("Avg Spend / Player", f"${avg_spend:.2f}")
    c5.metric("Avg Sessions",       f"{avg_sessions:.0f}")

    st.markdown("---")

    col_left, col_right = st.columns([1.4, 1])

    with col_left:
        # Churn by game mode
        if "game_mode_primary" in df.columns:
            mode_churn = df.groupby("game_mode_primary")["churn_label"].agg(["mean", "count"]).reset_index()
            mode_churn.columns = ["game_mode", "churn_rate", "player_count"]
            mode_churn["churn_pct"] = (mode_churn["churn_rate"] * 100).round(1)
            fig = px.bar(
                mode_churn, x="game_mode", y="churn_pct", color="churn_pct",
                color_continuous_scale="RdYlGn_r",
                text="churn_pct", title="Churn Rate by Game Mode",
                labels={"churn_pct": "Churn Rate (%)"},
            )
            fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig.update_layout(showlegend=False, height=350)
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Spend tier distribution
        if "spend_tier_label" in df.columns:
            tier_counts = df["spend_tier_label"].value_counts().reset_index()
            tier_counts.columns = ["tier", "count"]
            fig = px.pie(
                tier_counts, names="tier", values="count",
                color="tier",
                color_discrete_map={"Free": "#95a5a6", "Minnow": "#3498db",
                                    "Dolphin": "#2ecc71", "Whale": "#e74c3c"},
                title="Player Distribution by Spend Tier",
                hole=0.4,
            )
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🔴 Top At-Risk Players")

    risk_df = load_top_risk()
    if risk_df is not None:
        display = risk_df[["player_index", "churn_prob", "predicted_churn",
                            "actual_churn", "risk_narrative"]].head(20).copy()
        display["churn_prob"] = (display["churn_prob"] * 100).round(1).astype(str) + "%"
        st.dataframe(display, use_container_width=True)

        # CSV export
        csv_bytes = display.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Export Top Risk Players (CSV)",
            csv_bytes,
            file_name="top_risk_players.csv",
            mime="text/csv",
        )
    else:
        st.info("Run `src/explain.py` to generate the at-risk player table.")

    # Last login distribution
    st.markdown("---")
    if "last_login_days_ago" in df.columns:
        st.subheader("📅 Login Recency Distribution")
        fig = px.histogram(
            df, x="last_login_days_ago", color="churn_label",
            color_discrete_map={0: "#2ecc71", 1: "#e74c3c"},
            labels={"last_login_days_ago": "Days Since Last Login", "churn_label": "Churned"},
            title="Last Login Recency by Churn Status",
            nbins=60, barmode="overlay", opacity=0.7,
        )
        fig.add_vline(x=30, line_dash="dash", line_color="orange",
                      annotation_text="30-day churn threshold")
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Predict
# ---------------------------------------------------------------------------

def tab_predict(model_bundle: dict | None, preprocessor) -> None:
    """Render the single-player churn prediction tab with What-If simulator."""
    st.header("🔮 Player Churn Predictor")

    if model_bundle is None or preprocessor is None:
        st.warning("Model or preprocessor not found. Run `src/train.py` and `src/preprocessing.py` first.")
        return

    col_form, col_result = st.columns([1, 1])

    with col_form:
        st.subheader("🕹️ Player Stats")
        session_count = st.slider("Session Count", 1, 500, 45)
        avg_session_duration = st.slider("Avg Session Duration (min)", 5, 240, 45)
        last_login_days_ago = st.slider("Last Login (days ago)", 0, 180, 15)
        total_playtime_hours = st.slider("Total Playtime (hours)", 1, 1000, 60)
        win_rate = st.slider("Win Rate", 0.0, 1.0, 0.50, step=0.01, format="%.2f")
        total_spend_usd = st.number_input("Total Spend (USD)", min_value=0.0, max_value=5000.0, value=20.0, step=5.0)
        friend_count = st.slider("Friend Count", 0, 100, 5)
        game_mode = st.selectbox("Game Mode", ["PvP", "PvE", "Co-op", "Solo"])
        achievement_count = st.slider("Achievement Count", 0, 500, 30)
        consecutive_losses = st.slider("Consecutive Losses", 0, 15, 2)

        predict_clicked = st.button("🎯 Predict Churn Risk", type="primary", use_container_width=True)

    with col_result:
        if predict_clicked:
            player_dict = {
                "session_count": session_count,
                "avg_session_duration": avg_session_duration,
                "last_login_days_ago": last_login_days_ago,
                "total_playtime_hours": total_playtime_hours,
                "win_rate": win_rate,
                "total_spend_usd": total_spend_usd,
                "friend_count": friend_count,
                "game_mode_primary": game_mode,
                "achievement_count": achievement_count,
                "consecutive_losses": consecutive_losses,
            }
            with st.spinner("Computing churn probability …"):
                try:
                    proba, tier = predict_single(player_dict, model_bundle, preprocessor)
                    st.session_state["last_proba"] = proba
                    st.session_state["last_tier"] = tier
                except Exception as e:
                    st.error(f"Prediction error: {e}")
                    return

        proba = st.session_state.get("last_proba", 0.25)
        tier = st.session_state.get("last_tier", "Low Risk")

        st.subheader("📈 Prediction Result")
        tier_color = {"High Risk": "🔴", "Medium Risk": "🟡", "Low Risk": "🟢"}.get(tier, "⚪")
        st.markdown(f"### {tier_color} {tier}")
        st.plotly_chart(make_gauge(proba), use_container_width=True)

        # Risk interpretation
        if proba >= 0.75:
            st.error("**Urgent** — This player is highly likely to churn. "
                     "Trigger a win-back campaign immediately.")
        elif proba >= 0.50:
            st.warning("**At risk** — Monitor this player closely. "
                       "Consider a personalised offer or difficulty adjustment.")
        else:
            st.success("**Healthy** — This player shows strong engagement signals.")

    # What-If Simulator
    st.markdown("---")
    st.subheader("🎛️ What-If Simulator")
    st.markdown("Adjust the sliders below to explore how changes to player behaviour affect churn risk.")

    wi_col1, wi_col2, wi_col3 = st.columns(3)
    with wi_col1:
        wi_sessions = st.slider("Simulate Session Count", 1, 500, 45, key="wi_sessions")
        wi_login = st.slider("Simulate Last Login (days ago)", 0, 180, 15, key="wi_login")
    with wi_col2:
        wi_spend = st.slider("Simulate Total Spend ($)", 0.0, 500.0, 20.0, key="wi_spend")
        wi_losses = st.slider("Simulate Consecutive Losses", 0, 15, 2, key="wi_losses")
    with wi_col3:
        wi_winrate = st.slider("Simulate Win Rate", 0.0, 1.0, 0.50, 0.01, key="wi_winrate")
        wi_friends = st.slider("Simulate Friend Count", 0, 100, 5, key="wi_friends")

    if model_bundle is not None and preprocessor is not None:
        wi_dict = {
            "session_count": wi_sessions,
            "avg_session_duration": 45,
            "last_login_days_ago": wi_login,
            "total_playtime_hours": wi_sessions * 45 / 60,
            "win_rate": wi_winrate,
            "total_spend_usd": wi_spend,
            "friend_count": wi_friends,
            "game_mode_primary": "PvP",
            "achievement_count": int(wi_sessions * 0.6),
            "consecutive_losses": wi_losses,
        }
        try:
            wi_proba, wi_tier = predict_single(wi_dict, model_bundle, preprocessor)
            wi_color = {"High Risk": "#e74c3c", "Medium Risk": "#f39c12", "Low Risk": "#2ecc71"}.get(wi_tier, "grey")
            st.markdown(
                f"**What-If Result:** Churn Probability = "
                f"<span style='color:{wi_color}; font-size:1.4rem; font-weight:bold;'>{wi_proba*100:.1f}%</span> "
                f"({wi_tier})",
                unsafe_allow_html=True,
            )
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Tab 3: Segments
# ---------------------------------------------------------------------------

def tab_segments(df: pd.DataFrame) -> None:
    """Render the player segmentation tab."""
    st.header("🗺️ Player Segmentation")

    if "segment" not in df.columns:
        st.info("Run `src/segment.py` to generate player segments.")
        return

    SEGMENT_COLORS = {
        "Champions": "#2ecc71",
        "Whales": "#3498db",
        "Casual": "#f39c12",
        "At-Risk": "#e74c3c",
    }

    # Segment summary cards
    cols = st.columns(4)
    for i, seg in enumerate(["Champions", "Whales", "Casual", "At-Risk"]):
        seg_df = df[df["segment"] == seg]
        churn_r = seg_df["churn_label"].mean() * 100 if "churn_label" in seg_df.columns else 0
        avg_spend_seg = seg_df["total_spend_usd"].fillna(0).mean()
        color = SEGMENT_COLORS.get(seg, "grey")
        cols[i].markdown(
            f"""<div class='metric-card' style='border-top: 4px solid {color};'>
                <div class='metric-label'>{seg}</div>
                <div class='metric-value' style='color:{color};'>{len(seg_df):,}</div>
                <div class='metric-label'>players</div>
                <div class='metric-label'>Churn: {churn_r:.1f}% | Avg Spend: ${avg_spend_seg:.0f}</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # PCA scatter (interactive)
    col_scatter, col_strat = st.columns([1.4, 1])
    with col_scatter:
        st.subheader("2-D Player Cluster Scatter (PCA)")
        scatter_html = FIG_DIR / "cluster_scatter.html"
        if scatter_html.exists():
            from streamlit.components.v1 import html as st_html
            with open(scatter_html) as f:
                st_html(f.read(), height=500, scrolling=False)
        else:
            # Build scatter on the fly using engagement_score + rfm_score
            if "engagement_score" in df.columns and "rfm_score" in df.columns:
                sample = df.sample(min(5000, len(df)), random_state=42)
                fig = px.scatter(
                    sample, x="engagement_score", y="rfm_score",
                    color="segment", color_discrete_map=SEGMENT_COLORS,
                    hover_data=["session_count", "total_spend_usd", "win_rate"],
                    title="Engagement Score vs RFM Score by Segment",
                    opacity=0.6, height=500,
                )
                st.plotly_chart(fig, use_container_width=True)

    with col_strat:
        st.subheader("Retention Strategies")
        from src.segment import RETENTION_STRATEGIES
        selected_seg = st.selectbox("Select Segment", list(RETENTION_STRATEGIES.keys()))
        st.markdown(
            f"<div style='background:#1e1e2e; border-left: 4px solid {SEGMENT_COLORS[selected_seg]}; "
            f"padding: 1rem; border-radius: 8px; color: white;'>{RETENTION_STRATEGIES[selected_seg]}</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Radar chart
    radar_img = FIG_DIR / "segment_radar.png"
    if radar_img.exists():
        st.subheader("📡 Segment Radar Profiles")
        st.image(str(radar_img), use_column_width=True)

    # Churn by segment bar
    churn_img = FIG_DIR / "segment_churn_bar.png"
    if churn_img.exists():
        st.image(str(churn_img), use_column_width=True)

    # Segment data table with export
    st.markdown("---")
    st.subheader("📋 Segment Player Table")
    seg_filter = st.selectbox("Filter by segment", ["All"] + list(SEGMENT_COLORS.keys()), key="seg_table")
    view_df = df if seg_filter == "All" else df[df["segment"] == seg_filter]
    cols_to_show = ["player_id", "segment", "session_count", "last_login_days_ago",
                    "total_spend_usd", "win_rate", "churn_label"]
    available_cols = [c for c in cols_to_show if c in view_df.columns]
    st.dataframe(view_df[available_cols].head(100), use_container_width=True)
    csv_bytes = view_df[available_cols].to_csv(index=False).encode()
    st.download_button("⬇️ Export Segment Data (CSV)", csv_bytes,
                       file_name=f"segment_{seg_filter.lower()}.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Tab 4: Insights
# ---------------------------------------------------------------------------

def tab_insights() -> None:
    """Render the SHAP and model insights tab."""
    st.header("💡 Model Insights")

    # Model comparison
    comp_df = load_model_comparison()
    if comp_df is not None:
        st.subheader("🏆 Model Comparison")
        st.dataframe(
            comp_df.style.highlight_max(subset=["test_auc", "f1_churn"], color="#2d572c")
                         .highlight_min(subset=["test_auc", "f1_churn"], color="#572d2d"),
            use_container_width=True,
        )

        # AUC bar chart
        fig = px.bar(
            comp_df, x="model", y="test_auc", color="model",
            text="test_auc", title="Model AUC-ROC Comparison",
            color_discrete_sequence=px.colors.qualitative.Plotly,
        )
        fig.update_traces(texttemplate="%{text:.3f}", textposition="outside")
        fig.add_hline(y=0.85, line_dash="dash", line_color="red",
                      annotation_text="Target AUC=0.85")
        fig.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # SHAP plots
    col_bar, col_bee = st.columns(2)
    with col_bar:
        p = FIG_DIR / "shap_bar_summary.png"
        if p.exists():
            st.subheader("🎯 SHAP Feature Importance")
            st.image(str(p), use_column_width=True)
        else:
            st.info("Run `src/explain.py` to generate SHAP plots.")

    with col_bee:
        p = FIG_DIR / "shap_beeswarm.png"
        if p.exists():
            st.subheader("🐝 SHAP Beeswarm Plot")
            st.image(str(p), use_column_width=True)

    st.markdown("---")

    col_wf, col_calib = st.columns(2)
    with col_wf:
        p = FIG_DIR / "shap_waterfall.png"
        if p.exists():
            st.subheader("🌊 Waterfall — Highest Risk Player")
            st.image(str(p), use_column_width=True)

    with col_calib:
        p = FIG_DIR / "calibration_plot.png"
        if p.exists():
            st.subheader("📐 Calibration Plot")
            st.image(str(p), use_column_width=True)

    st.markdown("---")

    # ROC + PR curves
    col_roc, col_pr = st.columns(2)
    with col_roc:
        p = FIG_DIR / "roc_curves.png"
        if p.exists():
            st.subheader("📈 ROC Curves")
            st.image(str(p), use_column_width=True)
    with col_pr:
        p = FIG_DIR / "pr_curves.png"
        if p.exists():
            st.subheader("📊 Precision-Recall Curves")
            st.image(str(p), use_column_width=True)

    # Top features table
    top10_path = REPORT_DIR / "top10_features.csv"
    if top10_path.exists():
        st.markdown("---")
        st.subheader("🔑 Top 10 Predictive Features")
        top10_df = pd.read_csv(top10_path)
        st.dataframe(top10_df, use_container_width=True)


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    """Main Streamlit entry point."""
    df = load_players()
    model_bundle = load_model_bundle()
    preprocessor = load_preprocessor()

    filtered_df = render_sidebar(df)

    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Dashboard",
        "🔮 Predict",
        "🗺️ Segments",
        "💡 Insights",
    ])
    with tab1:
        tab_dashboard(filtered_df)
    with tab2:
        tab_predict(model_bundle, preprocessor)
    with tab3:
        tab_segments(filtered_df)
    with tab4:
        tab_insights()


if __name__ == "__main__":
    main()
