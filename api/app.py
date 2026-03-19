"""
api/app.py
----------
Flask REST API for player churn prediction.

Endpoints:
  GET  /health       — Health check
  POST /predict      — Single player churn prediction
  POST /predict/batch — Batch prediction (up to 1000 players)
  GET  /model/info   — Model metadata and performance metrics

Run:
    python api/app.py

Sample curl:
    curl -X POST http://localhost:5000/predict \\
         -H "Content-Type: application/json" \\
         -d '{
               "session_count": 45,
               "avg_session_duration": 40,
               "last_login_days_ago": 12,
               "total_playtime_hours": 80,
               "win_rate": 0.52,
               "total_spend_usd": 25.0,
               "friend_count": 8,
               "game_mode_primary": "PvP",
               "achievement_count": 35,
               "consecutive_losses": 2
             }'
"""

import logging
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from features import engineer_features
from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, _SafeOneHotEncoder
import __main__
setattr(__main__, '_SafeOneHotEncoder', _SafeOneHotEncoder)

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

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
# Flask app
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# Model loading (once at startup)
# ---------------------------------------------------------------------------
_model_bundle = None
_preprocessor = None
_model_comparison = None


def _prepare_loaded_model(model):
    """Force inference-safe settings on loaded estimators."""
    if model is None:
        return None

    if hasattr(model, "get_params") and "n_jobs" in model.get_params():
        try:
            model.set_params(n_jobs=1)
        except Exception:
            # Some wrapped estimators expose n_jobs but reject mutation post-fit.
            pass

    return model


def _load_artifacts() -> None:
    """Load the model bundle and preprocessor once at startup."""
    global _model_bundle, _preprocessor, _model_comparison

    model_path = MODEL_DIR / "best_model.pkl"
    preprocessor_path = MODEL_DIR / "preprocessor.pkl"
    comparison_path = MODEL_DIR / "model_comparison.csv"

    if model_path.exists():
        _model_bundle = joblib.load(model_path)
        _model_bundle["model"] = _prepare_loaded_model(_model_bundle.get("model"))
        log.info("Loaded model: %s", _model_bundle.get("name", "unknown"))
    else:
        log.warning("Model not found at %s. Run src/train.py first.", model_path)

    if preprocessor_path.exists():
        _preprocessor = joblib.load(preprocessor_path)
        log.info("Loaded preprocessor.")
    else:
        log.warning("Preprocessor not found. Run src/preprocessing.py first.")

    if comparison_path.exists():
        _model_comparison = pd.read_csv(comparison_path).to_dict(orient="records")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
REQUIRED_FIELDS = {
    "session_count": (int, float),
    "avg_session_duration": (int, float),
    "last_login_days_ago": (int, float),
    "total_playtime_hours": (int, float),
    "win_rate": (int, float),
    "total_spend_usd": (int, float),
    "friend_count": (int, float),
    "game_mode_primary": str,
    "achievement_count": (int, float),
    "consecutive_losses": (int, float),
}

FIELD_BOUNDS = {
    "session_count": (1, 10_000),
    "avg_session_duration": (1, 1440),
    "last_login_days_ago": (0, 3650),
    "total_playtime_hours": (0, 50_000),
    "win_rate": (0.0, 1.0),
    "total_spend_usd": (0.0, 100_000),
    "friend_count": (0, 10_000),
    "achievement_count": (0, 10_000),
    "consecutive_losses": (0, 100),
}

VALID_GAME_MODES = {"PvP", "PvE", "Co-op", "Solo"}


def validate_player_data(data: dict) -> list[str]:
    """
    Validate a single player data dict and return a list of error strings.

    Parameters
    ----------
    data : Dict from JSON request body.

    Returns
    -------
    List of error message strings (empty = valid).
    """
    errors = []
    for field, expected_types in REQUIRED_FIELDS.items():
        if field not in data:
            errors.append(f"Missing required field: '{field}'")
            continue
        val = data[field]
        if not isinstance(val, expected_types):
            errors.append(f"Field '{field}' must be {expected_types}, got {type(val).__name__}")
            continue
        if field in FIELD_BOUNDS:
            lo, hi = FIELD_BOUNDS[field]
            if not (lo <= val <= hi):
                errors.append(f"Field '{field}' = {val} is out of range [{lo}, {hi}]")
    if "game_mode_primary" in data and data["game_mode_primary"] not in VALID_GAME_MODES:
        errors.append(f"game_mode_primary must be one of {VALID_GAME_MODES}")
    return errors


def _preprocess_single(player_dict: dict) -> np.ndarray:
    """
    Apply RFM + feature engineering + preprocessor to a single player dict.

    Parameters
    ----------
    player_dict : Raw player fields.

    Returns
    -------
    Scaled numpy array (1 × n_features).
    """
    df = pd.DataFrame([player_dict])

    # RFM features (approximated for a single record; log1p monetary matches training)
    max_days = 180
    df["rfm_recency"] = max_days - df["last_login_days_ago"].clip(upper=max_days)
    df["rfm_frequency"] = df["session_count"]
    df["rfm_monetary"] = np.log1p(df["total_spend_usd"].fillna(0))
    df["rfm_score"] = 50.0  # percentile unknown for single record

    df = engineer_features(df)

    rfm_features = ["rfm_recency", "rfm_frequency", "rfm_monetary", "rfm_score"]
    all_numeric = NUMERIC_FEATURES + rfm_features
    feature_df = df[all_numeric + CATEGORICAL_FEATURES]

    return _preprocessor.transform(feature_df)


def _build_prediction_response(player_dict: dict, proba: float, threshold: float) -> dict:
    """
    Build the JSON response dict for a single prediction.

    Parameters
    ----------
    player_dict : Original input data.
    proba       : Churn probability.
    threshold   : Decision threshold.

    Returns
    -------
    Dict suitable for JSON serialisation.
    """
    churned = bool(proba >= threshold)

    if proba >= 0.75:
        risk_tier = "High"
        recommendation = ("Urgent re-engagement: trigger win-back push notification, "
                          "offer free premium currency, and adjust matchmaking difficulty.")
    elif proba >= 0.50:
        risk_tier = "Medium"
        recommendation = ("Proactive retention: send a personalised daily login bonus offer "
                          "and highlight new content or events.")
    elif proba >= 0.25:
        risk_tier = "Low"
        recommendation = ("Maintain engagement: standard email newsletter, "
                          "seasonal event notifications.")
    else:
        risk_tier = "Minimal"
        recommendation = "Player is highly engaged. Focus on monetisation opportunities."

    return {
        "churn_probability": round(float(proba), 4),
        "churn_probability_pct": round(float(proba) * 100, 1),
        "predicted_churn": churned,
        "risk_tier": risk_tier,
        "decision_threshold": round(threshold, 4),
        "recommendation": recommendation,
        "input_summary": {
            "session_count": player_dict.get("session_count"),
            "last_login_days_ago": player_dict.get("last_login_days_ago"),
            "total_spend_usd": player_dict.get("total_spend_usd"),
            "win_rate": player_dict.get("win_rate"),
            "consecutive_losses": player_dict.get("consecutive_losses"),
        },
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/", methods=["GET"])
def index() -> tuple:
    """Root endpoint."""
    return jsonify({"message": "Player Churn API is running. Check /health"}), 200


@app.route("/health", methods=["GET"])
def health() -> tuple:
    """
    Health check endpoint.

    Returns 200 OK with model status information.
    """
    return jsonify({
        "status": "ok",
        "model_loaded": _model_bundle is not None,
        "preprocessor_loaded": _preprocessor is not None,
        "model_name": _model_bundle.get("name") if _model_bundle else None,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }), 200


@app.route("/predict", methods=["POST"])
def predict() -> tuple:
    """
    Predict churn probability for a single player.

    Request body (JSON):
    --------------------
    {
        "session_count": int,
        "avg_session_duration": float,
        "last_login_days_ago": int,
        "total_playtime_hours": float,
        "win_rate": float (0–1),
        "total_spend_usd": float,
        "friend_count": int,
        "game_mode_primary": "PvP" | "PvE" | "Co-op" | "Solo",
        "achievement_count": int,
        "consecutive_losses": int
    }

    Response (JSON):
    ----------------
    {
        "churn_probability": float,
        "churn_probability_pct": float,
        "predicted_churn": bool,
        "risk_tier": "Minimal" | "Low" | "Medium" | "High",
        "decision_threshold": float,
        "recommendation": str,
        "input_summary": { ... }
    }
    """
    if _model_bundle is None or _preprocessor is None:
        return jsonify({"error": "Model not loaded. Run training pipeline first.", "code": "MODEL_UNAVAILABLE"}), 503

    if not request.is_json:
        return jsonify({"error": "Content-Type must be application/json.", "code": "INVALID_CONTENT_TYPE"}), 415

    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be valid JSON.", "code": "INVALID_JSON"}), 400

    errors = validate_player_data(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 422

    try:
        t0 = time.perf_counter()
        X = _preprocess_single(data)
        model = _model_bundle["model"]
        threshold = _model_bundle["threshold"]
        proba = float(model.predict_proba(X)[0, 1])
        latency_ms = round((time.perf_counter() - t0) * 1000, 2)

        response = _build_prediction_response(data, proba, threshold)
        response["latency_ms"] = latency_ms
        log.info("Predict  churn_prob=%.4f  risk=%s  latency=%dms",
                 proba, response["risk_tier"], latency_ms)
        return jsonify(response), 200

    except Exception as exc:
        log.exception("Prediction error: %s", exc)
        return jsonify({"error": "Internal prediction error", "detail": str(exc)}), 500


@app.route("/predict/batch", methods=["POST"])
def predict_batch() -> tuple:
    """
    Batch prediction for multiple players (up to 1000 per call).

    Request body (JSON):
    --------------------
    { "players": [ { ...player fields... }, ... ] }

    Response:
    ---------
    { "predictions": [ { ...response per player... } ], "count": int }
    """
    if _model_bundle is None or _preprocessor is None:
        return jsonify({"error": "Model not loaded."}), 503

    body = request.get_json(silent=True)
    if not body or "players" not in body:
        return jsonify({"error": "Request must contain a 'players' list."}), 400

    players = body["players"]
    if not isinstance(players, list) or len(players) == 0:
        return jsonify({"error": "'players' must be a non-empty list."}), 400
    if len(players) > 1000:
        return jsonify({"error": "Batch size limited to 1000 players per request."}), 400

    predictions = []
    errors_log = []

    for i, player in enumerate(players):
        player_errors = validate_player_data(player)
        if player_errors:
            errors_log.append({"index": i, "errors": player_errors})
            predictions.append({"index": i, "error": player_errors})
            continue
        try:
            X = _preprocess_single(player)
            proba = float(_model_bundle["model"].predict_proba(X)[0, 1])
            pred = _build_prediction_response(player, proba, _model_bundle["threshold"])
            pred["index"] = i
            predictions.append(pred)
        except Exception as exc:
            predictions.append({"index": i, "error": str(exc)})

    log.info("Batch predict: %d players, %d errors", len(players), len(errors_log))
    return jsonify({"predictions": predictions, "count": len(players)}), 200


@app.route("/model/info", methods=["GET"])
def model_info() -> tuple:
    """
    Return model metadata and test-set performance metrics.
    """
    if _model_bundle is None:
        return jsonify({"error": "Model not loaded."}), 503

    info = {
        "model_name": _model_bundle.get("name"),
        "decision_threshold": round(_model_bundle.get("threshold", 0.5), 4),
        "performance": _model_comparison,
    }
    return jsonify(info), 200


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Endpoint not found."}), 404


@app.errorhandler(405)
def method_not_allowed(e):
    return jsonify({"error": "Method not allowed."}), 405


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    _load_artifacts()
    log.info("Starting Flask API on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
