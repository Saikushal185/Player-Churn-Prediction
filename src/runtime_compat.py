"""
Runtime compatibility helpers for sandboxed and Windows environments.
"""

from __future__ import annotations

import logging


log = logging.getLogger(__name__)


def prepare_model_for_inference(model):
    """
    Clamp inference parallelism for models that expose ``n_jobs``.

    Some Windows/sandboxed environments fail when estimators attempt to spin up
    thread pools during ``predict``/``predict_proba``. For inference latency in
    this project, a single worker is sufficient and far more reliable.
    """
    if model is None or not hasattr(model, "n_jobs"):
        return model

    try:
        current_n_jobs = getattr(model, "n_jobs")
        if current_n_jobs != 1:
            setattr(model, "n_jobs", 1)
            log.info("Adjusted %s inference n_jobs from %s to 1.", type(model).__name__, current_n_jobs)
    except Exception as exc:
        log.warning("Could not adjust model parallelism for %s: %s", type(model).__name__, exc)
    return model
