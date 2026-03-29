"""
model_loader.py
---------------
Utilities for loading a trained sklearn/XGBoost pipeline from disk.
Falls back to a DummyModel when no model file is present so the engine
can always return a valid analytics payload (useful in CI / demo contexts).
"""

import os
import pickle
import logging
import numpy as np

logger = logging.getLogger(__name__)


class DummyModel:
    """Fallback model used when no trained model file is available.

    Returns balanced 50/50 probabilities for each observation so that
    downstream consumers can still receive a structurally valid analytics
    payload while a real model is being trained or deployed.
    """

    def predict_proba(self, X):
        """Return uniform class probabilities of shape (n_samples, 2)."""
        n = len(X)
        return np.full((n, 2), 0.5)


def load_model(model_path: str):
    """Load a pickled model from *model_path*.

    Parameters
    ----------
    model_path:
        Absolute or relative path to the ``*.pkl`` model file.  The value is
        read from the ``MODEL_PATH`` environment variable when the caller does
        not supply an explicit path, allowing twelve-factor configuration.

    Returns
    -------
    model
        A fitted sklearn estimator (or pipeline) that exposes
        ``predict_proba``, or a :class:`DummyModel` instance when the file
        cannot be found or loaded.
    """
    resolved = model_path or os.environ.get("MODEL_PATH", "")

    if not resolved or not os.path.exists(resolved):
        logger.warning(
            "Model file not found at '%s'. Using DummyModel.", resolved
        )
        return DummyModel()

    try:
        with open(resolved, "rb") as f:
            model = pickle.load(f)

        # Guard: the engine always calls predict_proba; make sure the loaded
        # object supports it before returning it.
        if not callable(getattr(model, "predict_proba", None)):
            logger.warning(
                "Loaded model does not expose predict_proba. Using DummyModel."
            )
            return DummyModel()

        logger.info("Model loaded successfully from '%s'.", resolved)
        return model

    except Exception as exc:
        logger.error("Error loading model from '%s': %s", resolved, exc)
        return DummyModel()
