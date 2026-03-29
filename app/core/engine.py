"""
engine.py
---------
Forex Market Intelligence — Core Analytics Engine.

This module is the heart of the SaaS.  It replaces the trade-execution bot
(``src/live_stream.py``) with a **read-only analytics pipeline** that:

  1. Ingests EUR/USD tick data and resamples it into OHLC-style bars.
  2. Engineers lag-return features identical to the training pipeline.
  3. Runs ``model.predict_proba()`` — never ``predict()`` — to obtain
     calibrated probabilities.
  4. Returns a structured analytics payload (dict / JSON).

⚠️  This module NEVER executes trades, creates orders, or connects to a
    broker API.  It is a decision-support system only.
"""

import os
import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yaml

from app.core.features import build_feature_row
from app.core.model_loader import load_model

logger = logging.getLogger(__name__)

# ── Default configuration (overridden by environment variables) ──────────────
_DEFAULT_INSTRUMENT = "EUR_USD"
_DEFAULT_BAR_LENGTH = "1min"
_DEFAULT_LAGS = 10
_DEFAULT_MODEL_PATH = os.path.join("models", "model.pkl")

# Fraction of uncertainty mass redistributed to the "neutral" probability
# bucket.  At 50 % the neutral bucket captures half the residual ambiguity
# while bullish and bearish each keep the other half, keeping all three
# probabilities well-separated at high confidence.
_NEUTRAL_UNCERTAINTY_FRACTION = 0.5


def _load_params() -> dict:
    """Load live-stream parameters from ``params.yaml`` if it exists."""
    params_path = os.environ.get("PARAMS_PATH", "params.yaml")
    if os.path.exists(params_path):
        with open(params_path, "r") as f:
            return yaml.safe_load(f)
    return {}


def _granularity_to_offset(granularity: str) -> str:
    """Map an OANDA-style granularity string to a pandas offset alias."""
    mapping = {
        "S5": "5s",
        "M1": "1min",
        "M5": "5min",
        "M15": "15min",
        "H1": "1h",
        "H4": "4h",
        "D": "1D",
    }
    return mapping.get(granularity, "1min")


class ForexAnalyticsEngine:
    """Real-time analytics engine for a single currency pair.

    Parameters
    ----------
    instrument:
        Currency pair identifier, e.g. ``"EUR_USD"``.
    bar_length:
        Pandas offset alias for bar resampling, e.g. ``"1min"``.
    lags:
        Number of lagged return features to generate.
    model_path:
        Path to the pickled model file.  Falls back to
        ``DummyModel`` when the file is absent.
    """

    def __init__(
        self,
        instrument: Optional[str] = None,
        bar_length: Optional[str] = None,
        lags: Optional[int] = None,
        model_path: Optional[str] = None,
    ) -> None:
        params = _load_params()
        live_params = params.get("live_stream", {})
        data_params = params.get("data_fetch", {})
        preprocess_params = params.get("preprocess", {})
        train_params = params.get("train", {})

        self.instrument: str = (
            instrument
            or os.environ.get("INSTRUMENT")
            or data_params.get("instrument", _DEFAULT_INSTRUMENT)
        )

        raw_granularity = data_params.get("granularity", "M1")
        default_bar = _granularity_to_offset(raw_granularity)
        self.bar_length: str = (
            bar_length
            or os.environ.get("BAR_LENGTH")
            or default_bar
        )
        self.bar_timedelta = pd.to_timedelta(self.bar_length)

        self.lags: int = int(
            lags
            or os.environ.get("LAGS", "")
            or preprocess_params.get("lags", _DEFAULT_LAGS)
        )

        resolved_model_path: str = (
            model_path
            or os.environ.get("MODEL_PATH")
            or train_params.get("model_path", _DEFAULT_MODEL_PATH)
        )
        self.model = load_model(resolved_model_path)

        # Internal state: price history for the current session.
        # Explicit float64 dtype avoids object-dtype coercion during pd.concat.
        self._price_history: pd.DataFrame = pd.DataFrame(
            columns=[self.instrument], dtype=float
        )
        self._tick_buffer: pd.DataFrame = pd.DataFrame(
            columns=[self.instrument], dtype=float
        )
        self._last_bar: Optional[pd.Timestamp] = None

        logger.info(
            "ForexAnalyticsEngine initialised | instrument=%s bar=%s lags=%d",
            self.instrument,
            self.bar_length,
            self.lags,
        )

    # ── Data ingestion ────────────────────────────────────────────────────────

    def ingest_tick(self, timestamp: str, bid: float, ask: float) -> Optional[dict]:
        """Process a single incoming tick.

        Parameters
        ----------
        timestamp:
            ISO-8601 string or any format parseable by ``pd.to_datetime``.
        bid:
            Best bid price.
        ask:
            Best ask price.

        Returns
        -------
        dict or None
            An analytics payload when a new bar is completed, otherwise
            ``None`` (bar still accumulating ticks).
        """
        ts = pd.to_datetime(timestamp)
        mid_price = (bid + ask) / 2.0

        tick_df = pd.DataFrame(
            {self.instrument: [mid_price]}, index=[ts]
        )
        self._tick_buffer = pd.concat([self._tick_buffer, tick_df])

        # Determine whether a new bar has closed
        if self._last_bar is None:
            # First tick without pre-loaded history — anchor the bar clock.
            self._last_bar = ts
            return None

        if ts - self._last_bar >= self.bar_timedelta:
            self._resample_and_join()
            return self._run_inference()

        return None

    def load_history(self, df: pd.DataFrame) -> None:
        """Seed the engine with historical price data.

        Parameters
        ----------
        df:
            DataFrame with a DatetimeIndex and a single column named after
            ``self.instrument`` (or ``"c"`` / ``"close"`` — renamed
            automatically).
        """
        df = df.copy()

        # Normalise column name
        rename_map = {col: self.instrument for col in df.columns
                      if col in ("c", "close", "price")}
        if rename_map:
            df.rename(columns=rename_map, inplace=True)

        if self.instrument not in df.columns:
            raise ValueError(
                f"Expected column '{self.instrument}' (or 'c'/'close'/'price') "
                f"in history DataFrame; got {list(df.columns)}."
            )

        df = df[[self.instrument]].resample(self.bar_length, label="right").last().dropna()
        self._price_history = df
        if not df.empty:
            self._last_bar = df.index[-1]

        logger.info(
            "History loaded: %d bars up to %s", len(df), self._last_bar
        )

    # ── Resampling ────────────────────────────────────────────────────────────

    def _resample_and_join(self) -> None:
        """Resample the tick buffer into completed bars and append to history."""
        if self._tick_buffer.empty:
            return

        resampled = (
            self._tick_buffer
            .resample(self.bar_length, label="right")
            .last()
            .ffill()
        )
        # Exclude the still-open bar (last row)
        completed = resampled.iloc[:-1]

        self._price_history = pd.concat(
            [self._price_history, completed]
        ).drop_duplicates()

        # Keep only the latest tick to start the next bar
        self._tick_buffer = self._tick_buffer.iloc[-1:]
        self._last_bar = self._price_history.index[-1]

    # ── Inference ─────────────────────────────────────────────────────────────

    def _run_inference(self) -> Optional[dict]:
        """Run the ML pipeline on the current price history.

        Returns
        -------
        dict or None
            Structured analytics payload, or ``None`` when there is
            insufficient data to compute features.
        """
        if self._price_history.empty or self.instrument not in self._price_history.columns:
            return None

        price_series = self._price_history[self.instrument]

        feature_df, returns, volatility, regime = build_feature_row(
            price_series, self.lags
        )

        if feature_df.empty:
            logger.debug("Not enough data to build features yet.")
            return None

        # Use only the most-recent feature row for inference
        X = feature_df.iloc[[-1]]

        try:
            proba = self.model.predict_proba(X)[0]
        except Exception as exc:
            logger.error("predict_proba failed: %s", exc)
            return None

        # The model is a binary classifier: index 0 = bearish (down),
        # index 1 = bullish (up).
        prob_bearish = float(proba[0])
        prob_bullish = float(proba[1])

        # Derive a neutral probability from model uncertainty.
        # When the model is near 50/50 the confidence is low, so we
        # redistribute a portion of the probability mass to "neutral".
        confidence = float(abs(prob_bullish - prob_bearish))
        uncertainty = 1.0 - confidence
        neutral_mass = uncertainty * _NEUTRAL_UNCERTAINTY_FRACTION
        prob_neutral = neutral_mass
        scale = 1.0 - neutral_mass
        prob_bullish = prob_bullish * scale
        prob_bearish = prob_bearish * scale
        # Renormalise to sum = 1
        total = prob_bullish + prob_bearish + prob_neutral
        prob_bullish /= total
        prob_bearish /= total
        prob_neutral /= total

        timestamp = (
            self._price_history.index[-1]
            .isoformat()
        )

        payload = {
            "pair": self.instrument.replace("_", ""),
            "timestamp": timestamp,
            "prob_bullish": round(prob_bullish, 6),
            "prob_bearish": round(prob_bearish, 6),
            "prob_neutral": round(prob_neutral, 6),
            "confidence": round(confidence, 6),
            "volatility": round(volatility, 8),
            "regime": regime,
        }

        logger.info("Analytics payload: %s", payload)
        return payload

    def run_on_history(self) -> Optional[dict]:
        """Run inference directly on the loaded history without streaming.

        Useful for batch analytics or back-testing the pipeline.

        Returns
        -------
        dict or None
            Latest analytics payload.
        """
        return self._run_inference()
