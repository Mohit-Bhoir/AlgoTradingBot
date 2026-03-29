"""
features.py
-----------
Feature engineering for the Forex Market Intelligence engine.

All functions are pure transformations — they accept pandas Series / DataFrames
and return new objects, keeping them easy to test in isolation.
"""

import numpy as np
import pandas as pd

# ── Regime thresholds ────────────────────────────────────────────────────────
# A rolling volatility (std of log-returns) above this value → "volatile".
_VOLATILE_THRESHOLD = 0.0015
# When mean absolute log-return exceeds this value the market is "trending".
_TREND_THRESHOLD = 0.0003


def compute_returns(price_series: pd.Series) -> pd.Series:
    """Compute log returns from a price series.

    Parameters
    ----------
    price_series:
        Time-indexed series of mid-prices.

    Returns
    -------
    pd.Series
        Log-return series of the same length (first value is NaN).
    """
    return np.log(price_series / price_series.shift(1))


def compute_lag_features(returns: pd.Series, lags: int) -> pd.DataFrame:
    """Build a lag-feature DataFrame from a returns series.

    Parameters
    ----------
    returns:
        Series of log returns.
    lags:
        Number of lagged columns to generate (``returns_lag_1`` …
        ``returns_lag_<lags>``).

    Returns
    -------
    pd.DataFrame
        DataFrame whose columns are ``returns_lag_1`` through
        ``returns_lag_<lags>``.  Rows with NaN values (the first *lags* rows)
        are dropped.
    """
    feature_df = pd.DataFrame(index=returns.index)
    for lag in range(1, lags + 1):
        feature_df[f"returns_lag_{lag}"] = returns.shift(lag)
    return feature_df.dropna()


def compute_volatility(returns: pd.Series, window: int = 20) -> float:
    """Return the rolling realised volatility of the most-recent bar.

    Volatility is expressed as the standard deviation of log returns over
    the last *window* observations.  Returns ``0.0`` when there is
    insufficient data.

    Parameters
    ----------
    returns:
        Series of log returns.
    window:
        Look-back window in bars.
    """
    if len(returns) < 2:
        return 0.0
    tail = returns.iloc[-window:] if len(returns) >= window else returns
    return float(tail.std())


def detect_regime(returns: pd.Series, volatility: float) -> str:
    """Classify the current market regime.

    Rules (applied in priority order):

    1. **volatile** — rolling volatility exceeds ``_VOLATILE_THRESHOLD``.
    2. **trend**    — mean absolute log-return exceeds ``_TREND_THRESHOLD``
                     and the market is not classed as volatile.
    3. **range**    — catch-all for low-volatility, low-directional periods.

    Parameters
    ----------
    returns:
        Recent log-return series (last ~20 bars is sufficient).
    volatility:
        Pre-computed rolling volatility (output of :func:`compute_volatility`).

    Returns
    -------
    str
        One of ``"volatile"``, ``"trend"``, or ``"range"``.
    """
    if volatility > _VOLATILE_THRESHOLD:
        return "volatile"

    recent = returns.iloc[-20:] if len(returns) >= 20 else returns
    mean_abs_return = float(recent.abs().mean()) if len(recent) > 0 else 0.0

    if mean_abs_return > _TREND_THRESHOLD:
        return "trend"

    return "range"


def build_feature_row(
    price_series: pd.Series, lags: int
) -> tuple[pd.DataFrame, pd.Series, float, str]:
    """Convenience wrapper that runs the full feature-engineering pipeline.

    Parameters
    ----------
    price_series:
        Time-indexed mid-price series.
    lags:
        Number of lag features to generate.

    Returns
    -------
    feature_df : pd.DataFrame
        Lag-feature DataFrame (all rows, NaN-dropped).
    returns : pd.Series
        Full log-return series.
    volatility : float
        Current realised volatility.
    regime : str
        Current market regime string.
    """
    returns = compute_returns(price_series)
    feature_df = compute_lag_features(returns, lags)
    volatility = compute_volatility(returns)
    regime = detect_regime(returns, volatility)
    return feature_df, returns, volatility, regime
