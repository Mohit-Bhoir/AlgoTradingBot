"""
market_state_store.py
---------------------
Persistence helper for saving ``engine.py`` analytics output into the
``market_states`` database table.

The :func:`save_market_state` function accepts the **exact** dictionary
returned by :meth:`ForexAnalyticsEngine._run_inference` and inserts
it as a new :class:`~app.core.models.MarketState` row.

Usage
-----
>>> from app.core.market_state_store import save_market_state
>>> payload = engine.run_on_history()   # dict from engine.py
>>> if payload:
...     save_market_state(payload)
"""

import logging
from datetime import datetime, timezone
from typing import Any, Dict

from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.core.models import MarketState

logger = logging.getLogger(__name__)


def save_market_state(
    payload: Dict[str, Any],
    session: Session | None = None,
) -> MarketState:
    """Persist an engine analytics payload to the ``market_states`` table.

    Parameters
    ----------
    payload : dict
        Dictionary produced by ``ForexAnalyticsEngine._run_inference()``.
        Expected keys: ``pair``, ``timestamp``, ``prob_bullish``,
        ``prob_bearish``, ``prob_neutral``, ``confidence``, ``volatility``,
        ``regime``.
    session : sqlalchemy.orm.Session, optional
        An existing database session.  When *None* a new session is created
        (and committed / closed automatically).

    Returns
    -------
    MarketState
        The newly created ORM instance (with ``id`` populated after commit).

    Raises
    ------
    KeyError
        If a required key is missing from *payload*.
    """
    _own_session = session is None
    if _own_session:
        session = SessionLocal()

    try:
        timestamp_raw = payload["timestamp"]
        if isinstance(timestamp_raw, str):
            timestamp_raw = datetime.fromisoformat(timestamp_raw)
        if timestamp_raw.tzinfo is None:
            timestamp_raw = timestamp_raw.replace(tzinfo=timezone.utc)

        market_state = MarketState(
            timestamp=timestamp_raw,
            pair=payload["pair"],
            prob_bullish=payload["prob_bullish"],
            prob_bearish=payload["prob_bearish"],
            prob_neutral=payload["prob_neutral"],
            confidence=payload["confidence"],
            volatility=payload["volatility"],
            regime=payload["regime"],
        )

        session.add(market_state)
        if _own_session:
            session.commit()
            session.refresh(market_state)
            logger.info("Saved MarketState id=%s for %s @ %s",
                        market_state.id, market_state.pair,
                        market_state.timestamp)
        return market_state
    except Exception:
        if _own_session:
            session.rollback()
        raise
    finally:
        if _own_session:
            session.close()
