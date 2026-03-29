"""
models.py
---------
SQLAlchemy ORM models for the Forex Market Intelligence SaaS.

Tables
------
- **users** — registered platform users.
- **subscriptions** — Stripe billing state per user.
- **market_states** — analytics snapshots produced by ``engine.py``.
- **trade_journal** — user-submitted trade notes and entries.
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
)
from sqlalchemy.orm import relationship

from app.core.database import Base


# ── Users ────────────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    email = Column(String(255), unique=True, nullable=False)
    password_hash = Column(String(255), nullable=False)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    subscriptions = relationship("Subscription", back_populates="user")
    trade_entries = relationship("TradeJournal", back_populates="user")

    def __repr__(self):
        return f"<User id={self.id} email={self.email!r}>"


# ── Subscriptions ────────────────────────────────────────────────────────────

class Subscription(Base):
    __tablename__ = "subscriptions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    stripe_customer_id = Column(String(255))
    stripe_subscription_id = Column(String(255))
    status = Column(String(50), nullable=False, default="active")

    user = relationship("User", back_populates="subscriptions")

    def __repr__(self):
        return (
            f"<Subscription id={self.id} user_id={self.user_id} "
            f"status={self.status!r}>"
        )


# ── Market States ────────────────────────────────────────────────────────────

class MarketState(Base):
    __tablename__ = "market_states"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime(timezone=True), nullable=False)
    pair = Column(String(20), nullable=False)
    prob_bullish = Column(Float, nullable=False)
    prob_bearish = Column(Float, nullable=False)
    prob_neutral = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    volatility = Column(Float, nullable=False)
    regime = Column(String(20), nullable=False)

    __table_args__ = (
        Index("ix_market_states_timestamp", "timestamp"),
        Index("ix_market_states_pair", "pair"),
        Index("ix_market_states_timestamp_pair", "timestamp", "pair"),
    )

    def __repr__(self):
        return (
            f"<MarketState id={self.id} pair={self.pair!r} "
            f"regime={self.regime!r}>"
        )


# ── Trade Journal ────────────────────────────────────────────────────────────

class TradeJournal(Base):
    __tablename__ = "trade_journal"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    entry_price = Column(Numeric(precision=18, scale=8), nullable=False)
    stop_loss = Column(Numeric(precision=18, scale=8))
    notes = Column(Text)
    created_at = Column(
        DateTime(timezone=True),
        default=lambda: datetime.now(timezone.utc),
        nullable=False,
    )

    user = relationship("User", back_populates="trade_entries")

    def __repr__(self):
        return (
            f"<TradeJournal id={self.id} user_id={self.user_id} "
            f"entry_price={self.entry_price}>"
        )
