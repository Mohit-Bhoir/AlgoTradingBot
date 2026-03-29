"""
routes.py
---------
REST API endpoints for the Forex Market Intelligence SaaS.

Endpoints
~~~~~~~~~
- ``GET  /api/market-state/latest``  — latest market snapshot
- ``GET  /api/market-state/history`` — historical market snapshots
- ``POST /api/auth/register``        — create a new user
- ``POST /api/auth/login``           — authenticate and receive JWT
- ``GET  /api/user/dashboard``       — authenticated dashboard data

Example requests (curl)
~~~~~~~~~~~~~~~~~~~~~~~
::

    # Latest market state (default pair EURUSD)
    curl http://localhost:5000/api/market-state/latest

    # Latest for a specific pair
    curl http://localhost:5000/api/market-state/latest?pair=GBPUSD

    # History (last 50 records)
    curl http://localhost:5000/api/market-state/history?pair=EURUSD&limit=50

    # Register
    curl -X POST http://localhost:5000/api/auth/register \
         -H "Content-Type: application/json" \
         -d '{"email":"user@example.com","password":"secret123"}'

    # Login
    curl -X POST http://localhost:5000/api/auth/login \
         -H "Content-Type: application/json" \
         -d '{"email":"user@example.com","password":"secret123"}'

    # Dashboard (use token from login response)
    curl http://localhost:5000/api/user/dashboard \
         -H "Authorization: Bearer <token>"
"""

import datetime
import functools
import logging

import jwt
from flask import Blueprint, current_app, jsonify, request
from sqlalchemy import desc, exists
from werkzeug.security import check_password_hash, generate_password_hash

from app.core.database import SessionLocal
from app.core.models import MarketState, User

logger = logging.getLogger(__name__)

api_bp = Blueprint("api", __name__, url_prefix="/api")

_JWT_ALGORITHM = "HS256"
_JWT_EXPIRATION_HOURS = 24


# ── Helpers ──────────────────────────────────────────────────────────────────

def _market_state_to_dict(ms: MarketState) -> dict:
    """Serialise a MarketState ORM instance to a JSON-safe dictionary."""
    return {
        "pair": ms.pair,
        "timestamp": ms.timestamp.isoformat(),
        "prob_bullish": ms.prob_bullish,
        "prob_bearish": ms.prob_bearish,
        "prob_neutral": ms.prob_neutral,
        "confidence": ms.confidence,
        "volatility": ms.volatility,
        "regime": ms.regime,
    }


def _get_secret_key() -> str:
    """Return the secret key from the Flask application config."""
    return current_app.config["SECRET_KEY"]


def _create_token(user_id: int) -> str:
    """Create a signed JWT for the given user."""
    now = datetime.datetime.now(datetime.timezone.utc)
    payload = {
        "user_id": user_id,
        "exp": now + datetime.timedelta(hours=_JWT_EXPIRATION_HOURS),
        "iat": now,
    }
    return jwt.encode(payload, _get_secret_key(), algorithm=_JWT_ALGORITHM)


def _decode_token(token: str) -> dict:
    """Decode and verify a JWT.  Raises on failure."""
    return jwt.decode(token, _get_secret_key(), algorithms=[_JWT_ALGORITHM])


def token_required(f):
    """Decorator that enforces a valid ``Authorization: Bearer <token>``."""

    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Missing or invalid Authorization header"}), 401

        token = auth_header.split(" ", 1)[1]
        try:
            data = _decode_token(token)
        except jwt.ExpiredSignatureError:
            return jsonify({"error": "Token has expired"}), 401
        except jwt.InvalidTokenError:
            return jsonify({"error": "Invalid token"}), 401

        kwargs["current_user_id"] = data["user_id"]
        return f(*args, **kwargs)

    return wrapper


# ── Market-state endpoints ───────────────────────────────────────────────────

@api_bp.route("/market-state/latest", methods=["GET"])
def market_state_latest():
    """Return the most recent market state for the given currency pair."""
    pair = request.args.get("pair", "EURUSD")

    session = SessionLocal()
    try:
        ms = (
            session.query(MarketState)
            .filter(MarketState.pair == pair)
            .order_by(desc(MarketState.timestamp))
            .first()
        )
        if ms is None:
            return jsonify({"error": f"No market state found for pair {pair}"}), 404
        return jsonify(_market_state_to_dict(ms)), 200
    finally:
        session.close()


@api_bp.route("/market-state/history", methods=["GET"])
def market_state_history():
    """Return historical market states ordered by timestamp descending."""
    pair = request.args.get("pair", "EURUSD")
    limit = request.args.get("limit", 100, type=int)
    limit = max(1, min(limit, 1000))  # clamp between 1 and 1000

    session = SessionLocal()
    try:
        rows = (
            session.query(MarketState)
            .filter(MarketState.pair == pair)
            .order_by(desc(MarketState.timestamp))
            .limit(limit)
            .all()
        )
        return jsonify([_market_state_to_dict(ms) for ms in rows]), 200
    finally:
        session.close()


# ── Auth endpoints ───────────────────────────────────────────────────────────

@api_bp.route("/auth/register", methods=["POST"])
def auth_register():
    """Register a new user with email and password."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    session = SessionLocal()
    try:
        if session.query(exists().where(User.email == email)).scalar():
            return jsonify({"error": "Email already registered"}), 409

        user = User(
            email=email,
            password_hash=generate_password_hash(password),
        )
        session.add(user)
        session.commit()
        session.refresh(user)

        return jsonify({"message": "User registered", "user_id": user.id}), 201
    except Exception:
        session.rollback()
        logger.exception("Failed to register user %s", email)
        return jsonify({"error": "Registration failed"}), 500
    finally:
        session.close()


@api_bp.route("/auth/login", methods=["POST"])
def auth_login():
    """Authenticate a user and return a JWT."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    email = data.get("email", "").strip()
    password = data.get("password", "")

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    session = SessionLocal()
    try:
        user = session.query(User).filter(User.email == email).first()
        if user is None or not check_password_hash(user.password_hash, password):
            return jsonify({"error": "Invalid credentials"}), 401

        token = _create_token(user.id)
        return jsonify({"token": token}), 200
    finally:
        session.close()


# ── User endpoints ───────────────────────────────────────────────────────────

@api_bp.route("/user/dashboard", methods=["GET"])
@token_required
def user_dashboard(*, current_user_id: int):
    """Return dashboard data for the authenticated user.

    Includes the latest market state and the last *N* historical records
    (controlled by the ``limit`` query parameter, default 10).
    """
    pair = request.args.get("pair", "EURUSD")
    limit = request.args.get("limit", 10, type=int)
    limit = max(1, min(limit, 100))

    session = SessionLocal()
    try:
        latest = (
            session.query(MarketState)
            .filter(MarketState.pair == pair)
            .order_by(desc(MarketState.timestamp))
            .first()
        )
        history = (
            session.query(MarketState)
            .filter(MarketState.pair == pair)
            .order_by(desc(MarketState.timestamp))
            .limit(limit)
            .all()
        )

        return jsonify({
            "user_id": current_user_id,
            "latest": _market_state_to_dict(latest) if latest else None,
            "history": [_market_state_to_dict(ms) for ms in history],
        }), 200
    finally:
        session.close()


# ── Error handlers ───────────────────────────────────────────────────────────

@api_bp.errorhandler(500)
def handle_internal_error(exc):
    logger.exception("Internal server error: %s", exc)
    return jsonify({"error": "Internal server error"}), 500


@api_bp.errorhandler(404)
def handle_not_found(exc):
    return jsonify({"error": "Resource not found"}), 404
