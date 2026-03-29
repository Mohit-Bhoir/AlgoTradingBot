"""
Forex Market Intelligence SaaS — Flask application factory.

Usage
-----
::

    from app import create_app
    app = create_app()
    app.run()
"""

import logging
import os

from flask import Flask

logger = logging.getLogger(__name__)


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    secret = os.environ.get("SECRET_KEY")
    if not secret:
        logger.warning(
            "SECRET_KEY is not set — using an insecure default. "
            "Set the SECRET_KEY environment variable in production."
        )
        secret = "change-me-in-production"
    app.config["SECRET_KEY"] = secret

    # ── Register blueprints ──────────────────────────────────────────────
    from app.api.routes import api_bp  # noqa: E402 — deferred to avoid circular imports

    app.register_blueprint(api_bp)

    return app
