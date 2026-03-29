"""
Forex Market Intelligence SaaS — Flask application factory.

Usage
-----
::

    from app import create_app
    app = create_app()
    app.run()
"""

import os

from flask import Flask


def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__)

    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "change-me-in-production")

    # ── Register blueprints ──────────────────────────────────────────────
    from app.api.routes import api_bp  # noqa: E402 — deferred to avoid circular imports

    app.register_blueprint(api_bp)

    return app
