"""Strategy Portfolio API Endpoints

Serves strategy portfolio entries (watchlist/name/underlying/legs) backed by
the strategy portfolio database.
Endpoints:
    POST   /api/v1/strategyportfolio       - Create a new portfolio entry
    POST   /api/v1/strategyportfolio/list - List portfolio entries (optional watchlist filter)
    POST   /api/v1/strategyportfolio/<id> - Get a single portfolio entry
    PUT    /api/v1/strategyportfolio/<id> - Update a portfolio entry
    DELETE /api/v1/strategyportfolio/<id> - Delete a portfolio entry
"""
import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from database.strategy_portfolio_db import (
    WATCHLISTS,
    delete_portfolio_entry,
    get_portfolio_entry,
    list_portfolio,
    save_portfolio_entry,
)
from limiter import limiter
from utils.logging import get_logger

from .data_schemas import StrategyPortfolioSchema

logger = get_logger(__name__)
api = Namespace("strategyportfolio", description="Strategy portfolio operations")
API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


def _validate_legs(legs):
    """Validate a legs list from a schema payload, returning an error string or None."""
    if not isinstance(legs, list) or not legs:
        return "legs must be a non-empty list"
    return None


@api.route("/", strict_slashes=False)
class StrategyPortfolioCreate(Resource):
    """Strategy portfolio creation resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Create a new strategy portfolio entry."""
        try:
            schema = StrategyPortfolioSchema()
            data = schema.load(request.json)
            legs = data["legs"]
            legs_error = _validate_legs(legs)
            if legs_error:
                return {"status": "error", "message": legs_error}, 400

            entry = save_portfolio_entry(
                name=data["name"],
                watchlist=data["watchlist"],
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry=data.get("expiry"),
                legs=legs,
                notes=data.get("notes"),
            )
            if not entry:
                return {"status": "error", "message": "Failed to create portfolio entry"}, 400
            return {"status": "success", "data": {"entry": entry}}, 200
        except ValidationError as e:
            logger.warning(f"Validation error in Strategy Portfolio create endpoint: {e.messages}")
            return ({"status": "error", "message": "Validation error", "errors": e.messages}, 400)
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy Portfolio create endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/list", strict_slashes=False)
class StrategyPortfolioList(Resource):
    """Strategy portfolio listing resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """List strategy portfolio entries, optionally filtered by watchlist."""
        try:
            body = request.json or {}
            watchlist = body.get("watchlist")
            if watchlist is not None and watchlist not in WATCHLISTS:
                return {"status": "error", "message": f"Invalid watchlist. Must be one of {', '.join(WATCHLISTS)}"}, 400
            entries = list_portfolio(watchlist=watchlist)
            return {"status": "success", "data": {"entries": entries}}, 200
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy Portfolio list endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/<int:entry_id>", strict_slashes=False)
class StrategyPortfolioDetail(Resource):
    """Strategy portfolio detail resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self, entry_id):
        """Get a single strategy portfolio entry."""
        try:
            entry = get_portfolio_entry(entry_id)
            if not entry:
                return {"status": "error", "message": "Portfolio entry not found"}, 404
            return {"status": "success", "data": {"entry": entry}}, 200
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy Portfolio detail endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500

    @limiter.limit(API_RATE_LIMIT)
    def put(self, entry_id):
        """Update a strategy portfolio entry."""
        try:
            schema = StrategyPortfolioSchema()
            data = schema.load(request.json)
            legs = data["legs"]
            legs_error = _validate_legs(legs)
            if legs_error:
                return {"status": "error", "message": legs_error}, 400

            entry = save_portfolio_entry(
                name=data["name"],
                watchlist=data["watchlist"],
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry=data.get("expiry"),
                legs=legs,
                notes=data.get("notes"),
                entry_id=entry_id,
            )
            if not entry:
                return {"status": "error", "message": "Portfolio entry not found or invalid"}, 404
            return {"status": "success", "data": {"entry": entry}}, 200
        except ValidationError as e:
            logger.warning(f"Validation error in Strategy Portfolio update endpoint: {e.messages}")
            return ({"status": "error", "message": "Validation error", "errors": e.messages}, 400)
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy Portfolio update endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500

    @limiter.limit(API_RATE_LIMIT)
    def delete(self, entry_id):
        """Delete a strategy portfolio entry."""
        try:
            if delete_portfolio_entry(entry_id):
                return {"status": "success", "message": "Portfolio entry deleted"}, 200
            return {"status": "error", "message": "Portfolio entry not found"}, 404
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy Portfolio delete endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
