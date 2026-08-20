"""Arbitrage API Endpoints

Serves the arbitrage universe (near/far futures pairs) built from the
master contract database.
Endpoints:
    POST /api/v1/arbitrage - Get arbitrage pairs across supported exchanges
"""
import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.arbitrage_service import get_arbitrage_universe
from utils.logging import get_logger

from .data_schemas import ArbitrageSchema

logger = get_logger(__name__)
api = Namespace("arbitrage", description="Arbitrage universe data operations")
API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class Arbitrage(Resource):
    """Arbitrage universe resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get the arbitrage universe for supported exchanges."""
        try:
            schema = ArbitrageSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]
            exchanges = data.get("exchanges")
            if exchanges:
                success, response, status_code = get_arbitrage_universe(
                    exchanges=exchanges, api_key=api_key
                )
            else:
                success, response, status_code = get_arbitrage_universe(api_key=api_key)
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in Arbitrage endpoint: {e.messages}")
            return ({"status": "error", "message": "Validation error", "errors": e.messages}, 400)
        except Exception as e:
            logger.exception(f"Unexpected error in Arbitrage endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
