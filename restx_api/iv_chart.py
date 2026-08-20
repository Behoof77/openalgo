"""IV Chart API Endpoints

Serves intraday Implied Volatility chart data.
Endpoints:
    POST /api/v1/ivchart - Get intraday IV data
    POST /api/v1/ivchart/default-symbols - Get default ATM option symbols
"""

import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.iv_chart_service import get_default_symbols, get_iv_chart_data
from utils.logging import get_logger

from .data_schemas import DefaultSymbolsSchema, IvChartSchema

logger = get_logger(__name__)

api = Namespace("ivchart", description="Intraday Implied Volatility chart data operations")

API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class IvChart(Resource):
    """IV chart data resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get intraday Implied Volatility data for an underlying/expiry."""
        try:
            schema = IvChartSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = get_iv_chart_data(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                interval=data["interval"],
                api_key=api_key,
                days=data["days"],
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in IV Chart endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in IV Chart endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/default-symbols", strict_slashes=False)
class DefaultSymbols(Resource):
    """Default ATM option symbols resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get default ATM option symbols for an underlying/expiry."""
        try:
            schema = DefaultSymbolsSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = get_default_symbols(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                api_key=api_key,
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in Default Symbols endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in Default Symbols endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
