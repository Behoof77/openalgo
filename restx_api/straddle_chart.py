"""Straddle Chart API Endpoints

Serves Dynamic ATM Straddle chart data.
Endpoints:
    POST /api/v1/straddle - Get Dynamic ATM Straddle time series
"""

import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.straddle_chart_service import get_straddle_chart_data
from utils.logging import get_logger

from .data_schemas import StraddleChartSchema

logger = get_logger(__name__)

api = Namespace("straddle", description="Dynamic ATM Straddle chart data operations")

API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class StraddleChart(Resource):
    """Straddle chart data resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get Dynamic ATM Straddle time series for an underlying/expiry."""
        try:
            schema = StraddleChartSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = get_straddle_chart_data(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                interval=data["interval"],
                api_key=api_key,
                days=data["days"],
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in Straddle Chart endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in Straddle Chart endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
