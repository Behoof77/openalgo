"""Multi Strike OI API Endpoints

Serves Open Interest data for multiple strikes of an underlying across an
intraday time range.
Endpoints:
    POST /api/v1/multistrikeoi - Get OI data for multiple option legs
"""
import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.multi_strike_oi_service import get_multi_strike_oi_data
from utils.logging import get_logger

from .data_schemas import MultiStrikeOISchema

logger = get_logger(__name__)
api = Namespace("multistrikeoi", description="Multi strike Open Interest data operations")
API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class MultiStrikeOI(Resource):
    """Multi strike Open Interest data resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get Open Interest data for multiple option legs of an underlying."""
        try:
            schema = MultiStrikeOISchema()
            data = schema.load(request.json)
            api_key = data["apikey"]
            success, response, status_code = get_multi_strike_oi_data(
                underlying=data["underlying"],
                exchange=data["exchange"],
                legs=data["legs"],
                interval=data["interval"],
                api_key=api_key,
                days=data["days"],
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in Multi Strike OI endpoint: {e.messages}")
            return ({"status": "error", "message": "Validation error", "errors": e.messages}, 400)
        except Exception as e:
            logger.exception(f"Unexpected error in Multi Strike OI endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
