"""OI Tracker API Endpoints

Serves Open Interest and Max Pain data for option chains.
Endpoints:
    POST /api/v1/oitracker - Get OI data for all strikes
    POST /api/v1/oitracker/maxpain - Calculate Max Pain
"""

import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.oi_tracker_service import calculate_max_pain, get_oi_data
from utils.logging import get_logger

from .data_schemas import MaxPainSchema, OiDataSchema

logger = get_logger(__name__)

api = Namespace("oitracker", description="Open Interest and Max Pain data operations")

API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class OiData(Resource):
    """Open Interest data resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get Open Interest data for all strikes of an expiry."""
        try:
            schema = OiDataSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = get_oi_data(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                api_key=api_key,
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in OI data endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in OI data endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/maxpain", strict_slashes=False)
class MaxPain(Resource):
    """Max Pain resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Calculate Max Pain for an underlying/expiry."""
        try:
            schema = MaxPainSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = calculate_max_pain(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                api_key=api_key,
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in Max Pain endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in Max Pain endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
