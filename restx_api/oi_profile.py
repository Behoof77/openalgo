"""OI Profile API Endpoints

Serves Open Interest Profile data with an intraday futures panel.
Endpoints:
    POST /api/v1/oiprofile - Get OI Profile data
"""

import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.oi_profile_service import get_oi_profile_data
from utils.logging import get_logger

from .data_schemas import OiProfileSchema

logger = get_logger(__name__)

api = Namespace("oiprofile", description="OI Profile data operations")

API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class OiProfile(Resource):
    """OI Profile data resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get Open Interest Profile data for an underlying/expiry."""
        try:
            schema = OiProfileSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = get_oi_profile_data(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                interval=data["interval"],
                days=data["days"],
                api_key=api_key,
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in OI Profile endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in OI Profile endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
