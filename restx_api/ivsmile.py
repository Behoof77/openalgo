"""IV Smile API Endpoints

Serves Implied Volatility Smile data computed from the option chain.
Endpoints:
    POST /api/v1/ivsmile - Get IV Smile data for all strikes
"""

import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.iv_smile_service import get_iv_smile_data
from utils.logging import get_logger

from .data_schemas import IvSmileSchema

logger = get_logger(__name__)

api = Namespace("ivsmile", description="IV Smile data operations")

API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class IvSmile(Resource):
    """IV Smile data resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get Implied Volatility Smile data for all strikes of an expiry."""
        try:
            schema = IvSmileSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = get_iv_smile_data(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                api_key=api_key,
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in IV Smile endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in IV Smile endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
