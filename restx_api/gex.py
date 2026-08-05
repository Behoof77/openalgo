"""GEX (Gamma Exposure) API Endpoints

Serves Gamma Exposure data computed from the option chain and Black-76 greeks.
Endpoints:
    POST /api/v1/gex - Get GEX data for all strikes of an expiry
"""

import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.gex_service import get_gex_data
from utils.logging import get_logger

from .data_schemas import GexSchema

logger = get_logger(__name__)

api = Namespace("gex", description="GEX (Gamma Exposure) data operations")

API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class GEX(Resource):
    """GEX data resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get Gamma Exposure data for all strikes of an expiry."""
        try:
            schema = GexSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = get_gex_data(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                api_key=api_key,
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in GEX endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in GEX endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
