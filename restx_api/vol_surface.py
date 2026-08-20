"""Volatility Surface API Endpoints

Serves 3D volatility surface data across expiries.
Endpoints:
    POST /api/v1/volsurface - Get Volatility Surface data
"""

import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.vol_surface_service import get_vol_surface_data
from utils.logging import get_logger

from .data_schemas import VolSurfaceSchema

logger = get_logger(__name__)

api = Namespace("volsurface", description="Volatility Surface data operations")

API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class VolSurface(Resource):
    """Volatility Surface data resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get 3D Volatility Surface data for a list of expiries."""
        try:
            schema = VolSurfaceSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = get_vol_surface_data(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_dates=data["expiry_dates"],
                strike_count=data["strike_count"],
                api_key=api_key,
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in Volatility Surface endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in Volatility Surface endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
