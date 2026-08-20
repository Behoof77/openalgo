"""Gamma Density API Endpoints

Serves Gamma Density data computed from the option chain and Black-76 greeks.
Endpoints:
    POST /api/v1/gammadensity - Get gamma density data for an underlying and expiry
"""
import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.gamma_density_service import calculate_gamma_density
from utils.logging import get_logger

from .data_schemas import GammaDensitySchema

logger = get_logger(__name__)
api = Namespace("gammadensity", description="Gamma Density data operations")
API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/", strict_slashes=False)
class GammaDensity(Resource):
    """Gamma Density data resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Get Gamma Density data for an underlying and expiry."""
        try:
            schema = GammaDensitySchema()
            data = schema.load(request.json)
            api_key = data["apikey"]
            success, response, status_code = calculate_gamma_density(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                api_key=api_key,
                interest_rate=data.get("interest_rate"),
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in Gamma Density endpoint: {e.messages}")
            return ({"status": "error", "message": "Validation error", "errors": e.messages}, 400)
        except Exception as e:
            logger.exception(f"Unexpected error in Gamma Density endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
