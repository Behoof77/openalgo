"""Custom Straddle API Endpoints

Serves custom straddle simulation (PnL) data.
Endpoints:
    POST /api/v1/straddlepnl/simulate - Simulate a custom straddle
"""

import os

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from limiter import limiter
from services.custom_straddle_service import get_custom_straddle_simulation
from utils.logging import get_logger

from .data_schemas import CustomStraddleSchema

logger = get_logger(__name__)

api = Namespace("straddlepnl", description="Custom straddle simulation operations")

API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


@api.route("/simulate", strict_slashes=False)
class CustomStraddleSimulation(Resource):
    """Custom straddle simulation resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Simulate a custom straddle with adjustment points and lot sizing."""
        try:
            schema = CustomStraddleSchema()
            data = schema.load(request.json)
            api_key = data["apikey"]

            success, response, status_code = get_custom_straddle_simulation(
                underlying=data["underlying"],
                exchange=data["exchange"],
                expiry_date=data["expiry_date"],
                interval=data["interval"],
                api_key=api_key,
                days=data["days"],
                adjustment_points=data["adjustment_points"],
                lot_size=data["lot_size"],
                lots=data["lots"],
            )
            return response, status_code
        except ValidationError as e:
            logger.warning(f"Validation error in Custom Straddle endpoint: {e.messages}")
            return (
                {"status": "error", "message": "Validation error", "errors": e.messages},
                400,
            )
        except Exception as e:
            logger.exception(f"Unexpected error in Custom Straddle endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
