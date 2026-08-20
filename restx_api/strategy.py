"""Strategy API Endpoints

Serves strategy management operations backed by the strategy database,
including symbol mappings, activation toggling, trading time updates and
square-off scheduling.
Endpoints:
    POST   /api/v1/strategy       - Create a new strategy
    POST   /api/v1/strategy/list  - List strategies for the authenticated user
    POST   /api/v1/strategy/<id>  - Get a single strategy with its symbol mappings
    POST   /api/v1/strategy/<id>/toggle    - Toggle strategy active status
    DELETE /api/v1/strategy/<id>           - Delete a strategy
    POST   /api/v1/strategy/<id>/symbols   - Add symbol mappings to a strategy
    DELETE /api/v1/strategy/<id>/symbol/<mapping_id> - Delete a symbol mapping
    POST   /api/v1/strategy/<id>/times     - Update strategy trading times
"""
import os
import uuid

from flask import request
from flask_restx import Namespace, Resource
from marshmallow import ValidationError

from blueprints.strategy import (
    schedule_squareoff,
    scheduler,
    validate_strategy_name,
    validate_strategy_times,
)
from database.auth_db import verify_api_key
from database.strategy_db import (
    bulk_add_symbol_mappings,
    create_strategy,
    delete_strategy,
    delete_symbol_mapping,
    get_strategy,
    get_symbol_mappings,
    get_user_strategies,
    toggle_strategy,
    update_strategy_times,
)
from limiter import limiter
from utils.logging import get_logger

from .data_schemas import StrategyCreateSchema, StrategySymbolsSchema, StrategyTimesSchema

logger = get_logger(__name__)
api = Namespace("strategy", description="Strategy management operations")
API_RATE_LIMIT = os.getenv("API_RATE_LIMIT", "10 per second")


def _serialize_strategy(strategy):
    """Serialize a Strategy model to a JSON-safe dict."""
    return {
        "id": strategy.id,
        "name": strategy.name,
        "webhook_id": strategy.webhook_id,
        "is_active": strategy.is_active,
        "is_intraday": strategy.is_intraday,
        "trading_mode": strategy.trading_mode,
        "platform": strategy.platform,
        "start_time": strategy.start_time,
        "end_time": strategy.end_time,
        "squareoff_time": strategy.squareoff_time,
        "created_at": strategy.created_at.isoformat() if strategy.created_at else None,
        "updated_at": strategy.updated_at.isoformat() if strategy.updated_at else None,
    }


def _serialize_mapping(mapping):
    """Serialize a StrategySymbolMapping model to a JSON-safe dict."""
    return {
        "id": mapping.id,
        "symbol": mapping.symbol,
        "exchange": mapping.exchange,
        "quantity": mapping.quantity,
        "product_type": mapping.product_type,
        "created_at": mapping.created_at.isoformat() if mapping.created_at else None,
    }


def _authenticate_user(data):
    """Resolve the authenticated user from the apikey or return None."""
    return verify_api_key(data["apikey"])


def _get_owned_strategy(strategy_id, user_id):
    """Return the strategy if it exists and belongs to the user, else None."""
    strategy = get_strategy(strategy_id)
    if not strategy or strategy.user_id != user_id:
        return None
    return strategy


@api.route("/", strict_slashes=False)
class StrategyCreate(Resource):
    """Strategy creation resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """Create a new strategy."""
        try:
            schema = StrategyCreateSchema()
            data = schema.load(request.json)
            user_id = _authenticate_user(data)
            if not user_id:
                return {"status": "error", "message": "Invalid API key"}, 401

            platform = data["platform"].strip()
            name = data["name"].strip()
            strategy_type = data["strategy_type"]
            trading_mode = data["trading_mode"]
            start_time = data.get("start_time")
            end_time = data.get("end_time")
            squareoff_time = data.get("squareoff_time")

            full_name = f"{platform}_{name}"
            valid_name, name_error = validate_strategy_name(full_name)
            if not valid_name:
                return {"status": "error", "message": name_error or "Invalid strategy name"}, 400

            is_intraday = strategy_type == "intraday"
            if is_intraday:
                valid_times, times_error = validate_strategy_times(start_time, end_time, squareoff_time)
                if not valid_times:
                    return {"status": "error", "message": times_error or "Invalid trading times"}, 400
            else:
                start_time = end_time = squareoff_time = None

            webhook_id = str(uuid.uuid4())
            strategy = create_strategy(
                name=full_name,
                webhook_id=webhook_id,
                user_id=user_id,
                is_intraday=is_intraday,
                trading_mode=trading_mode,
                start_time=start_time,
                end_time=end_time,
                squareoff_time=squareoff_time,
                platform=platform,
            )
            if not strategy:
                return {"status": "error", "message": "Failed to create strategy"}, 500

            if is_intraday and squareoff_time:
                schedule_squareoff(strategy.id)

            return {"status": "success", "data": {"strategy_id": strategy.id, "webhook_id": strategy.webhook_id}}, 200
        except ValidationError as e:
            logger.warning(f"Validation error in Strategy create endpoint: {e.messages}")
            return ({"status": "error", "message": "Validation error", "errors": e.messages}, 400)
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy create endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/list", strict_slashes=False)
class StrategyList(Resource):
    """Strategy listing resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self):
        """List strategies for the authenticated user."""
        try:
            data = {"apikey": (request.json or {}).get("apikey", "")}
            if not data["apikey"]:
                return {"status": "error", "message": "apikey is required"}, 400
            user_id = _authenticate_user(data)
            if not user_id:
                return {"status": "error", "message": "Invalid API key"}, 401

            strategies = get_user_strategies(user_id)
            return {"status": "success", "data": {"strategies": [_serialize_strategy(s) for s in strategies]}}, 200
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy list endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/<int:strategy_id>", strict_slashes=False)
class StrategyDetail(Resource):
    """Strategy detail resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self, strategy_id):
        """Get a single strategy with its symbol mappings."""
        try:
            data = {"apikey": (request.json or {}).get("apikey", "")}
            if not data["apikey"]:
                return {"status": "error", "message": "apikey is required"}, 400
            user_id = _authenticate_user(data)
            if not user_id:
                return {"status": "error", "message": "Invalid API key"}, 401

            strategy = _get_owned_strategy(strategy_id, user_id)
            if not strategy:
                return {"status": "error", "message": "Strategy not found"}, 404

            mappings = get_symbol_mappings(strategy_id)
            return {
                "status": "success",
                "data": {
                    "strategy": _serialize_strategy(strategy),
                    "mappings": [_serialize_mapping(m) for m in mappings],
                },
            }, 200
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy detail endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500

    @limiter.limit(API_RATE_LIMIT)
    def delete(self, strategy_id):
        """Delete a strategy."""
        try:
            data = {"apikey": (request.json or {}).get("apikey", "")}
            if not data["apikey"]:
                return {"status": "error", "message": "apikey is required"}, 400
            user_id = _authenticate_user(data)
            if not user_id:
                return {"status": "error", "message": "Invalid API key"}, 401

            strategy = _get_owned_strategy(strategy_id, user_id)
            if not strategy:
                return {"status": "error", "message": "Strategy not found"}, 404

            try:
                scheduler.remove_job(f"squareoff_{strategy_id}")
            except Exception:
                pass

            if delete_strategy(strategy_id):
                return {"status": "success", "message": "Strategy deleted"}, 200
            return {"status": "error", "message": "Failed to delete strategy"}, 500
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy delete endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/<int:strategy_id>/toggle", strict_slashes=False)
class StrategyToggle(Resource):
    """Strategy activation toggle resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self, strategy_id):
        """Toggle strategy active status."""
        try:
            data = {"apikey": (request.json or {}).get("apikey", "")}
            if not data["apikey"]:
                return {"status": "error", "message": "apikey is required"}, 400
            user_id = _authenticate_user(data)
            if not user_id:
                return {"status": "error", "message": "Invalid API key"}, 401

            strategy = _get_owned_strategy(strategy_id, user_id)
            if not strategy:
                return {"status": "error", "message": "Strategy not found"}, 404

            updated = toggle_strategy(strategy_id)
            if not updated:
                return {"status": "error", "message": "Failed to toggle strategy"}, 500

            if updated.is_active:
                schedule_squareoff(strategy_id)
            else:
                try:
                    scheduler.remove_job(f"squareoff_{strategy_id}")
                except Exception:
                    pass

            return {"status": "success", "data": {"is_active": updated.is_active}}, 200
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy toggle endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/<int:strategy_id>/symbols", strict_slashes=False)
class StrategySymbols(Resource):
    """Strategy symbol mapping resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self, strategy_id):
        """Add symbol mappings to a strategy."""
        try:
            schema = StrategySymbolsSchema()
            data = schema.load(request.json)
            user_id = _authenticate_user(data)
            if not user_id:
                return {"status": "error", "message": "Invalid API key"}, 401

            strategy = _get_owned_strategy(strategy_id, user_id)
            if not strategy:
                return {"status": "error", "message": "Strategy not found"}, 404

            mappings = []
            for item in data["symbols"]:
                symbol = item.get("symbol")
                exchange = item.get("exchange")
                quantity = item.get("quantity")
                product_type = item.get("product_type")
                if not all([symbol, exchange, quantity, product_type]):
                    return {"status": "error", "message": "Each symbol requires symbol, exchange, quantity and product_type"}, 400
                try:
                    quantity = int(quantity)
                except (TypeError, ValueError):
                    return {"status": "error", "message": "Quantity must be a valid number"}, 400
                if quantity <= 0:
                    return {"status": "error", "message": "Quantity must be greater than 0"}, 400
                mappings.append(
                    {
                        "symbol": symbol.strip(),
                        "exchange": exchange.strip(),
                        "quantity": quantity,
                        "product_type": product_type.strip(),
                    }
                )

            if bulk_add_symbol_mappings(strategy_id, mappings):
                return {"status": "success", "message": "Symbol mappings added"}, 200
            return {"status": "error", "message": "Failed to add symbol mappings"}, 500
        except ValidationError as e:
            logger.warning(f"Validation error in Strategy symbols endpoint: {e.messages}")
            return ({"status": "error", "message": "Validation error", "errors": e.messages}, 400)
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy symbols endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/<int:strategy_id>/symbol/<int:mapping_id>", strict_slashes=False)
class StrategySymbolDetail(Resource):
    """Strategy symbol mapping detail resource."""

    @limiter.limit(API_RATE_LIMIT)
    def delete(self, strategy_id, mapping_id):
        """Delete a symbol mapping from a strategy."""
        try:
            data = {"apikey": (request.json or {}).get("apikey", "")}
            if not data["apikey"]:
                return {"status": "error", "message": "apikey is required"}, 400
            user_id = _authenticate_user(data)
            if not user_id:
                return {"status": "error", "message": "Invalid API key"}, 401

            strategy = _get_owned_strategy(strategy_id, user_id)
            if not strategy:
                return {"status": "error", "message": "Strategy not found"}, 404

            if delete_symbol_mapping(mapping_id):
                return {"status": "success", "message": "Symbol mapping deleted"}, 200
            return {"status": "error", "message": "Symbol mapping not found"}, 404
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy symbol delete endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500


@api.route("/<int:strategy_id>/times", strict_slashes=False)
class StrategyTimes(Resource):
    """Strategy trading times update resource."""

    @limiter.limit(API_RATE_LIMIT)
    def post(self, strategy_id):
        """Update strategy trading times."""
        try:
            schema = StrategyTimesSchema()
            data = schema.load(request.json)
            user_id = _authenticate_user(data)
            if not user_id:
                return {"status": "error", "message": "Invalid API key"}, 401

            strategy = _get_owned_strategy(strategy_id, user_id)
            if not strategy:
                return {"status": "error", "message": "Strategy not found"}, 404

            start_time = data.get("start_time")
            end_time = data.get("end_time")
            squareoff_time = data.get("squareoff_time")

            if strategy.is_intraday:
                valid_times, times_error = validate_strategy_times(start_time, end_time, squareoff_time)
                if not valid_times:
                    return {"status": "error", "message": times_error or "Invalid trading times"}, 400

            if update_strategy_times(strategy_id, start_time, end_time, squareoff_time):
                if strategy.is_intraday and squareoff_time:
                    schedule_squareoff(strategy_id)
                return {"status": "success", "message": "Trading times updated"}, 200
            return {"status": "error", "message": "Failed to update trading times"}, 500
        except ValidationError as e:
            logger.warning(f"Validation error in Strategy times endpoint: {e.messages}")
            return ({"status": "error", "message": "Validation error", "errors": e.messages}, 400)
        except Exception as e:
            logger.exception(f"Unexpected error in Strategy times endpoint: {e}")
            return {"status": "error", "message": "An unexpected error occurred"}, 500
