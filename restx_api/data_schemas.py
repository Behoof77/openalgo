import re

from marshmallow import Schema, ValidationError, fields, validate

from utils.constants import VALID_EXCHANGES


# Custom validator for date or timestamp string
def validate_date_or_timestamp(data):
    """
    Validates that the input string is either in 'YYYY-MM-DD' format or a numeric timestamp.
    """
    date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
    timestamp_pattern = re.compile(r"^\d{10,13}$")  # Allows for seconds or milliseconds
    if not (isinstance(data, str) and (date_pattern.match(data) or timestamp_pattern.match(data))):
        raise ValidationError(
            "Field must be a string in 'YYYY-MM-DD' format or a numeric timestamp."
        )


# Custom validator for option offset
def validate_option_offset(data):
    """
    Validates option offset: ATM, ITM1-ITM50, OTM1-OTM50
    """
    data_upper = data.upper()
    if data_upper == "ATM":
        return True

    # Check for ITM pattern: ITM followed by 1-50
    itm_pattern = re.compile(r"^ITM([1-9]|[1-4][0-9]|50)$")
    otm_pattern = re.compile(r"^OTM([1-9]|[1-4][0-9]|50)$")

    if not (itm_pattern.match(data_upper) or otm_pattern.match(data_upper)):
        raise ValidationError("Offset must be ATM, ITM1-ITM50, or OTM1-OTM50")

    return True


class QuotesSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))
    symbol = fields.Str(required=True)  # Single symbol
    exchange = fields.Str(required=True, validate=validate.OneOf(VALID_EXCHANGES))  # Exchange (e.g., NSE, BSE)


class SymbolExchangePair(Schema):
    symbol = fields.Str(required=True)
    exchange = fields.Str(required=True, validate=validate.OneOf(VALID_EXCHANGES))


class MultiQuotesSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))
    symbols = fields.List(
        fields.Nested(SymbolExchangePair), required=True, validate=validate.Length(min=1)
    )


class HistorySchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))
    symbol = fields.Str(required=True)
    exchange = fields.Str(required=True, validate=validate.OneOf(VALID_EXCHANGES))  # Exchange (e.g., NSE, BSE)
    interval = fields.Str(
        required=True,
        validate=validate.OneOf(
            [
                # Seconds intervals
                "1s",
                "5s",
                "10s",
                "15s",
                "30s",
                "45s",
                # Minutes intervals
                "1m",
                "2m",
                "3m",
                "5m",
                "10m",
                "15m",
                "20m",
                "30m",
                # Hours intervals
                "1h",
                "2h",
                "3h",
                "4h",
                # Daily, Weekly, Monthly, Quarterly, Yearly intervals
                "D",
                "W",
                "M",
                "Q",
                "Y",
            ]
        ),
    )
    start_date = fields.Date(required=True, format="%Y-%m-%d")  # YYYY-MM-DD
    end_date = fields.Date(required=True, format="%Y-%m-%d")  # YYYY-MM-DD
    # Optional: Data source - 'api' (broker, default) or 'db' (DuckDB/Historify)
    source = fields.Str(required=False, load_default="api", validate=validate.OneOf(["api", "db"]))
    # OI is now always included by default for F&O exchanges


class DepthSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))
    symbol = fields.Str(required=True)
    exchange = fields.Str(required=True, validate=validate.OneOf(VALID_EXCHANGES))  # Exchange (e.g., NSE, BSE)


class IntervalsSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))


class SymbolSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    symbol = fields.Str(required=True)  # Symbol code (e.g., RELIANCE)
    exchange = fields.Str(required=True, validate=validate.OneOf(VALID_EXCHANGES))  # Exchange (e.g., NSE, BSE)


class TickerSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))
    symbol = fields.Str(required=True)  # Combined exchange:symbol format
    interval = fields.Str(
        required=True,
        validate=validate.OneOf(["1m", "5m", "15m", "30m", "1h", "4h", "D", "W", "M"]),
    )  # Supported intervals: 1m, 5m, 15m, 30m, 1h, 4h, D, W, M etc.
    from_ = fields.Str(
        data_key="from", required=True, validate=validate_date_or_timestamp
    )  # YYYY-MM-DD or millisecond timestamp
    to = fields.Str(
        required=True, validate=validate_date_or_timestamp
    )  # YYYY-MM-DD or millisecond timestamp
    adjusted = fields.Bool(required=False, default=True)  # Adjust for splits
    sort = fields.Str(
        required=False, default="asc", validate=validate.OneOf(["asc", "desc"])
    )  # Sort direction


class SearchSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    query = fields.Str(required=True)  # Search query/symbol name
    exchange = fields.Str(required=False, validate=validate.OneOf(VALID_EXCHANGES))  # Optional exchange filter (e.g., NSE, BSE)


class ExpirySchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    symbol = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(["NFO", "BFO", "MCX", "CDS", "CRYPTO"])
    )  # Exchange (e.g., NFO, BFO, MCX, CDS, CRYPTO)
    instrumenttype = fields.Str(
        required=True, validate=validate.OneOf(["futures", "options"])
    )  # futures or options


class OptionSymbolSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    strategy = fields.Str(
        required=False, allow_none=True
    )  # DEPRECATED: Strategy name (optional, will be removed in future versions)
    underlying = fields.Str(required=True)  # Underlying symbol (NIFTY, RELIANCE, NIFTY28OCT25FUT)
    exchange = fields.Str(required=True, validate=validate.OneOf(VALID_EXCHANGES))  # Exchange (NSE_INDEX, NSE, NFO)
    expiry_date = fields.Str(
        required=False
    )  # Expiry date in DDMMMYY format (e.g., 28OCT25). Optional if underlying includes expiry
    strike_int = fields.Int(
        required=False, validate=validate.Range(min=1), allow_none=True
    )  # OPTIONAL: Strike interval. If not provided, actual strikes from database will be used (RECOMMENDED for accuracy)
    offset = fields.Str(
        required=True, validate=validate_option_offset
    )  # Strike offset from ATM (ATM, ITM1-ITM50, OTM1-OTM50)
    option_type = fields.Str(
        required=True, validate=validate.OneOf(["CE", "PE", "ce", "pe"])
    )  # Call or Put option


class OptionGreeksSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    symbol = fields.Str(required=True)  # Option symbol (e.g., NIFTY28NOV2424000CE)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(["NFO", "BFO", "CDS", "MCX", "CRYPTO"])
    )  # Exchange (NFO, BFO, CDS, MCX, CRYPTO)
    interest_rate = fields.Float(
        required=False, validate=validate.Range(min=0, max=100)
    )  # Risk-free interest rate (annualized %). Optional, defaults per exchange
    forward_price = fields.Float(
        required=False, validate=validate.Range(min=0)
    )  # Optional: Custom forward/synthetic futures price. If provided, skips underlying price fetch
    underlying_symbol = fields.Str(
        required=False
    )  # Optional: Specify underlying symbol (e.g., NIFTY or NIFTY28NOV24FUT)
    underlying_exchange = fields.Str(
        required=False
    )  # Optional: Specify underlying exchange (NSE_INDEX, NFO, etc.)
    expiry_time = fields.Str(
        required=False
    )  # Optional: Custom expiry time in HH:MM format (e.g., "15:30", "19:00"). If not provided, uses exchange defaults


class InstrumentsSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    exchange = fields.Str(
        required=False,
        validate=validate.OneOf(VALID_EXCHANGES),
    )  # Optional exchange filter
    format = fields.Str(
        required=False, validate=validate.OneOf(["json", "csv"])
    )  # Output format (json or csv), defaults to json


class OptionChainSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY, RELIANCE)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(
        required=True
    )  # Expiry date in DDMMMYY format (e.g., 28NOV25) - MANDATORY
    strike_count = fields.Int(
        required=False, validate=validate.Range(min=1, max=100), allow_none=True
    )  # Number of strikes above/below ATM. If not provided, returns entire chain


class MarketHolidaysSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    year = fields.Int(
        required=False, validate=validate.Range(min=2020, max=2050)
    )  # Year to get holidays for (defaults to current year)


class MarketTimingsSchema(Schema):
    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    date = fields.Str(required=True)  # Date in YYYY-MM-DD format


class OptionSymbolRequest(Schema):
    """Schema for a single option symbol request in batch"""

    symbol = fields.Str(required=True)  # Option symbol (e.g., NIFTY28NOV2424000CE)
    exchange = fields.Str(required=True, validate=validate.OneOf(["NFO", "BFO", "CDS", "MCX", "CRYPTO"]))
    underlying_symbol = fields.Str(required=False)  # Optional: Specify underlying symbol
    underlying_exchange = fields.Str(required=False)  # Optional: Specify underlying exchange


class MultiOptionGreeksSchema(Schema):
    """Schema for batch option greeks requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    symbols = fields.List(
        fields.Nested(OptionSymbolRequest),
        required=True,
        validate=validate.Length(min=1, max=50),  # Max 50 symbols per request
    )
    interest_rate = fields.Float(
        required=False, validate=validate.Range(min=0, max=100)
    )  # Common interest rate for all
    expiry_time = fields.Str(required=False)  # Optional: Common expiry time for all


class GexSchema(Schema):
    """Schema for Gamma Exposure (GEX) data requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY, RELIANCE)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)


class IvSmileSchema(Schema):
    """Schema for Implied Volatility Smile data requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)


class OiDataSchema(Schema):
    """Schema for Open Interest data requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)


class MaxPainSchema(Schema):
    """Schema for Max Pain calculation requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)


class OiProfileSchema(Schema):
    """Schema for OI Profile data requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)
    interval = fields.Str(
        required=False, load_default="5m", validate=validate.OneOf(["1m", "5m", "15m"])
    )  # Candle interval for the futures panel
    days = fields.Int(
        required=False, load_default=5, validate=validate.Range(min=1, max=30)
    )  # Number of days of history to load


class StraddleChartSchema(Schema):
    """Schema for Dynamic ATM Straddle chart data requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)
    interval = fields.Str(required=False, load_default="1m")  # Candle interval (e.g., 1m, 5m, 15m)
    days = fields.Int(
        required=False, load_default=5, validate=validate.Range(min=1, max=30)
    )  # Number of days of history to load


class VolSurfaceSchema(Schema):
    """Schema for 3D Volatility Surface data requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_dates = fields.List(
        fields.Str(required=True), required=True, validate=validate.Length(min=1, max=8)
    )  # List of expiry dates in DDMMMYY format (e.g., 28NOV25), max 8
    strike_count = fields.Int(
        required=False, load_default=15, validate=validate.Range(min=5, max=40)
    )  # Number of strikes above and below ATM


class IvChartSchema(Schema):
    """Schema for intraday Implied Volatility chart data requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)
    interval = fields.Str(required=False, load_default="5m")  # Candle interval (e.g., 1m, 5m, 15m)
    days = fields.Int(
        required=False, load_default=1, validate=validate.Range(min=1, max=30)
    )  # Number of days of history to load


class DefaultSymbolsSchema(Schema):
    """Schema for ATM default symbol lookup requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)


class CustomStraddleSchema(Schema):
    """Schema for custom straddle simulation requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)
    interval = fields.Str(required=False, load_default="1m")  # Candle interval (e.g., 1m, 5m, 15m)
    days = fields.Int(
        required=False, load_default=1, validate=validate.Range(min=1, max=30)
    )  # Number of days of history to load
    adjustment_points = fields.Int(
        required=False, load_default=50, validate=validate.Range(min=1)
    )  # Straddle adjustment threshold in points
    lot_size = fields.Int(
        required=False, load_default=65, validate=validate.Range(min=1)
    )  # Contract lot size for PnL scaling
    lots = fields.Int(
        required=False, load_default=1, validate=validate.Range(min=1)
    )  # Number of lots to simulate


class GammaDensitySchema(Schema):
    """Schema for gamma density requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry_date = fields.Str(required=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)
    interest_rate = fields.Float(
        required=False, allow_none=True
    )  # Optional risk-free interest rate override (fraction, e.g. 0.065)


class ArbitrageSchema(Schema):
    """Schema for arbitrage universe requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    exchanges = fields.List(
        fields.Str(), required=False, allow_none=True
    )  # Optional exchange filter (e.g., NFO, MCX, BFO, CDS)


class MultiStrikeOISchema(Schema):
    """Schema for multi strike open interest requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    legs = fields.List(
        fields.Dict(), required=True, validate=validate.Length(min=1)
    )  # List of option legs (symbol, exchange, side, strike, optionType, expiry)
    interval = fields.Str(required=False, load_default="1m")  # Candle interval (e.g., 1m, 5m, 15m)
    days = fields.Int(
        required=False, load_default=5, validate=validate.Range(min=1, max=30)
    )  # Number of days of history to load


class StrategyCreateSchema(Schema):
    """Schema for strategy creation requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    platform = fields.Str(required=True)  # Platform type (e.g., tradingview, chartink)
    name = fields.Str(required=True)  # Strategy name
    strategy_type = fields.Str(required=False, load_default="intraday")  # Strategy type (intraday or positional)
    trading_mode = fields.Str(required=False, load_default="LONG")  # Trading mode (LONG, SHORT, or BOTH)
    start_time = fields.Str(required=False, allow_none=True)  # Entry window start (HH:MM, 24h)
    end_time = fields.Str(required=False, allow_none=True)  # Entry window end (HH:MM, 24h)
    squareoff_time = fields.Str(required=False, allow_none=True)  # Square off time (HH:MM, 24h)


class StrategySymbolsSchema(Schema):
    """Schema for adding symbol mappings to a strategy"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    symbols = fields.List(
        fields.Dict(), required=True, validate=validate.Length(min=1)
    )  # List of symbols with exchange, quantity, product_type


class StrategyTimesSchema(Schema):
    """Schema for updating strategy trading times"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    start_time = fields.Str(required=False, allow_none=True)  # Entry window start (HH:MM, 24h)
    end_time = fields.Str(required=False, allow_none=True)  # Entry window end (HH:MM, 24h)
    squareoff_time = fields.Str(required=False, allow_none=True)  # Square off time (HH:MM, 24h)


class StrategyPortfolioSchema(Schema):
    """Schema for strategy portfolio create/update requests"""

    apikey = fields.Str(required=True, validate=validate.Length(min=1, max=256))  # API Key for authentication
    name = fields.Str(required=True, validate=validate.Length(max=120))  # Portfolio entry name
    watchlist = fields.Str(
        required=True, validate=validate.OneOf(["mytrades", "simulation"])
    )  # Watchlist (mytrades or simulation)
    underlying = fields.Str(required=True)  # Underlying symbol (e.g., NIFTY, BANKNIFTY)
    exchange = fields.Str(
        required=True, validate=validate.OneOf(VALID_EXCHANGES)
    )  # Exchange (NSE_INDEX, NSE, NFO, BSE_INDEX, BSE, BFO, MCX, CDS)
    expiry = fields.Str(required=False, allow_none=True)  # Expiry date in DDMMMYY format (e.g., 28NOV25)
    legs = fields.List(
        fields.Dict(), required=True, validate=validate.Length(min=1)
    )  # List of strategy legs (option or future definitions)
    notes = fields.Str(required=False, allow_none=True)  # Optional notes
