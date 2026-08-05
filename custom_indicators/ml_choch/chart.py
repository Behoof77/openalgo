"""
ML CHoCH Chart -- Plotly visualization of ML CHoCH signals.

Renders a dark-themed candlestick chart with:
- CHoCH markers with probability scores
- TP1/TP2/TP3 target zones as colored rectangles
- Volume subplot
- Stats panel (model size, bias, latest signal)

Usage:
    python chart.py SYMBOL EXCHANGE INTERVAL
    python chart.py SBIN NSE D
    python chart.py NIFTY NSE_INDEX 15m
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dotenv import find_dotenv, load_dotenv

# Add parent directory for imports when running standalone
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from custom_indicators.ml_choch.signal_generator import generate_signals


def fetch_data(symbol: str, exchange: str, interval: str, days: int = 365) -> pd.DataFrame:
    """Fetch OHLCV data from OpenAlgo API.

    Args:
        symbol: Trading symbol
        exchange: Exchange code (NSE, NFO, etc.)
        interval: Candle interval (D, 15m, 1h, etc.)
        days: Number of calendar days of history

    Returns:
        DataFrame with columns [open, high, low, close, volume]
    """
    from openalgo import api

    load_dotenv(find_dotenv(), override=False)

    client = api(
        api_key=os.getenv("OPENALGO_API_KEY"),
        host=os.getenv("OPENALGO_HOST", "http://127.0.0.1:5000"),
    )

    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days)

    df = client.history(
        symbol=symbol,
        exchange=exchange,
        interval=interval,
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
    )

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        df = df.set_index("timestamp")
    else:
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)

    return df


def create_chart(
    df: pd.DataFrame,
    signals: list,
    symbol: str,
    exchange: str,
    interval: str,
) -> go.Figure:
    """Create a Plotly dark-theme chart with CHoCH signals and targets.

    Args:
        df: OHLCV DataFrame
        signals: List of SignalResult objects
        symbol: Symbol name for chart title
        exchange: Exchange name
        interval: Interval name

    Returns:
        Plotly Figure object
    """
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    opn = df["open"].values
    volume = df["volume"].values

    x_labels = df.index.strftime("%Y-%m-%d %H:%M") if interval != "D" else df.index.strftime("%Y-%m-%d")

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.75, 0.25], vertical_spacing=0.03,
        subplot_titles=[f"ML CHoCH | {symbol} ({exchange}) {interval}", "Volume"],
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=x_labels, open=opn, high=high, low=low, close=close,
        name="Price", increasing_line_color="#14c896", decreasing_line_color="#ff6b8a",
    ), row=1, col=1)

    # Volume bars (colored by direction)
    vol_colors = [
        "#14c896" if close[i] >= opn[i] else "#ff6b8a"
        for i in range(len(close))
    ]
    fig.add_trace(go.Bar(
        x=x_labels, y=volume, name="Volume",
        marker_color=vol_colors, opacity=0.5,
    ), row=2, col=1)

    # Valid signals only
    valid_signals = [s for s in signals if s.is_valid]

    # CHoCH markers
    bull_indices = [s.bar_index for s in valid_signals if s.direction]
    bear_indices = [s.bar_index for s in valid_signals if not s.direction]

    if bull_indices:
        bull_x = [x_labels[i] for i in bull_indices if i < len(x_labels)]
        bull_y = [low[i] - (high[i] - low[i]) * 0.3 for i in bull_indices if i < len(low)]
        bull_prob = [f"{s.probability:.0f}%" for s in valid_signals if s.direction]
        fig.add_trace(go.Scatter(
            x=bull_x, y=bull_y, mode="markers+text",
            marker=dict(symbol="triangle-up", size=12, color="#80dfff"),
            text=bull_prob, textposition="bottom center",
            textfont=dict(size=9, color="#80dfff"),
            name="Bullish CHoCH",
        ), row=1, col=1)

    if bear_indices:
        bear_x = [x_labels[i] for i in bear_indices if i < len(x_labels)]
        bear_y = [high[i] + (high[i] - low[i]) * 0.3 for i in bear_indices if i < len(high)]
        bear_prob = [f"{s.probability:.0f}%" for s in valid_signals if not s.direction]
        fig.add_trace(go.Scatter(
            x=bear_x, y=bear_y, mode="markers+text",
            marker=dict(symbol="triangle-down", size=12, color="#ff8ca0"),
            text=bear_prob, textposition="top center",
            textfont=dict(size=9, color="#ff8ca0"),
            name="Bearish CHoCH",
        ), row=1, col=1)

    # Target zones (last 3 valid signals)
    recent_targets = valid_signals[-3:] if len(valid_signals) > 3 else valid_signals
    for sig in recent_targets:
        if sig.tp1 <= 0 or sig.tp3 <= 0:
            continue
        idx = sig.bar_index
        if idx >= len(x_labels):
            continue

        x_start = x_labels[idx]
        # Extend target zone 10 bars right
        extend = min(idx + 10, len(x_labels) - 1)
        x_end = x_labels[extend]

        if sig.direction:
            # Bullish: TP3 on top, TP1 on bottom
            fig.add_shape(type="rect",
                x0=x_start, x1=x_end, y0=sig.tp1, y1=sig.tp3,
                fillcolor="rgba(128, 223, 255, 0.07)",
                line=dict(color="rgba(128, 223, 255, 0.3)", width=1),
                row=1, col=1,
            )
            # TP2 line
            fig.add_trace(go.Scatter(
                x=[x_start, x_end], y=[sig.tp2, sig.tp2],
                mode="lines", line=dict(color="rgba(180, 225, 255, 0.4)", dash="dot", width=1),
                showlegend=False,
            ), row=1, col=1)
        else:
            # Bearish: TP1 on top, TP3 on bottom
            fig.add_shape(type="rect",
                x0=x_start, x1=x_end, y0=sig.tp3, y1=sig.tp1,
                fillcolor="rgba(255, 140, 160, 0.07)",
                line=dict(color="rgba(255, 140, 160, 0.3)", width=1),
                row=1, col=1,
            )
            fig.add_trace(go.Scatter(
                x=[x_start, x_end], y=[sig.tp2, sig.tp2],
                mode="lines", line=dict(color="rgba(255, 185, 200, 0.4)", dash="dot", width=1),
                showlegend=False,
            ), row=1, col=1)

    # Stats annotation
    total = len(valid_signals)
    bull_count = sum(1 for s in valid_signals if s.direction)
    bear_count = total - bull_count
    last_prob = valid_signals[-1].probability if valid_signals else 0.0
    db_size = valid_signals[-1].db_size if valid_signals else 0

    bias_text = "BULLISH" if bull_count > bear_count else "BEARISH" if bear_count > bull_count else "NEUTRAL"
    bias_color = "#80ff90" if bull_count > bear_count else "#ff6080" if bear_count > bull_count else "#888"

    stats_text = (
        f"DB: {db_size} | Signals: {total} (B:{bull_count} S:{bear_count}) | "
        f"Last: {last_prob:.0f}% | Bias: {bias_text}"
    )
    fig.add_annotation(
        text=stats_text, xref="paper", yref="paper",
        x=0.5, y=1.08, showarrow=False,
        font=dict(size=10, color="#bbb"),
    )

    # Layout
    fig.update_layout(
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        xaxis_type="category",
        xaxis2_type="category",
        height=750,
        margin=dict(l=60, r=40, t=80, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


def main():
    """CLI entry point: python chart.py SYMBOL [EXCHANGE] [INTERVAL]"""
    if len(sys.argv) < 2:
        print("Usage: python chart.py SYMBOL [EXCHANGE] [INTERVAL]")
        print("Example: python chart.py SBIN NSE D")
        print("Example: python chart.py NIFTY NSE_INDEX 15m")
        sys.exit(1)

    symbol = sys.argv[1].upper()
    exchange = sys.argv[2].upper() if len(sys.argv) > 2 else "NSE"
    interval = sys.argv[3] if len(sys.argv) > 3 else "D"

    print(f"Fetching {symbol} ({exchange}) {interval} data...")
    df = fetch_data(symbol, exchange, interval)
    print(f"Loaded {len(df)} bars")

    print("Running ML CHoCH pipeline...")
    signals, model = generate_signals(
        df, lookahead=20, swing_len=5, atr_len=14,
        min_score=60, window=1500,
    )

    valid = [s for s in signals if s.is_valid]
    print(f"Generated {len(signals)} total signals, {len(valid)} valid (>= 60%)")

    if valid:
        last = valid[-1]
        direction = "BULLISH" if last.direction else "BEARISH"
        print(f"Latest signal: {direction} {last.probability:.1f}% "
              f"TP1={last.tp1:.2f} TP2={last.tp2:.2f} TP3={last.tp3:.2f}")

    fig = create_chart(df, signals, symbol, exchange, interval)
    fig.show()
    print("Chart displayed.")


if __name__ == "__main__":
    main()
