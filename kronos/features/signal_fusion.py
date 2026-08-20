"""
Multi-factor signal fusion combiner.

Takes signal inputs from three families — Kronos (price-pattern),
technicals (RSI, Bollinger, ATR, ADX) and optionally option-market
sentiment (PCR, IV skew) — and produces a single fused BUY / SELL /
HOLD decision with an accompanying conviction score.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from kronos.features.indicators import TechnicalSnapshot
from kronos.utils.helpers import get_logger

logger = get_logger(__name__)

# ── Constants used as default thresholds ─────────────────────────────
DEFAULT_FUSE_THRESHOLD = 0.30  # fused score above → BUY, below → SELL


@dataclass
class FusionFactors:
    """All factors that can contribute to the fused decision.

    Every field has a default ``None`` meaning "factor not available";
    the fusion engine will skip it.
    """

    # Kronos model
    kronos_signal: int | None = None       # 1 / -1 / 0
    kronos_confidence: float | None = None  # [0, 1]

    # Technicals
    technicals: TechnicalSnapshot | None = None

    # Option-market sentiment (optional, live only)
    pcr: float | None = None          # Put/Call OI ratio
    iv_skew: float | None = None      # OTM Put IV - ATM IV
    oi_change_pct: float | None = None  # Total OI change % (momentum)


@dataclass
class SignalFusionConfig:
    """Weights and thresholds for the fusion combiner.

    Default weights are rough heuristics.  Tune via config / .env.
    """

    weight_kronos: float = 3.0
    weight_rsi: float = 1.0
    weight_bb: float = 0.5
    weight_adx: float = 1.0
    weight_pcr: float = 0.5
    weight_iv_skew: float = 0.3
    weight_oi_momentum: float = 0.5

    # Kronos confidence floor — below this the Kronos factor abstains
    kronos_min_confidence: float = 0.55

    # Fused-score thresholds  (score in [-1, 1])
    buy_threshold: float = DEFAULT_FUSE_THRESHOLD
    sell_threshold: float = -DEFAULT_FUSE_THRESHOLD

    # Technical indicator thresholds
    rsi_overbought: float = 70.0
    rsi_oversold: float = 30.0
    rsi_extreme_ob: float = 80.0
    rsi_extreme_os: float = 20.0
    adx_trend_threshold: float = 22.0
    bb_width_low_pct: float = 2.0   # Below this → squeeze / mean-revert
    bb_width_high_pct: float = 8.0  # Above this → high vol / trend


def _clamp(v: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


# ---------------------------------------------------------------------------
# Factor scoring  (each returns a score in [-1, 1])
# ---------------------------------------------------------------------------


def _score_kronos(
    signal: int | None,
    confidence: float | None,
    cfg: SignalFusionConfig,
) -> float:
    if signal is None or confidence is None:
        return 0.0
    if confidence < cfg.kronos_min_confidence:
        return 0.0
    return float(signal) * confidence


def _score_rsi(tech: TechnicalSnapshot | None, cfg: SignalFusionConfig) -> float:
    if tech is None:
        return 0.0
    r = tech.rsi
    # Map RSI to score: extreme zones → strong bias, normal → neutral
    if r >= cfg.rsi_extreme_ob:
        return -1.0  # strongly overbought → bearish
    if r <= cfg.rsi_extreme_os:
        return 1.0  # strongly oversold → bullish
    if r >= cfg.rsi_overbought:
        # Linear: 70→-0.5, 80→-1.0
        return -_clamp((r - cfg.rsi_overbought) / (cfg.rsi_extreme_ob - cfg.rsi_overbought) * 0.5 + 0.5)
    if r <= cfg.rsi_oversold:
        # Linear: 30→0.5, 20→1.0
        return _clamp((cfg.rsi_oversold - r) / (cfg.rsi_oversold - cfg.rsi_extreme_os) * 0.5 + 0.5)
    return 0.0


def _score_bb(tech: TechnicalSnapshot | None, cfg: SignalFusionConfig) -> float:
    """Bollinger Band mean-reversion / squeeze signal."""
    if tech is None:
        return 0.0
    bp = tech.bb_position
    bw = tech.bb_width_pct
    # Squeeze detection: tight bands + price outside → strong mean-revert
    if bw < cfg.bb_width_low_pct and abs(bp) > 0.5:
        return -bp  # price above upper → sell (revert down)
    # Wide bands → trend
    if bw > cfg.bb_width_high_pct:
        return bp  # price at upper edge → trend up
    return -bp * 0.3  # mild mean-reversion bias


def _score_adx(tech: TechnicalSnapshot | None, cfg: SignalFusionConfig) -> float:
    """ADX trend confirmation score."""
    if tech is None:
        return 0.0
    a = tech.adx
    d = tech.adx_direction
    if a < cfg.adx_trend_threshold:
        return 0.0  # no trend → neutral
    # Strong trend: amplify direction
    strength = _clamp((a - cfg.adx_trend_threshold) / 30.0)  # 22→0, 52→1
    return d * strength


def _score_option(
    pcr: float | None,
    iv_skew: float | None,
    oi_chg: float | None,
    cfg: SignalFusionConfig,
) -> float:
    """Option-market sentiment score."""
    score = 0.0
    count = 0
    if pcr is not None:
        # PCR > 1.2 → excessive puts (bearish), PCR < 0.6 → excessive calls (bullish)
        if pcr > 1.2:
            score -= _clamp((pcr - 1.2) / 0.8)
        elif pcr < 0.6:
            score += _clamp((0.6 - pcr) / 0.4)
        count += 1
    if iv_skew is not None:
        # Positive skew → puts more expensive (fear), negative → calls (greed)
        score -= _clamp(iv_skew / 5.0)
        count += 1
    if oi_chg is not None and abs(oi_chg) > 2.0:
        # Sharp OI increase in one direction
        score += _clamp(oi_chg / 20.0)
        count += 1
    return score / max(count, 1) if count > 0 else 0.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def fuse_signals(
    factors: FusionFactors,
    config: SignalFusionConfig | None = None,
) -> tuple[int, float, dict]:
    """Combine all available factors into a single trading decision.

    Parameters
    ----------
    factors : FusionFactors
        Populated with whatever data is available.
    config : SignalFusionConfig | None
        Weights and thresholds.  Defaults to ``SignalFusionConfig()``.

    Returns
    -------
    signal : int
        ``1`` (BUY), ``-1`` (SELL), or ``0`` (HOLD).
    conviction : float
        Absolute value of the fused score (``[0, 1]``).
    breakdown : dict
        Per-factor scores for debugging / logging.
    """
    cfg = config or SignalFusionConfig()

    scores: dict[str, float] = {}
    weights: dict[str, float] = {}
    total_weight = 0.0

    # Kronos
    ks = _score_kronos(factors.kronos_signal, factors.kronos_confidence, cfg)
    scores["kronos"] = ks
    weights["kronos"] = cfg.weight_kronos
    total_weight += cfg.weight_kronos

    # RSI
    if factors.technicals is not None:
        rs = _score_rsi(factors.technicals, cfg)
        scores["rsi"] = rs
        weights["rsi"] = cfg.weight_rsi
        total_weight += cfg.weight_rsi
    else:
        scores["rsi"] = 0.0
        weights["rsi"] = cfg.weight_rsi  # Still include weight 0 effectively

    # BB
    if factors.technicals is not None:
        bs = _score_bb(factors.technicals, cfg)
        scores["bb"] = bs
        weights["bb"] = cfg.weight_bb
        total_weight += cfg.weight_bb
    else:
        scores["bb"] = 0.0

    # ADX
    if factors.technicals is not None:
        adx_s = _score_adx(factors.technicals, cfg)
        scores["adx"] = adx_s
        weights["adx"] = cfg.weight_adx
        total_weight += cfg.weight_adx
    else:
        scores["adx"] = 0.0

    # Option sentiment
    os_ = _score_option(factors.pcr, factors.iv_skew, factors.oi_change_pct, cfg)
    scores["option_sentiment"] = os_
    has_option = any(x is not None for x in (factors.pcr, factors.iv_skew, factors.oi_change_pct))
    if has_option:
        weights["option_sentiment"] = cfg.weight_pcr  # use PCR weight as base
        total_weight += cfg.weight_pcr
    else:
        weights["option_sentiment"] = 0.0

    # Weighted sum
    if total_weight < 1e-10:
        return 0, 0.0, scores

    fused = sum(scores.get(k, 0.0) * weights.get(k, 0.0) for k in set(scores) | set(weights)) / total_weight
    fused = _clamp(fused)

    # Decision
    if fused >= cfg.buy_threshold:
        signal = 1
    elif fused <= cfg.sell_threshold:
        signal = -1
    else:
        signal = 0

    conviction = abs(fused)

    breakdown = {
        "fused_score": round(fused, 3),
        "conviction": round(conviction, 3),
        "kronos": round(scores.get("kronos", 0.0), 3),
        "rsi": round(scores.get("rsi", 0.0), 3),
        "bb": round(scores.get("bb", 0.0), 3),
        "adx": round(scores.get("adx", 0.0), 3),
        "option_sentiment": round(scores.get("option_sentiment", 0.0), 3),
    }
    return signal, conviction, breakdown
