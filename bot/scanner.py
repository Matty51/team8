"""
Market scanner — analyzes candle data and produces technical signals.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .client import ExchangeClient
from .config import Config
from . import indicators as ind

logger = logging.getLogger(__name__)


@dataclass
class MarketSnapshot:
    """A point-in-time snapshot of market data with computed indicators."""
    symbol: str
    timeframe: str
    current_price: float
    open_price: float
    high: float
    low: float
    volume: float

    # Moving averages
    sma_fast: Optional[float] = None
    sma_slow: Optional[float] = None
    ema_12: Optional[float] = None
    ema_26: Optional[float] = None

    # RSI
    rsi: Optional[float] = None

    # MACD
    macd_line: Optional[float] = None
    macd_signal: Optional[float] = None
    macd_histogram: Optional[float] = None

    # Bollinger Bands
    bb_upper: Optional[float] = None
    bb_middle: Optional[float] = None
    bb_lower: Optional[float] = None

    # Volume
    avg_volume: Optional[float] = None
    volume_ratio: Optional[float] = None  # current / average

    # Volatility
    atr: Optional[float] = None

    # Trend
    trend: str = "neutral"  # "bullish", "bearish", "neutral"

    @property
    def has_volume_spike(self) -> bool:
        return self.volume_ratio is not None and self.volume_ratio > 2.0

    @property
    def is_oversold(self) -> bool:
        return self.rsi is not None and self.rsi < 30

    @property
    def is_overbought(self) -> bool:
        return self.rsi is not None and self.rsi > 70


class MarketScanner:
    """Fetches candle data and computes all indicators."""

    def __init__(self, client: ExchangeClient, config: Config):
        self.client = client
        self.config = config

    def scan(self, symbol: Optional[str] = None) -> Optional[MarketSnapshot]:
        """Fetch candles and compute a full market snapshot."""
        symbol = symbol or self.config.trading_pair
        candles = self.client.fetch_ohlcv(symbol)

        if not candles or len(candles) < self.config.sma_slow_period + 5:
            logger.warning(
                f"Not enough candle data for {symbol} "
                f"({len(candles)} candles)"
            )
            return None

        # Extract OHLCV columns
        timestamps = [c[0] for c in candles]
        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        volumes = [c[5] for c in candles]

        # Current candle
        price = closes[-1]
        curr_open = opens[-1]
        curr_high = highs[-1]
        curr_low = lows[-1]
        curr_volume = volumes[-1]

        # Compute indicators
        sma_fast = ind.sma(closes, self.config.sma_fast_period)
        sma_slow = ind.sma(closes, self.config.sma_slow_period)
        ema_12 = ind.ema(closes, 12)
        ema_26 = ind.ema(closes, 26)
        rsi_values = ind.rsi(closes, self.config.rsi_period)
        macd_line, macd_signal, macd_hist = ind.macd(closes)
        bb_upper, bb_middle, bb_lower = ind.bollinger_bands(closes)
        avg_vol = ind.average_volume(volumes, 20)
        atr_values = ind.atr(highs, lows, closes)

        # Volume ratio
        vol_ratio = None
        if avg_vol[-1] and avg_vol[-1] > 0:
            vol_ratio = curr_volume / avg_vol[-1]

        # Determine trend
        trend = "neutral"
        if sma_fast[-1] and sma_slow[-1]:
            if sma_fast[-1] > sma_slow[-1]:
                trend = "bullish"
            elif sma_fast[-1] < sma_slow[-1]:
                trend = "bearish"

        snapshot = MarketSnapshot(
            symbol=symbol,
            timeframe=self.config.timeframe,
            current_price=price,
            open_price=curr_open,
            high=curr_high,
            low=curr_low,
            volume=curr_volume,
            sma_fast=sma_fast[-1],
            sma_slow=sma_slow[-1],
            ema_12=ema_12[-1],
            ema_26=ema_26[-1],
            rsi=rsi_values[-1],
            macd_line=macd_line[-1],
            macd_signal=macd_signal[-1],
            macd_histogram=macd_hist[-1],
            bb_upper=bb_upper[-1],
            bb_middle=bb_middle[-1],
            bb_lower=bb_lower[-1],
            avg_volume=avg_vol[-1],
            volume_ratio=vol_ratio,
            atr=atr_values[-1],
            trend=trend,
        )

        logger.info(
            f"{symbol} @ ${price:,.2f} | "
            f"RSI: {snapshot.rsi:.1f} | "
            f"Trend: {trend} | "
            f"Vol ratio: {vol_ratio:.1f}x"
            if snapshot.rsi and vol_ratio
            else f"{symbol} @ ${price:,.2f}"
        )

        return snapshot
