"""
Market scanner — analyzes candle data and produces technical signals.
"""
import logging
from dataclasses import dataclass, field
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

    # Fibonacci
    fibonacci: Optional[ind.FibonacciLevels] = None

    # Support/Resistance
    support_levels: List[float] = field(default_factory=list)
    resistance_levels: List[float] = field(default_factory=list)

    # Pivot points
    pivots: Optional[Dict[str, float]] = None

    # Raw OHLCV (needed by LevelManager for exit planning)
    raw_highs: List[float] = field(default_factory=list)
    raw_lows: List[float] = field(default_factory=list)
    raw_closes: List[float] = field(default_factory=list)

    # Pin bar detection
    pin_bar: Optional[ind.PinBar] = None

    # Stochastics (from Wealth Training: 12,3,3 fast and 20,12,9 slow)
    stoch_fast_k: Optional[float] = None
    stoch_fast_d: Optional[float] = None
    stoch_slow_k: Optional[float] = None
    stoch_slow_d: Optional[float] = None

    # CCI (from Wealth Training: period 90)
    cci_value: Optional[float] = None

    # Long-term moving averages (50/100 from Wealth Training)
    sma_long_fast: Optional[float] = None
    sma_long_slow: Optional[float] = None

    # Chart patterns
    chart_patterns: List[ind.ChartPattern] = field(default_factory=list)

    # Doji and pipe bottom detection
    is_doji: bool = False
    is_pipe_bottom: bool = False

    # Market behaviour (from 6-step analysis)
    market_behaviour: str = "unknown"  # "trending" or "oscillating"

    # Raw opens (needed for pipe bottom detection)
    raw_opens: List[float] = field(default_factory=list)

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
        sma_long_fast = ind.sma(closes, self.config.sma_long_fast_period)
        sma_long_slow = ind.sma(closes, self.config.sma_long_slow_period)
        ema_12 = ind.ema(closes, 12)
        ema_26 = ind.ema(closes, 26)
        rsi_values = ind.rsi(closes, self.config.rsi_period)
        macd_line, macd_signal, macd_hist = ind.macd(
            closes, self.config.macd_fast, self.config.macd_slow, self.config.macd_signal
        )
        bb_upper, bb_middle, bb_lower = ind.bollinger_bands(closes)
        avg_vol = ind.average_volume(volumes, 20)
        atr_values = ind.atr(highs, lows, closes)

        # Stochastics (Wealth Training: fast 12,3,3 and slow 20,12,9)
        stoch_fk, stoch_fd = ind.stochastic(
            highs, lows, closes,
            self.config.stoch_fast_k, self.config.stoch_fast_d, self.config.stoch_fast_smooth,
        )
        stoch_sk, stoch_sd = ind.stochastic(
            highs, lows, closes,
            self.config.stoch_slow_k, self.config.stoch_slow_d, self.config.stoch_slow_smooth,
        )

        # CCI (Wealth Training: period 90)
        cci_values = ind.cci(highs, lows, closes, self.config.cci_period)

        # Chart pattern detection
        chart_patterns = ind.detect_chart_patterns(highs, lows, closes)

        # Doji and pipe bottom
        is_doji = ind.detect_doji(curr_open, curr_high, curr_low, price, atr_values[-1])
        is_pipe_bottom = ind.detect_pipe_bottom(opens, highs, lows, closes, atr_values[-1])

        # Volume ratio
        vol_ratio = None
        if avg_vol[-1] and avg_vol[-1] > 0:
            vol_ratio = curr_volume / avg_vol[-1]

        # Determine trend (using both short and long MAs)
        trend = "neutral"
        if sma_fast[-1] and sma_slow[-1]:
            if sma_fast[-1] > sma_slow[-1]:
                trend = "bullish"
            elif sma_fast[-1] < sma_slow[-1]:
                trend = "bearish"

        # Determine market behaviour (Step 1 of 6-step analysis)
        # Trending = smooth directional move; Oscillating = bouncing in range
        market_behaviour = "unknown"
        if len(closes) >= 20:
            recent = closes[-20:]
            direction_changes = sum(
                1 for i in range(1, len(recent) - 1)
                if (recent[i] - recent[i-1]) * (recent[i+1] - recent[i]) < 0
            )
            if direction_changes < 8:
                market_behaviour = "trending"
            else:
                market_behaviour = "oscillating"

        # Fibonacci levels
        fib_levels = ind.calculate_fibonacci(highs, lows, closes)

        # Support/Resistance
        support, resistance = ind.find_support_resistance(
            highs, lows, closes
        )

        # Pivot points (from previous candle)
        pivots = None
        if len(highs) >= 2:
            pivots = ind.pivot_points(highs[-2], lows[-2], closes[-2])

        # Pin bar detection on current candle
        pin_bar = ind.detect_pin_bar(
            open_price=curr_open,
            high=curr_high,
            low=curr_low,
            close=price,
            atr_value=atr_values[-1],
        )

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
            fibonacci=fib_levels,
            support_levels=support,
            resistance_levels=resistance,
            pivots=pivots,
            raw_highs=highs,
            raw_lows=lows,
            raw_closes=closes,
            raw_opens=opens,
            pin_bar=pin_bar,
            stoch_fast_k=stoch_fk[-1],
            stoch_fast_d=stoch_fd[-1],
            stoch_slow_k=stoch_sk[-1],
            stoch_slow_d=stoch_sd[-1],
            cci_value=cci_values[-1],
            sma_long_fast=sma_long_fast[-1],
            sma_long_slow=sma_long_slow[-1],
            chart_patterns=chart_patterns,
            is_doji=is_doji,
            is_pipe_bottom=is_pipe_bottom,
            market_behaviour=market_behaviour,
            trend=trend,
        )

        log_parts = [f"{symbol} @ ${price:,.2f}"]
        if snapshot.rsi:
            log_parts.append(f"RSI({self.config.rsi_period}): {snapshot.rsi:.1f}")
        log_parts.append(f"Trend: {trend}")
        if vol_ratio:
            log_parts.append(f"Vol: {vol_ratio:.1f}x")
        if snapshot.stoch_fast_k:
            log_parts.append(f"Stoch: {snapshot.stoch_fast_k:.1f}")
        if snapshot.cci_value:
            log_parts.append(f"CCI: {snapshot.cci_value:.0f}")
        log_parts.append(f"Mkt: {market_behaviour}")
        if chart_patterns:
            names = [p.name for p in chart_patterns]
            log_parts.append(f"Patterns: {', '.join(names)}")
        logger.info(" | ".join(log_parts))

        return snapshot
