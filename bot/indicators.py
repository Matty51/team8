"""
Technical indicators — computed from OHLCV candle data.
No external TA library needed; pure Python for transparency.
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


def sma(closes: List[float], period: int) -> List[Optional[float]]:
    """Simple Moving Average."""
    result = [None] * len(closes)
    if len(closes) < period:
        return result
    for i in range(period - 1, len(closes)):
        result[i] = sum(closes[i - period + 1: i + 1]) / period
    return result


def ema(closes: List[float], period: int) -> List[Optional[float]]:
    """Exponential Moving Average."""
    result: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period:
        return result
    # Seed with SMA
    seed = sum(closes[:period]) / period
    result[period - 1] = seed
    multiplier = 2.0 / (period + 1)
    for i in range(period, len(closes)):
        prev = result[i - 1]
        if prev is not None:
            result[i] = (closes[i] - prev) * multiplier + prev
    return result


def rsi(closes: List[float], period: int = 14) -> List[Optional[float]]:
    """Relative Strength Index (0–100)."""
    result: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return result

    # Calculate price changes
    deltas = [closes[i] - closes[i - 1] for i in range(1, len(closes))]

    # Initial average gain/loss
    gains = [max(d, 0) for d in deltas[:period]]
    losses = [abs(min(d, 0)) for d in deltas[:period]]
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100.0 - (100.0 / (1.0 + rs))

    # Smoothed RSI
    for i in range(period, len(deltas)):
        gain = max(deltas[i], 0)
        loss = abs(min(deltas[i], 0))
        avg_gain = (avg_gain * (period - 1) + gain) / period
        avg_loss = (avg_loss * (period - 1) + loss) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100.0 - (100.0 / (1.0 + rs))

    return result


def bollinger_bands(
    closes: List[float], period: int = 20, num_std: float = 2.0
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    Bollinger Bands.
    Returns (upper, middle, lower) bands.
    """
    middle = sma(closes, period)
    upper: List[Optional[float]] = [None] * len(closes)
    lower: List[Optional[float]] = [None] * len(closes)

    for i in range(period - 1, len(closes)):
        if middle[i] is None:
            continue
        window = closes[i - period + 1: i + 1]
        std = (sum((x - middle[i]) ** 2 for x in window) / period) ** 0.5
        upper[i] = middle[i] + num_std * std
        lower[i] = middle[i] - num_std * std

    return upper, middle, lower


def macd(
    closes: List[float],
    fast: int = 12,
    slow: int = 26,
    signal_period: int = 9,
) -> Tuple[List[Optional[float]], List[Optional[float]], List[Optional[float]]]:
    """
    MACD (Moving Average Convergence Divergence).
    Returns (macd_line, signal_line, histogram).
    """
    ema_fast = ema(closes, fast)
    ema_slow = ema(closes, slow)

    macd_line: List[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    # Signal line = EMA of MACD line
    macd_values = [v if v is not None else 0.0 for v in macd_line]
    signal_line = ema(macd_values, signal_period)

    # Histogram
    histogram: List[Optional[float]] = [None] * len(closes)
    for i in range(len(closes)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]

    return macd_line, signal_line, histogram


def average_volume(volumes: List[float], period: int = 20) -> List[Optional[float]]:
    """Average volume over a period."""
    return sma(volumes, period)


def atr(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> List[Optional[float]]:
    """Average True Range — measures volatility."""
    result: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return result

    true_ranges = []
    for i in range(1, len(closes)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    # Initial ATR
    if len(true_ranges) < period:
        return result
    current_atr = sum(true_ranges[:period]) / period
    result[period] = current_atr

    for i in range(period, len(true_ranges)):
        current_atr = (current_atr * (period - 1) + true_ranges[i]) / period
        result[i + 1] = current_atr

    return result


# ── Fibonacci ────────────────────────────────────────────────────────

# Standard Fibonacci retracement levels
FIB_RETRACEMENT_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]

# Fibonacci extension levels (for take-profit targets beyond the move)
FIB_EXTENSION_LEVELS = [1.000, 1.272, 1.618, 2.000, 2.618]


@dataclass
class FibLevel:
    """A single Fibonacci level with its price and label."""
    ratio: float
    price: float
    label: str


@dataclass
class FibonacciLevels:
    """Complete set of Fibonacci retracement and extension levels."""
    swing_high: float
    swing_low: float
    direction: str  # "bullish" (low→high) or "bearish" (high→low)
    retracements: List[FibLevel]
    extensions: List[FibLevel]

    def get_take_profit_levels(self) -> List[FibLevel]:
        """Get TP levels appropriate for the direction."""
        if self.direction == "bullish":
            # For long trades: extensions above the swing high
            return self.extensions
        else:
            # For short trades: extensions below the swing low
            return self.extensions

    def get_support_resistance(self) -> List[FibLevel]:
        """Get retracement levels as support/resistance."""
        return self.retracements

    def nearest_level_above(self, price: float) -> Optional[FibLevel]:
        """Find the nearest Fibonacci level above current price."""
        all_levels = sorted(
            self.retracements + self.extensions,
            key=lambda l: l.price,
        )
        for level in all_levels:
            if level.price > price:
                return level
        return None

    def nearest_level_below(self, price: float) -> Optional[FibLevel]:
        """Find the nearest Fibonacci level below current price."""
        all_levels = sorted(
            self.retracements + self.extensions,
            key=lambda l: l.price,
            reverse=True,
        )
        for level in all_levels:
            if level.price < price:
                return level
        return None


def fibonacci_retracements(
    swing_high: float, swing_low: float, direction: str = "bullish"
) -> List[FibLevel]:
    """
    Calculate Fibonacci retracement levels.

    For bullish (uptrend retracement): levels between high and low
    where price might find support on a pullback.

    For bearish (downtrend retracement): levels between low and high
    where price might find resistance on a bounce.
    """
    diff = swing_high - swing_low
    levels = []

    for ratio in FIB_RETRACEMENT_LEVELS:
        if direction == "bullish":
            # Retracements from the high going down
            price = swing_high - (diff * ratio)
        else:
            # Retracements from the low going up
            price = swing_low + (diff * ratio)

        levels.append(FibLevel(
            ratio=ratio,
            price=price,
            label=f"Fib {ratio:.1%}",
        ))

    return levels


def fibonacci_extensions(
    swing_high: float, swing_low: float, direction: str = "bullish"
) -> List[FibLevel]:
    """
    Calculate Fibonacci extension levels (for take-profit targets).

    For bullish: levels above the swing high where price might reach.
    For bearish: levels below the swing low where price might reach.
    """
    diff = swing_high - swing_low
    levels = []

    for ratio in FIB_EXTENSION_LEVELS:
        if direction == "bullish":
            # Extensions above the swing high
            price = swing_low + (diff * ratio)
        else:
            # Extensions below the swing low
            price = swing_high - (diff * ratio)

        levels.append(FibLevel(
            ratio=ratio,
            price=price,
            label=f"Fib Ext {ratio:.1%}",
        ))

    return levels


def calculate_fibonacci(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    lookback: int = 50,
) -> Optional[FibonacciLevels]:
    """
    Calculate Fibonacci levels from recent swing high/low.

    Automatically detects the trend direction and finds the most
    significant swing high and swing low in the lookback period.
    """
    if len(closes) < lookback:
        lookback = len(closes)
    if lookback < 10:
        return None

    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    recent_closes = closes[-lookback:]

    swing_high = max(recent_highs)
    swing_low = min(recent_lows)

    if swing_high == swing_low:
        return None

    high_idx = recent_highs.index(swing_high)
    low_idx = recent_lows.index(swing_low)

    # Direction: if the low came before the high, we're in an uptrend
    direction = "bullish" if low_idx < high_idx else "bearish"

    retracements = fibonacci_retracements(swing_high, swing_low, direction)
    extensions = fibonacci_extensions(swing_high, swing_low, direction)

    return FibonacciLevels(
        swing_high=swing_high,
        swing_low=swing_low,
        direction=direction,
        retracements=retracements,
        extensions=extensions,
    )


def find_support_resistance(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    lookback: int = 50,
    tolerance_pct: float = 0.3,
) -> Tuple[List[float], List[float]]:
    """
    Find support and resistance levels from price action.
    Returns (support_levels, resistance_levels) sorted by strength.

    Looks for price levels that have been tested multiple times
    (swing highs = resistance, swing lows = support).
    """
    if len(closes) < lookback:
        lookback = len(closes)
    if lookback < 5:
        return [], []

    recent_highs = highs[-lookback:]
    recent_lows = lows[-lookback:]
    current_price = closes[-1]

    # Find swing highs (local maxima)
    swing_highs = []
    for i in range(2, len(recent_highs) - 2):
        if (recent_highs[i] > recent_highs[i-1] and
            recent_highs[i] > recent_highs[i-2] and
            recent_highs[i] > recent_highs[i+1] and
            recent_highs[i] > recent_highs[i+2]):
            swing_highs.append(recent_highs[i])

    # Find swing lows (local minima)
    swing_lows = []
    for i in range(2, len(recent_lows) - 2):
        if (recent_lows[i] < recent_lows[i-1] and
            recent_lows[i] < recent_lows[i-2] and
            recent_lows[i] < recent_lows[i+1] and
            recent_lows[i] < recent_lows[i+2]):
            swing_lows.append(recent_lows[i])

    # Cluster nearby levels (within tolerance)
    def cluster_levels(levels: List[float]) -> List[float]:
        if not levels:
            return []
        levels = sorted(levels)
        clusters: List[List[float]] = [[levels[0]]]
        for price in levels[1:]:
            if abs(price - clusters[-1][-1]) / clusters[-1][-1] * 100 < tolerance_pct:
                clusters[-1].append(price)
            else:
                clusters.append([price])
        # Return the average of each cluster, sorted by cluster size (strength)
        result = [(sum(c) / len(c), len(c)) for c in clusters]
        result.sort(key=lambda x: x[1], reverse=True)
        return [price for price, _ in result]

    resistance = [p for p in cluster_levels(swing_highs) if p > current_price]
    support = [p for p in cluster_levels(swing_lows) if p < current_price]

    return support, resistance


def pivot_points(
    high: float, low: float, close: float
) -> Dict[str, float]:
    """
    Calculate classic pivot points from the previous candle.
    Used as intraday support/resistance levels.
    """
    pivot = (high + low + close) / 3.0
    return {
        "R3": pivot + 2 * (high - low),
        "R2": pivot + (high - low),
        "R1": (2 * pivot) - low,
        "P": pivot,
        "S1": (2 * pivot) - high,
        "S2": pivot - (high - low),
        "S3": pivot - 2 * (high - low),
    }
