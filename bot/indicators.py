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


def stochastic(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    k_period: int = 12,
    d_period: int = 3,
    smooth: int = 3,
) -> Tuple[List[Optional[float]], List[Optional[float]]]:
    """
    Stochastic Oscillator (%K and %D).

    From Wealth Training course settings: fast (12,3,3) and slow (20,12,9).
    Signals: two-line crossovers and divergences.
    """
    n = len(closes)
    raw_k: List[Optional[float]] = [None] * n

    for i in range(k_period - 1, n):
        window_highs = highs[i - k_period + 1: i + 1]
        window_lows = lows[i - k_period + 1: i + 1]
        highest = max(window_highs)
        lowest = min(window_lows)
        if highest == lowest:
            raw_k[i] = 50.0
        else:
            raw_k[i] = ((closes[i] - lowest) / (highest - lowest)) * 100.0

    # Smooth %K
    k_values = [v if v is not None else 0.0 for v in raw_k]
    smoothed_k = sma(k_values, smooth)

    # %D = SMA of smoothed %K
    sk_values = [v if v is not None else 0.0 for v in smoothed_k]
    d_line = sma(sk_values, d_period)

    return smoothed_k, d_line


def cci(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 90,
) -> List[Optional[float]]:
    """
    Commodity Channel Index.

    From Wealth Training course: period 90.
    Signals: cross above/below +100/-100, trendline breaks.
    """
    n = len(closes)
    result: List[Optional[float]] = [None] * n

    if n < period:
        return result

    for i in range(period - 1, n):
        # Typical Price
        tp_values = []
        for j in range(i - period + 1, i + 1):
            tp = (highs[j] + lows[j] + closes[j]) / 3.0
            tp_values.append(tp)

        tp_mean = sum(tp_values) / period
        # Mean deviation
        mean_dev = sum(abs(tp - tp_mean) for tp in tp_values) / period

        if mean_dev == 0:
            result[i] = 0.0
        else:
            current_tp = (highs[i] + lows[i] + closes[i]) / 3.0
            result[i] = (current_tp - tp_mean) / (0.015 * mean_dev)

    return result


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


def adx(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    period: int = 14,
) -> List[Optional[float]]:
    """
    Average Directional Index — measures trend strength (not direction).

    ADX > 25 = strong trend (good for trend-following strategies)
    ADX < 20 = weak/no trend (choppy, ranging — avoid trend strategies)
    ADX 20-25 = emerging trend

    Returns a list of ADX values (same length as input).
    """
    n = len(closes)
    result: List[Optional[float]] = [None] * n
    if n < period * 2 + 1:
        return result

    # Step 1: Calculate +DM, -DM and TR
    plus_dm = []
    minus_dm = []
    true_ranges = []

    for i in range(1, n):
        up_move = highs[i] - highs[i - 1]
        down_move = lows[i - 1] - lows[i]

        pdm = up_move if (up_move > down_move and up_move > 0) else 0.0
        mdm = down_move if (down_move > up_move and down_move > 0) else 0.0
        plus_dm.append(pdm)
        minus_dm.append(mdm)

        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i - 1]),
            abs(lows[i] - closes[i - 1]),
        )
        true_ranges.append(tr)

    if len(true_ranges) < period:
        return result

    # Step 2: Smoothed +DM, -DM, TR using Wilder's smoothing
    smooth_plus = sum(plus_dm[:period])
    smooth_minus = sum(minus_dm[:period])
    smooth_tr = sum(true_ranges[:period])

    # Step 3: Calculate +DI, -DI, DX
    dx_values = []

    for i in range(period - 1, len(true_ranges)):
        if i == period - 1:
            pass  # Use initial sums
        else:
            smooth_plus = smooth_plus - smooth_plus / period + plus_dm[i]
            smooth_minus = smooth_minus - smooth_minus / period + minus_dm[i]
            smooth_tr = smooth_tr - smooth_tr / period + true_ranges[i]

        if smooth_tr == 0:
            dx_values.append(0.0)
            continue

        plus_di = 100 * smooth_plus / smooth_tr
        minus_di = 100 * smooth_minus / smooth_tr

        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_values.append(0.0)
        else:
            dx = 100 * abs(plus_di - minus_di) / di_sum
            dx_values.append(dx)

    if len(dx_values) < period:
        return result

    # Step 4: ADX = smoothed average of DX
    adx_val = sum(dx_values[:period]) / period
    start_idx = period * 2
    if start_idx < n:
        result[start_idx] = adx_val

    for j in range(period, len(dx_values)):
        adx_val = (adx_val * (period - 1) + dx_values[j]) / period
        idx = j + period
        if idx < n:
            result[idx] = adx_val

    return result


# ── Fibonacci ────────────────────────────────────────────────────────

# Standard Fibonacci retracement levels
FIB_RETRACEMENT_LEVELS = [0.236, 0.382, 0.500, 0.618, 0.786]

# Fibonacci extension levels (for take-profit targets beyond the move)
# Includes higher extensions from TradingView (3.618, 4.236)
FIB_EXTENSION_LEVELS = [1.000, 1.272, 1.618, 2.000, 2.618, 3.618, 4.236]


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


# ── Pin Bar Detection ────────────────────────────────────────────────


@dataclass
class PinBar:
    """Detected pin bar candlestick pattern."""
    direction: str           # "bullish" or "bearish"
    strength: str            # "strongest", "stronger", "strong", "indecision"
    score: float             # 0.0–1.0 numeric strength
    wick_ratio: float        # How long the wick is vs the body
    body_position: float     # Where the body sits (0=bottom, 1=top)
    candle_range: float      # High - Low
    open: float
    high: float
    low: float
    close: float


def detect_pin_bar(
    open_price: float,
    high: float,
    low: float,
    close: float,
    atr_value: Optional[float] = None,
) -> Optional[PinBar]:
    """
    Detect a pin bar (hammer/shooting star) candlestick pattern.

    Pin bar anatomy:
    - Long wick (shadow) on one side — shows rejection
    - Small body — shows indecision resolved
    - Little/no wick on the other side

    Strength ranking (matches "The Pin Bar Story"):
    - STRONGEST: Tiny body at the very end of the wick, no opposite wick
    - STRONGER:  Small body near the end, tiny opposite wick
    - STRONG:    Small body, noticeable opposite wick
    - INDECISION: Equal wicks on both sides (doji-like)

    Returns None if the candle is not a pin bar.
    """
    candle_range = high - low
    if candle_range == 0:
        return None

    # Minimum candle size (filter noise) — use ATR if available
    if atr_value and candle_range < atr_value * 0.5:
        return None

    body = abs(close - open_price)
    body_ratio = body / candle_range

    # Pin bars need a small body relative to the range
    if body_ratio > 0.35:
        return None

    # Calculate wick sizes
    upper_wick = high - max(open_price, close)
    lower_wick = min(open_price, close) - low

    upper_wick_ratio = upper_wick / candle_range
    lower_wick_ratio = lower_wick / candle_range

    # Body position: 0 = body at bottom, 1 = body at top
    body_mid = (open_price + close) / 2
    body_position = (body_mid - low) / candle_range

    direction = None
    dominant_wick = 0.0
    opposite_wick = 0.0

    # Bullish pin bar: long LOWER wick (rejection of lower prices)
    if lower_wick_ratio > 0.55:
        direction = "bullish"
        dominant_wick = lower_wick_ratio
        opposite_wick = upper_wick_ratio

    # Bearish pin bar: long UPPER wick (rejection of higher prices)
    elif upper_wick_ratio > 0.55:
        direction = "bearish"
        dominant_wick = upper_wick_ratio
        opposite_wick = lower_wick_ratio

    # Indecision: roughly equal wicks, small body
    elif (body_ratio < 0.15 and
          abs(upper_wick_ratio - lower_wick_ratio) < 0.15):
        direction = "bullish" if close > open_price else "bearish"
        dominant_wick = max(upper_wick_ratio, lower_wick_ratio)
        opposite_wick = min(upper_wick_ratio, lower_wick_ratio)
        return PinBar(
            direction=direction,
            strength="indecision",
            score=0.3,
            wick_ratio=dominant_wick / max(body_ratio, 0.01),
            body_position=body_position,
            candle_range=candle_range,
            open=open_price,
            high=high,
            low=low,
            close=close,
        )
    else:
        return None

    # Rank strength based on body position and opposite wick
    wick_ratio = dominant_wick / max(body_ratio, 0.01)

    if opposite_wick < 0.05 and body_ratio < 0.15:
        strength = "strongest"
        score = min(0.95, 0.80 + wick_ratio * 0.01)
    elif opposite_wick < 0.12 and body_ratio < 0.25:
        strength = "stronger"
        score = min(0.85, 0.65 + wick_ratio * 0.01)
    elif opposite_wick < 0.20:
        strength = "strong"
        score = min(0.75, 0.55 + wick_ratio * 0.01)
    else:
        return None  # Not a clean enough pin bar

    return PinBar(
        direction=direction,
        strength=strength,
        score=score,
        wick_ratio=wick_ratio,
        body_position=body_position,
        candle_range=candle_range,
        open=open_price,
        high=high,
        low=low,
        close=close,
    )


# ── Chart Pattern Detection ─────────────────────────────────────────
# From Wealth Training course: reversal and continuation patterns


@dataclass
class ChartPattern:
    """A detected chart pattern."""
    name: str              # e.g. "double_bottom", "head_and_shoulders", "cup_and_handle"
    pattern_type: str      # "reversal" or "continuation"
    direction: str         # "bullish" or "bearish"
    confidence: float      # 0.0–1.0
    entry_price: float     # Suggested entry (breakout level)
    target_price: float    # Profit target from pattern measurement
    stop_price: float      # Suggested stop loss
    pattern_height: float  # Height of the pattern (for profit target calc)
    description: str       # Human-readable description


def _find_swing_points(
    data: List[float], order: int = 5
) -> Tuple[List[Tuple[int, float]], List[Tuple[int, float]]]:
    """Find swing highs and swing lows with their indices."""
    swing_highs = []
    swing_lows = []
    for i in range(order, len(data) - order):
        is_high = all(data[i] >= data[i - j] and data[i] >= data[i + j]
                       for j in range(1, order + 1))
        is_low = all(data[i] <= data[i - j] and data[i] <= data[i + j]
                      for j in range(1, order + 1))
        if is_high:
            swing_highs.append((i, data[i]))
        if is_low:
            swing_lows.append((i, data[i]))
    return swing_highs, swing_lows


def detect_double_bottom(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    tolerance_pct: float = 2.0,
) -> Optional[ChartPattern]:
    """
    Detect double bottom reversal pattern.
    Two troughs at approximately the same level, with a peak between them.
    Profit target = pattern height projected above the neckline.
    """
    if len(closes) < 20:
        return None

    swing_highs, swing_lows = _find_swing_points(lows, order=3)
    if len(swing_lows) < 2:
        return None

    # Check last two swing lows
    for i in range(len(swing_lows) - 1, 0, -1):
        idx2, low2 = swing_lows[i]
        idx1, low1 = swing_lows[i - 1]

        if idx2 - idx1 < 5:
            continue

        # Bottoms within tolerance
        diff_pct = abs(low1 - low2) / min(low1, low2) * 100
        if diff_pct > tolerance_pct:
            continue

        # Find neckline (highest point between the two bottoms)
        between_highs = highs[idx1:idx2 + 1]
        if not between_highs:
            continue
        neckline = max(between_highs)
        pattern_bottom = min(low1, low2)
        pattern_height = neckline - pattern_bottom

        # Price must be near or above neckline for breakout
        current_price = closes[-1]
        if current_price < neckline * 0.98:
            continue

        target = neckline + pattern_height
        stop = pattern_bottom - pattern_height * 0.1

        confidence = 0.65
        if diff_pct < 1.0:
            confidence += 0.1
        if current_price > neckline:
            confidence += 0.1

        return ChartPattern(
            name="double_bottom",
            pattern_type="reversal",
            direction="bullish",
            confidence=min(confidence, 0.90),
            entry_price=neckline,
            target_price=target,
            stop_price=stop,
            pattern_height=pattern_height,
            description=(
                f"Double bottom at ${pattern_bottom:,.2f}, "
                f"neckline ${neckline:,.2f}, target ${target:,.2f}"
            ),
        )
    return None


def detect_triple_bottom(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    tolerance_pct: float = 2.0,
) -> Optional[ChartPattern]:
    """
    Detect triple bottom reversal pattern.
    Three troughs at approximately the same level.
    """
    if len(closes) < 30:
        return None

    _, swing_lows = _find_swing_points(lows, order=3)
    if len(swing_lows) < 3:
        return None

    for i in range(len(swing_lows) - 2, 0, -1):
        idx1, low1 = swing_lows[i - 1] if i >= 1 else swing_lows[0]
        idx2, low2 = swing_lows[i]
        idx3, low3 = swing_lows[i + 1] if i + 1 < len(swing_lows) else (0, 0)
        if idx3 == 0:
            continue

        if idx3 - idx1 < 10:
            continue

        avg_low = (low1 + low2 + low3) / 3
        if all(abs(l - avg_low) / avg_low * 100 < tolerance_pct
               for l in [low1, low2, low3]):
            neckline = max(highs[idx1:idx3 + 1])
            pattern_height = neckline - avg_low
            current_price = closes[-1]

            if current_price < neckline * 0.98:
                continue

            target = neckline + pattern_height
            stop = avg_low - pattern_height * 0.1

            return ChartPattern(
                name="triple_bottom",
                pattern_type="reversal",
                direction="bullish",
                confidence=0.75,
                entry_price=neckline,
                target_price=target,
                stop_price=stop,
                pattern_height=pattern_height,
                description=(
                    f"Triple bottom at ${avg_low:,.2f}, "
                    f"neckline ${neckline:,.2f}, target ${target:,.2f}"
                ),
            )
    return None


def detect_head_and_shoulders(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    tolerance_pct: float = 3.0,
) -> Optional[ChartPattern]:
    """
    Detect head and shoulders top reversal pattern.
    Three peaks with the middle (head) higher than the two shoulders.
    Profit target = pattern depth projected below the neckline.
    """
    if len(closes) < 30:
        return None

    swing_highs, swing_lows = _find_swing_points(highs, order=3)
    if len(swing_highs) < 3:
        return None

    for i in range(len(swing_highs) - 2):
        idx1, h1 = swing_highs[i]      # Left shoulder
        idx2, h2 = swing_highs[i + 1]  # Head
        idx3, h3 = swing_highs[i + 2]  # Right shoulder

        # Head must be highest
        if not (h2 > h1 and h2 > h3):
            continue

        # Shoulders approximately equal
        shoulder_diff = abs(h1 - h3) / min(h1, h3) * 100
        if shoulder_diff > tolerance_pct:
            continue

        # Neckline = lowest point between shoulders
        neckline_region = lows[idx1:idx3 + 1]
        if not neckline_region:
            continue
        neckline = min(neckline_region)
        pattern_height = h2 - neckline

        current_price = closes[-1]
        # Pattern complete when price breaks below neckline
        if current_price > neckline * 1.02:
            continue

        target = neckline - pattern_height
        stop = h3 + pattern_height * 0.1

        return ChartPattern(
            name="head_and_shoulders",
            pattern_type="reversal",
            direction="bearish",
            confidence=0.75,
            entry_price=neckline,
            target_price=target,
            stop_price=stop,
            pattern_height=pattern_height,
            description=(
                f"Head & Shoulders: head ${h2:,.2f}, "
                f"neckline ${neckline:,.2f}, target ${target:,.2f}"
            ),
        )
    return None


def detect_cup_and_handle(
    highs: List[float],
    lows: List[float],
    closes: List[float],
    tolerance_pct: float = 5.0,
) -> Optional[ChartPattern]:
    """
    Detect cup and handle pattern (bullish reversal).

    Quality filters from Wealth Training:
    1. Smoother cup = better (avoid jagged)
    2. Shallower cup = better (avoid deep V)
    3. No spike in handle
    4. Handle above 50% midpoint of cup depth
    """
    if len(closes) < 30:
        return None

    # Look for a U-shaped bottom in the last N candles
    lookback = min(len(closes), 60)
    segment = closes[-lookback:]
    segment_lows = lows[-lookback:]
    segment_highs = highs[-lookback:]

    # Find the cup: rim (start high) -> bottom -> back to rim level
    rim_left = max(segment_highs[:lookback // 4])  # Left rim
    cup_bottom_idx = segment_lows.index(min(segment_lows))
    cup_bottom = min(segment_lows)

    # Cup bottom should be in the middle portion
    if cup_bottom_idx < lookback * 0.2 or cup_bottom_idx > lookback * 0.8:
        return None

    # Right side should recover near left rim level
    right_segment = segment_highs[cup_bottom_idx:]
    if not right_segment:
        return None
    rim_right = max(right_segment)

    rim = min(rim_left, rim_right)
    cup_depth = rim - cup_bottom

    if cup_depth <= 0:
        return None

    # Quality filter 2: cup shouldn't be too deep (V-shape check)
    depth_pct = cup_depth / rim * 100
    if depth_pct > 40:
        return None

    # Quality filter 1: smoothness check (count direction changes)
    direction_changes = 0
    for i in range(1, len(segment) - 1):
        if ((segment[i] - segment[i-1]) * (segment[i+1] - segment[i])) < 0:
            direction_changes += 1
    smoothness = 1.0 - min(direction_changes / lookback, 1.0)

    # Quality filter 4: handle should be above 50% of cup depth
    midpoint = cup_bottom + cup_depth * 0.5
    handle_region = segment[-max(lookback // 6, 3):]
    handle_low = min(handle_region)

    if handle_low < midpoint:
        return None

    # Quality filter 3: no spike in handle
    handle_highs_region = segment_highs[-max(lookback // 6, 3):]
    handle_range = max(handle_highs_region) - min(handle_region)
    if handle_range > cup_depth * 0.5:
        return None

    current_price = closes[-1]
    # Cup target = 50% of cup depth above the rim (per Wealth Training)
    target = rim + cup_depth * 0.5
    stop = handle_low - cup_depth * 0.1

    confidence = 0.60 + smoothness * 0.15
    if depth_pct < 20:
        confidence += 0.05
    if current_price >= rim * 0.98:
        confidence += 0.10

    return ChartPattern(
        name="cup_and_handle",
        pattern_type="reversal",
        direction="bullish",
        confidence=min(confidence, 0.90),
        entry_price=rim,
        target_price=target,
        stop_price=stop,
        pattern_height=cup_depth,
        description=(
            f"Cup & Handle: rim ${rim:,.2f}, depth {depth_pct:.1f}%, "
            f"target ${target:,.2f} (50% of cup depth)"
        ),
    )


def detect_flag_pattern(
    highs: List[float],
    lows: List[float],
    closes: List[float],
) -> Optional[ChartPattern]:
    """
    Detect flag/pennant continuation pattern.

    After a strong move (the pole), price consolidates in a small channel.
    Profit target = pole height projected from breakout.
    """
    if len(closes) < 20:
        return None

    # Look for a strong prior move (the pole)
    lookback = min(len(closes), 40)
    segment = closes[-lookback:]

    # Find the pole: strong directional move in first half
    half = lookback // 2
    pole_start = segment[0]
    pole_end = segment[half]
    pole_height = abs(pole_end - pole_start)
    pole_pct = pole_height / pole_start * 100

    if pole_pct < 2.0:
        return None  # Need meaningful pole

    bullish_pole = pole_end > pole_start

    # Flag: consolidation in second half (smaller range, counter-trend slope)
    flag_segment = segment[half:]
    flag_highs = highs[-lookback + half:]
    flag_lows = lows[-lookback + half:]

    flag_range = max(flag_highs) - min(flag_lows)
    flag_range_pct = flag_range / pole_end * 100

    # Flag should be much smaller than the pole
    if flag_range_pct > pole_pct * 0.5:
        return None

    current_price = closes[-1]

    if bullish_pole:
        target = current_price + pole_height
        stop = min(flag_lows) - pole_height * 0.1
        direction = "bullish"
    else:
        target = current_price - pole_height
        stop = max(flag_highs) + pole_height * 0.1
        direction = "bearish"

    return ChartPattern(
        name="flag",
        pattern_type="continuation",
        direction=direction,
        confidence=0.65,
        entry_price=current_price,
        target_price=target,
        stop_price=stop,
        pattern_height=pole_height,
        description=(
            f"{'Bull' if bullish_pole else 'Bear'} flag: "
            f"pole height ${pole_height:,.2f}, target ${target:,.2f}"
        ),
    )


def detect_chart_patterns(
    highs: List[float],
    lows: List[float],
    closes: List[float],
) -> List[ChartPattern]:
    """Run all chart pattern detectors and return any found patterns."""
    patterns = []
    detectors = [
        detect_double_bottom,
        detect_triple_bottom,
        detect_head_and_shoulders,
        detect_cup_and_handle,
        detect_flag_pattern,
    ]
    for detector in detectors:
        result = detector(highs, lows, closes)
        if result:
            patterns.append(result)
    return patterns


def detect_doji(
    open_price: float,
    high: float,
    low: float,
    close: float,
    atr_value: Optional[float] = None,
) -> bool:
    """
    Detect doji candlestick (from Wealth Training reversal patterns).
    Open and close at nearly the same level, signals indecision/reversal.
    """
    candle_range = high - low
    if candle_range == 0:
        return False
    if atr_value and candle_range < atr_value * 0.3:
        return False
    body = abs(close - open_price)
    return body / candle_range < 0.1


def detect_pipe_bottom(
    opens: List[float],
    highs: List[float],
    lows: List[float],
    closes: List[float],
    atr_value: Optional[float] = None,
) -> bool:
    """
    Detect pipe bottom (from Wealth Training reversal patterns).
    Two adjacent candles with long lower shadows at a market bottom.
    """
    if len(closes) < 2:
        return False

    for i in [-2, -1]:
        candle_range = highs[i] - lows[i]
        if candle_range == 0:
            return False
        lower_wick = min(opens[i], closes[i]) - lows[i]
        if lower_wick / candle_range < 0.5:
            return False

    return True
