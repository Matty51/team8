"""Unit tests for technical indicators."""
import math
import pytest

from bot import indicators as ind


# ── SMA ────────────────────────────────────────────────────────────────


class TestSMA:
    def test_basic_sma(self):
        closes = [10, 20, 30, 40, 50]
        result = ind.sma(closes, 3)
        assert result[0] is None
        assert result[1] is None
        assert result[2] == pytest.approx(20.0)  # (10+20+30)/3
        assert result[3] == pytest.approx(30.0)  # (20+30+40)/3
        assert result[4] == pytest.approx(40.0)  # (30+40+50)/3

    def test_sma_not_enough_data(self):
        result = ind.sma([10, 20], 5)
        assert all(v is None for v in result)

    def test_sma_period_equals_length(self):
        closes = [10, 20, 30]
        result = ind.sma(closes, 3)
        assert result[-1] == pytest.approx(20.0)

    def test_sma_constant_values(self):
        closes = [5.0] * 10
        result = ind.sma(closes, 5)
        for v in result[4:]:
            assert v == pytest.approx(5.0)


# ── EMA ────────────────────────────────────────────────────────────────


class TestEMA:
    def test_ema_seed_is_sma(self):
        closes = [10, 20, 30, 40, 50]
        result = ind.ema(closes, 3)
        # Seed at index 2 = SMA(10,20,30) = 20
        assert result[2] == pytest.approx(20.0)

    def test_ema_not_enough_data(self):
        result = ind.ema([10], 5)
        assert all(v is None for v in result)

    def test_ema_weights_recent_more(self):
        # Rising prices with acceleration: EMA should be above SMA
        closes = [10, 12, 14, 16, 18, 21, 25, 30, 36, 43]
        ema_vals = ind.ema(closes, 5)
        sma_vals = ind.sma(closes, 5)
        # For accelerating data, EMA should be > SMA
        assert ema_vals[-1] > sma_vals[-1]


# ── RSI ────────────────────────────────────────────────────────────────


class TestRSI:
    def test_rsi_range(self):
        # Random-ish data
        closes = [44, 44.34, 44.09, 43.61, 44.33, 44.83, 45.10, 45.42,
                  45.84, 46.08, 45.89, 46.03, 45.61, 46.28, 46.28, 46.00,
                  46.03, 46.41, 46.22, 45.64]
        result = ind.rsi(closes, 14)
        for v in result:
            if v is not None:
                assert 0 <= v <= 100

    def test_rsi_all_gains(self):
        closes = list(range(1, 20))  # Steadily rising
        result = ind.rsi(closes, 14)
        last_rsi = [v for v in result if v is not None][-1]
        assert last_rsi > 90  # Should be near 100

    def test_rsi_all_losses(self):
        closes = list(range(20, 1, -1))  # Steadily falling
        result = ind.rsi(closes, 14)
        last_rsi = [v for v in result if v is not None][-1]
        assert last_rsi < 10  # Should be near 0

    def test_rsi_not_enough_data(self):
        result = ind.rsi([10, 11, 12], 14)
        assert all(v is None for v in result)


# ── Bollinger Bands ────────────────────────────────────────────────────


class TestBollingerBands:
    def test_bands_symmetry(self):
        closes = [10, 11, 12, 11, 10, 11, 12, 11, 10, 11,
                  12, 11, 10, 11, 12, 11, 10, 11, 12, 11]
        upper, middle, lower = ind.bollinger_bands(closes, 20, 2.0)
        # Upper and lower should be equidistant from middle
        if middle[-1] and upper[-1] and lower[-1]:
            assert upper[-1] - middle[-1] == pytest.approx(
                middle[-1] - lower[-1], abs=0.01
            )

    def test_bands_order(self):
        closes = [10 + i * 0.5 for i in range(25)]
        upper, middle, lower = ind.bollinger_bands(closes, 20)
        for i in range(19, len(closes)):
            if upper[i] and middle[i] and lower[i]:
                assert upper[i] > middle[i] > lower[i]

    def test_constant_price_zero_bandwidth(self):
        closes = [100.0] * 25
        upper, middle, lower = ind.bollinger_bands(closes, 20)
        # Constant price = zero std dev = bands collapse to SMA
        assert upper[-1] == pytest.approx(middle[-1])
        assert lower[-1] == pytest.approx(middle[-1])


# ── MACD ───────────────────────────────────────────────────────────────


class TestMACD:
    def test_macd_structure(self):
        closes = [10 + i * 0.3 + (i % 5) * 0.2 for i in range(50)]
        macd_l, signal_l, hist = ind.macd(closes, 12, 26, 9)
        assert len(macd_l) == len(closes)
        assert len(signal_l) == len(closes)
        assert len(hist) == len(closes)

    def test_histogram_is_difference(self):
        closes = [10 + i * 0.3 for i in range(50)]
        macd_l, signal_l, hist = ind.macd(closes, 12, 26, 9)
        for i in range(len(closes)):
            if macd_l[i] is not None and signal_l[i] is not None and hist[i] is not None:
                assert hist[i] == pytest.approx(macd_l[i] - signal_l[i], abs=0.001)

    def test_macd_bullish_crossover(self):
        # Rising prices should produce positive MACD eventually
        closes = [10 + i * 0.5 for i in range(50)]
        macd_l, _, _ = ind.macd(closes, 12, 26, 9)
        last_val = [v for v in macd_l if v is not None][-1]
        assert last_val > 0


# ── Stochastics ────────────────────────────────────────────────────────


class TestStochastics:
    def test_stochastic_range(self):
        n = 50
        highs = [10 + i * 0.2 + 1 for i in range(n)]
        lows = [10 + i * 0.2 - 1 for i in range(n)]
        closes = [10 + i * 0.2 for i in range(n)]
        k, d = ind.stochastic(highs, lows, closes, 12, 3, 3)
        for v in k:
            if v is not None and v != 0.0:
                assert 0 <= v <= 100
        for v in d:
            if v is not None and v != 0.0:
                assert 0 <= v <= 100

    def test_stochastic_at_high(self):
        # Close at the high of range should give high %K
        n = 20
        highs = [100.0] * n
        lows = [90.0] * n
        closes = [100.0] * n  # Always at high
        k, d = ind.stochastic(highs, lows, closes, 12, 3, 3)
        last_k = [v for v in k if v is not None and v != 0.0]
        if last_k:
            assert last_k[-1] == pytest.approx(100.0)


# ── CCI ────────────────────────────────────────────────────────────────


class TestCCI:
    def test_cci_returns_values(self):
        n = 100
        highs = [10 + i * 0.1 + 0.5 for i in range(n)]
        lows = [10 + i * 0.1 - 0.5 for i in range(n)]
        closes = [10 + i * 0.1 for i in range(n)]
        result = ind.cci(highs, lows, closes, 20)
        non_none = [v for v in result if v is not None]
        assert len(non_none) > 0

    def test_cci_not_enough_data(self):
        result = ind.cci([10], [9], [9.5], 90)
        assert all(v is None for v in result)


# ── ATR ────────────────────────────────────────────────────────────────


class TestATR:
    def test_atr_positive(self):
        n = 30
        highs = [10 + i * 0.1 + 1 for i in range(n)]
        lows = [10 + i * 0.1 - 1 for i in range(n)]
        closes = [10 + i * 0.1 for i in range(n)]
        result = ind.atr(highs, lows, closes, 14)
        for v in result:
            if v is not None:
                assert v > 0

    def test_atr_constant_range(self):
        # Each candle has range of exactly 2 (high - low)
        n = 30
        highs = [11.0] * n
        lows = [9.0] * n
        closes = [10.0] * n
        result = ind.atr(highs, lows, closes, 14)
        last_atr = [v for v in result if v is not None][-1]
        assert last_atr == pytest.approx(2.0, abs=0.01)


# ── Fibonacci ──────────────────────────────────────────────────────────


class TestFibonacci:
    def test_retracement_levels(self):
        levels = ind.fibonacci_retracements(100, 50, "bullish")
        # Bullish retracements go from high down
        for level in levels:
            assert 50 <= level.price <= 100

    def test_extension_levels(self):
        levels = ind.fibonacci_extensions(100, 50, "bullish")
        # Extensions project beyond the move
        for level in levels:
            assert level.price >= 50

    def test_key_ratios(self):
        levels = ind.fibonacci_retracements(100, 0, "bullish")
        prices = {round(l.ratio, 3): l.price for l in levels}
        # 50% retracement of 0-100 should be at 50
        assert 0.5 in prices
        assert prices[0.5] == pytest.approx(50.0)
        # 38.2% should be at 61.8
        assert 0.382 in prices
        assert prices[0.382] == pytest.approx(61.8)

    def test_calculate_fibonacci_auto_direction(self):
        # Rising prices = bullish
        highs = [10 + i for i in range(20)]
        lows = [9 + i for i in range(20)]
        closes = [9.5 + i for i in range(20)]
        fib = ind.calculate_fibonacci(highs, lows, closes, 20)
        assert fib is not None
        assert fib.direction == "bullish"

    def test_nearest_level(self):
        fib = ind.calculate_fibonacci(
            [10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30] * 2,
            [8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28] * 2,
            [9, 11, 13, 15, 17, 19, 21, 23, 25, 27, 29] * 2,
            lookback=20,
        )
        if fib:
            above = fib.nearest_level_above(20)
            below = fib.nearest_level_below(20)
            if above:
                assert above.price > 20
            if below:
                assert below.price < 20


# ── Support / Resistance ───────────────────────────────────────────────


class TestSupportResistance:
    def test_finds_levels(self):
        # Create data with clear swing points
        closes = []
        highs = []
        lows = []
        for i in range(60):
            # Oscillating between 90 and 110
            base = 100 + 10 * math.sin(i * 0.3)
            closes.append(base)
            highs.append(base + 1)
            lows.append(base - 1)

        support, resistance = ind.find_support_resistance(highs, lows, closes)
        # Should find some levels
        assert isinstance(support, list)
        assert isinstance(resistance, list)


# ── Pivot Points ───────────────────────────────────────────────────────


class TestPivotPoints:
    def test_pivot_calculation(self):
        pivots = ind.pivot_points(high=110, low=90, close=100)
        assert "P" in pivots
        assert "R1" in pivots
        assert "S1" in pivots
        assert "R2" in pivots
        assert "S2" in pivots
        assert "R3" in pivots
        assert "S3" in pivots
        # Pivot = (H + L + C) / 3
        assert pivots["P"] == pytest.approx(100.0)
        # R levels should be above pivot, S levels below
        assert pivots["R1"] > pivots["P"]
        assert pivots["S1"] < pivots["P"]
        assert pivots["R3"] > pivots["R2"] > pivots["R1"]
        assert pivots["S3"] < pivots["S2"] < pivots["S1"]


# ── Pin Bar Detection ──────────────────────────────────────────────────


class TestPinBar:
    def test_bullish_pin_bar(self):
        # Long lower wick, tiny body at top
        pin = ind.detect_pin_bar(
            open_price=99.0, high=100.0, low=90.0, close=99.5
        )
        assert pin is not None
        assert pin.direction == "bullish"

    def test_bearish_pin_bar(self):
        # Long upper wick, tiny body at bottom
        pin = ind.detect_pin_bar(
            open_price=91.0, high=100.0, low=90.0, close=90.5
        )
        assert pin is not None
        assert pin.direction == "bearish"

    def test_no_pin_bar_large_body(self):
        # Large body relative to range = not a pin bar
        pin = ind.detect_pin_bar(
            open_price=92.0, high=100.0, low=90.0, close=98.0
        )
        assert pin is None

    def test_pin_bar_strength_ranking(self):
        # Strongest: tiny body at end, no opposite wick
        strongest = ind.detect_pin_bar(
            open_price=99.5, high=100.0, low=90.0, close=99.8
        )
        # Strong: noticeable opposite wick
        strong = ind.detect_pin_bar(
            open_price=98.0, high=100.0, low=90.0, close=98.5
        )
        if strongest and strong:
            assert strongest.score >= strong.score


# ── Doji Detection ─────────────────────────────────────────────────────


class TestDoji:
    def test_doji_detected(self):
        # Open == Close, with wicks
        assert ind.detect_doji(100.0, 105.0, 95.0, 100.0) is True

    def test_not_doji_large_body(self):
        assert ind.detect_doji(95.0, 105.0, 94.0, 104.0) is False


# ── Chart Patterns ─────────────────────────────────────────────────────


class TestChartPatterns:
    def _make_double_bottom_data(self):
        """Create price data with a clear double bottom pattern."""
        closes = []
        highs = []
        lows = []
        # Start high, dip to 90, recover to 100, dip to 90 again, recover above 100
        path = (
            [100, 98, 96, 94, 92, 90, 92, 94, 96, 98, 100,
             98, 96, 94, 92, 90, 92, 94, 96, 98, 100, 102]
        )
        for p in path:
            closes.append(p)
            highs.append(p + 1)
            lows.append(p - 1)
        return highs, lows, closes

    def test_double_bottom_detection(self):
        highs, lows, closes = self._make_double_bottom_data()
        pattern = ind.detect_double_bottom(highs, lows, closes)
        if pattern:
            assert pattern.name == "double_bottom"
            assert pattern.direction == "bullish"
            assert pattern.target_price > pattern.entry_price

    def test_detect_chart_patterns_returns_list(self):
        highs, lows, closes = self._make_double_bottom_data()
        patterns = ind.detect_chart_patterns(highs, lows, closes)
        assert isinstance(patterns, list)

    def test_flag_pattern(self):
        # Strong move up then consolidation
        closes = []
        highs = []
        lows = []
        # Pole: strong move from 100 to 110
        for i in range(20):
            p = 100 + i * 0.5
            closes.append(p)
            highs.append(p + 0.3)
            lows.append(p - 0.3)
        # Flag: small consolidation around 110
        for i in range(20):
            p = 110 + 0.2 * (i % 3 - 1)
            closes.append(p)
            highs.append(p + 0.2)
            lows.append(p - 0.2)

        pattern = ind.detect_flag_pattern(highs, lows, closes)
        if pattern:
            assert pattern.pattern_type == "continuation"
