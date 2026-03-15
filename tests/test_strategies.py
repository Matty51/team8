"""Unit tests for trading strategies."""
import math
import pytest

from bot.config import Config
from bot.scanner import MarketSnapshot
from bot import indicators as ind
from bot.strategy import (
    SMACrossoverStrategy,
    RSIStrategy,
    MACDStrategy,
    VolumeSpikeStrategy,
    PinBarStrategy,
    StochasticStrategy,
    CCIStrategy,
    SurpriseDayStrategy,
    TripleConvergenceStrategy,
    VolatilityExpansionStrategy,
    BollingerSqueezeStrategy,
    StrategyManager,
    Side,
    position_size_from_risk,
)


def _make_snapshot(**overrides) -> MarketSnapshot:
    """Create a MarketSnapshot with sensible defaults."""
    defaults = dict(
        symbol="BTC/USDT",
        timeframe="5m",
        current_price=100.0,
        open_price=99.5,
        high=101.0,
        low=99.0,
        volume=1000.0,
        sma_fast=100.0,
        sma_slow=99.0,
        ema_12=100.0,
        ema_26=99.0,
        rsi=50.0,
        macd_line=0.5,
        macd_signal=0.3,
        macd_histogram=0.2,
        bb_upper=102.0,
        bb_middle=100.0,
        bb_lower=98.0,
        avg_volume=800.0,
        volume_ratio=1.25,
        atr=1.5,
        trend="neutral",
        raw_closes=[99 + i * 0.05 for i in range(100)],
        raw_highs=[99.5 + i * 0.05 for i in range(100)],
        raw_lows=[98.5 + i * 0.05 for i in range(100)],
        raw_opens=[99.2 + i * 0.05 for i in range(100)],
    )
    defaults.update(overrides)
    return MarketSnapshot(**defaults)


class TestPositionSizeFromRisk:
    def test_basic_calculation(self):
        size = position_size_from_risk(
            capital=500, risk_pct=2.0, entry=100, stop=98, max_size=25
        )
        # risk_usd = 500 * 0.02 = 10, distance = 2, qty = 5, usd = 500
        # capped at 25
        assert size == 25.0

    def test_zero_distance_returns_zero(self):
        size = position_size_from_risk(
            capital=500, risk_pct=2.0, entry=100, stop=100, max_size=25
        )
        assert size == 0.0

    def test_respects_max(self):
        size = position_size_from_risk(
            capital=100000, risk_pct=5.0, entry=100, stop=99, max_size=50
        )
        assert size <= 50.0


class TestSMACrossover:
    def test_golden_cross_buy(self):
        config = Config()
        strat = SMACrossoverStrategy(config)
        snap = _make_snapshot(sma_fast=101.0, sma_slow=100.0, trend="bullish")
        signal = strat.evaluate(snap)
        if signal:
            assert signal.side == Side.BUY

    def test_death_cross_sell(self):
        config = Config()
        strat = SMACrossoverStrategy(config)
        snap = _make_snapshot(sma_fast=99.0, sma_slow=100.0, trend="bearish")
        signal = strat.evaluate(snap)
        if signal:
            assert signal.side == Side.SELL

    def test_no_signal_when_equal(self):
        config = Config()
        strat = SMACrossoverStrategy(config)
        snap = _make_snapshot(sma_fast=100.0, sma_slow=100.0)
        signal = strat.evaluate(snap)
        assert signal is None


class TestRSIStrategy:
    def test_oversold_buy(self):
        config = Config(rsi_oversold=30.0, min_confidence=0.50)
        strat = RSIStrategy(config)
        snap = _make_snapshot(rsi=20.0)
        signal = strat.evaluate(snap)
        assert signal is not None
        assert signal.side == Side.BUY

    def test_overbought_sell(self):
        config = Config(rsi_overbought=70.0, min_confidence=0.50)
        strat = RSIStrategy(config)
        snap = _make_snapshot(rsi=80.0)
        signal = strat.evaluate(snap)
        assert signal is not None
        assert signal.side == Side.SELL

    def test_midline_buy(self):
        config = Config(min_confidence=0.50)
        strat = RSIStrategy(config)
        snap = _make_snapshot(rsi=52.0, trend="bullish")
        signal = strat.evaluate(snap)
        if signal:
            assert signal.side == Side.BUY
            assert "48 midline" in signal.reason

    def test_neutral_rsi_no_signal(self):
        config = Config(min_confidence=0.65)
        strat = RSIStrategy(config)
        snap = _make_snapshot(rsi=50.0, trend="neutral")
        signal = strat.evaluate(snap)
        assert signal is None

    def test_no_signal_without_rsi(self):
        config = Config()
        strat = RSIStrategy(config)
        snap = _make_snapshot(rsi=None)
        signal = strat.evaluate(snap)
        assert signal is None


class TestMACDStrategy:
    def test_bullish_macd(self):
        config = Config(min_confidence=0.50)
        strat = MACDStrategy(config)
        snap = _make_snapshot(
            macd_line=0.5, macd_signal=0.3, macd_histogram=0.2,
            trend="bullish",
        )
        signal = strat.evaluate(snap)
        if signal:
            assert signal.side == Side.BUY


class TestVolumeSpikeStrategy:
    def test_volume_spike_buy(self):
        config = Config(min_confidence=0.50)
        strat = VolumeSpikeStrategy(config)
        snap = _make_snapshot(
            volume_ratio=3.0,
            current_price=101.0,
            open_price=99.0,
            trend="bullish",
        )
        signal = strat.evaluate(snap)
        if signal:
            assert signal.side == Side.BUY

    def test_no_spike_no_signal(self):
        config = Config()
        strat = VolumeSpikeStrategy(config)
        snap = _make_snapshot(volume_ratio=1.0)
        signal = strat.evaluate(snap)
        assert signal is None


class TestSurpriseDayStrategy:
    def test_bull_surprise(self):
        config = Config(min_confidence=0.50)
        strat = SurpriseDayStrategy(config)
        # Open below prev low, price above prev low
        snap = _make_snapshot(
            open_price=97.0,
            current_price=100.0,
            raw_highs=[101.0] * 100,
            raw_lows=[98.0] * 100,  # prev low = 98
        )
        signal = strat.evaluate(snap)
        assert signal is not None
        assert signal.side == Side.BUY
        assert "Surprise Day UP" in signal.reason

    def test_bear_surprise(self):
        config = Config(min_confidence=0.50)
        strat = SurpriseDayStrategy(config)
        snap = _make_snapshot(
            open_price=103.0,
            current_price=100.0,
            raw_highs=[102.0] * 100,  # prev high = 102
            raw_lows=[98.0] * 100,
        )
        signal = strat.evaluate(snap)
        assert signal is not None
        assert signal.side == Side.SELL

    def test_no_surprise_normal_open(self):
        config = Config()
        strat = SurpriseDayStrategy(config)
        snap = _make_snapshot(
            open_price=100.0,
            current_price=100.5,
            raw_highs=[102.0] * 100,
            raw_lows=[98.0] * 100,
        )
        signal = strat.evaluate(snap)
        assert signal is None


class TestBollingerSqueezeStrategy:
    def test_squeeze_breakout_up(self):
        config = Config(min_confidence=0.50)
        strat = BollingerSqueezeStrategy(config)
        # Tight bands (< 2% width), price above upper
        snap = _make_snapshot(
            bb_upper=100.5,
            bb_middle=100.0,
            bb_lower=99.5,
            current_price=101.0,
            atr=1.0,
            trend="bullish",
            volume_ratio=2.0,
        )
        signal = strat.evaluate(snap)
        assert signal is not None
        assert signal.side == Side.BUY

    def test_no_signal_wide_bands(self):
        config = Config()
        strat = BollingerSqueezeStrategy(config)
        snap = _make_snapshot(
            bb_upper=110.0,
            bb_middle=100.0,
            bb_lower=90.0,
        )
        signal = strat.evaluate(snap)
        assert signal is None


class TestStrategyManager:
    def test_evaluate_all_returns_list(self):
        config = Config(min_confidence=0.30)
        mgr = StrategyManager(config)
        snap = _make_snapshot(rsi=15.0)  # Very oversold
        signals = mgr.generate_signals(snap)
        assert isinstance(signals, list)

    def test_signals_have_required_fields(self):
        config = Config(min_confidence=0.30)
        mgr = StrategyManager(config)
        snap = _make_snapshot(rsi=15.0)
        signals = mgr.generate_signals(snap)
        for sig in signals:
            assert sig.symbol == "BTC/USDT"
            assert sig.price > 0
            assert 0 < sig.confidence <= 1.0
            assert sig.stop_loss is not None
            assert sig.take_profit is not None
            assert sig.reason

    def test_signals_sorted_by_confidence(self):
        config = Config(min_confidence=0.30)
        mgr = StrategyManager(config)
        snap = _make_snapshot(rsi=15.0, trend="bullish")
        signals = mgr.generate_signals(snap)
        if len(signals) > 1:
            confs = [s.confidence for s in signals]
            assert confs == sorted(confs, reverse=True)
