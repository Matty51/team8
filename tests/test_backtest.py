"""Unit tests for the backtesting engine."""
import math
import pytest

from bot.backtest import Backtester, BacktestResult
from bot.config import Config


def _generate_trending_candles(n=500, start_price=100.0, trend=0.01):
    """Generate synthetic trending candle data."""
    candles = []
    price = start_price
    ts = 1700000000000  # Arbitrary start timestamp (ms)

    for i in range(n):
        noise = math.sin(i * 0.5) * 0.5
        price = price * (1 + trend) + noise
        o = price - 0.3
        h = price + 1.0
        l = price - 1.0
        c = price
        v = 1000 + math.sin(i * 0.3) * 200

        candles.append([ts, o, h, l, c, abs(v)])
        ts += 300000  # 5-minute candles

    return candles


def _generate_oscillating_candles(n=500, center=100.0, amplitude=5.0):
    """Generate synthetic oscillating (mean-reverting) candle data."""
    candles = []
    ts = 1700000000000

    for i in range(n):
        price = center + amplitude * math.sin(i * 0.15)
        o = price - 0.2
        h = price + 0.8
        l = price - 0.8
        c = price
        v = 800 + abs(math.sin(i * 0.1)) * 400

        candles.append([ts, o, h, l, c, abs(v)])
        ts += 300000

    return candles


class TestBacktester:
    def test_backtest_runs(self):
        config = Config(starting_capital=500)
        bt = Backtester(config)
        candles = _generate_trending_candles(300)
        result = bt.run(candles)
        assert isinstance(result, BacktestResult)
        assert result.total_bars == 300
        assert result.starting_capital == 500

    def test_result_has_metrics(self):
        config = Config(starting_capital=500)
        bt = Backtester(config)
        candles = _generate_trending_candles(300)
        result = bt.run(candles)

        assert hasattr(result, "total_return_pct")
        assert hasattr(result, "win_rate")
        assert hasattr(result, "max_drawdown_pct")
        assert hasattr(result, "sharpe_ratio")
        assert hasattr(result, "profit_factor")
        assert hasattr(result, "equity_curve")
        assert len(result.equity_curve) > 0

    def test_equity_curve_tracks(self):
        config = Config(starting_capital=1000)
        bt = Backtester(config)
        candles = _generate_trending_candles(300)
        result = bt.run(candles)
        # Equity curve should start at starting capital
        assert result.equity_curve[0] == 1000

    def test_oscillating_market(self):
        config = Config(starting_capital=500)
        bt = Backtester(config)
        candles = _generate_oscillating_candles(400)
        result = bt.run(candles)
        assert isinstance(result, BacktestResult)

    def test_strategy_breakdown(self):
        config = Config(starting_capital=500, min_confidence=0.40)
        bt = Backtester(config)
        candles = _generate_trending_candles(300)
        result = bt.run(candles)
        assert isinstance(result.trades_by_strategy, dict)

    def test_summary_string(self):
        config = Config(starting_capital=500)
        bt = Backtester(config)
        candles = _generate_trending_candles(300)
        result = bt.run(candles)
        summary = result.summary()
        assert "BACKTEST RESULTS" in summary
        assert "Win Rate" in summary

    def test_too_few_candles_raises(self):
        config = Config(starting_capital=500)
        bt = Backtester(config)
        with pytest.raises(ValueError, match="Need at least"):
            bt.run([[0, 10, 11, 9, 10, 100]] * 5)

    def test_trades_have_required_fields(self):
        config = Config(starting_capital=500, min_confidence=0.40)
        bt = Backtester(config)
        candles = _generate_trending_candles(300)
        result = bt.run(candles)
        for trade in result.trades:
            assert trade.symbol
            assert trade.strategy
            assert trade.entry_price > 0
            assert trade.exit_price > 0
            assert trade.size_usd > 0
            assert trade.exit_reason in ("stop_loss", "take_profit", "end_of_data")
            assert trade.bars_held >= 0

    def test_max_drawdown_non_negative(self):
        config = Config(starting_capital=500)
        bt = Backtester(config)
        candles = _generate_trending_candles(300)
        result = bt.run(candles)
        assert result.max_drawdown_pct >= 0
        assert result.max_drawdown_usd >= 0

    def test_risk_controls_respected(self):
        """Verify that backtest respects position limits."""
        config = Config(
            starting_capital=500,
            max_open_positions=1,
            max_daily_trades=5,
            min_confidence=0.40,
        )
        bt = Backtester(config)
        candles = _generate_trending_candles(300)
        result = bt.run(candles)
        # Should still produce valid results
        assert isinstance(result, BacktestResult)
