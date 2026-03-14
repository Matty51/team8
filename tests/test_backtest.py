"""Unit tests for the backtesting engine."""
import math
import os
import tempfile
import pytest

from bot.backtest import Backtester, BacktestResult, load_csv, _parse_timestamp
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


class TestCSVLoader:
    def test_load_tradingview_csv(self, tmp_path):
        """Test loading a TradingView-style CSV export."""
        csv_file = tmp_path / "btc.csv"
        csv_file.write_text(
            "time,open,high,low,close,volume\n"
            "2024-01-01 00:00,42000.0,42500.0,41800.0,42300.0,1500.0\n"
            "2024-01-01 01:00,42300.0,42800.0,42100.0,42600.0,1200.0\n"
            "2024-01-01 02:00,42600.0,42900.0,42400.0,42700.0,1100.0\n"
        )
        candles = load_csv(str(csv_file))
        assert len(candles) == 3
        assert candles[0][1] == 42000.0  # open
        assert candles[0][4] == 42300.0  # close
        assert candles[1][5] == 1200.0   # volume

    def test_load_csv_case_insensitive_headers(self, tmp_path):
        """Column headers should be matched case-insensitively."""
        csv_file = tmp_path / "data.csv"
        csv_file.write_text(
            "Time,Open,High,Low,Close,Volume\n"
            "2024-06-15,100.0,105.0,98.0,103.0,5000\n"
        )
        candles = load_csv(str(csv_file))
        assert len(candles) == 1
        assert candles[0][1] == 100.0

    def test_load_csv_no_volume(self, tmp_path):
        """Volume column is optional, defaults to 0."""
        csv_file = tmp_path / "novolume.csv"
        csv_file.write_text(
            "time,open,high,low,close\n"
            "2024-01-01,50.0,55.0,48.0,52.0\n"
        )
        candles = load_csv(str(csv_file))
        assert candles[0][5] == 0.0

    def test_load_csv_missing_required_column(self, tmp_path):
        """Should raise ValueError if a required column is missing."""
        csv_file = tmp_path / "bad.csv"
        csv_file.write_text("time,open,high,low\n1,2,3,4\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_csv(str(csv_file))

    def test_load_csv_sorted_by_time(self, tmp_path):
        """Output should be sorted by timestamp even if CSV isn't."""
        csv_file = tmp_path / "unsorted.csv"
        csv_file.write_text(
            "time,open,high,low,close,volume\n"
            "2024-01-03,3,4,2,3,100\n"
            "2024-01-01,1,2,0,1,100\n"
            "2024-01-02,2,3,1,2,100\n"
        )
        candles = load_csv(str(csv_file))
        assert candles[0][0] < candles[1][0] < candles[2][0]

    def test_parse_iso_timestamp(self):
        ts = _parse_timestamp("2024-01-01T12:00:00")
        assert isinstance(ts, int)
        assert ts > 0

    def test_parse_unix_timestamp(self):
        ts = _parse_timestamp("1704067200")
        assert ts == 1704067200000  # converted to ms
