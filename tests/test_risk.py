"""Unit tests for risk management."""
import pytest

from bot.config import Config
from bot.risk import DailyStats, Position, RiskManager
from bot.strategy import Side, Signal


def _make_signal(
    symbol="BTC/USDT",
    side=Side.BUY,
    price=100.0,
    size_usd=10.0,
    stop_loss=98.0,
    take_profit=104.0,
    confidence=0.70,
) -> Signal:
    return Signal(
        strategy_name="test",
        symbol=symbol,
        side=side,
        price=price,
        size_usd=size_usd,
        stop_loss=stop_loss,
        take_profit=take_profit,
        confidence=confidence,
        reason="test signal",
    )


class TestPositionSizing:
    def test_basic_position_size(self):
        config = Config(starting_capital=500, risk_per_trade_pct=2.0,
                        stop_loss_pct=1.5, max_position_size_usd=25.0)
        rm = RiskManager(config)
        size = rm.calculate_position_size(500)
        # Risk = 500 * 0.02 = $10
        # Size = $10 / 0.015 = $666.67 -> capped at $25
        assert size == 25.0

    def test_position_size_respects_cap(self):
        config = Config(starting_capital=10000, risk_per_trade_pct=5.0,
                        stop_loss_pct=0.5, max_position_size_usd=50.0)
        rm = RiskManager(config)
        size = rm.calculate_position_size(10000)
        assert size <= 50.0

    def test_position_size_scales_with_balance(self):
        config = Config(starting_capital=1000, risk_per_trade_pct=2.0,
                        stop_loss_pct=1.5, max_position_size_usd=10000.0)
        rm = RiskManager(config)
        size_500 = rm.calculate_position_size(500)
        size_1000 = rm.calculate_position_size(1000)
        assert size_1000 > size_500


class TestRiskChecks:
    def test_signal_passes_checks(self):
        config = Config()
        rm = RiskManager(config)
        signal = _make_signal()
        allowed, reason = rm.check_signal(signal)
        assert allowed is True

    def test_daily_trade_limit(self):
        config = Config(max_daily_trades=2)
        rm = RiskManager(config)
        rm.daily_stats.trades_count = 2
        rm.daily_stats.date = __import__("datetime").date.today().isoformat()
        signal = _make_signal()
        allowed, reason = rm.check_signal(signal)
        assert allowed is False
        assert "Daily trade limit" in reason

    def test_daily_loss_limit(self):
        config = Config(max_daily_loss_usd=25.0)
        rm = RiskManager(config)
        rm.daily_stats.realized_pnl = -25.0
        rm.daily_stats.date = __import__("datetime").date.today().isoformat()
        signal = _make_signal()
        allowed, reason = rm.check_signal(signal)
        assert allowed is False
        assert "Daily loss limit" in reason

    def test_max_open_positions(self):
        config = Config(max_open_positions=1)
        rm = RiskManager(config)
        # Add a position
        rm.positions["ETH/USDT"] = Position(
            symbol="ETH/USDT", side="buy", entry_price=3000,
            size_usd=20, amount=0.006, stop_loss=2900,
            take_profit=3200, strategy="test", timestamp="2024-01-01",
        )
        signal = _make_signal(symbol="BTC/USDT")
        allowed, reason = rm.check_signal(signal)
        assert allowed is False
        assert "Max open positions" in reason

    def test_duplicate_position_blocked(self):
        config = Config()
        rm = RiskManager(config)
        rm.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT", side="buy", entry_price=100,
            size_usd=10, amount=0.1, stop_loss=98,
            take_profit=104, strategy="test", timestamp="2024-01-01",
        )
        signal = _make_signal(symbol="BTC/USDT")
        allowed, reason = rm.check_signal(signal)
        assert allowed is False
        assert "Already have" in reason

    def test_confidence_threshold(self):
        config = Config(min_confidence=0.60)
        rm = RiskManager(config)
        signal = _make_signal(confidence=0.50)
        allowed, reason = rm.check_signal(signal)
        assert allowed is False
        assert "Confidence" in reason

    def test_max_risk_per_trade(self):
        config = Config(max_risk_per_trade_pct=5.0)
        rm = RiskManager(config)
        # 10% risk distance
        signal = _make_signal(price=100.0, stop_loss=90.0)
        allowed, reason = rm.check_signal(signal)
        assert allowed is False
        assert "risk" in reason.lower()

    def test_reward_risk_ratio(self):
        config = Config(min_reward_risk_ratio=2.0)
        rm = RiskManager(config)
        # 1:1 R:R (risk=2, reward=2)
        signal = _make_signal(price=100, stop_loss=98, take_profit=102)
        allowed, reason = rm.check_signal(signal)
        assert allowed is False
        assert "R:R ratio" in reason

    def test_position_size_exceeds_max(self):
        config = Config(max_position_size_usd=25.0)
        rm = RiskManager(config)
        signal = _make_signal(size_usd=50.0)
        allowed, reason = rm.check_signal(signal)
        assert allowed is False
        assert "exceeds max" in reason


class TestStopLossTakeProfit:
    def test_stop_loss_triggered_buy(self):
        config = Config()
        rm = RiskManager(config)
        rm.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT", side="buy", entry_price=100,
            size_usd=10, amount=0.1, stop_loss=95,
            take_profit=110, strategy="test", timestamp="2024-01-01",
        )
        result = rm.check_stop_loss_take_profit("BTC/USDT", 94.0)
        assert result == "stop_loss"

    def test_stop_loss_triggered_sell(self):
        config = Config()
        rm = RiskManager(config)
        rm.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT", side="sell", entry_price=100,
            size_usd=10, amount=0.1, stop_loss=105,
            take_profit=90, strategy="test", timestamp="2024-01-01",
        )
        result = rm.check_stop_loss_take_profit("BTC/USDT", 106.0)
        assert result == "stop_loss"

    def test_take_profit_triggered(self):
        config = Config()
        rm = RiskManager(config)
        rm.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT", side="buy", entry_price=100,
            size_usd=10, amount=0.1, stop_loss=95,
            take_profit=110, strategy="test", timestamp="2024-01-01",
        )
        result = rm.check_stop_loss_take_profit("BTC/USDT", 111.0)
        assert result == "take_profit_final"

    def test_no_exit_in_range(self):
        config = Config()
        rm = RiskManager(config)
        rm.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT", side="buy", entry_price=100,
            size_usd=10, amount=0.1, stop_loss=95,
            take_profit=110, strategy="test", timestamp="2024-01-01",
        )
        result = rm.check_stop_loss_take_profit("BTC/USDT", 102.0)
        assert result is None


class TestPositionTracking:
    def test_record_and_close_position(self):
        config = Config()
        rm = RiskManager(config)
        signal = _make_signal(price=100, size_usd=10)
        rm.record_trade(signal, fill_price=100.0, amount=0.1)

        assert "BTC/USDT" in rm.positions
        assert rm.daily_stats.trades_count == 1

        # Close with profit
        pnl = rm.close_position("BTC/USDT", exit_price=105.0, reason="tp")
        assert pnl > 0
        assert "BTC/USDT" not in rm.positions
        assert rm.total_pnl > 0

    def test_close_with_loss(self):
        config = Config()
        rm = RiskManager(config)
        signal = _make_signal(price=100, size_usd=10)
        rm.record_trade(signal, fill_price=100.0, amount=0.1)

        pnl = rm.close_position("BTC/USDT", exit_price=95.0, reason="sl")
        assert pnl < 0

    def test_total_risk_calculation(self):
        config = Config()
        rm = RiskManager(config)
        rm.positions["BTC/USDT"] = Position(
            symbol="BTC/USDT", side="buy", entry_price=100,
            size_usd=10, amount=0.1, stop_loss=95,
            take_profit=110, strategy="test", timestamp="2024-01-01",
        )
        risk = rm.get_total_risk_usd()
        # Risk = (100 - 95) * 0.1 = 0.5
        assert risk == pytest.approx(0.5)


class TestDailyStats:
    def test_reset_on_new_day(self):
        stats = DailyStats(date="2020-01-01", trades_count=5, realized_pnl=-10.0)
        stats.reset_if_new_day()
        # Should reset since date is old
        assert stats.trades_count == 0
        assert stats.realized_pnl == 0.0
