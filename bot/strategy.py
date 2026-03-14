"""
Trading strategies — analyze market snapshots and produce trade signals.

Each strategy is independent and can be enabled/disabled.
Start with conservative ones, then add more aggressive ones as you learn.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .config import Config
from .levels import ExitPlan, LevelManager
from .scanner import MarketSnapshot

logger = logging.getLogger(__name__)


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Signal:
    """A concrete trade signal."""
    strategy_name: str
    symbol: str
    side: Side
    price: float
    size_usd: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float          # 0.0 – 1.0
    reason: str
    exit_plan: Optional[ExitPlan] = None  # Multi-level TP via Fibonacci/S/R


class SMACrossoverStrategy:
    """
    SMA Crossover — the classic trend-following strategy.

    BUY when fast SMA crosses above slow SMA (golden cross).
    SELL when fast SMA crosses below slow SMA (death cross).

    Best for: trending markets, higher timeframes (15m+).
    Risk level: Low-Medium.
    """
    NAME = "sma_crossover"

    def __init__(self, config: Config):
        self.config = config

    def _position_size(self) -> float:
        """Risk-based position sizing: risk_pct of capital / stop_loss distance."""
        risk_usd = self.config.starting_capital * (self.config.risk_per_trade_pct / 100)
        size = risk_usd / (self.config.stop_loss_pct / 100)
        return min(size, self.config.max_position_size_usd)

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if snapshot.sma_fast is None or snapshot.sma_slow is None:
            return None

        price = snapshot.current_price
        fast = snapshot.sma_fast
        slow = snapshot.sma_slow
        gap_pct = abs(fast - slow) / slow * 100

        # Need meaningful separation (avoid noise)
        if gap_pct < 0.1:
            return None

        side = None
        reason = ""
        confidence = 0.0

        if fast > slow and snapshot.trend == "bullish":
            side = Side.BUY
            confidence = min(0.5 + gap_pct * 0.05, 0.85)
            reason = (
                f"Golden cross: SMA{self.config.sma_fast_period}"
                f"({fast:,.2f}) > SMA{self.config.sma_slow_period}"
                f"({slow:,.2f}), gap {gap_pct:.2f}%"
            )
            # Boost if RSI confirms
            if snapshot.rsi and snapshot.rsi < 60:
                confidence += 0.05
                reason += f", RSI {snapshot.rsi:.0f} confirms room to run"

        elif fast < slow and snapshot.trend == "bearish":
            side = Side.SELL
            confidence = min(0.5 + gap_pct * 0.05, 0.85)
            reason = (
                f"Death cross: SMA{self.config.sma_fast_period}"
                f"({fast:,.2f}) < SMA{self.config.sma_slow_period}"
                f"({slow:,.2f}), gap {gap_pct:.2f}%"
            )

        if side is None or confidence < self.config.min_confidence:
            return None

        stop_loss = (
            price * (1 - self.config.stop_loss_pct / 100)
            if side == Side.BUY
            else price * (1 + self.config.stop_loss_pct / 100)
        )
        take_profit = (
            price * (1 + self.config.take_profit_pct / 100)
            if side == Side.BUY
            else price * (1 - self.config.take_profit_pct / 100)
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=self._position_size(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class RSIStrategy:
    """
    RSI Mean Reversion — buy oversold, sell overbought.

    BUY when RSI < 30 (oversold).
    SELL when RSI > 70 (overbought).

    Best for: ranging/sideways markets.
    Risk level: Medium.
    """
    NAME = "rsi"

    def __init__(self, config: Config):
        self.config = config

    def _position_size(self) -> float:
        risk_usd = self.config.starting_capital * (self.config.risk_per_trade_pct / 100)
        size = risk_usd / (self.config.stop_loss_pct / 100)
        return min(size, self.config.max_position_size_usd)

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if snapshot.rsi is None:
            return None

        price = snapshot.current_price
        rsi_val = snapshot.rsi
        side = None
        reason = ""
        confidence = 0.0

        if rsi_val < self.config.rsi_oversold:
            side = Side.BUY
            # More oversold = more confident
            intensity = (self.config.rsi_oversold - rsi_val) / self.config.rsi_oversold
            confidence = min(0.55 + intensity * 0.3, 0.90)
            reason = f"RSI oversold at {rsi_val:.1f}"

            # Boost if price near Bollinger lower band
            if snapshot.bb_lower and price <= snapshot.bb_lower * 1.005:
                confidence += 0.05
                reason += f", price near BB lower ({snapshot.bb_lower:,.2f})"

        elif rsi_val > self.config.rsi_overbought:
            side = Side.SELL
            intensity = (rsi_val - self.config.rsi_overbought) / (100 - self.config.rsi_overbought)
            confidence = min(0.55 + intensity * 0.3, 0.90)
            reason = f"RSI overbought at {rsi_val:.1f}"

            if snapshot.bb_upper and price >= snapshot.bb_upper * 0.995:
                confidence += 0.05
                reason += f", price near BB upper ({snapshot.bb_upper:,.2f})"

        if side is None or confidence < self.config.min_confidence:
            return None

        stop_loss = (
            price * (1 - self.config.stop_loss_pct / 100)
            if side == Side.BUY
            else price * (1 + self.config.stop_loss_pct / 100)
        )
        take_profit = (
            price * (1 + self.config.take_profit_pct / 100)
            if side == Side.BUY
            else price * (1 - self.config.take_profit_pct / 100)
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=self._position_size() * 0.7,  # smaller for mean reversion
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class MACDStrategy:
    """
    MACD Momentum — trade momentum shifts.

    BUY when MACD crosses above signal line (bullish momentum).
    SELL when MACD crosses below signal line (bearish momentum).

    Best for: confirming trends, works on all timeframes.
    Risk level: Medium.
    """
    NAME = "macd"

    def __init__(self, config: Config):
        self.config = config

    def _position_size(self) -> float:
        risk_usd = self.config.starting_capital * (self.config.risk_per_trade_pct / 100)
        size = risk_usd / (self.config.stop_loss_pct / 100)
        return min(size, self.config.max_position_size_usd)

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if snapshot.macd_line is None or snapshot.macd_signal is None:
            return None
        if snapshot.macd_histogram is None:
            return None

        price = snapshot.current_price
        hist = snapshot.macd_histogram
        side = None
        reason = ""
        confidence = 0.0

        # Histogram positive = bullish momentum
        if hist > 0 and snapshot.trend == "bullish":
            side = Side.BUY
            confidence = min(0.50 + abs(hist) / price * 1000, 0.80)
            reason = (
                f"MACD bullish: histogram {hist:+.4f}, "
                f"MACD({snapshot.macd_line:.4f}) > "
                f"Signal({snapshot.macd_signal:.4f})"
            )

        elif hist < 0 and snapshot.trend == "bearish":
            side = Side.SELL
            confidence = min(0.50 + abs(hist) / price * 1000, 0.80)
            reason = (
                f"MACD bearish: histogram {hist:+.4f}, "
                f"MACD({snapshot.macd_line:.4f}) < "
                f"Signal({snapshot.macd_signal:.4f})"
            )

        if side is None or confidence < self.config.min_confidence:
            return None

        stop_loss = (
            price * (1 - self.config.stop_loss_pct / 100)
            if side == Side.BUY
            else price * (1 + self.config.stop_loss_pct / 100)
        )
        take_profit = (
            price * (1 + self.config.take_profit_pct / 100)
            if side == Side.BUY
            else price * (1 - self.config.take_profit_pct / 100)
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=self._position_size() * 0.6,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class VolumeSpikeStrategy:
    """
    Volume Spike Breakout — trade when volume surges with price move.

    BUY when volume spikes 2x+ above average with bullish candle.
    SELL when volume spikes 2x+ above average with bearish candle.

    Best for: catching breakouts and momentum moves.
    Risk level: Medium-High.
    """
    NAME = "volume_spike"

    def __init__(self, config: Config):
        self.config = config

    def _position_size(self) -> float:
        risk_usd = self.config.starting_capital * (self.config.risk_per_trade_pct / 100)
        size = risk_usd / (self.config.stop_loss_pct / 100)
        return min(size, self.config.max_position_size_usd)

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if not snapshot.has_volume_spike:
            return None

        price = snapshot.current_price
        candle_bullish = price > snapshot.open_price
        side = None
        reason = ""
        confidence = 0.0

        vol_str = f"Volume spike {snapshot.volume_ratio:.1f}x avg"

        if candle_bullish and snapshot.trend == "bullish":
            side = Side.BUY
            confidence = min(
                0.50 + (snapshot.volume_ratio - 2.0) * 0.1, 0.80
            )
            reason = f"{vol_str} with bullish candle, trend confirms"

        elif not candle_bullish and snapshot.trend == "bearish":
            side = Side.SELL
            confidence = min(
                0.50 + (snapshot.volume_ratio - 2.0) * 0.1, 0.80
            )
            reason = f"{vol_str} with bearish candle, trend confirms"

        if side is None or confidence < self.config.min_confidence:
            return None

        # Wider stops for volatile breakouts
        sl_pct = self.config.stop_loss_pct * 1.5
        tp_pct = self.config.take_profit_pct * 1.5

        stop_loss = (
            price * (1 - sl_pct / 100)
            if side == Side.BUY
            else price * (1 + sl_pct / 100)
        )
        take_profit = (
            price * (1 + tp_pct / 100)
            if side == Side.BUY
            else price * (1 - tp_pct / 100)
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=self._position_size() * 0.5,  # smaller for breakouts
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class StrategyManager:
    """Runs all strategies, aggregates signals, and attaches exit plans."""

    def __init__(self, config: Config):
        self.config = config
        self.level_manager = LevelManager(config)
        self.strategies = [
            SMACrossoverStrategy(config),
            RSIStrategy(config),
            MACDStrategy(config),
            VolumeSpikeStrategy(config),
        ]

    def generate_signals(
        self, snapshot: MarketSnapshot
    ) -> List[Signal]:
        signals = []
        for strategy in self.strategies:
            signal = strategy.evaluate(snapshot)
            if signal:
                # Attach Fibonacci/S/R exit plan
                if snapshot.raw_closes:
                    signal.exit_plan = self.level_manager.calculate_exit_plan(
                        entry_price=signal.price,
                        side=signal.side.value,
                        highs=snapshot.raw_highs,
                        lows=snapshot.raw_lows,
                        closes=snapshot.raw_closes,
                        current_atr=snapshot.atr,
                    )
                    # Override flat SL/TP with fib-based levels
                    if signal.exit_plan:
                        signal.stop_loss = signal.exit_plan.stop_loss.price
                        if signal.exit_plan.take_profits:
                            # Primary TP = first level (nearest)
                            signal.take_profit = (
                                signal.exit_plan.take_profits[0].price
                            )

                logger.info(
                    f"[{strategy.NAME}] {signal.side.value.upper()} "
                    f"@ ${signal.price:,.2f} — {signal.reason}"
                )
                if signal.exit_plan:
                    for tp in signal.exit_plan.take_profits:
                        logger.info(f"  {tp}")

                signals.append(signal)

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
