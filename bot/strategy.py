"""
Trading strategies — analyze market snapshots and produce trade signals.

Each strategy is independent and can be enabled/disabled.
Start with conservative ones, then add more aggressive ones as you learn.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from . import indicators as ind
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

    # Position sizing now handled by position_size_from_risk()

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

        size_usd = position_size_from_risk(
            capital=self.config.starting_capital,
            risk_pct=self.config.risk_per_trade_pct,
            entry=price,
            stop=stop_loss,
            max_size=self.config.max_position_size_usd,
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=size_usd,
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

        size_usd = position_size_from_risk(
            capital=self.config.starting_capital,
            risk_pct=self.config.risk_per_trade_pct * 0.7,  # smaller for mean reversion
            entry=price,
            stop=stop_loss,
            max_size=self.config.max_position_size_usd,
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class MACDStrategy:
    """
    MACD Momentum — trade momentum shifts.

    From Wealth Training course: MACD (5, 35, 5).
    BUY when MACD crosses above signal line (bullish momentum).
    SELL when MACD crosses below signal line (bearish momentum).
    Also monitors divergences between price and MACD.

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

        size_usd = position_size_from_risk(
            capital=self.config.starting_capital,
            risk_pct=self.config.risk_per_trade_pct * 0.6,
            entry=price,
            stop=stop_loss,
            max_size=self.config.max_position_size_usd,
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=size_usd,
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

        size_usd = position_size_from_risk(
            capital=self.config.starting_capital,
            risk_pct=self.config.risk_per_trade_pct * 0.5,  # smaller for breakouts
            entry=price,
            stop=stop_loss,
            max_size=self.config.max_position_size_usd,
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


def position_size_from_risk(
    capital: float,
    risk_pct: float,
    entry: float,
    stop: float,
    max_size: float,
) -> float:
    """
    Calculate position size using: $$$ Risk / (entry - stop).

    This is the proper risk-based sizing formula shown in the
    TradingView strategy. Instead of a flat dollar amount, it
    calculates how many units you can buy such that if you get
    stopped out, you only lose risk_pct of your capital.

    Args:
        capital: Total account capital
        risk_pct: Percentage of capital to risk (e.g. 1.0 = 1%)
        entry: Entry price
        stop: Stop-loss price
        max_size: Maximum position size in USD

    Returns:
        Position size in USD
    """
    risk_usd = capital * (risk_pct / 100)
    distance = abs(entry - stop)
    if distance == 0:
        return 0.0

    # Number of units = risk_usd / distance_per_unit
    units = risk_usd / distance
    size_usd = units * entry

    return min(size_usd, max_size)


class PinBarStrategy:
    """
    Pin Bar Reversal — trade candlestick rejection patterns.

    Detects pin bars (hammers/shooting stars) and ranks them
    by strength: STRONGEST > STRONGER > STRONG > INDECISION.

    - Bullish pin bar (hammer): Long lower wick rejecting lower prices
    - Bearish pin bar (shooting star): Long upper wick rejecting higher prices

    Best for: reversals at key levels (Fib, S/R, pivot points).
    Risk level: Medium (tight stops behind the wick).
    """
    NAME = "pin_bar"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        # Detect pin bar on the current candle
        pin = ind.detect_pin_bar(
            open_price=snapshot.open_price,
            high=snapshot.high,
            low=snapshot.low,
            close=snapshot.current_price,
            atr_value=snapshot.atr,
        )
        if not pin:
            return None

        # Skip indecision candles (not strong enough signal alone)
        if pin.strength == "indecision":
            return None

        price = snapshot.current_price
        side = Side.BUY if pin.direction == "bullish" else Side.SELL

        # Stop-loss goes behind the wick (the rejection point)
        buffer = pin.candle_range * 0.1  # Small buffer past the wick
        if side == Side.BUY:
            stop_loss = pin.low - buffer
        else:
            stop_loss = pin.high + buffer

        # Confidence from pin bar score + confirmation factors
        confidence = pin.score

        # Boost confidence when pin bar aligns with trend
        if side == Side.BUY and snapshot.trend == "bullish":
            confidence += 0.05
        elif side == Side.SELL and snapshot.trend == "bearish":
            confidence += 0.05

        # Boost when pin bar appears at a key level (Fib, S/R)
        at_key_level = False
        if snapshot.fibonacci:
            for ret in snapshot.fibonacci.retracements:
                dist = abs(price - ret.price) / price * 100
                if dist < 0.5:  # Within 0.5% of a fib level
                    confidence += 0.08
                    at_key_level = True
                    break

        if not at_key_level and snapshot.support_levels:
            for s in snapshot.support_levels:
                if abs(price - s) / price * 100 < 0.5:
                    confidence += 0.05
                    at_key_level = True
                    break

        if not at_key_level and snapshot.resistance_levels:
            for r in snapshot.resistance_levels:
                if abs(price - r) / price * 100 < 0.5:
                    confidence += 0.05
                    at_key_level = True
                    break

        # Boost if RSI confirms
        if side == Side.BUY and snapshot.rsi and snapshot.rsi < 40:
            confidence += 0.05
        elif side == Side.SELL and snapshot.rsi and snapshot.rsi > 60:
            confidence += 0.05

        confidence = min(confidence, 0.95)

        if confidence < self.config.min_confidence:
            return None

        # Take-profit at 2:1 R:R minimum
        risk = abs(price - stop_loss)
        if side == Side.BUY:
            take_profit = price + (risk * 2)
        else:
            take_profit = price - (risk * 2)

        # Position sizing: $$$ Risk / (entry - stop)
        size_usd = position_size_from_risk(
            capital=self.config.starting_capital,
            risk_pct=self.config.risk_per_trade_pct,
            entry=price,
            stop=stop_loss,
            max_size=self.config.max_position_size_usd,
        )

        level_str = " at key level" if at_key_level else ""
        reason = (
            f"Pin bar {pin.strength.upper()} {pin.direction} "
            f"(wick ratio {pin.wick_ratio:.1f}x){level_str}"
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class StochasticStrategy:
    """
    Stochastic Oscillator — trade oversold/overbought crossovers.

    From Wealth Training course:
    - Fast stochastic (12,3,3): for shorter-term signals
    - Slow stochastic (20,12,9): for confirmation
    - BUY when %K crosses above %D below oversold (20)
    - SELL when %K crosses below %D above overbought (80)
    - Also checks for divergences

    Best for: oscillating/ranging markets.
    Risk level: Medium.
    """
    NAME = "stochastic"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        # Use fast stochastic for signals, slow for confirmation
        if snapshot.stoch_fast_k is None or snapshot.stoch_fast_d is None:
            return None

        k = snapshot.stoch_fast_k
        d = snapshot.stoch_fast_d
        price = snapshot.current_price
        side = None
        reason = ""
        confidence = 0.0

        # BUY: %K crosses above %D in oversold territory
        if k > d and k < self.config.stoch_oversold + 10:
            side = Side.BUY
            confidence = 0.55 + (self.config.stoch_oversold - k) / 100 * 0.3
            reason = f"Stochastic bullish crossover: %K({k:.1f}) > %D({d:.1f}) in oversold zone"

            # Boost if slow stochastic confirms
            if (snapshot.stoch_slow_k is not None and
                    snapshot.stoch_slow_d is not None and
                    snapshot.stoch_slow_k > snapshot.stoch_slow_d):
                confidence += 0.08
                reason += ", slow stochastic confirms"

        # SELL: %K crosses below %D in overbought territory
        elif k < d and k > self.config.stoch_overbought - 10:
            side = Side.SELL
            confidence = 0.55 + (k - self.config.stoch_overbought) / 100 * 0.3
            reason = f"Stochastic bearish crossover: %K({k:.1f}) < %D({d:.1f}) in overbought zone"

            if (snapshot.stoch_slow_k is not None and
                    snapshot.stoch_slow_d is not None and
                    snapshot.stoch_slow_k < snapshot.stoch_slow_d):
                confidence += 0.08
                reason += ", slow stochastic confirms"

        if side is None:
            return None

        confidence = min(max(confidence, 0.0), 0.90)
        if confidence < self.config.min_confidence:
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

        size_usd = position_size_from_risk(
            capital=self.config.starting_capital,
            risk_pct=self.config.risk_per_trade_pct * 0.7,
            entry=price,
            stop=stop_loss,
            max_size=self.config.max_position_size_usd,
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class CCIStrategy:
    """
    Commodity Channel Index strategy.

    From Wealth Training course (CCI 90,1):
    - BUY when CCI crosses above -100 (leaving oversold)
    - SELL when CCI crosses below +100 (leaving overbought)
    - Also monitors trendline breaks in CCI

    Best for: catching trend shifts in both trending and oscillating markets.
    Risk level: Medium.
    """
    NAME = "cci"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if snapshot.cci_value is None:
            return None

        cci_val = snapshot.cci_value
        price = snapshot.current_price
        side = None
        reason = ""
        confidence = 0.0

        # BUY: CCI crossing above -100 (emerging from oversold)
        if -120 < cci_val < -80:
            side = Side.BUY
            confidence = 0.55 + abs(cci_val + 100) / 200 * 0.2
            reason = f"CCI emerging from oversold at {cci_val:.1f}"

        # SELL: CCI crossing below +100 (falling from overbought)
        elif 80 < cci_val < 120:
            side = Side.SELL
            confidence = 0.55 + abs(cci_val - 100) / 200 * 0.2
            reason = f"CCI falling from overbought at {cci_val:.1f}"

        # Strong signals at extreme readings
        elif cci_val < -200:
            side = Side.BUY
            confidence = 0.70
            reason = f"CCI extreme oversold at {cci_val:.1f}"
        elif cci_val > 200:
            side = Side.SELL
            confidence = 0.70
            reason = f"CCI extreme overbought at {cci_val:.1f}"

        if side is None:
            return None

        # Boost if trend confirms
        if side == Side.BUY and snapshot.trend == "bullish":
            confidence += 0.05
        elif side == Side.SELL and snapshot.trend == "bearish":
            confidence += 0.05

        confidence = min(confidence, 0.85)
        if confidence < self.config.min_confidence:
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

        size_usd = position_size_from_risk(
            capital=self.config.starting_capital,
            risk_pct=self.config.risk_per_trade_pct * 0.6,
            entry=price,
            stop=stop_loss,
            max_size=self.config.max_position_size_usd,
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class ChartPatternStrategy:
    """
    Chart Pattern Recognition strategy.

    From Wealth Training course reversal and continuation patterns:
    - Double Bottom, Triple Bottom, Head & Shoulders
    - Cup and Handle (with 4 quality filters)
    - Flag/Pennant continuation patterns

    Profit targets follow the course rules:
    - Cup patterns: 50% of depth above breakout
    - All other patterns: full height projected from breakout

    Best for: higher timeframes (15m+), swing trading.
    Risk level: Medium.
    """
    NAME = "chart_pattern"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if not snapshot.chart_patterns:
            return None

        # Take the highest confidence pattern
        best = max(snapshot.chart_patterns, key=lambda p: p.confidence)

        price = snapshot.current_price
        side = Side.BUY if best.direction == "bullish" else Side.SELL
        confidence = best.confidence

        # Boost if volume confirms (per course: price + volume same direction = good)
        if snapshot.volume_ratio and snapshot.volume_ratio > 1.5:
            confidence += 0.05
            desc_extra = ", volume confirms"
        else:
            desc_extra = ""

        # Boost if at longer-term trend alignment
        if side == Side.BUY and snapshot.sma_long_fast and snapshot.sma_long_slow:
            if snapshot.sma_long_fast > snapshot.sma_long_slow:
                confidence += 0.05

        confidence = min(confidence, 0.95)
        if confidence < self.config.min_confidence:
            return None

        size_usd = position_size_from_risk(
            capital=self.config.starting_capital,
            risk_pct=self.config.risk_per_trade_pct,
            entry=price,
            stop=best.stop_price,
            max_size=self.config.max_position_size_usd,
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=side,
            price=price,
            size_usd=size_usd,
            stop_loss=best.stop_price,
            take_profit=best.target_price,
            confidence=confidence,
            reason=f"{best.name}: {best.description}{desc_extra}",
        )


def check_trading_checklist(
    snapshot: MarketSnapshot, signal: Signal, config: Config
) -> tuple[bool, List[str]]:
    """
    Pre-trade checklist from Wealth Training course.

    7 items must be checked before every trade:
    1. Does the trade have a 2+ to 1 reward to risk ratio?
    2. Buying close to support with bounce?
    3. Trading with the trend or after reversal pattern?
    4. Capital spread across trades (checked by risk manager)?
    5. Have both up and down trades?
    6. Checked longer time frame for resistance?
    7. Done the 6-step analysis?

    Returns (passed, list_of_warnings).
    """
    warnings = []

    # 1. Reward:Risk ratio check
    if signal.stop_loss and signal.take_profit:
        risk = abs(signal.price - signal.stop_loss)
        reward = abs(signal.take_profit - signal.price)
        if risk > 0:
            rr_ratio = reward / risk
            if rr_ratio < config.min_reward_risk_ratio:
                warnings.append(
                    f"R:R ratio {rr_ratio:.1f}:1 below minimum "
                    f"{config.min_reward_risk_ratio:.0f}:1"
                )

    # 2. Buying close to support with bounce
    if signal.side == Side.BUY and snapshot.support_levels:
        nearest_support = max(
            (s for s in snapshot.support_levels if s < signal.price),
            default=None,
        )
        if nearest_support:
            dist_to_support = (signal.price - nearest_support) / signal.price * 100
            if dist_to_support > 3.0:
                warnings.append(
                    f"Not close to support (nearest ${nearest_support:,.2f}, "
                    f"{dist_to_support:.1f}% away)"
                )

    # 3. Trading with trend or after reversal
    trend_aligned = (
        (signal.side == Side.BUY and snapshot.trend == "bullish") or
        (signal.side == Side.SELL and snapshot.trend == "bearish")
    )
    has_reversal = any(
        p.pattern_type == "reversal" for p in snapshot.chart_patterns
    )
    if not trend_aligned and not has_reversal:
        warnings.append("Not with trend and no reversal pattern detected")

    # 6. Check longer timeframe for resistance (via long MAs)
    if signal.side == Side.BUY and snapshot.sma_long_fast and snapshot.sma_long_slow:
        if snapshot.sma_long_fast < snapshot.sma_long_slow:
            warnings.append(
                "Long-term trend bearish (MA50 < MA100) — "
                "check higher timeframe for resistance"
            )

    # 7. Market behaviour check (part of 6-step analysis)
    if snapshot.market_behaviour == "unknown":
        warnings.append("Could not determine market behaviour (trending vs oscillating)")

    # Volume confirmation (from course: price + volume same direction)
    if snapshot.volume_ratio and snapshot.volume_ratio < 0.5:
        warnings.append("Low volume — weak conviction")

    passed = len(warnings) == 0
    return passed, warnings


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
            PinBarStrategy(config),
            StochasticStrategy(config),
            CCIStrategy(config),
            ChartPatternStrategy(config),
        ]

    def generate_signals(
        self, snapshot: MarketSnapshot
    ) -> List[Signal]:
        signals = []

        # Log market behaviour (Step 1 of 6-step analysis)
        if snapshot.market_behaviour != "unknown":
            logger.info(
                f"Market behaviour: {snapshot.market_behaviour} | "
                f"Chart patterns: {len(snapshot.chart_patterns)}"
            )

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

                # Run the 7-point trading checklist from Wealth Training
                passed, warnings = check_trading_checklist(
                    snapshot, signal, self.config
                )
                if warnings:
                    for w in warnings:
                        logger.warning(f"  Checklist: {w}")
                    # Reduce confidence for each warning
                    signal.confidence = max(
                        signal.confidence - len(warnings) * 0.05,
                        0.0,
                    )

                logger.info(
                    f"[{strategy.NAME}] {signal.side.value.upper()} "
                    f"@ ${signal.price:,.2f} — {signal.reason} "
                    f"(conf: {signal.confidence:.0%})"
                )
                if signal.exit_plan:
                    for tp in signal.exit_plan.take_profits:
                        logger.info(f"  {tp}")

                # Only keep signals above confidence threshold after checklist
                if signal.confidence >= self.config.min_confidence:
                    signals.append(signal)

        # Sort by confidence
        signals.sort(key=lambda s: s.confidence, reverse=True)
        return signals
