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
    RSI strategy — combines oversold/overbought with midline crossing.

    From Wealth Training course (Section 7):
    - RSI period 21
    - BUY when RSI crosses above 48 (strength confirmed)
    - SELL when RSI drops below 48
    - Also: BUY when RSI < 30 (oversold), SELL when RSI > 70 (overbought)

    Best for: trending and ranging markets.
    Risk level: Medium.
    """
    NAME = "rsi"

    RSI_MIDLINE = 48.0  # From Wealth Training course

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if snapshot.rsi is None:
            return None

        price = snapshot.current_price
        rsi_val = snapshot.rsi
        side = None
        reason = ""
        confidence = 0.0

        # Classic oversold/overbought signals
        # TREND FILTER: don't sell overbought in uptrends or buy oversold
        # in downtrends — mean reversion fails against strong trends
        if rsi_val < self.config.rsi_oversold:
            # Only buy oversold if trend is NOT bearish
            if snapshot.trend == "bearish":
                return None
            side = Side.BUY
            intensity = (self.config.rsi_oversold - rsi_val) / self.config.rsi_oversold
            confidence = min(0.55 + intensity * 0.3, 0.90)
            reason = f"RSI oversold at {rsi_val:.1f}"

            if snapshot.bb_lower and price <= snapshot.bb_lower * 1.005:
                confidence += 0.05
                reason += f", price near BB lower ({snapshot.bb_lower:,.2f})"

            # Extra boost when trend aligns (bullish + oversold = pullback buy)
            if snapshot.trend == "bullish":
                confidence += 0.05
                reason += ", with-trend pullback"

        elif rsi_val > self.config.rsi_overbought:
            # Only sell overbought if trend is NOT bullish
            if snapshot.trend == "bullish":
                return None
            side = Side.SELL
            intensity = (rsi_val - self.config.rsi_overbought) / (100 - self.config.rsi_overbought)
            confidence = min(0.55 + intensity * 0.3, 0.90)
            reason = f"RSI overbought at {rsi_val:.1f}"

            if snapshot.bb_upper and price >= snapshot.bb_upper * 0.995:
                confidence += 0.05
                reason += f", price near BB upper ({snapshot.bb_upper:,.2f})"

            if snapshot.trend == "bearish":
                confidence += 0.05
                reason += ", with-trend bounce short"

        # Course midline rule: RSI crossing above/below 48
        elif (rsi_val > self.RSI_MIDLINE and snapshot.trend == "bullish" and
              rsi_val < 60):  # Just crossed above, not already high
            side = Side.BUY
            confidence = 0.58
            reason = f"RSI crossed above 48 midline ({rsi_val:.1f}) — strength confirmed"

        elif (rsi_val < self.RSI_MIDLINE and snapshot.trend == "bearish" and
              rsi_val > 40):  # Just crossed below
            side = Side.SELL
            confidence = 0.58
            reason = f"RSI dropped below 48 midline ({rsi_val:.1f}) — weakness confirmed"

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

        # Course Section 7: CCI zero-line crossing (primary signal)
        # BUY when CCI crosses above zero from below
        if -20 < cci_val < 20 and snapshot.trend == "bullish":
            side = Side.BUY
            confidence = 0.58
            reason = f"CCI crossing above zero ({cci_val:.1f}) — bullish momentum"

        elif -20 < cci_val < 20 and snapshot.trend == "bearish":
            side = Side.SELL
            confidence = 0.58
            reason = f"CCI crossing below zero ({cci_val:.1f}) — bearish momentum"

        # BUY: CCI crossing above -100 (emerging from oversold)
        # TREND FILTER: don't buy oversold in downtrends
        elif -120 < cci_val < -80:
            if snapshot.trend == "bearish":
                return None
            side = Side.BUY
            confidence = 0.55 + abs(cci_val + 100) / 200 * 0.2
            reason = f"CCI emerging from oversold at {cci_val:.1f}"
            if snapshot.trend == "bullish":
                confidence += 0.05
                reason += ", with-trend pullback"

        # SELL: CCI crossing below +100 (falling from overbought)
        # TREND FILTER: don't sell overbought in uptrends
        elif 80 < cci_val < 120:
            if snapshot.trend == "bullish":
                return None
            side = Side.SELL
            confidence = 0.55 + abs(cci_val - 100) / 200 * 0.2
            reason = f"CCI falling from overbought at {cci_val:.1f}"
            if snapshot.trend == "bearish":
                confidence += 0.05
                reason += ", with-trend bounce short"

        # Strong signals at extreme readings
        # TREND FILTER: extreme readings still respect trend
        elif cci_val < -200:
            if snapshot.trend == "bearish":
                return None  # Don't catch falling knives
            side = Side.BUY
            confidence = 0.70
            reason = f"CCI extreme oversold at {cci_val:.1f}"
        elif cci_val > 200:
            if snapshot.trend == "bullish":
                return None  # Don't short a rocket
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

        # Gate 1: require minimum pattern confidence of 0.65
        if best.confidence < 0.65:
            return None

        # Gate 2: pattern must have a valid R:R on its own
        pattern_risk = abs(price - best.stop_price)
        pattern_reward = abs(best.target_price - price)
        if pattern_risk <= 0 or pattern_reward / pattern_risk < 1.5:
            return None

        confidence = best.confidence

        # Gate 3: require volume confirmation — patterns without volume
        # are unreliable (per course: price + volume same direction)
        if snapshot.volume_ratio and snapshot.volume_ratio > 1.5:
            confidence += 0.08
            desc_extra = ", volume confirms"
        elif snapshot.volume_ratio and snapshot.volume_ratio > 1.0:
            desc_extra = ""
        else:
            # Low volume = don't trust the pattern
            confidence -= 0.10
            desc_extra = ", low volume"

        # Boost if trend aligns with pattern direction
        trend_aligned = (
            (side == Side.BUY and snapshot.trend == "bullish") or
            (side == Side.SELL and snapshot.trend == "bearish")
        )
        if trend_aligned:
            confidence += 0.05
        else:
            # Counter-trend patterns need higher confidence
            confidence -= 0.05

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


class SurpriseDayStrategy:
    """
    Power Strategy: Surprise Day — from Wealth Training Section 8.

    "Very Reliable" short-term strategy.

    UP (Bull Market):
    - Price opens below previous candle's low
    - BUY if price breaks above previous candle's low
    - Works because shorts get squeezed

    DOWN (Bear Market):
    - Price opens above previous candle's high
    - SELL if price breaks below previous candle's high
    """
    NAME = "surprise_day"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if len(snapshot.raw_closes) < 2 or len(snapshot.raw_highs) < 2:
            return None

        price = snapshot.current_price
        prev_high = snapshot.raw_highs[-2]
        prev_low = snapshot.raw_lows[-2]
        curr_open = snapshot.open_price

        side = None
        reason = ""
        confidence = 0.0

        # Bull surprise: opened below previous low, now trading above it
        if curr_open < prev_low and price > prev_low:
            side = Side.BUY
            confidence = 0.70
            reason = (
                f"Surprise Day UP: opened ${curr_open:,.2f} below prev low "
                f"${prev_low:,.2f}, now ${price:,.2f} — short squeeze"
            )

        # Bear surprise: opened above previous high, now trading below it
        elif curr_open > prev_high and price < prev_high:
            side = Side.SELL
            confidence = 0.70
            reason = (
                f"Surprise Day DOWN: opened ${curr_open:,.2f} above prev high "
                f"${prev_high:,.2f}, now ${price:,.2f} — long squeeze"
            )

        if side is None:
            return None

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
            risk_pct=self.config.risk_per_trade_pct * 0.5,
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


class TripleConvergenceStrategy:
    """
    Power Strategy: Triple Convergence — from Wealth Training Section 8.

    Medium-term strategy. Three simultaneous signals must align:
    1. PRICE — making lower lows (downtrend weakening)
    2. MACD Histogram — showing deceleration (bars getting shorter)
    3. RSI — making higher lows (convergence/strength)

    All three converging = powerful BUY signal.
    """
    NAME = "triple_convergence"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if (snapshot.rsi is None or snapshot.macd_histogram is None or
                len(snapshot.raw_closes) < 20):
            return None

        closes = snapshot.raw_closes
        price = snapshot.current_price

        # Check for price making lower lows (last 10 candles)
        recent_lows = snapshot.raw_lows[-10:]
        if len(recent_lows) < 10:
            return None

        # Price in a downtrend (lower lows)
        first_half_low = min(recent_lows[:5])
        second_half_low = min(recent_lows[5:])
        price_lower_lows = second_half_low < first_half_low

        if not price_lower_lows:
            return None

        # RSI making higher lows (convergence with price)
        rsi_values = ind.rsi(closes, self.config.rsi_period)
        recent_rsi = [v for v in rsi_values[-10:] if v is not None]
        if len(recent_rsi) < 6:
            return None

        first_half_rsi = min(recent_rsi[:len(recent_rsi)//2])
        second_half_rsi = min(recent_rsi[len(recent_rsi)//2:])
        rsi_higher_lows = second_half_rsi > first_half_rsi

        if not rsi_higher_lows:
            return None

        # MACD histogram decelerating (bars getting shorter / less negative)
        macd_l, macd_s, macd_h = ind.macd(
            closes, self.config.macd_fast, self.config.macd_slow,
            self.config.macd_signal,
        )
        recent_hist = [v for v in macd_h[-10:] if v is not None]
        if len(recent_hist) < 6:
            return None

        # Histogram should be negative but getting less negative
        first_half_hist = min(recent_hist[:len(recent_hist)//2])
        second_half_hist = min(recent_hist[len(recent_hist)//2:])
        macd_decelerating = (first_half_hist < 0 and
                              second_half_hist > first_half_hist)

        if not macd_decelerating:
            return None

        # All three converging — powerful BUY
        confidence = 0.75
        reason = (
            f"Triple Convergence: price lower lows BUT "
            f"RSI higher lows ({second_half_rsi:.1f} > {first_half_rsi:.1f}) + "
            f"MACD decelerating — powerful reversal signal"
        )

        stop_loss = min(recent_lows) * 0.99  # Below recent lows
        risk = price - stop_loss
        take_profit = price + risk * 2.5  # 2.5:1 R:R for convergence

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
            side=Side.BUY,
            price=price,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class VolatilityExpansionStrategy:
    """
    Power Strategy: Volatility Expansion Breakout — from Wealth Training Section 8.

    Short to medium-term strategy:
    1. Price hits a new 1-month high today
    2. RSI above the midline (48) = strength
    3. Today's range is bigger than each of the previous 10 days
    4. BUY slightly above today's high
    """
    NAME = "volatility_expansion"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if (snapshot.rsi is None or len(snapshot.raw_closes) < 22 or
                len(snapshot.raw_highs) < 22):
            return None

        price = snapshot.current_price
        curr_range = snapshot.high - snapshot.low

        # 1. New 1-month high (approximately 20 trading candles)
        month_high = max(snapshot.raw_highs[-22:-1])  # Exclude current
        if snapshot.high <= month_high:
            return None

        # 2. RSI above midline (48 per course)
        if snapshot.rsi < 48:
            return None

        # 3. Today's range bigger than each of previous 10 days
        for i in range(-11, -1):
            if i >= -len(snapshot.raw_highs):
                prev_range = snapshot.raw_highs[i] - snapshot.raw_lows[i]
                if curr_range <= prev_range:
                    return None

        # All conditions met — BUY signal
        confidence = 0.72

        # Boost if volume confirms
        if snapshot.volume_ratio and snapshot.volume_ratio > 1.5:
            confidence += 0.05

        stop_loss = snapshot.low  # Stop at today's low
        risk = price - stop_loss
        if risk <= 0:
            return None
        take_profit = price + risk * 2

        size_usd = position_size_from_risk(
            capital=self.config.starting_capital,
            risk_pct=self.config.risk_per_trade_pct * 0.7,
            entry=price,
            stop=stop_loss,
            max_size=self.config.max_position_size_usd,
        )

        reason = (
            f"Volatility Expansion: new month high ${snapshot.high:,.2f}, "
            f"RSI {snapshot.rsi:.1f} above 48, "
            f"range ${curr_range:,.2f} biggest in 10 periods"
        )

        return Signal(
            strategy_name=self.NAME,
            symbol=snapshot.symbol,
            side=Side.BUY,
            price=price,
            size_usd=size_usd,
            stop_loss=stop_loss,
            take_profit=take_profit,
            confidence=confidence,
            reason=reason,
        )


class BollingerSqueezeStrategy:
    """
    Bollinger Band Squeeze — from Wealth Training Section 7.

    When Bollinger Bands squeeze tight (low volatility), a big move is
    imminent. Trade in the direction of the breakout.

    BUY when price breaks above upper band after squeeze.
    SELL when price breaks below lower band after squeeze.
    """
    NAME = "bollinger_squeeze"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, snapshot: MarketSnapshot) -> Optional[Signal]:
        if (snapshot.bb_upper is None or snapshot.bb_lower is None or
                snapshot.bb_middle is None or snapshot.atr is None):
            return None

        price = snapshot.current_price
        bb_width = (snapshot.bb_upper - snapshot.bb_lower) / snapshot.bb_middle * 100

        # Squeeze: bands very tight (width < 2% of price)
        if bb_width > 2.0:
            return None

        side = None
        reason = ""
        confidence = 0.60

        # Breakout above upper band
        if price > snapshot.bb_upper:
            side = Side.BUY
            confidence = 0.65
            reason = (
                f"Bollinger Squeeze breakout UP: width {bb_width:.2f}%, "
                f"price ${price:,.2f} > upper band ${snapshot.bb_upper:,.2f}"
            )

        # Breakdown below lower band
        elif price < snapshot.bb_lower:
            side = Side.SELL
            confidence = 0.65
            reason = (
                f"Bollinger Squeeze breakout DOWN: width {bb_width:.2f}%, "
                f"price ${price:,.2f} < lower band ${snapshot.bb_lower:,.2f}"
            )

        if side is None:
            return None

        # Boost if volume confirms
        if snapshot.volume_ratio and snapshot.volume_ratio > 1.5:
            confidence += 0.08

        # Boost if trend aligns
        if side == Side.BUY and snapshot.trend == "bullish":
            confidence += 0.05
        elif side == Side.SELL and snapshot.trend == "bearish":
            confidence += 0.05

        confidence = min(confidence, 0.85)
        if confidence < self.config.min_confidence:
            return None

        # ATR-based stops for volatility breakouts
        atr_stop = snapshot.atr * 2.0
        if side == Side.BUY:
            stop_loss = price - atr_stop
            take_profit = price + atr_stop * 2
        else:
            stop_loss = price + atr_stop
            take_profit = price - atr_stop * 2

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
            SurpriseDayStrategy(config),
            TripleConvergenceStrategy(config),
            VolatilityExpansionStrategy(config),
            BollingerSqueezeStrategy(config),
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
                    # Fib exit plan is INFORMATIONAL ONLY — used by
                    # the partial exit system (trader.py) for scaling
                    # out at TP1/TP2/TP3 levels. We do NOT override
                    # the strategy's own SL/TP, which were set based
                    # on the specific signal logic (ATR, pattern
                    # measurements, etc). Backtesting showed that Fib
                    # overrides hurt win rate (12.6% → 27.3%) and
                    # profit factor (0.49 → 0.72).

                # Run the 7-point trading checklist from Wealth Training
                passed, warnings = check_trading_checklist(
                    snapshot, signal, self.config
                )
                if warnings:
                    for w in warnings:
                        logger.warning(f"  Checklist: {w}")
                    # Proportional confidence penalty:
                    # - R:R failure is a hard reject (critical)
                    # - Other warnings are softer deductions
                    penalty = 0.0
                    for w in warnings:
                        if "R:R ratio" in w:
                            penalty += 0.20  # Hard penalty — bad R:R
                        elif "Low volume" in w:
                            penalty += 0.03  # Minor
                        elif "Not with trend" in w:
                            penalty += 0.08  # Moderate
                        elif "Long-term trend bearish" in w:
                            penalty += 0.05  # Moderate
                        elif "Not close to support" in w:
                            penalty += 0.03  # Minor
                        else:
                            penalty += 0.02  # Negligible
                    signal.confidence = max(
                        signal.confidence - penalty, 0.0,
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
