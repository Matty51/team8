"""
Risk management — enforces position limits, stop-losses, and circuit breakers.
"""
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

from .config import Config
from .levels import ExitPlan, TPLevel
from .strategy import Signal, Side

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Tracks an open position."""
    symbol: str
    side: str  # "buy" or "sell"
    entry_price: float
    size_usd: float
    amount: float  # quantity of asset
    stop_loss: Optional[float]
    take_profit: Optional[float]
    strategy: str
    timestamp: str
    exit_plan: Optional[ExitPlan] = None
    tp_levels_hit: int = 0  # How many TP levels we've exited at


@dataclass
class DailyStats:
    """Tracks daily trading stats for risk limits."""
    date: str = ""
    trades_count: int = 0
    realized_pnl: float = 0.0
    total_volume: float = 0.0

    def reset_if_new_day(self):
        today = str(date.today())
        if self.date != today:
            logger.info(f"New trading day: {today}")
            self.date = today
            self.trades_count = 0
            self.realized_pnl = 0.0
            self.total_volume = 0.0


class RiskManager:
    """Enforces all risk controls before and during trades."""

    def __init__(self, config: Config):
        self.config = config
        self.positions: Dict[str, Position] = {}  # symbol -> Position
        self.daily_stats = DailyStats()
        self.total_pnl = 0.0

    def calculate_position_size(self, current_balance: float) -> float:
        """
        Calculate position size based on risk percentage of current balance.
        With $500 and 2% risk: $10 max risk per trade.
        Capped by max_position_size_usd.
        """
        risk_amount = current_balance * (self.config.risk_per_trade_pct / 100)
        # Position size = risk amount / stop loss distance
        # Since stop loss is 1.5%, position = risk / 0.015
        position = risk_amount / (self.config.stop_loss_pct / 100)
        return min(position, self.config.max_position_size_usd)

    def get_total_risk_usd(self) -> float:
        """Total USD at risk across all open positions."""
        total = 0.0
        for pos in self.positions.values():
            if pos.stop_loss:
                if pos.side == "buy":
                    risk = (pos.entry_price - pos.stop_loss) * pos.amount
                else:
                    risk = (pos.stop_loss - pos.entry_price) * pos.amount
                total += max(risk, 0)
        return total

    def check_signal(self, signal: Signal) -> tuple[bool, str]:
        """
        Returns (allowed, reason).
        If allowed is False, the trade should be skipped.
        """
        self.daily_stats.reset_if_new_day()

        # Daily trade count limit
        if self.daily_stats.trades_count >= self.config.max_daily_trades:
            return False, (
                f"Daily trade limit reached ({self.config.max_daily_trades})"
            )

        # Daily loss limit
        if self.daily_stats.realized_pnl <= -self.config.max_daily_loss_usd:
            return False, (
                f"Daily loss limit reached "
                f"(${self.daily_stats.realized_pnl:.2f})"
            )

        # Max open positions
        if len(self.positions) >= self.config.max_open_positions:
            if signal.symbol not in self.positions:
                return False, (
                    f"Max open positions reached "
                    f"({self.config.max_open_positions})"
                )

        # Per-trade size limit
        if signal.size_usd > self.config.max_position_size_usd:
            return False, (
                f"Trade size ${signal.size_usd:.2f} exceeds max "
                f"${self.config.max_position_size_usd:.2f}"
            )

        # Total portfolio risk check
        max_risk = self.config.starting_capital * (
            self.config.max_portfolio_risk_pct / 100
        )
        current_risk = self.get_total_risk_usd()
        new_risk = signal.size_usd * (self.config.stop_loss_pct / 100)
        if current_risk + new_risk > max_risk:
            return False, (
                f"Portfolio risk ${current_risk + new_risk:.2f} would exceed "
                f"max ${max_risk:.2f} ({self.config.max_portfolio_risk_pct}%)"
            )

        # Duplicate position — don't double up
        if signal.symbol in self.positions:
            existing = self.positions[signal.symbol]
            return False, (
                f"Already have {existing.side} position in {signal.symbol} "
                f"@ {existing.entry_price:,.2f}"
            )

        # Confidence threshold
        if signal.confidence < self.config.min_confidence:
            return False, (
                f"Confidence {signal.confidence:.2f} below threshold "
                f"{self.config.min_confidence:.2f}"
            )

        # Wealth Training: max 5% risk per trade (hard cap)
        if signal.stop_loss:
            risk_distance = abs(signal.price - signal.stop_loss)
            risk_pct = risk_distance / signal.price * 100
            if risk_pct > self.config.max_risk_per_trade_pct:
                return False, (
                    f"Trade risk {risk_pct:.1f}% exceeds max "
                    f"{self.config.max_risk_per_trade_pct}% per trade"
                )

        # Wealth Training checklist: minimum 2:1 reward:risk ratio
        if signal.stop_loss and signal.take_profit:
            risk = abs(signal.price - signal.stop_loss)
            reward = abs(signal.take_profit - signal.price)
            if risk > 0:
                rr_ratio = reward / risk
                if rr_ratio < self.config.min_reward_risk_ratio:
                    return False, (
                        f"R:R ratio {rr_ratio:.1f}:1 below minimum "
                        f"{self.config.min_reward_risk_ratio:.0f}:1"
                    )

        return True, "Passed all risk checks"

    def check_stop_loss_take_profit(
        self, symbol: str, current_price: float
    ) -> Optional[str]:
        """
        Check if an open position has hit stop-loss or take-profit.
        Returns "stop_loss", "take_profit_partial", "take_profit_final", or None.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        # Check stop-loss first
        if pos.side == "buy":
            if pos.stop_loss and current_price <= pos.stop_loss:
                return "stop_loss"
        elif pos.side == "sell":
            if pos.stop_loss and current_price >= pos.stop_loss:
                return "stop_loss"

        # Check multi-level take-profits (Fibonacci/S/R)
        if pos.exit_plan and pos.exit_plan.take_profits:
            tp_levels = pos.exit_plan.take_profits
            next_tp_idx = pos.tp_levels_hit

            if next_tp_idx < len(tp_levels):
                next_tp = tp_levels[next_tp_idx]
                hit = False

                if pos.side == "buy" and current_price >= next_tp.price:
                    hit = True
                elif pos.side == "sell" and current_price <= next_tp.price:
                    hit = True

                if hit:
                    # Is this the last TP level?
                    if next_tp_idx >= len(tp_levels) - 1:
                        return "take_profit_final"
                    else:
                        return "take_profit_partial"

        # Fallback: check flat take-profit
        elif pos.take_profit:
            if pos.side == "buy" and current_price >= pos.take_profit:
                return "take_profit_final"
            elif pos.side == "sell" and current_price <= pos.take_profit:
                return "take_profit_final"

        return None

    def get_next_tp_level(self, symbol: str) -> Optional[TPLevel]:
        """Get the next TP level to exit at for a position."""
        if symbol not in self.positions:
            return None
        pos = self.positions[symbol]
        if not pos.exit_plan or not pos.exit_plan.take_profits:
            return None
        idx = pos.tp_levels_hit
        if idx < len(pos.exit_plan.take_profits):
            return pos.exit_plan.take_profits[idx]
        return None

    def record_partial_exit(self, symbol: str) -> None:
        """Record that a TP level was hit (for partial exit tracking)."""
        if symbol in self.positions:
            self.positions[symbol].tp_levels_hit += 1

    def record_trade(
        self, signal: Signal, fill_price: float, amount: float
    ):
        """Record a new position after a trade executes."""
        self.daily_stats.trades_count += 1
        self.daily_stats.total_volume += signal.size_usd

        self.positions[signal.symbol] = Position(
            symbol=signal.symbol,
            side=signal.side.value,
            entry_price=fill_price,
            size_usd=signal.size_usd,
            amount=amount,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            strategy=signal.strategy_name,
            timestamp=datetime.utcnow().isoformat(),
            exit_plan=signal.exit_plan,
        )
        logger.info(
            f"Position opened: {signal.side.value.upper()} "
            f"{amount:.6f} {signal.symbol} @ ${fill_price:,.2f} "
            f"(${signal.size_usd:.2f}) | "
            f"SL: ${signal.stop_loss:,.2f} | TP: ${signal.take_profit:,.2f}"
        )

    def close_position(
        self, symbol: str, exit_price: float, reason: str = ""
    ) -> float:
        """Close a position and return realized PnL."""
        if symbol not in self.positions:
            return 0.0

        pos = self.positions.pop(symbol)

        if pos.side == "buy":
            pnl = (exit_price - pos.entry_price) * pos.amount
        else:
            pnl = (pos.entry_price - exit_price) * pos.amount

        self.daily_stats.realized_pnl += pnl
        self.total_pnl += pnl

        emoji = "+" if pnl >= 0 else ""
        logger.info(
            f"Position closed ({reason}): {pos.side.upper()} "
            f"{pos.amount:.6f} {symbol} @ ${exit_price:,.2f} "
            f"(entry: ${pos.entry_price:,.2f}) | "
            f"PnL: ${emoji}{pnl:.2f}"
        )
        return pnl

    def get_summary(self) -> Dict:
        """Return current risk state summary."""
        return {
            "open_positions": len(self.positions),
            "max_positions": self.config.max_open_positions,
            "daily_trades": self.daily_stats.trades_count,
            "max_daily_trades": self.config.max_daily_trades,
            "daily_pnl": f"${self.daily_stats.realized_pnl:+.2f}",
            "total_pnl": f"${self.total_pnl:+.2f}",
            "daily_volume": f"${self.daily_stats.total_volume:.2f}",
            "positions": {
                sym: {
                    "side": p.side,
                    "entry": f"${p.entry_price:,.2f}",
                    "size": f"${p.size_usd:.2f}",
                    "SL": f"${p.stop_loss:,.2f}" if p.stop_loss else "none",
                    "TP": f"${p.take_profit:,.2f}" if p.take_profit else "none",
                    "strategy": p.strategy,
                }
                for sym, p in self.positions.items()
            },
        }
