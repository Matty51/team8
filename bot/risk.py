"""
Risk management — enforces position limits, stop-losses, and circuit breakers.
"""
import logging
from dataclasses import dataclass
from datetime import date, datetime
from typing import Dict, List, Optional

from .config import Config
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

        return True, "Passed all risk checks"

    def check_stop_loss_take_profit(
        self, symbol: str, current_price: float
    ) -> Optional[str]:
        """
        Check if an open position has hit stop-loss or take-profit.
        Returns "stop_loss", "take_profit", or None.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]

        if pos.side == "buy":
            if pos.stop_loss and current_price <= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit and current_price >= pos.take_profit:
                return "take_profit"
        elif pos.side == "sell":
            if pos.stop_loss and current_price >= pos.stop_loss:
                return "stop_loss"
            if pos.take_profit and current_price <= pos.take_profit:
                return "take_profit"

        return None

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
