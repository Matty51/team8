"""
Risk management — enforces position limits, daily loss caps, and circuit breakers.
"""
import logging
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List

from .config import Config
from .strategy import Signal

logger = logging.getLogger(__name__)


@dataclass
class Position:
    """Tracks an open position."""
    market_id: str
    token_id: str
    outcome: str
    entry_price: float
    size_usd: float
    shares: float
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
            self.date = today
            self.trades_count = 0
            self.realized_pnl = 0.0
            self.total_volume = 0.0


class RiskManager:
    """Enforces all risk controls before trades execute."""

    def __init__(self, config: Config):
        self.config = config
        self.positions: Dict[str, Position] = {}  # token_id -> Position
        self.daily_stats = DailyStats()

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
            if signal.token_id not in self.positions:
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

        # Duplicate position check (don't double up)
        if signal.token_id in self.positions:
            existing = self.positions[signal.token_id]
            return False, (
                f"Already have position in {existing.outcome} "
                f"@ {existing.entry_price:.3f}"
            )

        # Confidence threshold
        if signal.confidence < self.config.confidence_threshold:
            return False, (
                f"Confidence {signal.confidence:.2f} below threshold "
                f"{self.config.confidence_threshold:.2f}"
            )

        return True, "Passed all risk checks"

    def record_trade(self, signal: Signal, fill_price: float, shares: float):
        """Record a new position after a trade executes."""
        from datetime import datetime

        self.daily_stats.trades_count += 1
        self.daily_stats.total_volume += signal.size_usd

        self.positions[signal.token_id] = Position(
            market_id=signal.opportunity.market_id,
            token_id=signal.token_id,
            outcome=signal.outcome,
            entry_price=fill_price,
            size_usd=signal.size_usd,
            shares=shares,
            strategy=signal.strategy_name,
            timestamp=datetime.utcnow().isoformat(),
        )
        logger.info(
            f"Position opened: {signal.outcome} {shares:.2f} shares "
            f"@ {fill_price:.3f} (${signal.size_usd:.2f})"
        )

    def close_position(self, token_id: str, exit_price: float) -> float:
        """Close a position and return realized PnL."""
        if token_id not in self.positions:
            return 0.0

        pos = self.positions.pop(token_id)
        pnl = (exit_price - pos.entry_price) * pos.shares
        self.daily_stats.realized_pnl += pnl
        logger.info(
            f"Position closed: {pos.outcome} {pos.shares:.2f} shares "
            f"@ {exit_price:.3f} (PnL: ${pnl:+.2f})"
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
            "daily_volume": f"${self.daily_stats.total_volume:.2f}",
            "daily_loss_limit": f"${self.config.max_daily_loss_usd:.2f}",
            "positions": {
                tid: {
                    "outcome": p.outcome,
                    "entry": p.entry_price,
                    "size": p.size_usd,
                    "strategy": p.strategy,
                }
                for tid, p in self.positions.items()
            },
        }
