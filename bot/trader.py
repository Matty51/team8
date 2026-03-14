"""
Trade executor — handles paper trading and (future) live execution.
"""
import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .config import Config
from .risk import RiskManager
from .strategy import Signal, Side

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of an executed trade (paper or live)."""
    timestamp: str
    market_question: str
    market_id: str
    strategy: str
    side: str
    outcome: str
    token_id: str
    target_price: float
    fill_price: float
    size_usd: float
    shares: float
    confidence: float
    paper: bool
    reason: str


class PaperTrader:
    """Simulates trade execution without real money."""

    def __init__(self, config: Config, risk_manager: RiskManager):
        self.config = config
        self.risk = risk_manager
        self.trade_log: List[TradeRecord] = []
        self.balance = 1000.0  # Starting paper balance

    def execute_signals(self, signals: List[Signal]) -> List[TradeRecord]:
        """Process signals through risk checks and simulate execution."""
        executed = []

        for signal in signals:
            # Risk check
            allowed, reason = self.risk.check_signal(signal)
            if not allowed:
                logger.info(f"BLOCKED: {reason} — {signal.reason}")
                continue

            # Check paper balance
            if signal.size_usd > self.balance:
                logger.info(
                    f"BLOCKED: Insufficient paper balance "
                    f"(${self.balance:.2f} < ${signal.size_usd:.2f})"
                )
                continue

            # Simulate fill (assume we get the target price)
            fill_price = signal.target_price
            shares = signal.size_usd / fill_price if fill_price > 0 else 0

            if shares <= 0:
                continue

            # Record the trade
            record = TradeRecord(
                timestamp=datetime.utcnow().isoformat(),
                market_question=signal.opportunity.market_question,
                market_id=signal.opportunity.market_id,
                strategy=signal.strategy_name,
                side=signal.side.value,
                outcome=signal.outcome,
                token_id=signal.token_id,
                target_price=signal.target_price,
                fill_price=fill_price,
                size_usd=signal.size_usd,
                shares=shares,
                confidence=signal.confidence,
                paper=True,
                reason=signal.reason,
            )

            # Update state
            self.balance -= signal.size_usd
            self.risk.record_trade(signal, fill_price, shares)
            self.trade_log.append(record)
            executed.append(record)

            logger.info(
                f"PAPER TRADE: {signal.side.value.upper()} "
                f"{signal.outcome} {shares:.2f} shares @ {fill_price:.3f} "
                f"(${signal.size_usd:.2f}) — {signal.reason}"
            )

        return executed

    def get_stats(self) -> Dict:
        """Get paper trading statistics."""
        total_trades = len(self.trade_log)
        total_volume = sum(t.size_usd for t in self.trade_log)
        strategies_used = set(t.strategy for t in self.trade_log)

        return {
            "mode": "PAPER TRADING",
            "balance": f"${self.balance:.2f}",
            "total_trades": total_trades,
            "total_volume": f"${total_volume:.2f}",
            "strategies_used": list(strategies_used),
            "risk_summary": self.risk.get_summary(),
        }

    def save_trade_log(self, filepath: str = "paper_trades.json"):
        """Save trade history to JSON file."""
        with open(filepath, "w") as f:
            json.dump(
                [asdict(t) for t in self.trade_log],
                f,
                indent=2,
            )
        logger.info(f"Trade log saved to {filepath}")
