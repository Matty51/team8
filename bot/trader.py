"""
Trade executor — handles paper trading and live execution.
"""
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Dict, List, Optional

from .client import ExchangeClient
from .config import Config
from .risk import RiskManager
from .scanner import MarketSnapshot
from .strategy import Signal, Side

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    """Record of an executed trade."""
    timestamp: str
    symbol: str
    strategy: str
    side: str
    price: float
    amount: float
    size_usd: float
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    paper: bool
    reason: str
    pnl: Optional[float] = None
    close_reason: Optional[str] = None


class Trader:
    """Executes trades in paper or live mode with full risk management."""

    def __init__(
        self,
        client: ExchangeClient,
        config: Config,
        risk_manager: RiskManager,
    ):
        self.client = client
        self.config = config
        self.risk = risk_manager
        self.trade_log: List[TradeRecord] = []
        self.paper_balance = config.starting_capital

    def execute_signals(self, signals: List[Signal]) -> List[TradeRecord]:
        """Process signals through risk checks and execute."""
        executed = []

        for signal in signals:
            # Risk check
            allowed, reason = self.risk.check_signal(signal)
            if not allowed:
                logger.info(f"BLOCKED: {reason}")
                continue

            # Execute
            record = self._execute(signal)
            if record:
                executed.append(record)

        return executed

    def check_exits(self, snapshot: MarketSnapshot) -> List[TradeRecord]:
        """Check open positions for stop-loss / take-profit exits."""
        records = []
        price = snapshot.current_price
        symbol = snapshot.symbol

        exit_reason = self.risk.check_stop_loss_take_profit(symbol, price)
        if not exit_reason:
            return records

        if exit_reason == "take_profit_partial":
            record = self._partial_exit(symbol, price)
            if record:
                records.append(record)
        else:
            # stop_loss or take_profit_final — close entire position
            record = self._close_position(symbol, price, exit_reason)
            if record:
                records.append(record)

        return records

    def _partial_exit(
        self, symbol: str, price: float
    ) -> Optional[TradeRecord]:
        """Exit a portion of the position at a Fibonacci TP level."""
        if symbol not in self.risk.positions:
            return None

        pos = self.risk.positions[symbol]
        tp_level = self.risk.get_next_tp_level(symbol)
        if not tp_level:
            return None

        # Calculate how much to exit
        exit_amount = pos.amount * tp_level.pct_to_exit
        exit_usd = pos.size_usd * tp_level.pct_to_exit

        # Calculate PnL for this partial exit
        if pos.side == "buy":
            pnl = (price - pos.entry_price) * exit_amount
        else:
            pnl = (pos.entry_price - price) * exit_amount

        # Execute the partial close on exchange
        close_side = "sell" if pos.side == "buy" else "buy"
        if not self.config.paper_trading:
            order = self.client.create_market_order(
                side=close_side,
                amount=exit_amount,
                symbol=symbol,
            )
            if not order:
                logger.error(f"Failed partial exit for {symbol}")
                return None

        # Update position (reduce size)
        pos.amount -= exit_amount
        pos.size_usd -= exit_usd
        self.risk.record_partial_exit(symbol)
        self.risk.daily_stats.realized_pnl += pnl
        self.risk.total_pnl += pnl

        if self.config.paper_trading:
            self.paper_balance += exit_usd + pnl

        record = TradeRecord(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            strategy=pos.strategy,
            side=close_side,
            price=price,
            amount=exit_amount,
            size_usd=exit_usd,
            stop_loss=None,
            take_profit=tp_level.price,
            confidence=0.0,
            paper=self.config.paper_trading,
            reason=(
                f"Partial exit ({tp_level.pct_to_exit:.0%}) at "
                f"{tp_level.label} (${tp_level.price:,.2f}, "
                f"{tp_level.distance_pct:+.1f}%)"
            ),
            pnl=pnl,
            close_reason=f"tp_partial_{pos.tp_levels_hit}",
        )
        self.trade_log.append(record)

        logger.info(
            f"PARTIAL EXIT: {close_side.upper()} {exit_amount:.6f} "
            f"{symbol} @ ${price:,.2f} | {tp_level.label} | "
            f"PnL: ${pnl:+.2f} | Remaining: {pos.amount:.6f}"
        )
        return record

    def _execute(self, signal: Signal) -> Optional[TradeRecord]:
        """Execute a single trade (paper or live)."""
        if self.config.paper_trading:
            return self._paper_execute(signal)
        else:
            return self._live_execute(signal)

    def _paper_execute(self, signal: Signal) -> Optional[TradeRecord]:
        """Simulate trade execution."""
        if signal.size_usd > self.paper_balance:
            logger.info(
                f"Insufficient paper balance "
                f"(${self.paper_balance:.2f} < ${signal.size_usd:.2f})"
            )
            return None

        fill_price = signal.price
        amount = signal.size_usd / fill_price

        self.paper_balance -= signal.size_usd
        self.risk.record_trade(signal, fill_price, amount)

        record = TradeRecord(
            timestamp=datetime.utcnow().isoformat(),
            symbol=signal.symbol,
            strategy=signal.strategy_name,
            side=signal.side.value,
            price=fill_price,
            amount=amount,
            size_usd=signal.size_usd,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            paper=True,
            reason=signal.reason,
        )
        self.trade_log.append(record)

        logger.info(
            f"PAPER {signal.side.value.upper()} "
            f"{amount:.6f} {signal.symbol} @ ${fill_price:,.2f} "
            f"(${signal.size_usd:.2f}) [{signal.strategy_name}]"
        )
        return record

    def _live_execute(self, signal: Signal) -> Optional[TradeRecord]:
        """Execute a real trade via the exchange."""
        amount = signal.size_usd / signal.price

        order = self.client.create_market_order(
            side=signal.side.value,
            amount=amount,
            symbol=signal.symbol,
        )

        if not order:
            logger.error(f"Live order failed for {signal.symbol}")
            return None

        fill_price = order.get("average", order.get("price", signal.price))
        filled_amount = order.get("filled", amount)

        self.risk.record_trade(signal, fill_price, filled_amount)

        record = TradeRecord(
            timestamp=datetime.utcnow().isoformat(),
            symbol=signal.symbol,
            strategy=signal.strategy_name,
            side=signal.side.value,
            price=fill_price,
            amount=filled_amount,
            size_usd=signal.size_usd,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            paper=False,
            reason=signal.reason,
        )
        self.trade_log.append(record)

        logger.info(
            f"LIVE {signal.side.value.upper()} "
            f"{filled_amount:.6f} {signal.symbol} @ ${fill_price:,.2f} "
            f"(Order: {order.get('id')})"
        )
        return record

    def _close_position(
        self, symbol: str, price: float, reason: str
    ) -> Optional[TradeRecord]:
        """Close an open position."""
        if symbol not in self.risk.positions:
            return None

        pos = self.risk.positions[symbol]
        close_side = "sell" if pos.side == "buy" else "buy"

        if not self.config.paper_trading:
            order = self.client.create_market_order(
                side=close_side,
                amount=pos.amount,
                symbol=symbol,
            )
            if not order:
                logger.error(f"Failed to close position in {symbol}")
                return None

        pnl = self.risk.close_position(symbol, price, reason)

        if self.config.paper_trading:
            self.paper_balance += pos.size_usd + pnl

        record = TradeRecord(
            timestamp=datetime.utcnow().isoformat(),
            symbol=symbol,
            strategy=pos.strategy,
            side=close_side,
            price=price,
            amount=pos.amount,
            size_usd=pos.size_usd,
            stop_loss=None,
            take_profit=None,
            confidence=0.0,
            paper=self.config.paper_trading,
            reason=f"Exit: {reason}",
            pnl=pnl,
            close_reason=reason,
        )
        self.trade_log.append(record)
        return record

    def get_stats(self) -> Dict:
        """Get trading statistics."""
        total_trades = len(self.trade_log)
        total_volume = sum(t.size_usd for t in self.trade_log)
        closed = [t for t in self.trade_log if t.pnl is not None]
        wins = [t for t in closed if t.pnl and t.pnl > 0]
        losses = [t for t in closed if t.pnl and t.pnl < 0]
        win_rate = len(wins) / len(closed) * 100 if closed else 0

        return {
            "mode": "PAPER" if self.config.paper_trading else "LIVE",
            "exchange": self.config.exchange_id,
            "pair": self.config.trading_pair,
            "balance": f"${self.paper_balance:,.2f}" if self.config.paper_trading else "N/A",
            "total_trades": total_trades,
            "closed_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": f"{win_rate:.1f}%",
            "total_volume": f"${total_volume:,.2f}",
            "risk_summary": self.risk.get_summary(),
        }

    def save_trade_log(self, filepath: str = "trade_log.json"):
        """Save trade history to JSON file."""
        with open(filepath, "w") as f:
            json.dump(
                [asdict(t) for t in self.trade_log],
                f,
                indent=2,
            )
        logger.info(f"Trade log saved to {filepath}")
