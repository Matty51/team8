"""
Trading strategies — decide what to buy/sell and when.
"""
import logging
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional

from .config import Config
from .scanner import Opportunity

logger = logging.getLogger(__name__)


class Side(Enum):
    BUY = "buy"
    SELL = "sell"


@dataclass
class Signal:
    """A concrete trade signal produced by a strategy."""
    opportunity: Opportunity
    strategy_name: str
    side: Side
    token_id: str              # which token to trade
    outcome: str               # "YES" or "NO"
    target_price: float        # price we want to buy/sell at
    size_usd: float            # how much to spend
    confidence: float          # 0.0 – 1.0
    reason: str

    @property
    def risk_reward_ratio(self) -> float:
        if self.target_price <= 0 or self.target_price >= 1:
            return 0.0
        potential_profit = 1.0 - self.target_price
        potential_loss = self.target_price
        return potential_profit / potential_loss


class SpreadArbStrategy:
    """
    Buy both YES and NO when their combined price < $1.00.
    Guaranteed profit at resolution (minus fees).
    This is the safest strategy — your starting point.
    """
    NAME = "spread_arb"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, opportunities: List[Opportunity]) -> List[Signal]:
        signals = []
        for opp in opportunities:
            if opp.opportunity_type != "spread_arb":
                continue
            if opp.spread_discount_pct < self.config.min_spread_pct:
                continue

            # Split capital evenly between YES and NO
            half_size = min(
                self.config.max_position_size_usd / 2,
                5.0,  # conservative starting cap
            )

            confidence = min(opp.spread_discount_pct / 10.0, 1.0)

            if confidence < self.config.confidence_threshold:
                continue

            signals.append(Signal(
                opportunity=opp,
                strategy_name=self.NAME,
                side=Side.BUY,
                token_id=opp.token_id_yes,
                outcome="YES",
                target_price=opp.yes_price,
                size_usd=half_size,
                confidence=confidence,
                reason=(
                    f"Spread arb: buy YES@{opp.yes_price:.3f} + "
                    f"NO@{opp.no_price:.3f} = {opp.combined_price:.3f} "
                    f"({opp.spread_discount_pct:.1f}% guaranteed edge)"
                ),
            ))
            signals.append(Signal(
                opportunity=opp,
                strategy_name=self.NAME,
                side=Side.BUY,
                token_id=opp.token_id_no,
                outcome="NO",
                target_price=opp.no_price,
                size_usd=half_size,
                confidence=confidence,
                reason=f"Spread arb counterpart: buy NO@{opp.no_price:.3f}",
            ))

        return signals


class ValueStrategy:
    """
    Buy tokens that look underpriced based on extreme probabilities.
    Higher risk than spread arb, but higher potential return.
    """
    NAME = "value"

    def __init__(self, config: Config):
        self.config = config

    def evaluate(self, opportunities: List[Opportunity]) -> List[Signal]:
        signals = []
        for opp in opportunities:
            if opp.opportunity_type != "value":
                continue
            if opp.estimated_edge_pct < self.config.min_edge_pct:
                continue

            # Determine which side to trade
            if opp.yes_price < 0.10:
                token_id = opp.token_id_yes
                outcome = "YES"
                price = opp.yes_price
            elif opp.no_price < 0.10:
                token_id = opp.token_id_no
                outcome = "NO"
                price = opp.no_price
            elif opp.yes_price > 0.90:
                token_id = opp.token_id_yes
                outcome = "YES"
                price = opp.yes_price
            elif opp.no_price > 0.90:
                token_id = opp.token_id_no
                outcome = "NO"
                price = opp.no_price
            else:
                continue

            # Smaller size for higher-risk value bets
            size = min(self.config.max_position_size_usd * 0.3, 3.0)
            confidence = min(opp.estimated_edge_pct / 15.0, 0.9)

            if confidence < self.config.confidence_threshold:
                continue

            signals.append(Signal(
                opportunity=opp,
                strategy_name=self.NAME,
                side=Side.BUY,
                token_id=token_id,
                outcome=outcome,
                target_price=price,
                size_usd=size,
                confidence=confidence,
                reason=(
                    f"Value bet: {outcome}@{price:.3f} with "
                    f"{opp.estimated_edge_pct:.1f}% estimated edge"
                ),
            ))

        return signals


class StrategyManager:
    """Runs all strategies and aggregates signals."""

    def __init__(self, config: Config):
        self.strategies = [
            SpreadArbStrategy(config),
            ValueStrategy(config),
        ]

    def generate_signals(
        self, opportunities: List[Opportunity]
    ) -> List[Signal]:
        all_signals = []
        for strategy in self.strategies:
            signals = strategy.evaluate(opportunities)
            logger.info(
                f"Strategy '{strategy.NAME}' produced {len(signals)} signals"
            )
            all_signals.extend(signals)

        # Sort by confidence
        all_signals.sort(key=lambda s: s.confidence, reverse=True)
        return all_signals
