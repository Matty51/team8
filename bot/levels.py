"""
Take-profit and exit level manager.

Combines Fibonacci extensions, support/resistance, and pivot points
to create multi-level take-profit targets. The bot scales out of
positions at each level instead of a single flat TP.
"""
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from . import indicators as ind
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class TPLevel:
    """A single take-profit level."""
    price: float
    pct_to_exit: float        # What % of position to close at this level
    source: str               # "fib_ext", "fib_ret", "resistance", "pivot", "fixed"
    label: str                # Human-readable label
    distance_pct: float       # Distance from entry as percentage

    def __repr__(self) -> str:
        return (
            f"TP ${self.price:,.2f} ({self.distance_pct:+.1f}%) "
            f"— exit {self.pct_to_exit:.0%} [{self.source}: {self.label}]"
        )


@dataclass
class SLLevel:
    """Stop-loss level with source."""
    price: float
    source: str
    label: str
    distance_pct: float


@dataclass
class ExitPlan:
    """Complete exit plan for a trade: multi-level TPs + stop-loss."""
    entry_price: float
    side: str  # "buy" or "sell"
    stop_loss: SLLevel
    take_profits: List[TPLevel]  # Ordered from nearest to farthest

    @property
    def risk_reward_ratio(self) -> float:
        """Average R:R across all TP levels."""
        if not self.take_profits or self.stop_loss.distance_pct == 0:
            return 0.0
        avg_tp_dist = sum(
            tp.distance_pct * tp.pct_to_exit for tp in self.take_profits
        )
        return abs(avg_tp_dist / self.stop_loss.distance_pct)

    def summary(self) -> str:
        lines = [f"Exit Plan (R:R = 1:{self.risk_reward_ratio:.1f}):"]
        lines.append(
            f"  SL: ${self.stop_loss.price:,.2f} "
            f"({self.stop_loss.distance_pct:+.1f}%) "
            f"[{self.stop_loss.source}]"
        )
        for tp in self.take_profits:
            lines.append(f"  {tp}")
        return "\n".join(lines)


class LevelManager:
    """
    Calculates intelligent exit levels using multiple technical factors.

    For each trade, produces an ExitPlan with:
    - Stop-loss based on ATR or Fibonacci support
    - Multi-level take-profits from Fibonacci extensions, resistance, pivots
    """

    def __init__(self, config: Config):
        self.config = config

    def calculate_exit_plan(
        self,
        entry_price: float,
        side: str,
        highs: List[float],
        lows: List[float],
        closes: List[float],
        current_atr: Optional[float] = None,
    ) -> ExitPlan:
        """
        Build a complete exit plan combining all technical levels.

        Args:
            entry_price: The trade entry price
            side: "buy" or "sell"
            highs/lows/closes: OHLCV data for computing levels
            current_atr: Current ATR value (for volatility-adjusted SL)
        """
        # Gather all potential TP levels
        raw_tp_levels: List[Tuple[float, str, str]] = []  # (price, source, label)

        # 1. Fibonacci levels
        fib = ind.calculate_fibonacci(highs, lows, closes)
        if fib:
            for ext in fib.extensions:
                if side == "buy" and ext.price > entry_price:
                    raw_tp_levels.append((ext.price, "fib_ext", ext.label))
                elif side == "sell" and ext.price < entry_price:
                    raw_tp_levels.append((ext.price, "fib_ext", ext.label))

            # Retracement levels can act as targets too
            for ret in fib.retracements:
                if side == "buy" and ret.price > entry_price:
                    raw_tp_levels.append((ret.price, "fib_ret", ret.label))
                elif side == "sell" and ret.price < entry_price:
                    raw_tp_levels.append((ret.price, "fib_ret", ret.label))

        # 2. Support/Resistance levels
        support, resistance = ind.find_support_resistance(highs, lows, closes)
        if side == "buy":
            for r in resistance[:3]:  # Top 3 resistance levels
                if r > entry_price:
                    raw_tp_levels.append((r, "resistance", f"Resistance ${r:,.2f}"))
        else:
            for s in support[:3]:  # Top 3 support levels
                if s < entry_price:
                    raw_tp_levels.append((s, "support", f"Support ${s:,.2f}"))

        # 3. Pivot points (from the last candle)
        if len(highs) >= 2 and len(lows) >= 2 and len(closes) >= 2:
            pivots = ind.pivot_points(highs[-2], lows[-2], closes[-2])
            for label, price in pivots.items():
                if side == "buy" and price > entry_price and label.startswith("R"):
                    raw_tp_levels.append((price, "pivot", f"Pivot {label}"))
                elif side == "sell" and price < entry_price and label.startswith("S"):
                    raw_tp_levels.append((price, "pivot", f"Pivot {label}"))

        # 4. Fixed percentage fallbacks (always have at least these)
        fixed_pcts = [1.5, 3.0, 5.0]
        for pct in fixed_pcts:
            if side == "buy":
                price = entry_price * (1 + pct / 100)
            else:
                price = entry_price * (1 - pct / 100)
            raw_tp_levels.append((price, "fixed", f"Fixed {pct}%"))

        # Sort and deduplicate TP levels
        tp_levels = self._build_tp_levels(entry_price, side, raw_tp_levels)

        # Calculate stop-loss
        stop_loss = self._calculate_stop_loss(
            entry_price, side, fib, current_atr, support, resistance
        )

        plan = ExitPlan(
            entry_price=entry_price,
            side=side,
            stop_loss=stop_loss,
            take_profits=tp_levels,
        )

        logger.info(f"\n{plan.summary()}")
        return plan

    def _build_tp_levels(
        self,
        entry_price: float,
        side: str,
        raw_levels: List[Tuple[float, str, str]],
    ) -> List[TPLevel]:
        """
        Convert raw price levels into a structured multi-TP plan.

        Allocates exit percentages across levels:
        - TP1 (nearest):  40% of position — lock in profits early
        - TP2 (middle):   30% of position — let winners run
        - TP3 (farthest): 30% of position — max profit target
        """
        # Filter valid levels and compute distance
        valid = []
        for price, source, label in raw_levels:
            if side == "buy":
                dist_pct = (price - entry_price) / entry_price * 100
            else:
                dist_pct = (entry_price - price) / entry_price * 100

            if dist_pct > 0.3:  # At least 0.3% away
                valid.append((price, source, label, dist_pct))

        if not valid:
            return []

        # Sort by distance (nearest first)
        valid.sort(key=lambda x: x[3])

        # Deduplicate levels within 0.2% of each other (keep best source)
        source_priority = {"fib_ext": 0, "fib_ret": 1, "resistance": 2,
                          "support": 2, "pivot": 3, "fixed": 4}
        deduped = []
        for price, source, label, dist in valid:
            is_dup = False
            for existing in deduped:
                if abs(dist - existing[3]) < 0.2:
                    # Keep the one with better source priority
                    if source_priority.get(source, 5) < source_priority.get(existing[1], 5):
                        deduped.remove(existing)
                        deduped.append((price, source, label, dist))
                    is_dup = True
                    break
            if not is_dup:
                deduped.append((price, source, label, dist))

        deduped.sort(key=lambda x: x[3])

        # Pick up to 3 levels with good spacing
        selected = self._select_spaced_levels(deduped, max_levels=3)

        # Assign exit percentages
        exit_pcts = self._get_exit_allocation(len(selected))

        tp_levels = []
        for i, (price, source, label, dist) in enumerate(selected):
            tp_levels.append(TPLevel(
                price=price,
                pct_to_exit=exit_pcts[i],
                source=source,
                label=label,
                distance_pct=dist,
            ))

        return tp_levels

    def _select_spaced_levels(
        self,
        levels: List[Tuple],
        max_levels: int = 3,
        min_spacing_pct: float = 0.8,
    ) -> List[Tuple]:
        """Select up to max_levels with minimum spacing between them."""
        if len(levels) <= max_levels:
            return levels

        selected = [levels[0]]  # Always take the nearest
        for level in levels[1:]:
            if len(selected) >= max_levels:
                break
            last_dist = selected[-1][3]
            if level[3] - last_dist >= min_spacing_pct:
                selected.append(level)

        # If we don't have enough, fill from remaining
        if len(selected) < max_levels:
            for level in levels:
                if level not in selected and len(selected) < max_levels:
                    selected.append(level)
            selected.sort(key=lambda x: x[3])

        return selected[:max_levels]

    def _get_exit_allocation(self, num_levels: int) -> List[float]:
        """Get exit percentage allocation for N levels."""
        if num_levels == 1:
            return [1.0]
        elif num_levels == 2:
            return [0.50, 0.50]
        else:
            return [0.40, 0.30, 0.30]

    def _calculate_stop_loss(
        self,
        entry_price: float,
        side: str,
        fib: Optional[ind.FibonacciLevels],
        current_atr: Optional[float],
        support: List[float],
        resistance: List[float],
    ) -> SLLevel:
        """
        Calculate stop-loss from the best available source.

        Priority:
        1. Fibonacci support/resistance level (most reliable)
        2. ATR-based (1.5x ATR from entry — volatility-adjusted)
        3. Fixed percentage fallback
        """
        candidates: List[SLLevel] = []

        # Fibonacci-based SL
        if fib:
            if side == "buy":
                # SL below nearest Fib support
                fib_sl = fib.nearest_level_below(entry_price)
                if fib_sl:
                    dist = (entry_price - fib_sl.price) / entry_price * 100
                    if 0.5 < dist < 5.0:  # Reasonable range
                        candidates.append(SLLevel(
                            price=fib_sl.price,
                            source="fibonacci",
                            label=fib_sl.label,
                            distance_pct=-dist,
                        ))
            else:
                fib_sl = fib.nearest_level_above(entry_price)
                if fib_sl:
                    dist = (fib_sl.price - entry_price) / entry_price * 100
                    if 0.5 < dist < 5.0:
                        candidates.append(SLLevel(
                            price=fib_sl.price,
                            source="fibonacci",
                            label=fib_sl.label,
                            distance_pct=-dist,
                        ))

        # S/R-based SL
        if side == "buy" and support:
            nearest_support = max(s for s in support if s < entry_price) if any(s < entry_price for s in support) else None
            if nearest_support:
                dist = (entry_price - nearest_support) / entry_price * 100
                if 0.5 < dist < 5.0:
                    candidates.append(SLLevel(
                        price=nearest_support,
                        source="support",
                        label=f"Support ${nearest_support:,.2f}",
                        distance_pct=-dist,
                    ))
        elif side == "sell" and resistance:
            nearest_resistance = min(r for r in resistance if r > entry_price) if any(r > entry_price for r in resistance) else None
            if nearest_resistance:
                dist = (nearest_resistance - entry_price) / entry_price * 100
                if 0.5 < dist < 5.0:
                    candidates.append(SLLevel(
                        price=nearest_resistance,
                        source="resistance",
                        label=f"Resistance ${nearest_resistance:,.2f}",
                        distance_pct=-dist,
                    ))

        # ATR-based SL (1.5x ATR)
        if current_atr and current_atr > 0:
            atr_distance = current_atr * 1.5
            if side == "buy":
                sl_price = entry_price - atr_distance
            else:
                sl_price = entry_price + atr_distance
            dist = atr_distance / entry_price * 100
            candidates.append(SLLevel(
                price=sl_price,
                source="atr",
                label=f"1.5x ATR (${current_atr:,.2f})",
                distance_pct=-dist,
            ))

        # Pick the best candidate (prefer fib > support > atr > fixed)
        source_priority = {"fibonacci": 0, "support": 1, "resistance": 1, "atr": 2}
        if candidates:
            candidates.sort(key=lambda c: source_priority.get(c.source, 3))
            return candidates[0]

        # Fixed fallback
        sl_pct = self.config.stop_loss_pct
        if side == "buy":
            sl_price = entry_price * (1 - sl_pct / 100)
        else:
            sl_price = entry_price * (1 + sl_pct / 100)

        return SLLevel(
            price=sl_price,
            source="fixed",
            label=f"Fixed {sl_pct}%",
            distance_pct=-sl_pct,
        )
