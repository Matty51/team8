"""
Market scanner — finds tradeable opportunities across Polymarket.
"""
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .client import PolymarketClient
from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class Opportunity:
    """A detected trading opportunity."""
    market_question: str
    market_id: str
    opportunity_type: str          # "spread_arb" | "value" | "mispriced"
    yes_price: float
    no_price: float
    combined_price: float          # YES + NO (should be ~1.00)
    spread_discount_pct: float     # how far below $1.00 the pair is
    estimated_edge_pct: float      # estimated profit edge
    volume_24h: float
    token_id_yes: str
    token_id_no: str
    details: str

    @property
    def is_actionable(self) -> bool:
        return self.estimated_edge_pct > 0


class MarketScanner:
    """Scans Polymarket for opportunities based on configured thresholds."""

    def __init__(self, client: PolymarketClient, config: Config):
        self.client = client
        self.config = config

    def scan(self) -> List[Opportunity]:
        """Run a full scan and return ranked opportunities."""
        markets = self.client.get_markets(limit=self.config.markets_to_scan)
        logger.info(f"Scanning {len(markets)} markets...")

        opportunities: List[Opportunity] = []

        for market in markets:
            try:
                opps = self._analyze_market(market)
                opportunities.extend(opps)
            except Exception as e:
                logger.debug(f"Error analyzing market: {e}")

        # Sort by estimated edge, best first
        opportunities.sort(key=lambda o: o.estimated_edge_pct, reverse=True)
        logger.info(f"Found {len(opportunities)} opportunities")
        return opportunities

    def _analyze_market(self, market: Dict[str, Any]) -> List[Opportunity]:
        """Analyze a single market for opportunities."""
        opps = []

        tokens = market.get("tokens", [])
        if len(tokens) < 2:
            return opps

        question = market.get("question", "Unknown")
        market_id = market.get("condition_id", "")
        volume = float(market.get("volume", 0) or 0)

        # Get YES and NO tokens
        yes_token = None
        no_token = None
        for t in tokens:
            outcome = t.get("outcome", "").upper()
            if outcome == "YES":
                yes_token = t
            elif outcome == "NO":
                no_token = t

        if not yes_token or not no_token:
            return opps

        yes_price = float(yes_token.get("price", 0) or 0)
        no_price = float(no_token.get("price", 0) or 0)

        if yes_price <= 0 or no_price <= 0:
            return opps

        combined = yes_price + no_price

        # ── Spread arbitrage: YES + NO < $1.00 ──────────────────────
        # If combined < 1.00, buying both guarantees a profit at resolution
        if combined < 1.0:
            discount_pct = (1.0 - combined) * 100
            if discount_pct >= self.config.min_spread_pct:
                opps.append(Opportunity(
                    market_question=question,
                    market_id=market_id,
                    opportunity_type="spread_arb",
                    yes_price=yes_price,
                    no_price=no_price,
                    combined_price=combined,
                    spread_discount_pct=discount_pct,
                    estimated_edge_pct=discount_pct,
                    volume_24h=volume,
                    token_id_yes=yes_token.get("token_id", ""),
                    token_id_no=no_token.get("token_id", ""),
                    details=(
                        f"YES({yes_price:.3f}) + NO({no_price:.3f}) = "
                        f"{combined:.3f} → {discount_pct:.1f}% discount"
                    ),
                ))

        # ── Mispriced detection: YES + NO > $1.00 ───────────────────
        # If combined > 1.00 significantly, one side is overpriced
        if combined > 1.02:
            premium_pct = (combined - 1.0) * 100
            opps.append(Opportunity(
                market_question=question,
                market_id=market_id,
                opportunity_type="mispriced",
                yes_price=yes_price,
                no_price=no_price,
                combined_price=combined,
                spread_discount_pct=-premium_pct,
                estimated_edge_pct=premium_pct * 0.5,  # conservative
                volume_24h=volume,
                token_id_yes=yes_token.get("token_id", ""),
                token_id_no=no_token.get("token_id", ""),
                details=(
                    f"YES({yes_price:.3f}) + NO({no_price:.3f}) = "
                    f"{combined:.3f} → {premium_pct:.1f}% overpriced"
                ),
            ))

        # ── Value bet: extreme prices near 0 or 1 ───────────────────
        # Markets near resolution often have edge for fast actors
        for side, price, token in [
            ("YES", yes_price, yes_token),
            ("NO", no_price, no_token),
        ]:
            if price < 0.10 or price > 0.90:
                edge = min(price, 1.0 - price) * 100
                if edge >= self.config.min_edge_pct:
                    opps.append(Opportunity(
                        market_question=question,
                        market_id=market_id,
                        opportunity_type="value",
                        yes_price=yes_price,
                        no_price=no_price,
                        combined_price=combined,
                        spread_discount_pct=0,
                        estimated_edge_pct=edge,
                        volume_24h=volume,
                        token_id_yes=yes_token.get("token_id", ""),
                        token_id_no=no_token.get("token_id", ""),
                        details=(
                            f"{side} at {price:.3f} — potential value "
                            f"if market reverts ({edge:.1f}% edge)"
                        ),
                    ))

        return opps
