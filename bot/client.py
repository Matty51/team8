"""
Polymarket API client.
Wraps the public CLOB and Gamma APIs for fetching markets, prices, and orderbooks.
"""
import logging
import time
from typing import Any, Dict, List, Optional

import requests

from .config import Config

logger = logging.getLogger(__name__)


class PolymarketClient:
    """Thin wrapper around Polymarket's public REST APIs."""

    def __init__(self, config: Config):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/json",
            "Content-Type": "application/json",
        })
        self._rate_limit_until = 0.0

    # ── Public market data ───────────────────────────────────────────

    def get_markets(
        self,
        limit: int = 50,
        active: bool = True,
        closed: bool = False,
    ) -> List[Dict[str, Any]]:
        """Fetch markets from the Gamma API (richer metadata)."""
        params = {
            "limit": limit,
            "active": active,
            "closed": closed,
            "order": "volume",
            "ascending": False,
        }
        data = self._gamma_get("/markets", params=params)
        return data if isinstance(data, list) else []

    def get_market(self, condition_id: str) -> Optional[Dict[str, Any]]:
        """Fetch a single market by condition ID."""
        data = self._gamma_get(f"/markets/{condition_id}")
        return data if isinstance(data, dict) else None

    def get_orderbook(self, token_id: str) -> Dict[str, Any]:
        """Fetch the live orderbook for a token from the CLOB API."""
        data = self._clob_get(f"/book", params={"token_id": token_id})
        return data if isinstance(data, dict) else {"bids": [], "asks": []}

    def get_midpoint(self, token_id: str) -> Optional[float]:
        """Get the midpoint price for a token."""
        data = self._clob_get(f"/midpoint", params={"token_id": token_id})
        if data and "mid" in data:
            return float(data["mid"])
        return None

    def get_price(self, token_id: str, side: str = "buy") -> Optional[float]:
        """Get the best price for a token (buy or sell side)."""
        data = self._clob_get(f"/price", params={
            "token_id": token_id,
            "side": side,
        })
        if data and "price" in data:
            return float(data["price"])
        return None

    def get_spread(self, token_id: str) -> Optional[Dict[str, float]]:
        """Get bid/ask spread for a token."""
        data = self._clob_get(f"/spread", params={"token_id": token_id})
        if data and "spread" in data:
            return {
                "spread": float(data["spread"]),
                "bid": float(data.get("bid", 0)),
                "ask": float(data.get("ask", 0)),
            }
        return None

    # ── Internal HTTP helpers ────────────────────────────────────────

    def _clob_get(
        self, path: str, params: Optional[Dict] = None
    ) -> Optional[Any]:
        return self._get(self.config.clob_api_url + path, params)

    def _gamma_get(
        self, path: str, params: Optional[Dict] = None
    ) -> Optional[Any]:
        return self._get(self.config.gamma_api_url + path, params)

    def _get(
        self, url: str, params: Optional[Dict] = None
    ) -> Optional[Any]:
        """GET with basic rate-limit handling and retries."""
        now = time.time()
        if now < self._rate_limit_until:
            wait = self._rate_limit_until - now
            logger.debug(f"Rate-limited, sleeping {wait:.1f}s")
            time.sleep(wait)

        for attempt in range(3):
            try:
                resp = self.session.get(url, params=params, timeout=10)
                if resp.status_code == 429:
                    retry_after = float(resp.headers.get("Retry-After", 2))
                    self._rate_limit_until = time.time() + retry_after
                    logger.warning(f"Rate limited, backing off {retry_after}s")
                    time.sleep(retry_after)
                    continue
                resp.raise_for_status()
                return resp.json()
            except requests.RequestException as e:
                logger.warning(f"Request failed (attempt {attempt+1}): {e}")
                if attempt < 2:
                    time.sleep(2 ** attempt)
        return None
