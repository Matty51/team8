"""
Unified exchange client using ccxt.
Supports Bybit, OKX, Gate.io, Binance, Bitget, KuCoin, and 100+ others.
"""
import logging
from typing import Any, Dict, List, Optional

import ccxt

from .config import Config

logger = logging.getLogger(__name__)


class ExchangeClient:
    """Wraps ccxt for unified access to any supported crypto exchange."""

    def __init__(self, config: Config):
        self.config = config
        self.exchange = self._create_exchange()

    def _create_exchange(self) -> ccxt.Exchange:
        """Initialize the ccxt exchange instance."""
        exchange_class = getattr(ccxt, self.config.exchange_id, None)
        if exchange_class is None:
            raise ValueError(
                f"Exchange '{self.config.exchange}' not supported by ccxt. "
                f"Supported: bybit, okx, gateio, binance, bitget, kucoin, ..."
            )

        params = {
            "enableRateLimit": True,
        }

        # Add API credentials if provided
        if self.config.api_key:
            params["apiKey"] = self.config.api_key
        if self.config.api_secret:
            params["secret"] = self.config.api_secret
        if self.config.api_passphrase:
            params["password"] = self.config.api_passphrase

        # Use sandbox/testnet if paper trading and exchange supports it
        if self.config.paper_trading:
            params["sandbox"] = True

        exchange = exchange_class(params)

        # Set market type
        if self.config.market_type == "future":
            exchange.options["defaultType"] = "swap"

        logger.info(
            f"Connected to {self.config.exchange_id} "
            f"({'sandbox' if self.config.paper_trading else 'LIVE'}) "
            f"— {self.config.market_type}"
        )
        return exchange

    # ── Market data ──────────────────────────────────────────────────

    def fetch_ohlcv(
        self,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> List[List]:
        """
        Fetch OHLCV candles.
        Returns list of [timestamp, open, high, low, close, volume].
        """
        symbol = symbol or self.config.trading_pair
        timeframe = timeframe or self.config.timeframe
        limit = limit or self.config.candle_lookback

        try:
            candles = self.exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, limit=limit
            )
            logger.debug(
                f"Fetched {len(candles)} candles for {symbol} ({timeframe})"
            )
            return candles
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch OHLCV: {e}")
            return []

    def fetch_ticker(self, symbol: Optional[str] = None) -> Optional[Dict]:
        """Fetch current ticker (price, volume, bid, ask, etc.)."""
        symbol = symbol or self.config.trading_pair
        try:
            return self.exchange.fetch_ticker(symbol)
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch ticker: {e}")
            return None

    def fetch_orderbook(
        self, symbol: Optional[str] = None, limit: int = 20
    ) -> Optional[Dict]:
        """Fetch orderbook (bids and asks)."""
        symbol = symbol or self.config.trading_pair
        try:
            return self.exchange.fetch_order_book(symbol, limit=limit)
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch orderbook: {e}")
            return None

    def fetch_balance(self) -> Optional[Dict]:
        """Fetch account balance."""
        try:
            return self.exchange.fetch_balance()
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch balance: {e}")
            return None

    # ── Order execution ──────────────────────────────────────────────

    def create_market_order(
        self, side: str, amount: float, symbol: Optional[str] = None
    ) -> Optional[Dict]:
        """Place a market order. side = 'buy' or 'sell'."""
        symbol = symbol or self.config.trading_pair
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="market",
                side=side,
                amount=amount,
            )
            logger.info(
                f"Market {side} {amount} {symbol} — "
                f"Order ID: {order.get('id')}"
            )
            return order
        except ccxt.BaseError as e:
            logger.error(f"Market order failed: {e}")
            return None

    def create_limit_order(
        self,
        side: str,
        amount: float,
        price: float,
        symbol: Optional[str] = None,
    ) -> Optional[Dict]:
        """Place a limit order."""
        symbol = symbol or self.config.trading_pair
        try:
            order = self.exchange.create_order(
                symbol=symbol,
                type="limit",
                side=side,
                amount=amount,
                price=price,
            )
            logger.info(
                f"Limit {side} {amount} {symbol} @ {price} — "
                f"Order ID: {order.get('id')}"
            )
            return order
        except ccxt.BaseError as e:
            logger.error(f"Limit order failed: {e}")
            return None

    def cancel_order(
        self, order_id: str, symbol: Optional[str] = None
    ) -> bool:
        """Cancel an open order."""
        symbol = symbol or self.config.trading_pair
        try:
            self.exchange.cancel_order(order_id, symbol)
            logger.info(f"Cancelled order {order_id}")
            return True
        except ccxt.BaseError as e:
            logger.error(f"Cancel order failed: {e}")
            return False

    def fetch_open_orders(
        self, symbol: Optional[str] = None
    ) -> List[Dict]:
        """Fetch all open orders."""
        symbol = symbol or self.config.trading_pair
        try:
            return self.exchange.fetch_open_orders(symbol)
        except ccxt.BaseError as e:
            logger.error(f"Failed to fetch open orders: {e}")
            return []

    # ── Utility ──────────────────────────────────────────────────────

    def get_price(self, symbol: Optional[str] = None) -> Optional[float]:
        """Get the current mid price."""
        ticker = self.fetch_ticker(symbol)
        if ticker:
            return ticker.get("last")
        return None

    def get_min_order_amount(self, symbol: Optional[str] = None) -> float:
        """Get minimum order amount for a symbol."""
        symbol = symbol or self.config.trading_pair
        try:
            self.exchange.load_markets()
            market = self.exchange.market(symbol)
            return market.get("limits", {}).get("amount", {}).get("min", 0.0)
        except Exception:
            return 0.0
