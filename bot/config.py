"""
Bot configuration. Override via environment variables or .env file.
"""
import os
from dataclasses import dataclass


@dataclass
class Config:
    # ── Exchange ─────────────────────────────────────────────────────
    # Supported: bybit, okx, gateio, binance, kucoin, bitget, etc.
    exchange: str = "bybit"

    # Auth (set via env vars — never hardcode)
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""  # Required for OKX

    # ── Trading mode ─────────────────────────────────────────────────
    paper_trading: bool = True       # Sandbox/paper mode — no real money
    trading_pair: str = "BTC/USDT"   # Primary pair to trade
    timeframe: str = "5m"            # Candle timeframe for analysis
    market_type: str = "spot"        # "spot" or "future"

    # ── Capital ──────────────────────────────────────────────────────
    starting_capital: float = 500.0        # Account size for risk calculations

    # ── Risk management ──────────────────────────────────────────────
    # From Wealth Training: max 5% risk per trade
    risk_per_trade_pct: float = 2.0        # Risk 2% of capital per trade ($10 on $500)
    max_risk_per_trade_pct: float = 5.0    # Hard cap: never risk more than 5% per trade
    max_position_size_usd: float = 25.0    # Max per-trade size (5% of $500)
    max_open_positions: int = 3            # Max simultaneous positions
    max_portfolio_risk_pct: float = 6.0    # Max total capital at risk (3 x 2%)
    max_daily_loss_usd: float = 25.0       # Stop trading after 5% daily loss
    max_daily_trades: int = 20             # Circuit breaker
    stop_loss_pct: float = 1.5             # Stop-loss percentage
    take_profit_pct: float = 3.0           # Take-profit percentage (2:1 reward:risk)

    # ── Strategy parameters ──────────────────────────────────────────
    # Moving averages (from Wealth Training: 10/20 short-term, 50/100 long-term)
    sma_fast_period: int = 10
    sma_slow_period: int = 20
    sma_long_fast_period: int = 50     # Long-term trend MA fast
    sma_long_slow_period: int = 100    # Long-term trend MA slow

    # RSI (from Wealth Training: period 21)
    rsi_period: int = 21
    rsi_oversold: float = 30.0
    rsi_overbought: float = 70.0

    # MACD (from Wealth Training: 5, 35, 5)
    macd_fast: int = 5
    macd_slow: int = 35
    macd_signal: int = 5

    # Stochastics (from Wealth Training: 12,3,3 fast and 20,12,9 slow)
    stoch_fast_k: int = 12
    stoch_fast_d: int = 3
    stoch_fast_smooth: int = 3
    stoch_slow_k: int = 20
    stoch_slow_d: int = 12
    stoch_slow_smooth: int = 9
    stoch_oversold: float = 20.0
    stoch_overbought: float = 80.0

    # CCI (from Wealth Training: 90, 1)
    cci_period: int = 90
    cci_overbought: float = 100.0
    cci_oversold: float = -100.0

    # Volume spike
    volume_spike_multiplier: float = 2.0

    # General
    min_confidence: float = 0.55

    # Minimum reward:risk ratio (from Wealth Training checklist: 2:1)
    min_reward_risk_ratio: float = 2.0

    # ── Scan settings ────────────────────────────────────────────────
    scan_interval_seconds: int = 30
    candle_lookback: int = 100  # How many candles to fetch for analysis

    # ── Logging ──────────────────────────────────────────────────────
    log_level: str = "INFO"
    log_file: str = "bot.log"

    @classmethod
    def from_env(cls) -> "Config":
        """Load config, overriding defaults with env vars where set."""
        kwargs = {}
        for f in cls.__dataclass_fields__:
            env_val = os.environ.get(f"BOT_{f.upper()}")
            if env_val is not None:
                field_type = cls.__dataclass_fields__[f].type
                if field_type == "bool":
                    kwargs[f] = env_val.lower() in ("1", "true", "yes")
                elif field_type == "float":
                    kwargs[f] = float(env_val)
                elif field_type == "int":
                    kwargs[f] = int(env_val)
                else:
                    kwargs[f] = env_val
        return cls(**kwargs)

    @property
    def exchange_id(self) -> str:
        """Normalize exchange name to ccxt ID."""
        mapping = {
            "gate": "gateio",
            "gate.io": "gateio",
            "bitget": "bitget",
            "bybit": "bybit",
            "okx": "okx",
            "binance": "binance",
            "kucoin": "kucoin",
        }
        return mapping.get(self.exchange.lower(), self.exchange.lower())
