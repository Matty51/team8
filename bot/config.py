"""
Bot configuration. Override via environment variables or .env file.
"""
import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Polymarket CLOB API ──────────────────────────────────────────
    clob_api_url: str = "https://clob.polymarket.com"
    gamma_api_url: str = "https://gamma-api.polymarket.com"

    # Auth (only needed for live trading)
    api_key: str = ""
    api_secret: str = ""
    api_passphrase: str = ""
    private_key: str = ""  # Wallet private key for signing

    # ── Trading mode ─────────────────────────────────────────────────
    paper_trading: bool = True  # Start in paper mode — no real money

    # ── Risk management ──────────────────────────────────────────────
    max_position_size_usd: float = 10.0    # Max per-trade size in USDC
    max_open_positions: int = 5            # Max simultaneous positions
    max_daily_loss_usd: float = 20.0       # Stop trading after this loss
    max_daily_trades: int = 50             # Circuit breaker

    # ── Strategy parameters ──────────────────────────────────────────
    min_spread_pct: float = 2.0            # Min YES+NO discount to act on (%)
    min_edge_pct: float = 3.0              # Min estimated edge for value bets (%)
    confidence_threshold: float = 0.60     # Minimum confidence to enter

    # ── Scan settings ────────────────────────────────────────────────
    scan_interval_seconds: int = 30        # How often to scan markets
    markets_to_scan: int = 50              # How many active markets to check

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
