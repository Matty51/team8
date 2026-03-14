# Team 8: Covid-19 Contest

Instructions to run the starter application: Enter the root directory of this
project and install dependencies from `requirements.txt`:

```bash
    python -m venv venv/
    venv/Scripts/Activate.ps1
    pip install -r requirements.txt
    flask run
```

---

## Polymarket Trading Bot

A modular paper-trading bot for Polymarket prediction markets.

### Quick Start

```bash
pip install -r requirements.txt

# Scan markets once (no trading)
python -m bot.main --scan-only

# Run the paper trading loop
python -m bot.main

# Custom interval and market count
python -m bot.main --interval 60 --markets 100
```

### Architecture

```
bot/
├── config.py      # Configuration (env vars or defaults)
├── client.py      # Polymarket API client (CLOB + Gamma)
├── scanner.py     # Market scanner — finds opportunities
├── strategy.py    # Trading strategies (spread arb, value)
├── risk.py        # Risk management (limits, circuit breakers)
├── trader.py      # Trade executor (paper mode)
└── main.py        # Entry point and main loop
```

### Strategies

1. **Spread Arbitrage** — Buys both YES and NO when their combined price < $1.00.
   Guaranteed profit at market resolution (minus fees). Low risk.
2. **Value Betting** — Buys tokens at extreme prices (near 0 or 1) where
   reversion is likely. Higher risk, higher potential return.

### Configuration

Override defaults with environment variables (prefix `BOT_`):

| Variable | Default | Description |
|---|---|---|
| `BOT_PAPER_TRADING` | `true` | Paper mode (no real money) |
| `BOT_MAX_POSITION_SIZE_USD` | `10.0` | Max per-trade size |
| `BOT_MAX_DAILY_LOSS_USD` | `20.0` | Daily loss circuit breaker |
| `BOT_MIN_SPREAD_PCT` | `2.0` | Min spread discount for arb |
| `BOT_MIN_EDGE_PCT` | `3.0` | Min edge for value bets |
| `BOT_SCAN_INTERVAL_SECONDS` | `30` | Scan frequency |

### Risk Controls

- Per-trade position size limits
- Maximum concurrent open positions
- Daily loss circuit breaker
- Daily trade count limit
- Confidence threshold gating
