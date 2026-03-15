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

## Crypto Trading Bot

A modular crypto trading bot supporting **Bybit, OKX, Gate.io, Binance, Bitget, KuCoin**, and 100+ exchanges via [ccxt](https://github.com/ccxt/ccxt). Starts in paper trading mode — no real money at risk.

### Quick Start

```bash
pip install -r requirements.txt

# Paper trade BTC/USDT on Bybit (default)
python -m bot

# Use a different exchange
python -m bot --exchange okx --pair ETH/USDT
python -m bot --exchange gateio --pair SOL/USDT
python -m bot --exchange bitget --pair BTC/USDT

# Just scan, no trading
python -m bot --scan-only

# Slower scan with higher timeframe
python -m bot --timeframe 15m --interval 60
```

### Architecture

```
bot/
├── config.py       # Configuration (env vars or defaults)
├── client.py       # Unified exchange client (ccxt)
├── indicators.py   # Technical indicators (SMA, EMA, RSI, MACD, BB, ATR)
├── scanner.py      # Market scanner — fetches candles, computes indicators
├── strategy.py     # Trading strategies (SMA crossover, RSI, MACD, volume)
├── risk.py         # Risk management (limits, stop-loss, circuit breakers)
├── trader.py       # Trade executor (paper + live mode)
└── main.py         # Entry point and main loop
```

### Strategies

| Strategy | Signal | Best For | Risk |
|---|---|---|---|
| **SMA Crossover** | Fast SMA crosses slow SMA | Trending markets | Low-Med |
| **RSI** | Buy oversold (<30), sell overbought (>70) | Ranging markets | Medium |
| **MACD** | MACD crosses signal line | Momentum confirmation | Medium |
| **Volume Spike** | 2x+ volume with directional candle | Breakouts | Med-High |

### Configuration

Override defaults with environment variables (prefix `BOT_`):

| Variable | Default | Description |
|---|---|---|
| `BOT_EXCHANGE` | `bybit` | Exchange (bybit, okx, gateio, binance, bitget) |
| `BOT_TRADING_PAIR` | `BTC/USDT` | Trading pair |
| `BOT_TIMEFRAME` | `5m` | Candle timeframe |
| `BOT_PAPER_TRADING` | `true` | Paper mode (no real money) |
| `BOT_MAX_POSITION_SIZE_USD` | `50.0` | Max per-trade size |
| `BOT_MAX_DAILY_LOSS_USD` | `30.0` | Daily loss circuit breaker |
| `BOT_STOP_LOSS_PCT` | `1.5` | Stop-loss percentage |
| `BOT_TAKE_PROFIT_PCT` | `3.0` | Take-profit percentage |
| `BOT_API_KEY` | ` ` | Exchange API key (for live trading) |
| `BOT_API_SECRET` | ` ` | Exchange API secret |
| `BOT_API_PASSPHRASE` | ` ` | API passphrase (OKX only) |

### Risk Controls

- Per-trade position size limits
- Stop-loss and take-profit on every trade
- Maximum concurrent open positions (default: 3)
- Daily loss circuit breaker (default: $30)
- Daily trade count limit (default: 30)
- Confidence threshold gating (default: 55%)
- No duplicate positions in the same pair

### Going Live

1. Create API keys on your exchange (with trading permissions only — no withdrawal)
2. Set env vars: `BOT_API_KEY`, `BOT_API_SECRET`, `BOT_API_PASSPHRASE` (OKX)
3. Run with `--live` flag: `python -m bot --live`
