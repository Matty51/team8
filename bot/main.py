#!/usr/bin/env python3
"""
Crypto Trading Bot — main entry point.

Usage:
    python -m bot                                    # Paper trade BTC/USDT on Bybit
    python -m bot --exchange okx --pair ETH/USDT     # Paper trade ETH on OKX
    python -m bot --exchange gateio --pair SOL/USDT   # Paper trade SOL on Gate.io
    python -m bot --scan-only                         # Just scan, no trading
    python -m bot --interval 60 --timeframe 15m       # Custom timing
"""
import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime

from .client import ExchangeClient
from .config import Config
from .risk import RiskManager
from .scanner import MarketScanner
from .strategy import StrategyManager
from .trader import Trader

# ── Globals for graceful shutdown ────────────────────────────────────
running = True


def handle_shutdown(signum, frame):
    global running
    print("\nShutting down gracefully...")
    running = False


def setup_logging(config: Config):
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.log_file),
    ]
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def print_banner(config: Config):
    mode = "PAPER TRADING" if config.paper_trading else "LIVE TRADING"
    print("=" * 60)
    print(f"  Crypto Trading Bot — {mode}")
    print(f"  Exchange: {config.exchange_id}")
    print(f"  Pair: {config.trading_pair} ({config.timeframe})")
    print("=" * 60)
    print()
    print(f"  Max position:    ${config.max_position_size_usd:.2f}")
    print(f"  Daily loss limit: ${config.max_daily_loss_usd:.2f}")
    print(f"  Stop-loss:        {config.stop_loss_pct}%")
    print(f"  Take-profit:      {config.take_profit_pct}%")
    print(f"  Scan interval:    {config.scan_interval_seconds}s")
    print()
    print("  Strategies: SMA Crossover, RSI, MACD, Volume Spike")
    print()


def print_snapshot(snapshot):
    """Pretty-print market snapshot."""
    print(f"  Price:  ${snapshot.current_price:,.2f}")
    print(f"  Trend:  {snapshot.trend.upper()}")
    if snapshot.rsi:
        rsi_label = ""
        if snapshot.is_oversold:
            rsi_label = " (OVERSOLD)"
        elif snapshot.is_overbought:
            rsi_label = " (OVERBOUGHT)"
        print(f"  RSI:    {snapshot.rsi:.1f}{rsi_label}")
    if snapshot.sma_fast and snapshot.sma_slow:
        print(f"  SMA:    fast={snapshot.sma_fast:,.2f} "
              f"slow={snapshot.sma_slow:,.2f}")
    if snapshot.volume_ratio:
        spike = " (SPIKE)" if snapshot.has_volume_spike else ""
        print(f"  Volume: {snapshot.volume_ratio:.1f}x avg{spike}")
    if snapshot.bb_upper and snapshot.bb_lower:
        print(f"  BB:     [{snapshot.bb_lower:,.2f} — "
              f"{snapshot.bb_upper:,.2f}]")


def print_signals(signals):
    if not signals:
        print("  No trade signals.")
        return
    for s in signals:
        print(f"  >> [{s.strategy_name}] {s.side.value.upper()} "
              f"@ ${s.price:,.2f} | "
              f"conf: {s.confidence:.0%} | {s.reason}")


def print_trades(trades, label="Executed"):
    if not trades:
        return
    for t in trades:
        pnl_str = f" | PnL: ${t.pnl:+.2f}" if t.pnl is not None else ""
        print(f"  ** {label}: {t.side.upper()} {t.amount:.6f} "
              f"{t.symbol} @ ${t.price:,.2f}{pnl_str} — {t.reason}")


def run_scan_only(config: Config):
    """One-shot scan: show market snapshot and exit."""
    client = ExchangeClient(config)
    scanner = MarketScanner(client, config)
    strategy_mgr = StrategyManager(config)

    print(f"Scanning {config.trading_pair} on {config.exchange_id}...\n")
    snapshot = scanner.scan()
    if snapshot:
        print_snapshot(snapshot)
        print()
        signals = strategy_mgr.generate_signals(snapshot)
        print_signals(signals)
    else:
        print("  Could not fetch market data.")


def run_bot(config: Config):
    """Main trading loop: scan → strategize → execute."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    client = ExchangeClient(config)
    scanner = MarketScanner(client, config)
    strategy_mgr = StrategyManager(config)
    risk_mgr = RiskManager(config)
    trader = Trader(client, config, risk_mgr)

    print_banner(config)

    cycle = 0
    while running:
        cycle += 1
        now = datetime.utcnow().strftime("%H:%M:%S")
        print(f"── Cycle {cycle} ({now}) " + "─" * 40)

        try:
            # Step 1: Scan market
            snapshot = scanner.scan()
            if not snapshot:
                print("  No data, retrying next cycle...")
                _sleep(config.scan_interval_seconds)
                continue

            print_snapshot(snapshot)

            # Step 2: Check exits (stop-loss / take-profit)
            exits = trader.check_exits(snapshot)
            print_trades(exits, label="Closed")

            # Step 3: Generate signals
            signals = strategy_mgr.generate_signals(snapshot)
            print_signals(signals)

            # Step 4: Execute
            trades = trader.execute_signals(signals)
            print_trades(trades)

            # Step 5: Show stats
            stats = trader.get_stats()
            risk = stats["risk_summary"]
            print(f"\n  Balance: {stats['balance']} | "
                  f"Trades: {stats['total_trades']} | "
                  f"Win rate: {stats['win_rate']}")
            print(f"  Positions: {risk['open_positions']}/"
                  f"{risk['max_positions']} | "
                  f"Day PnL: {risk['daily_pnl']} | "
                  f"Total PnL: {risk['total_pnl']}")

        except Exception as e:
            logging.getLogger(__name__).error(f"Cycle error: {e}")
            print(f"  Error: {e}")

        print()
        _sleep(config.scan_interval_seconds)

    # Shutdown
    trader.save_trade_log()
    print("\nFinal stats:")
    print(json.dumps(trader.get_stats(), indent=2))
    print(f"\nTrade log saved to trade_log.json")


def _sleep(seconds: int):
    """Interruptible sleep."""
    for _ in range(seconds):
        if not running:
            break
        time.sleep(1)


def main():
    parser = argparse.ArgumentParser(
        description="Crypto Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m bot                                    Paper trade BTC/USDT on Bybit
  python -m bot --exchange okx --pair ETH/USDT     Trade ETH on OKX
  python -m bot --exchange gateio --pair SOL/USDT   Trade SOL on Gate.io
  python -m bot --exchange bitget --pair BTC/USDT   Trade BTC on Bitget
  python -m bot --scan-only                         Just scan, no trading
  python -m bot --timeframe 15m --interval 60       Slower, higher timeframe
        """,
    )
    parser.add_argument(
        "--exchange", type=str, default=None,
        help="Exchange to use (bybit, okx, gateio, binance, bitget, kucoin)",
    )
    parser.add_argument(
        "--pair", type=str, default=None,
        help="Trading pair (e.g., BTC/USDT, ETH/USDT, SOL/USDT)",
    )
    parser.add_argument(
        "--timeframe", type=str, default=None,
        help="Candle timeframe (1m, 5m, 15m, 1h, 4h, 1d)",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Scan interval in seconds",
    )
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Scan markets once and exit (no trading)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Enable live trading (requires API keys)",
    )
    args = parser.parse_args()

    config = Config.from_env()
    if args.exchange:
        config.exchange = args.exchange
    if args.pair:
        config.trading_pair = args.pair
    if args.timeframe:
        config.timeframe = args.timeframe
    if args.interval:
        config.scan_interval_seconds = args.interval
    if args.live:
        config.paper_trading = False

    setup_logging(config)

    if args.scan_only:
        run_scan_only(config)
    else:
        run_bot(config)


if __name__ == "__main__":
    main()
