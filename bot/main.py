#!/usr/bin/env python3
"""
Polymarket Trading Bot — main entry point.

Usage:
    python -m bot.main              # Run scanner + paper trading loop
    python -m bot.main --scan-only  # Just scan and print opportunities
    python -m bot.main --stats      # Show current paper trading stats
"""
import argparse
import json
import logging
import signal
import sys
import time
from datetime import datetime

from .client import PolymarketClient
from .config import Config
from .risk import RiskManager
from .scanner import MarketScanner
from .strategy import StrategyManager
from .trader import PaperTrader

# ── Globals for graceful shutdown ────────────────────────────────────
running = True


def handle_shutdown(signum, frame):
    global running
    print("\nShutting down gracefully...")
    running = False


def setup_logging(config: Config):
    """Configure logging to console and file."""
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(config.log_file),
    ]
    logging.basicConfig(
        level=getattr(logging, config.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


def print_banner():
    print("=" * 60)
    print("  Polymarket Trading Bot")
    print("  Paper Trading Mode — No Real Money")
    print("=" * 60)
    print()


def print_opportunities(opportunities):
    """Pretty-print discovered opportunities."""
    if not opportunities:
        print("  No opportunities found this scan.")
        return

    for i, opp in enumerate(opportunities, 1):
        print(f"  [{i}] {opp.opportunity_type.upper()}")
        print(f"      Market: {opp.market_question[:70]}")
        print(f"      {opp.details}")
        print(f"      Volume: ${opp.volume_24h:,.0f} | "
              f"Edge: {opp.estimated_edge_pct:.1f}%")
        print()


def print_trades(trades):
    """Pretty-print executed trades."""
    if not trades:
        return
    print(f"  Executed {len(trades)} paper trade(s):")
    for t in trades:
        print(f"    {t.side.upper()} {t.outcome} "
              f"{t.shares:.2f} shares @ {t.fill_price:.3f} "
              f"(${t.size_usd:.2f}) — {t.strategy}")


def run_scan_only(config: Config):
    """One-shot scan: show opportunities and exit."""
    client = PolymarketClient(config)
    scanner = MarketScanner(client, config)

    print("Scanning markets...\n")
    opportunities = scanner.scan()
    print_opportunities(opportunities)


def run_bot(config: Config):
    """Main trading loop: scan → strategize → execute (paper)."""
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)

    client = PolymarketClient(config)
    scanner = MarketScanner(client, config)
    strategy_mgr = StrategyManager(config)
    risk_mgr = RiskManager(config)
    trader = PaperTrader(config, risk_mgr)

    print_banner()
    print(f"  Scan interval: {config.scan_interval_seconds}s")
    print(f"  Max position: ${config.max_position_size_usd:.2f}")
    print(f"  Daily loss limit: ${config.max_daily_loss_usd:.2f}")
    print(f"  Min spread for arb: {config.min_spread_pct}%")
    print(f"  Min edge for value: {config.min_edge_pct}%")
    print()

    cycle = 0
    while running:
        cycle += 1
        now = datetime.utcnow().strftime("%H:%M:%S")
        print(f"── Cycle {cycle} ({now}) " + "─" * 40)

        try:
            # Step 1: Scan
            opportunities = scanner.scan()
            print_opportunities(opportunities)

            # Step 2: Generate signals
            signals = strategy_mgr.generate_signals(opportunities)
            if signals:
                print(f"  Generated {len(signals)} signal(s)")

            # Step 3: Execute (paper)
            trades = trader.execute_signals(signals)
            print_trades(trades)

            # Step 4: Show stats
            stats = trader.get_stats()
            print(f"\n  Balance: {stats['balance']} | "
                  f"Trades: {stats['total_trades']} | "
                  f"Volume: {stats['total_volume']}")
            risk = stats['risk_summary']
            print(f"  Positions: {risk['open_positions']}/"
                  f"{risk['max_positions']} | "
                  f"Daily PnL: {risk['daily_pnl']}")

        except Exception as e:
            logging.getLogger(__name__).error(f"Cycle error: {e}")
            print(f"  Error: {e}")

        print()

        # Wait for next cycle
        for _ in range(config.scan_interval_seconds):
            if not running:
                break
            time.sleep(1)

    # Save trade log on exit
    trader.save_trade_log()
    print("\nFinal stats:")
    print(json.dumps(trader.get_stats(), indent=2))
    print(f"\nTrade log saved to paper_trades.json")


def main():
    parser = argparse.ArgumentParser(description="Polymarket Trading Bot")
    parser.add_argument(
        "--scan-only", action="store_true",
        help="Scan markets once and exit",
    )
    parser.add_argument(
        "--interval", type=int, default=None,
        help="Override scan interval (seconds)",
    )
    parser.add_argument(
        "--markets", type=int, default=None,
        help="Override number of markets to scan",
    )
    args = parser.parse_args()

    config = Config.from_env()
    if args.interval:
        config.scan_interval_seconds = args.interval
    if args.markets:
        config.markets_to_scan = args.markets

    setup_logging(config)

    if args.scan_only:
        run_scan_only(config)
    else:
        run_bot(config)


if __name__ == "__main__":
    main()
