#!/usr/bin/env python3
"""
Multi-pair backtest runner — fetches live data from Bybit and tests all pairs.

Usage:
    python run_multi_backtest.py
    python run_multi_backtest.py --tf 4h --days 365
"""
import argparse
import sys
from dotenv import load_dotenv

load_dotenv()

from bot.backtest import Backtester, fetch_historical_data
from bot.config import Config


PAIRS = [
    "BTC/USDT",
    "ETH/USDT",
    "SOL/USDT",
    "XRP/USDT",
    "DOGE/USDT",
    "ADA/USDT",
    "AVAX/USDT",
    "LINK/USDT",
]


def main():
    parser = argparse.ArgumentParser(description="Multi-pair backtest")
    parser.add_argument("--tf", default="1h", help="Timeframe (default: 1h)")
    parser.add_argument("--days", type=int, default=180, help="Days of history (default: 180)")
    parser.add_argument("--capital", type=float, default=500, help="Starting capital (default: 500)")
    parser.add_argument("--exchange", default="bybit", help="Exchange (default: bybit)")
    args = parser.parse_args()

    config = Config(
        exchange=args.exchange,
        timeframe=args.tf,
        starting_capital=args.capital,
        paper_trading=True,
    )

    bt = Backtester(config)
    results = []

    print(f"\n{'=' * 72}")
    print(f"  MULTI-PAIR BACKTEST — {args.exchange.upper()} {args.tf} — {args.days} days")
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"{'=' * 72}\n")

    for pair in PAIRS:
        print(f"\n--- {pair} ---")
        try:
            candles = fetch_historical_data(
                config.exchange_id, pair, args.tf, args.days, config
            )
            if not candles or len(candles) < 150:
                print(f"  Skipped: not enough data ({len(candles) if candles else 0} candles)")
                continue

            result = bt.run(candles, symbol=pair)
            results.append((pair, result))
            print(result.summary())

        except Exception as e:
            print(f"  Error: {e}")
            continue

    # Print combined summary
    if not results:
        print("\nNo results. Check your API connection.")
        sys.exit(1)

    total_trades = sum(r.total_trades for _, r in results)
    total_wins = sum(r.winning_trades for _, r in results)
    overall_wr = total_wins / total_trades * 100 if total_trades else 0

    print(f"\n{'=' * 72}")
    print(f"  COMBINED RESULTS — {len(results)} pairs")
    print(f"{'=' * 72}")
    print(f"\n  {'Pair':12s} | Trades | Win%   | Return%  | PF    | MaxDD%")
    print(f"  {'-' * 62}")

    for pair, r in results:
        ret = r.total_return_pct
        print(
            f"  {pair:12s} | {r.total_trades:6d} | {r.win_rate:5.1f}% | "
            f"{ret:+7.2f}% | {r.profit_factor:5.2f} | {r.max_drawdown_pct:.1f}%"
        )

        # Strategy breakdown per pair
        for strat, st in sorted(r.trades_by_strategy.items()):
            print(
                f"    {strat:20s} {st['trades']:3d}t  "
                f"{st['win_rate']:.0f}% win  ${st['total_pnl']:+.2f}"
            )

    print(f"\n  {'-' * 62}")
    print(f"  OVERALL: {total_trades} trades, {overall_wr:.1f}% win rate")
    print(f"{'=' * 72}\n")

    # Identify best/worst
    if results:
        best = max(results, key=lambda x: x[1].win_rate)
        worst = min(results, key=lambda x: x[1].win_rate)
        print(f"  Best pair:  {best[0]} — {best[1].win_rate:.1f}% win rate")
        print(f"  Worst pair: {worst[0]} — {worst[1].win_rate:.1f}% win rate")

    # Strategy aggregate across all pairs
    all_strats = {}
    for _, r in results:
        for strat, st in r.trades_by_strategy.items():
            if strat not in all_strats:
                all_strats[strat] = {"trades": 0, "wins": 0, "pnl": 0.0}
            all_strats[strat]["trades"] += st["trades"]
            all_strats[strat]["wins"] += st["wins"]
            all_strats[strat]["pnl"] += st["total_pnl"]

    print(f"\n  Strategy Performance (all pairs combined):")
    print(f"  {'Strategy':25s} | Trades | Win%   | Total PnL")
    print(f"  {'-' * 55}")
    for strat, st in sorted(all_strats.items(), key=lambda x: -x[1]["trades"]):
        wr = st["wins"] / st["trades"] * 100 if st["trades"] else 0
        print(f"  {strat:25s} | {st['trades']:6d} | {wr:5.1f}% | ${st['pnl']:+.2f}")

    print()


if __name__ == "__main__":
    main()
