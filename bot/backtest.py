"""
Backtesting engine — replay historical data through strategies.

Usage:
    python -m bot.backtest                              # BTC/USDT 1h, 6 months
    python -m bot.backtest --pair ETH/USDT --tf 15m     # ETH 15m
    python -m bot.backtest --days 365                   # 1 year of data
    python -m bot.backtest --exchange binance            # Use Binance data
"""
import argparse
import json
import logging
import math
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

from . import indicators as ind
from .config import Config
from .levels import LevelManager
from .scanner import MarketSnapshot
from .strategy import Side, Signal, StrategyManager, check_trading_checklist

logger = logging.getLogger(__name__)


# ── Data structures ────────────────────────────────────────────────────


@dataclass
class BacktestTrade:
    """A single completed backtest trade."""
    entry_time: str
    exit_time: str
    symbol: str
    side: str
    strategy: str
    entry_price: float
    exit_price: float
    size_usd: float
    pnl: float
    pnl_pct: float
    exit_reason: str  # "stop_loss", "take_profit", "end_of_data"
    confidence: float
    reason: str
    bars_held: int


@dataclass
class BacktestPosition:
    """An open position during backtest."""
    entry_idx: int
    entry_price: float
    side: str
    size_usd: float
    amount: float
    stop_loss: float
    take_profit: float
    strategy: str
    confidence: float
    reason: str
    symbol: str


@dataclass
class BacktestResult:
    """Complete backtest result with performance metrics."""
    symbol: str
    timeframe: str
    start_date: str
    end_date: str
    total_bars: int
    starting_capital: float
    ending_capital: float
    total_return_pct: float
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    largest_win: float
    largest_loss: float
    profit_factor: float
    max_drawdown_pct: float
    max_drawdown_usd: float
    sharpe_ratio: float
    avg_bars_held: float
    trades_by_strategy: Dict[str, Dict]
    trades: List[BacktestTrade]
    equity_curve: List[float]

    def summary(self) -> str:
        lines = [
            "",
            "=" * 60,
            f"  BACKTEST RESULTS: {self.symbol} ({self.timeframe})",
            f"  {self.start_date} → {self.end_date} ({self.total_bars} bars)",
            "=" * 60,
            "",
            f"  Starting Capital:   ${self.starting_capital:,.2f}",
            f"  Ending Capital:     ${self.ending_capital:,.2f}",
            f"  Total Return:       {self.total_return_pct:+.2f}%",
            "",
            f"  Total Trades:       {self.total_trades}",
            f"  Win Rate:           {self.win_rate:.1f}%",
            f"  Winners:            {self.winning_trades}",
            f"  Losers:             {self.losing_trades}",
            "",
            f"  Avg Win:            ${self.avg_win:+.2f}",
            f"  Avg Loss:           ${self.avg_loss:+.2f}",
            f"  Largest Win:        ${self.largest_win:+.2f}",
            f"  Largest Loss:       ${self.largest_loss:+.2f}",
            f"  Profit Factor:      {self.profit_factor:.2f}",
            "",
            f"  Max Drawdown:       {self.max_drawdown_pct:.2f}% (${self.max_drawdown_usd:,.2f})",
            f"  Sharpe Ratio:       {self.sharpe_ratio:.2f}",
            f"  Avg Bars Held:      {self.avg_bars_held:.1f}",
            "",
            "  Strategy Breakdown:",
        ]
        for strat, stats in sorted(self.trades_by_strategy.items()):
            lines.append(
                f"    {strat:25s}  {stats['trades']:3d} trades  "
                f"{stats['win_rate']:.0f}% win  "
                f"PnL ${stats['total_pnl']:+.2f}"
            )
        lines.append("")
        lines.append("=" * 60)
        return "\n".join(lines)


# ── Backtest engine ────────────────────────────────────────────────────


class Backtester:
    """
    Replays historical candle data bar-by-bar through all strategies.

    For each bar:
    1. Build a MarketSnapshot from candles seen so far
    2. Check open positions for stop-loss / take-profit
    3. Generate new signals from strategies
    4. Execute trades (paper only, with risk checks)
    5. Track equity and PnL
    """

    def __init__(self, config: Config):
        self.config = config
        self.strategy_mgr = StrategyManager(config)
        self.level_mgr = LevelManager(config)

    def run(
        self,
        candles: List[List],
        symbol: Optional[str] = None,
    ) -> BacktestResult:
        """
        Run backtest on OHLCV candle data.

        candles: list of [timestamp, open, high, low, close, volume]
        """
        symbol = symbol or self.config.trading_pair

        # Need enough data for indicators
        min_bars = max(
            self.config.sma_slow_period,
            self.config.cci_period,
            self.config.sma_long_slow_period,
        ) + 20

        if len(candles) < min_bars:
            raise ValueError(
                f"Need at least {min_bars} candles, got {len(candles)}"
            )

        # Extract OHLCV columns
        timestamps = [c[0] for c in candles]
        opens = [c[1] for c in candles]
        highs = [c[2] for c in candles]
        lows = [c[3] for c in candles]
        closes = [c[4] for c in candles]
        volumes = [c[5] for c in candles]

        # State
        capital = self.config.starting_capital
        equity_curve = [capital]
        open_positions: List[BacktestPosition] = []
        closed_trades: List[BacktestTrade] = []
        daily_trades = 0
        daily_pnl = 0.0
        last_date = None

        # Walk through bars
        for i in range(min_bars, len(candles)):
            # Reset daily limits
            ts = timestamps[i]
            if isinstance(ts, (int, float)):
                current_date = datetime.utcfromtimestamp(ts / 1000).date()
            else:
                current_date = None
            if current_date and current_date != last_date:
                daily_trades = 0
                daily_pnl = 0.0
                last_date = current_date

            # Daily loss circuit breaker
            if daily_pnl <= -self.config.max_daily_loss_usd:
                continue

            price = closes[i]
            curr_high = highs[i]
            curr_low = lows[i]

            # Check exits on open positions
            positions_to_close = []
            for pos in open_positions:
                exit_reason = None
                exit_price = price

                if pos.side == "buy":
                    if curr_low <= pos.stop_loss:
                        exit_reason = "stop_loss"
                        exit_price = pos.stop_loss
                    elif curr_high >= pos.take_profit:
                        exit_reason = "take_profit"
                        exit_price = pos.take_profit
                else:
                    if curr_high >= pos.stop_loss:
                        exit_reason = "stop_loss"
                        exit_price = pos.stop_loss
                    elif curr_low <= pos.take_profit:
                        exit_reason = "take_profit"
                        exit_price = pos.take_profit

                if exit_reason:
                    if pos.side == "buy":
                        pnl = (exit_price - pos.entry_price) * pos.amount
                    else:
                        pnl = (pos.entry_price - exit_price) * pos.amount

                    pnl_pct = pnl / pos.size_usd * 100
                    capital += pos.size_usd + pnl
                    daily_pnl += pnl

                    entry_ts = timestamps[pos.entry_idx]
                    if isinstance(entry_ts, (int, float)):
                        entry_time = datetime.utcfromtimestamp(entry_ts / 1000).isoformat()
                    else:
                        entry_time = str(entry_ts)

                    exit_ts = timestamps[i]
                    if isinstance(exit_ts, (int, float)):
                        exit_time = datetime.utcfromtimestamp(exit_ts / 1000).isoformat()
                    else:
                        exit_time = str(exit_ts)

                    closed_trades.append(BacktestTrade(
                        entry_time=entry_time,
                        exit_time=exit_time,
                        symbol=symbol,
                        side=pos.side,
                        strategy=pos.strategy,
                        entry_price=pos.entry_price,
                        exit_price=exit_price,
                        size_usd=pos.size_usd,
                        pnl=pnl,
                        pnl_pct=pnl_pct,
                        exit_reason=exit_reason,
                        confidence=pos.confidence,
                        reason=pos.reason,
                        bars_held=i - pos.entry_idx,
                    ))
                    positions_to_close.append(pos)

            for pos in positions_to_close:
                open_positions.remove(pos)

            # Build snapshot from data seen so far
            snapshot = self._build_snapshot(
                symbol, opens[:i+1], highs[:i+1], lows[:i+1],
                closes[:i+1], volumes[:i+1],
            )
            if snapshot is None:
                equity_curve.append(capital)
                continue

            # Generate signals
            signals = self.strategy_mgr.generate_signals(snapshot)

            # Execute new trades (respect position limits)
            for signal in signals:
                if daily_trades >= self.config.max_daily_trades:
                    break
                if len(open_positions) >= self.config.max_open_positions:
                    break
                if signal.size_usd > capital * 0.95:
                    continue

                # Check for duplicate positions
                has_position = any(
                    p.symbol == signal.symbol and p.side == signal.side.value
                    for p in open_positions
                )
                if has_position:
                    continue

                # Minimum 2:1 R:R check
                if signal.stop_loss and signal.take_profit:
                    risk = abs(signal.price - signal.stop_loss)
                    reward = abs(signal.take_profit - signal.price)
                    if risk <= 0 or reward / risk < self.config.min_reward_risk_ratio:
                        continue

                # Max risk per trade check
                if signal.stop_loss:
                    risk_pct = abs(signal.price - signal.stop_loss) / signal.price * 100
                    if risk_pct > self.config.max_risk_per_trade_pct:
                        continue

                # Execute
                amount = signal.size_usd / signal.price
                capital -= signal.size_usd
                daily_trades += 1

                open_positions.append(BacktestPosition(
                    entry_idx=i,
                    entry_price=signal.price,
                    side=signal.side.value,
                    size_usd=signal.size_usd,
                    amount=amount,
                    stop_loss=signal.stop_loss,
                    take_profit=signal.take_profit,
                    strategy=signal.strategy_name,
                    confidence=signal.confidence,
                    reason=signal.reason,
                    symbol=signal.symbol,
                ))

            # Track equity (capital + unrealized PnL)
            unrealized = 0.0
            for pos in open_positions:
                if pos.side == "buy":
                    unrealized += (price - pos.entry_price) * pos.amount
                else:
                    unrealized += (pos.entry_price - price) * pos.amount
            equity_curve.append(capital + unrealized + sum(
                p.size_usd for p in open_positions
            ))

        # Close remaining positions at last price
        last_price = closes[-1]
        for pos in open_positions:
            if pos.side == "buy":
                pnl = (last_price - pos.entry_price) * pos.amount
            else:
                pnl = (pos.entry_price - last_price) * pos.amount

            capital += pos.size_usd + pnl

            entry_ts = timestamps[pos.entry_idx]
            if isinstance(entry_ts, (int, float)):
                entry_time = datetime.utcfromtimestamp(entry_ts / 1000).isoformat()
            else:
                entry_time = str(entry_ts)

            closed_trades.append(BacktestTrade(
                entry_time=entry_time,
                exit_time=datetime.utcfromtimestamp(timestamps[-1] / 1000).isoformat()
                    if isinstance(timestamps[-1], (int, float)) else str(timestamps[-1]),
                symbol=symbol,
                side=pos.side,
                strategy=pos.strategy,
                entry_price=pos.entry_price,
                exit_price=last_price,
                size_usd=pos.size_usd,
                pnl=pnl,
                pnl_pct=pnl / pos.size_usd * 100 if pos.size_usd else 0,
                exit_reason="end_of_data",
                confidence=pos.confidence,
                reason=pos.reason,
                bars_held=len(candles) - 1 - pos.entry_idx,
            ))

        # Compute metrics
        return self._compute_results(
            symbol=symbol,
            candles=candles,
            closed_trades=closed_trades,
            equity_curve=equity_curve,
            ending_capital=capital,
        )

    def _build_snapshot(
        self,
        symbol: str,
        opens: List[float],
        highs: List[float],
        lows: List[float],
        closes: List[float],
        volumes: List[float],
    ) -> Optional[MarketSnapshot]:
        """Build a MarketSnapshot from historical data (no exchange call)."""
        price = closes[-1]

        # Compute all indicators
        sma_fast = ind.sma(closes, self.config.sma_fast_period)
        sma_slow = ind.sma(closes, self.config.sma_slow_period)
        sma_long_fast = ind.sma(closes, self.config.sma_long_fast_period)
        sma_long_slow = ind.sma(closes, self.config.sma_long_slow_period)
        ema_12 = ind.ema(closes, 12)
        ema_26 = ind.ema(closes, 26)
        rsi_values = ind.rsi(closes, self.config.rsi_period)
        macd_l, macd_s, macd_h = ind.macd(
            closes, self.config.macd_fast, self.config.macd_slow,
            self.config.macd_signal,
        )
        bb_upper, bb_middle, bb_lower = ind.bollinger_bands(closes)
        avg_vol = ind.average_volume(volumes, 20)
        atr_values = ind.atr(highs, lows, closes)

        stoch_fk, stoch_fd = ind.stochastic(
            highs, lows, closes,
            self.config.stoch_fast_k, self.config.stoch_fast_d,
            self.config.stoch_fast_smooth,
        )
        stoch_sk, stoch_sd = ind.stochastic(
            highs, lows, closes,
            self.config.stoch_slow_k, self.config.stoch_slow_d,
            self.config.stoch_slow_smooth,
        )
        cci_values = ind.cci(highs, lows, closes, self.config.cci_period)
        adx_values = ind.adx(highs, lows, closes)
        chart_patterns = ind.detect_chart_patterns(highs, lows, closes)

        is_doji = ind.detect_doji(
            opens[-1], highs[-1], lows[-1], price,
            atr_values[-1] if atr_values[-1] else None,
        )
        is_pipe = ind.detect_pipe_bottom(
            opens, highs, lows, closes,
            atr_values[-1] if atr_values[-1] else None,
        )

        vol_ratio = None
        if avg_vol[-1] and avg_vol[-1] > 0:
            vol_ratio = volumes[-1] / avg_vol[-1]

        # Trend
        trend = "neutral"
        if sma_fast[-1] and sma_slow[-1]:
            if sma_fast[-1] > sma_slow[-1]:
                trend = "bullish"
            elif sma_fast[-1] < sma_slow[-1]:
                trend = "bearish"

        # Market behaviour
        market_behaviour = "unknown"
        if len(closes) >= 20:
            recent = closes[-20:]
            changes = sum(
                1 for j in range(1, len(recent) - 1)
                if (recent[j] - recent[j-1]) * (recent[j+1] - recent[j]) < 0
            )
            market_behaviour = "trending" if changes < 8 else "oscillating"

        fib = ind.calculate_fibonacci(highs, lows, closes)
        support, resistance = ind.find_support_resistance(highs, lows, closes)
        pivots = None
        if len(highs) >= 2:
            pivots = ind.pivot_points(highs[-2], lows[-2], closes[-2])

        pin_bar = ind.detect_pin_bar(
            opens[-1], highs[-1], lows[-1], price,
            atr_values[-1] if atr_values[-1] else None,
        )

        return MarketSnapshot(
            symbol=symbol,
            timeframe=self.config.timeframe,
            current_price=price,
            open_price=opens[-1],
            high=highs[-1],
            low=lows[-1],
            volume=volumes[-1],
            sma_fast=sma_fast[-1],
            sma_slow=sma_slow[-1],
            ema_12=ema_12[-1],
            ema_26=ema_26[-1],
            rsi=rsi_values[-1],
            macd_line=macd_l[-1],
            macd_signal=macd_s[-1],
            macd_histogram=macd_h[-1],
            bb_upper=bb_upper[-1],
            bb_middle=bb_middle[-1],
            bb_lower=bb_lower[-1],
            avg_volume=avg_vol[-1],
            volume_ratio=vol_ratio,
            atr=atr_values[-1],
            fibonacci=fib,
            support_levels=support,
            resistance_levels=resistance,
            pivots=pivots,
            raw_highs=highs,
            raw_lows=lows,
            raw_closes=closes,
            raw_opens=opens,
            pin_bar=pin_bar,
            stoch_fast_k=stoch_fk[-1],
            stoch_fast_d=stoch_fd[-1],
            stoch_slow_k=stoch_sk[-1],
            stoch_slow_d=stoch_sd[-1],
            cci_value=cci_values[-1],
            sma_long_fast=sma_long_fast[-1],
            sma_long_slow=sma_long_slow[-1],
            chart_patterns=chart_patterns,
            is_doji=is_doji,
            is_pipe_bottom=is_pipe,
            market_behaviour=market_behaviour,
            adx=adx_values[-1],
            trend=trend,
        )

    def _compute_results(
        self,
        symbol: str,
        candles: List[List],
        closed_trades: List[BacktestTrade],
        equity_curve: List[float],
        ending_capital: float,
    ) -> BacktestResult:
        """Compute performance metrics from completed trades."""
        start_ts = candles[0][0]
        end_ts = candles[-1][0]

        if isinstance(start_ts, (int, float)):
            start_date = datetime.utcfromtimestamp(start_ts / 1000).strftime("%Y-%m-%d")
            end_date = datetime.utcfromtimestamp(end_ts / 1000).strftime("%Y-%m-%d")
        else:
            start_date = str(start_ts)
            end_date = str(end_ts)

        total_trades = len(closed_trades)
        wins = [t for t in closed_trades if t.pnl > 0]
        losses = [t for t in closed_trades if t.pnl <= 0]

        win_rate = len(wins) / total_trades * 100 if total_trades else 0
        avg_win = sum(t.pnl for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.pnl for t in losses) / len(losses) if losses else 0
        largest_win = max((t.pnl for t in closed_trades), default=0)
        largest_loss = min((t.pnl for t in closed_trades), default=0)

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

        # Max drawdown
        peak = equity_curve[0]
        max_dd_usd = 0
        max_dd_pct = 0
        for eq in equity_curve:
            if eq > peak:
                peak = eq
            dd = peak - eq
            dd_pct = dd / peak * 100 if peak > 0 else 0
            if dd > max_dd_usd:
                max_dd_usd = dd
                max_dd_pct = dd_pct

        # Sharpe ratio (annualized, assuming 252 trading days)
        if len(equity_curve) > 1:
            returns = [
                (equity_curve[j] - equity_curve[j-1]) / equity_curve[j-1]
                for j in range(1, len(equity_curve))
                if equity_curve[j-1] > 0
            ]
            if returns:
                avg_ret = sum(returns) / len(returns)
                std_ret = (sum((r - avg_ret)**2 for r in returns) / len(returns)) ** 0.5
                sharpe = (avg_ret / std_ret * math.sqrt(252)) if std_ret > 0 else 0
            else:
                sharpe = 0.0
        else:
            sharpe = 0.0

        avg_bars = (
            sum(t.bars_held for t in closed_trades) / total_trades
            if total_trades else 0
        )

        # Strategy breakdown
        strat_stats: Dict[str, Dict] = {}
        for trade in closed_trades:
            s = trade.strategy
            if s not in strat_stats:
                strat_stats[s] = {"trades": 0, "wins": 0, "total_pnl": 0.0}
            strat_stats[s]["trades"] += 1
            if trade.pnl > 0:
                strat_stats[s]["wins"] += 1
            strat_stats[s]["total_pnl"] += trade.pnl

        for s in strat_stats:
            t = strat_stats[s]["trades"]
            strat_stats[s]["win_rate"] = strat_stats[s]["wins"] / t * 100 if t else 0

        total_return = (ending_capital - self.config.starting_capital) / self.config.starting_capital * 100

        return BacktestResult(
            symbol=symbol,
            timeframe=self.config.timeframe,
            start_date=start_date,
            end_date=end_date,
            total_bars=len(candles),
            starting_capital=self.config.starting_capital,
            ending_capital=ending_capital,
            total_return_pct=total_return,
            total_trades=total_trades,
            winning_trades=len(wins),
            losing_trades=len(losses),
            win_rate=win_rate,
            avg_win=avg_win,
            avg_loss=avg_loss,
            largest_win=largest_win,
            largest_loss=largest_loss,
            profit_factor=profit_factor,
            max_drawdown_pct=max_dd_pct,
            max_drawdown_usd=max_dd_usd,
            sharpe_ratio=sharpe,
            avg_bars_held=avg_bars,
            trades_by_strategy=strat_stats,
            trades=closed_trades,
            equity_curve=equity_curve,
        )


# ── Data fetching ──────────────────────────────────────────────────────


def fetch_historical_data(
    exchange_id: str,
    symbol: str,
    timeframe: str,
    days: int,
    config: Optional[Config] = None,
) -> List[List]:
    """Fetch historical candle data from exchange via ccxt."""
    import ccxt

    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        raise ValueError(f"Exchange '{exchange_id}' not supported")

    exchange = exchange_class({"enableRateLimit": True})

    since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    limit = 1000

    print(f"Fetching {symbol} {timeframe} data from {exchange_id} ({days} days)...")

    while True:
        candles = exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, since=since, limit=limit
        )
        if not candles:
            break

        all_candles.extend(candles)
        since = candles[-1][0] + 1  # Next batch after last candle

        if len(candles) < limit:
            break  # No more data

        print(f"  ...fetched {len(all_candles)} candles so far")

    print(f"  Total: {len(all_candles)} candles")
    return all_candles


# ── CLI entry point ────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Backtest trading strategies")
    parser.add_argument("--pair", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--tf", default="1h", help="Timeframe (1m, 5m, 15m, 1h, 4h, 1d)")
    parser.add_argument("--days", type=int, default=180, help="Days of history")
    parser.add_argument("--exchange", default="bybit", help="Exchange for data")
    parser.add_argument("--capital", type=float, default=500, help="Starting capital")
    parser.add_argument("--data-file", help="Load candles from JSON file instead")
    parser.add_argument("--save-trades", help="Save trade log to JSON file")
    parser.add_argument("--save-equity", help="Save equity curve to JSON file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)

    config = Config(
        exchange=args.exchange,
        trading_pair=args.pair,
        timeframe=args.tf,
        starting_capital=args.capital,
        paper_trading=True,
    )

    # Load data
    if args.data_file:
        with open(args.data_file) as f:
            candles = json.load(f)
        print(f"Loaded {len(candles)} candles from {args.data_file}")
    else:
        candles = fetch_historical_data(
            config.exchange_id, args.pair, args.tf, args.days, config
        )

    if not candles:
        print("No data available. Check pair, exchange, and timeframe.")
        sys.exit(1)

    # Run backtest
    bt = Backtester(config)
    result = bt.run(candles, symbol=args.pair)
    print(result.summary())

    # Save outputs
    if args.save_trades:
        with open(args.save_trades, "w") as f:
            json.dump([asdict(t) for t in result.trades], f, indent=2)
        print(f"\nTrade log saved to {args.save_trades}")

    if args.save_equity:
        with open(args.save_equity, "w") as f:
            json.dump(result.equity_curve, f)
        print(f"Equity curve saved to {args.save_equity}")


if __name__ == "__main__":
    main()
