from website import app

from flask import render_template, request, jsonify

import sys
import os

# Add project root so we can import the bot package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from bot.backtest import Backtester, fetch_yahoo_data
from bot.config import Config


@app.route('/')
@app.route('/home')
def home():
    return render_template('pages/home.html')


@app.route('/about')
def about():
    return render_template('pages/about.html')


@app.route('/backtest')
def backtest():
    return render_template('pages/backtest.html')


@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run a backtest and return results as JSON."""
    data = request.get_json() or {}

    pair = data.get('pair', 'BTC/USDT')
    timeframe = data.get('timeframe', '1h')
    days = int(data.get('days', 90))
    capital = float(data.get('capital', 500))

    # Clamp values
    days = max(7, min(days, 730))
    capital = max(100, min(capital, 100000))

    try:
        candles = fetch_yahoo_data(pair, timeframe, days)
        if not candles:
            return jsonify({'error': f'No data returned for {pair}. Check the pair name.'}), 400

        config = Config(
            trading_pair=pair,
            timeframe=timeframe,
            starting_capital=capital,
            paper_trading=True,
        )
        bt = Backtester(config)
        result = bt.run(candles, symbol=pair)

        return jsonify({
            'symbol': result.symbol,
            'timeframe': result.timeframe,
            'start_date': result.start_date,
            'end_date': result.end_date,
            'total_bars': result.total_bars,
            'starting_capital': result.starting_capital,
            'ending_capital': round(result.ending_capital, 2),
            'total_return_pct': round(result.total_return_pct, 2),
            'total_trades': result.total_trades,
            'winning_trades': result.winning_trades,
            'losing_trades': result.losing_trades,
            'win_rate': round(result.win_rate, 1),
            'avg_win': round(result.avg_win, 2),
            'avg_loss': round(result.avg_loss, 2),
            'largest_win': round(result.largest_win, 2),
            'largest_loss': round(result.largest_loss, 2),
            'profit_factor': round(result.profit_factor, 2),
            'max_drawdown_pct': round(result.max_drawdown_pct, 2),
            'max_drawdown_usd': round(result.max_drawdown_usd, 2),
            'sharpe_ratio': round(result.sharpe_ratio, 2),
            'avg_bars_held': round(result.avg_bars_held, 1),
            'trades_by_strategy': result.trades_by_strategy,
            'equity_curve': result.equity_curve,
            'trades': [
                {
                    'entry_time': t.entry_time,
                    'exit_time': t.exit_time,
                    'side': t.side,
                    'strategy': t.strategy,
                    'entry_price': round(t.entry_price, 2),
                    'exit_price': round(t.exit_price, 2),
                    'pnl': round(t.pnl, 2),
                    'pnl_pct': round(t.pnl_pct, 2),
                    'exit_reason': t.exit_reason,
                }
                for t in result.trades
            ],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
