import streamlit as st
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from tiingo import TiingoClient
import alpaca_trade_api as tradeapi

# --- Backtesting function (no changes needed here) ---
def run_backtest_for_dashboard(symbol, start_date, end_date, config, regime_window=200):
    # ... (This function is the same as before) ...
    try:
        client = TiingoClient(config)
        data = client.get_dataframe(symbol, frequency='daily', startDate=start_date, endDate=end_date)
        data.rename(columns={'adjClose': 'Adj Close'}, inplace=True)
        if data.empty:
            return None, None
    except Exception as e:
        st.error(f"An error occurred during download: {e}")
        return None, None

    data['regime_ma'] = data['Adj Close'].rolling(window=regime_window).mean()
    buffer = 0.02
    data['upper_band'] = data['regime_ma'] * (1 + buffer)
    data['lower_band'] = data['regime_ma'] * (1 - buffer)
    data['signal'] = 0
    for i in range(regime_window, len(data)):
        if data['Adj Close'][i] > data['upper_band'][i]:
            data['signal'][i] = 1
        elif data['Adj Close'][i] < data['lower_band'][i]:
            data['signal'][i] = 0
        else:
            data['signal'][i] = data['signal'][i-1]
    data['signal'] = data['signal'].shift(1)

    data['daily_return'] = data['Adj Close'].pct_change()
    data['strategy_return'] = data['daily_return'] * data['signal']
    data['buy_hold_cumulative'] = (1 + data['daily_return']).cumprod()
    data['strategy_cumulative'] = (1 + data['strategy_return']).cumprod()
    data.dropna(inplace=True)

    buy_hold_return = (data['buy_hold_cumulative'].iloc[-1] - 1) * 100
    strategy_return = (data['strategy_cumulative'].iloc[-1] - 1) * 100
    results = {"buy_and_hold": f"{buy_hold_return:.2f}%", "strategy": f"{strategy_return:.2f}%"}

    fig, ax1 = plt.subplots(figsize=(14, 7))
    ax1.plot(data.index, data['buy_hold_cumulative'], label='Buy & Hold', color='black', linestyle='--')
    ax1.plot(data.index, data['strategy_cumulative'], label='Adaptive Strategy', color='blue', linewidth=2)
    ax1.set_title(f'Adaptive Momentum Strategy vs. Buy & Hold for {symbol}')
    ax1.set_ylabel('Cumulative Return')
    ax1.legend()
    ax1.grid(True)
    
    return results, fig

# --- STREAMLIT WEB APPLICATION ---
st.set_page_config(layout="wide")
st.title("Quantitative Trading Dashboard")

# --- Load API Keys ---
try:
    tiingo_key = st.secrets["tiingo"]["api_key"]
    alpaca_key_id = st.secrets["alpaca"]["api_key_id"]
    alpaca_secret_key = st.secrets["alpaca"]["secret_key"]
except:
    config = configparser.ConfigParser()
    config.read('config.ini')
    tiingo_key = config['tiingo']['api_key']
    alpaca_key_id = config['alpaca']['api_key_id']
    alpaca_secret_key = config['alpaca']['secret_key']

tiingo_config = {'api_key': tiingo_key, 'session': True}


# --- NEW SECTION: ALPACA ACCOUNT STATUS ---
st.header("Live Alpaca Account Status")

try:
    # Connect to Alpaca API
    base_url = 'https://paper-api.alpaca.markets' # Paper trading URL
    api = tradeapi.REST(alpaca_key_id, alpaca_secret_key, base_url, api_version='v2')
    
    account = api.get_account()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Portfolio Value", f"${float(account.portfolio_value):,}")
    col2.metric("Buying Power", f"${float(account.buying_power):,}")
    col3.metric("Account Status", account.status)

    # Get and display positions
    positions = api.list_positions()
    if positions:
        st.subheader("Current Positions")
        pos_data = [{'Symbol': p.symbol, 'Qty': float(p.qty), 'Market Value': f"${float(p.market_value):,}", 'Current Price': f"${float(p.current_price):,}", 'Unrealized P/L': f"${float(p.unrealized_pl):,}"} for p in positions]
        positions_df = pd.DataFrame(pos_data)
        st.dataframe(positions_df, use_container_width=True)
    else:
        st.info("You have no open positions.")

    # Get and display recent trades
    trades = api.get_activities(activity_types='fill', direction='desc', limit=20)
    if trades:
        st.subheader("Recent Trades")
        trade_data = [{'Time': t.transaction_time.strftime('%Y-%m-%d %H:%M'), 'Symbol': t.symbol, 'Side': t.side, 'Qty': float(t.qty), 'Price': f"${float(t.price):,}"} for t in trades]
        trades_df = pd.DataFrame(trade_data)
        st.dataframe(trades_df, use_container_width=True)
    else:
        st.info("No recent trades found.")

except Exception as e:
    st.error(f"Could not connect to Alpaca or fetch account data. Error: {e}")


# --- EXISTING SECTION: BACKTESTING DASHBOARD ---
st.header("Strategy Backtester")

# NEW: Collapsible expander for the strategy description
with st.expander("About the Adaptive Momentum Strategy"):
    st.markdown("""
    This strategy is a **trend-following system** designed to adapt to different market regimes. It uses a long-term moving average to identify the overall trend.

    **Core Logic:**
    - **Regime Filter:** A 200-day simple moving average (SMA) determines the market "regime."
    - **Buffer Zone:** A 2% buffer is applied above and below the 200-day SMA to create a neutral zone. This helps prevent "whipsaws" (bad trades) during choppy, non-trending periods.
    
    **Trading Rules:**
    1.  **Buy Signal (Risk-On):** A position is entered only if the price moves **more than 2% above** the 200-day SMA.
    2.  **Sell Signal (Risk-Off):** The position is sold only if the price drops **more than 2% below** the 200-day SMA.
    3.  **Hold:** If the price is within the +/- 2% buffer zone, the strategy holds its current position (either in the stock or in cash) and does nothing.
    """)

st.write("Enter a stock ticker to backtest the strategy.")

symbol = st.text_input("Stock Ticker (e.g., AAPL, MSFT, SPY)", "NVDA").upper()

if st.button("Run Backtest"):
    if symbol:
        with st.spinner(f"Running backtest for {symbol}..."):
            results, chart_figure = run_backtest_for_dashboard(
                symbol=symbol, start_date='2015-01-01', end_date='2025-06-09', config=tiingo_config
            )
        if results:
            st.success(f"Backtest for {symbol} complete!")
            col1, col2 = st.columns(2)
            col1.metric("Buy & Hold Return", results["buy_and_hold"])
            col2.metric("Strategy Return", results["strategy"])
            st.pyplot(chart_figure)
        else:
            st.error(f"Could not retrieve data or run backtest for {symbol}.")
    else:
        st.warning("Please enter a stock ticker.")