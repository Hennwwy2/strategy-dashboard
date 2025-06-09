import streamlit as st
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from tiingo import TiingoClient

# This is our main backtesting function from before
def run_backtest_for_dashboard(symbol, start_date, end_date, config, regime_window=200):
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
st.title("Quantitative Strategy Backtesting Dashboard")

# This code correctly reads secrets for deployment and falls back to config.ini for local use
try:
    tiingo_key = st.secrets["tiingo"]["api_key"]
except:
    config = configparser.ConfigParser()
    config.read('config.ini')
    tiingo_key = config['tiingo']['api_key']

tiingo_config = {'api_key': tiingo_key, 'session': True}

st.write("Enter a stock ticker to backtest our Adaptive Momentum strategy with a 2% buffer zone.")
symbol = st.text_input("Stock Ticker (e.g., AAPL, MSFT, SPY)", "NVDA").upper()

if st.button("Run Backtest"):
    if symbol:
        with st.spinner(f"Running backtest for {symbol}..."):
            results, chart_figure = run_backtest_for_dashboard(
                symbol=symbol, start_date='2015-01-01', end_date='2025-06-08', config=tiingo_config
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