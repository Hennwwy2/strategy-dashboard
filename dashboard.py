import streamlit as st
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from tiingo import TiingoClient
import alpaca_trade_api as tradeapi
import plotly.graph_objects as go
from datetime import datetime, timedelta

def run_backtest_for_dashboard(symbol, start_date, end_date, config, regime_window=200):
    try:
        client = TiingoClient(config)
        data = client.get_dataframe(symbol, frequency='daily', startDate=start_date, endDate=end_date)
        data.rename(columns={'adjClose': 'Adj Close'}, inplace=True)
        if data.empty:
            return None, "No data returned from Tiingo."
    except Exception as e:
        return None, str(e)

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
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Quantitative Trading Dashboard")

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
tiingo_client = TiingoClient(tiingo_config)

tab_live, tab_backtest = st.tabs(["Live Account Dashboard", "Strategy Backtester"])

with tab_live:
    st.header("Live Alpaca Account Status")
    try:
        base_url = 'https://paper-api.alpaca.markets'
        api = tradeapi.REST(alpaca_key_id, alpaca_secret_key, base_url, api_version='v2')
        account = api.get_account()
        col1, col2, col3 = st.columns(3)
        col1.metric("Portfolio Value", f"${float(account.portfolio_value):,}")
        col2.metric("Buying Power", f"${float(account.buying_power):,}")
        col3.metric("Account Status", account.status)
        st.divider()
        st.subheader("Live Quote by Ticker")
        col1_quote, col2_quote = st.columns([1, 3])
        with col1_quote:
            quote_symbol = st.text_input("Enter ticker:", "NVDA", key="live_quote_symbol").upper()
            if st.button("Get Live Quote"):
                with st.spinner(f"Getting quote for {quote_symbol}..."):
                    quote_data = tiingo_client.get_ticker_price(quote_symbol)
                    if quote_data:
                        latest = quote_data[0]
                        price = latest['adjClose']
                        open_price = latest['adjOpen']
                        change = price - open_price
                        change_percent = (change / open_price) * 100
                        st.session_state.quote_result = {"symbol": quote_symbol, "price": f"${price:,.2f}", "delta": f"${change:,.2f} ({change_percent:.2f}%)", "timestamp": pd.to_datetime(latest['date']).strftime('%Y-%m-%d')}
                    else:
                        st.session_state.quote_result = None
                        st.error("Could not retrieve quote.")
        with col2_quote:
            if "quote_result" in st.session_state and st.session_state.quote_result:
                res = st.session_state.quote_result
                st.metric(label=f"Last Price for {res['symbol']}", value=res["price"], delta=res["delta"])
                st.caption(f"Based on intraday change from open. Date: {res['timestamp']}")
        st.divider()
        positions = api.list_positions()
        if positions:
            st.subheader("Current Positions")
            pos_data = [{'Symbol': p.symbol, 'Qty': float(p.qty), 'Market Value': f"${float(p.market_value):,}", 'Current Price': f"${float(p.current_price):,}", 'Unrealized P/L': f"${float(p.unrealized_pl):,}"} for p in positions]
            positions_df = pd.DataFrame(pos_data)
            st.dataframe(positions_df, use_container_width=True)
            st.subheader("Position Chart")
            position_symbols = [p.symbol for p in positions]
            selected_symbol = st.selectbox("Choose a stock to chart:", position_symbols)
            if selected_symbol:
                chart_start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
                chart_end_date = datetime.now().strftime('%Y-%m-%d')
                chart_df = tiingo_client.get_dataframe(selected_symbol, frequency='daily', startDate=chart_start_date, endDate=chart_end_date)
                fig = go.Figure(data=[go.Candlestick(x=chart_df.index, open=chart_df['open'], high=chart_df['high'], low=chart_df['low'], close=chart_df['close'])])
                fig.update_layout(title=f'{selected_symbol} - 1 Year Price Chart', yaxis_title='Price (USD)', xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("You have no open positions.")
        trades = api.get_activities(activity_types='FILL', direction='desc')[:20]
        if trades:
            st.subheader("Recent Trades")
            trade_data = [{'Time': t.transaction_time.strftime('%Y-%m-%d %H:%M'), 'Symbol': t.symbol, 'Side': t.side, 'Qty': float(t.qty), 'Price': f"${float(t.price):,}"} for t in trades]
            trades_df = pd.DataFrame(trade_data)
            st.dataframe(trades_df, use_container_width=True)
        else:
            st.info("No recent trades found.")
    except Exception as e:
        st.error(f"Could not connect to Alpaca or fetch account data. Error: {e}")

with tab_backtest:
    st.header("Individual Strategy Backtester")
    with st.expander("About the Adaptive Momentum Strategy"):
        st.markdown("""...""")
    st.write("Enter a stock ticker to backtest the strategy.")
    symbol = st.text_input("Stock Ticker", "NVDA", key="backtest_symbol").upper()
    if st.button("Run Single Backtest"):
        if symbol:
            with st.spinner(f"Running backtest for {symbol}..."):
                results, fig_or_error = run_backtest_for_dashboard(symbol=symbol, start_date='2015-01-01', end_date='2025-06-09', config=tiingo_config)
            if results:
                st.success(f"Backtest for {symbol} complete!")
                col1, col2 = st.columns(2)
                col1.metric("Buy & Hold Return", results["buy_and_hold"])
                col2.metric("Strategy Return", results["strategy"])
                st.pyplot(fig_or_error)
            else:
                st.error(f"Could not retrieve data or run backtest. Error: {fig_or_error}")
        else:
            st.warning("Please enter a stock ticker.")
    st.divider()
    st.header("Batch Test on Recommended Stocks")
    st.write("Click the button below to run the backtest on a curated list of historically trending stocks.")
    recommended_tickers = ["AAPL", "MSFT", "AMZN", "META", "TSLA"]
    st.write("Recommended Tickers:", ", ".join(recommended_tickers))
    if st.button("Run Batch Backtest"):
        batch_results = []
        results_placeholder = st.empty()
        for ticker in recommended_tickers:
            with st.spinner(f"Testing {ticker}..."):
                results, fig = run_backtest_for_dashboard(symbol=ticker, start_date='2015-01-01', end_date='2025-06-09', config=tiingo_config)
                if results:
                    batch_results.append({'Symbol': ticker, 'Buy & Hold Return': results['buy_and_hold'], 'Strategy Return': results['strategy']})
                else:
                    batch_results.append({'Symbol': ticker, 'Buy & Hold Return': 'Error', 'Strategy Return': 'Error'})
                results_df = pd.DataFrame(batch_results)
                results_placeholder.dataframe(results_df, use_container_width=True)
        st.success("Batch backtest complete!")