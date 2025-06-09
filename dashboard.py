# --- ENHANCED DASHBOARD WITH OPTIONS SUPPORT ---
import streamlit as st
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from tiingo import TiingoClient
import alpaca_trade_api as tradeapi
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math

def run_backtest_for_dashboard(symbol, start_date, end_date, config, regime_window=200):
    """Original backtest function - unchanged"""
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

def get_options_chain(api, symbol):
    """Get options chain for a symbol"""
    try:
        # Get next 4 expiration dates
        expirations = []
        for i in range(0, 120, 30):  # Check next 4 months
            exp_date = (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d')
            expirations.append(exp_date)
        
        all_contracts = []
        for exp in expirations:
            contracts = api.list_options_contracts(
                underlying_symbols=symbol,
                expiration_date=exp,
                status='active'
            )
            all_contracts.extend(contracts)
        
        return all_contracts
    except Exception as e:
        st.error(f"Error fetching options chain: {e}")
        return []

def display_options_chain(api, symbol):
    """Display options chain in a formatted table"""
    st.subheader(f"Options Chain for {symbol}")
    
    # Get current stock price
    try:
        quote = api.get_latest_trade(symbol)
        current_price = quote.price
        st.metric("Current Stock Price", f"${current_price:.2f}")
    except:
        current_price = None
        st.warning("Could not fetch current price")
    
    contracts = get_options_chain(api, symbol)
    
    if not contracts:
        st.info("No options contracts available")
        return
    
    # Separate calls and puts
    calls = [c for c in contracts if c.type == 'call']
    puts = [c for c in contracts if c.type == 'put']
    
    # Create tabs for calls and puts
    call_tab, put_tab = st.tabs(["Calls", "Puts"])
    
    with call_tab:
        if calls:
            call_data = []
            for contract in calls[:20]:  # Limit to 20 for performance
                try:
                    quote = api.get_latest_quote(contract.symbol)
                    call_data.append({
                        'Symbol': contract.symbol,
                        'Strike': f"${contract.strike_price}",
                        'Expiration': contract.expiration_date,
                        'Bid': f"${quote.bid_price:.2f}" if quote.bid_price else "N/A",
                        'Ask': f"${quote.ask_price:.2f}" if quote.ask_price else "N/A",
                        'Volume': contract.day_volume if hasattr(contract, 'day_volume') else "N/A"
                    })
                except:
                    continue
            
            if call_data:
                df = pd.DataFrame(call_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No call options available")
    
    with put_tab:
        if puts:
            put_data = []
            for contract in puts[:20]:  # Limit to 20 for performance
                try:
                    quote = api.get_latest_quote(contract.symbol)
                    put_data.append({
                        'Symbol': contract.symbol,
                        'Strike': f"${contract.strike_price}",
                        'Expiration': contract.expiration_date,
                        'Bid': f"${quote.bid_price:.2f}" if quote.bid_price else "N/A",
                        'Ask': f"${quote.ask_price:.2f}" if quote.ask_price else "N/A",
                        'Volume': contract.day_volume if hasattr(contract, 'day_volume') else "N/A"
                    })
                except:
                    continue
            
            if put_data:
                df = pd.DataFrame(put_data)
                st.dataframe(df, use_container_width=True)
        else:
            st.info("No put options available")

# --- STREAMLIT WEB APPLICATION ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Quantitative Trading Dashboard - Stocks & Options")

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

# Create tabs for different sections
tab_live, tab_options, tab_backtest = st.tabs(["Live Account", "Options Trading", "Strategy Backtester"])

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
        
        # Enhanced positions display - now includes options
        positions = api.list_positions()
        if positions:
            st.subheader("Current Positions")
            
            stock_positions = []
            options_positions = []
            
            for p in positions:
                try:
                    asset = api.get_asset(p.symbol)
                    # Check if this is an options position
                    # Using getattr to safely access the class attribute
                    asset_class = getattr(asset, 'class', None)
                    if asset_class == 'us_option':
                        # This is an options position
                        quote = api.get_latest_quote(p.symbol)
                        options_positions.append({
                            'Symbol': p.symbol,
                            'Type': 'Call' if 'C' in p.symbol else 'Put',
                            'Qty': float(p.qty),
                            'Avg Cost': f"${float(p.avg_entry_price):.2f}",
                            'Current': f"${float(quote.ask_price):.2f}",
                            'Market Value': f"${float(p.market_value):,}",
                            'P/L': f"${float(p.unrealized_pl):,}"
                        })
                    else:
                        # Regular stock position
                        stock_positions.append({
                            'Symbol': p.symbol,
                            'Qty': float(p.qty),
                            'Market Value': f"${float(p.market_value):,}",
                            'Current Price': f"${float(p.current_price):,}",
                            'Unrealized P/L': f"${float(p.unrealized_pl):,}"
                        })
                except:
                    continue
            
            if stock_positions:
                st.write("**Stock Positions:**")
                st.dataframe(pd.DataFrame(stock_positions), use_container_width=True)
            
            if options_positions:
                st.write("**Options Positions:**")
                st.dataframe(pd.DataFrame(options_positions), use_container_width=True)
        else:
            st.info("You have no open positions.")
        
    except Exception as e:
        st.error(f"Could not connect to Alpaca. Error: {e}")

with tab_options:
    st.header("Options Trading Center")
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader("Options Tools")
        
        # Options calculator
        st.write("**Quick Options Calculator**")
        calc_type = st.selectbox("Calculate:", ["Breakeven", "Max Profit", "Max Loss"])
        
        if calc_type == "Breakeven":
            strike = st.number_input("Strike Price", value=150.0, step=1.0)
            premium = st.number_input("Premium Paid", value=5.0, step=0.1)
            call_or_put = st.radio("Type", ["Call", "Put"])
            
            if call_or_put == "Call":
                breakeven = strike + premium
                st.success(f"Breakeven: ${breakeven:.2f}")
                st.caption("Stock must be above this price at expiration to profit")
            else:
                breakeven = strike - premium
                st.success(f"Breakeven: ${breakeven:.2f}")
                st.caption("Stock must be below this price at expiration to profit")
        
        elif calc_type == "Max Profit":
            strategy = st.selectbox("Strategy", ["Long Call", "Long Put", "Covered Call"])
            if strategy == "Long Call":
                st.info("Max Profit: Unlimited")
                st.caption("Calls have unlimited upside potential")
            elif strategy == "Long Put":
                strike = st.number_input("Strike Price", value=150.0)
                premium = st.number_input("Premium Paid", value=5.0)
                max_profit = strike - premium
                st.success(f"Max Profit: ${max_profit:.2f} per share")
                st.caption("If stock goes to $0")
            else:  # Covered Call
                premium = st.number_input("Premium Collected", value=5.0)
                st.success(f"Max Profit: ${premium:.2f} per share")
                st.caption("If stock stays below strike")
        
    with col2:
        st.subheader("Options Chain Explorer")
        
        # Symbol input for options chain
        options_symbol = st.text_input("Enter symbol for options chain:", "NVDA")
        
        if st.button("Load Options Chain"):
            with st.spinner(f"Loading options for {options_symbol}..."):
                try:
                    display_options_chain(api, options_symbol)
                except Exception as e:
                    st.error(f"Error loading options chain: {e}")
        
        # Educational content
        with st.expander("Options Strategies Guide"):
            st.markdown("""
            ### Common Options Strategies
            
            **1. Long Call (Bullish)**
            - Buy call options when expecting price to rise
            - Limited risk (premium paid), unlimited profit potential
            - Best for: Strong bullish conviction
            
            **2. Long Put (Bearish)**
            - Buy put options when expecting price to fall
            - Limited risk (premium paid), profit if stock falls
            - Best for: Protecting gains or betting on decline
            
            **3. Covered Call (Neutral/Bullish)**
            - Own stock + sell call options
            - Collect premium, but cap upside
            - Best for: Generating income on holdings
            
            **4. Cash-Secured Put (Neutral/Bullish)**
            - Sell puts backed by cash
            - Collect premium, may be assigned stock
            - Best for: Entering positions at lower prices
            
            **5. Vertical Spreads**
            - Buy one option, sell another at different strike
            - Limited risk, limited reward
            - Best for: Defined risk directional trades
            """)

with tab_backtest:
    # Original backtest code remains unchanged
    st.header("Strategy Backtester")
    
    # Strategy selector
    strategy_type = st.selectbox(
        "Select Strategy Type:",
        ["Stock Strategy (Original)", "Options Strategy (Simulated)"]
    )
    
    if strategy_type == "Stock Strategy (Original)":
        # Your original backtesting code here
        symbol = st.text_input("Stock Ticker", "NVDA", key="backtest_symbol").upper()
        if st.button("Run Backtest"):
            if symbol:
                with st.spinner(f"Running backtest for {symbol}..."):
                    results, fig_or_error = run_backtest_for_dashboard(
                        symbol=symbol,
                        start_date='2015-01-01',
                        end_date='2025-06-09',
                        config=tiingo_config
                    )
                if results:
                    st.success(f"Backtest for {symbol} complete!")
                    col1, col2 = st.columns(2)
                    col1.metric("Buy & Hold Return", results["buy_and_hold"])
                    col2.metric("Strategy Return", results["strategy"])
                    st.pyplot(fig_or_error)
    
    else:  # Options Strategy
        st.info("Options backtesting requires historical options data (expensive). Below is a simplified simulation.")
        
        symbol = st.text_input("Stock Ticker", "NVDA", key="options_backtest_symbol").upper()
        option_type = st.selectbox("Option Strategy", ["Long Calls", "Long Puts", "Covered Calls"])
        
        if st.button("Run Options Simulation"):
            st.warning("This is a simplified simulation using Black-Scholes approximations, not real options data.")
            
            # Placeholder for options backtesting
            # In reality, you'd need historical options prices from a provider like CBOE or ORATS
            st.markdown("""
            ### Options Backtesting Considerations:
            - Need historical implied volatility data
            - Must account for bid-ask spreads
            - Consider early assignment risk
            - Factor in theta decay
            - Model dividend impacts
            
            For production options backtesting, consider:
            - CBOE DataShop
            - ORATS
            - OptionMetrics
            """)