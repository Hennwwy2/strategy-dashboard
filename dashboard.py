# --- COMPLETE ENHANCED DASHBOARD WITH POLYGON INTEGRATION ---
import streamlit as st
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from tiingo import TiingoClient
import requests
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import numpy as np
import json
import sqlite3
import os

# Polygon API wrapper class
class PolygonAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    def get_quote(self, symbol):
        """Get real-time quote"""
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        params = {"apikey": self.api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching quote: {response.status_code}")
            return None
    
    def get_options_chain(self, underlying_symbol, expiration_date=None):
        """Get options chain for a symbol"""
        # Get options contracts
        url = f"{self.base_url}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying_symbol,
            "apikey": self.api_key,
            "limit": 1000
        }
        
        if expiration_date:
            params["expiration_date"] = expiration_date
            
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching options: {response.status_code}")
            return None
    
    def get_options_quotes(self, options_ticker):
        """Get options quote"""
        url = f"{self.base_url}/v3/last/trade/options/{options_ticker}"
        params = {"apikey": self.api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return None

# Paper Trading Database Manager
class PaperTradingDB:
    def __init__(self, db_path="paper_trading.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                position_type TEXT NOT NULL, -- 'stock' or 'option'
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                price REAL NOT NULL,
                side TEXT NOT NULL, -- 'buy' or 'sell'
                trade_type TEXT NOT NULL, -- 'stock' or 'option'
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Create account table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS account (
                id INTEGER PRIMARY KEY,
                cash REAL NOT NULL,
                portfolio_value REAL NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Initialize account with $100,000 if not exists
        cursor.execute('SELECT COUNT(*) FROM account')
        if cursor.fetchone()[0] == 0:
            cursor.execute('INSERT INTO account (id, cash, portfolio_value) VALUES (1, 100000, 100000)')
        
        conn.commit()
        conn.close()
    
    def get_positions(self):
        """Get all current positions"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM positions WHERE quantity != 0', conn)
        conn.close()
        return df
    
    def get_account_info(self):
        """Get account information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT cash, portfolio_value FROM account WHERE id = 1')
        result = cursor.fetchone()
        conn.close()
        return {"cash": result[0], "portfolio_value": result[1]} if result else {"cash": 100000, "portfolio_value": 100000}
    
    def add_trade(self, symbol, quantity, price, side, trade_type='stock'):
        """Add a trade to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add trade to trades table
        cursor.execute('''
            INSERT INTO trades (symbol, quantity, price, side, trade_type)
            VALUES (?, ?, ?, ?, ?)
        ''', (symbol, quantity, price, side, trade_type))
        
        # Update positions
        cursor.execute('SELECT quantity, avg_price FROM positions WHERE symbol = ?', (symbol,))
        existing = cursor.fetchone()
        
        if existing:
            existing_qty, existing_avg = existing
            if side == 'buy':
                new_qty = existing_qty + quantity
                new_avg = ((existing_qty * existing_avg) + (quantity * price)) / new_qty
            else:  # sell
                new_qty = existing_qty - quantity
                new_avg = existing_avg  # Keep same average price
            
            cursor.execute('''
                UPDATE positions SET quantity = ?, avg_price = ? WHERE symbol = ?
            ''', (new_qty, new_avg, symbol))
        else:
            if side == 'buy':
                cursor.execute('''
                    INSERT INTO positions (symbol, quantity, avg_price, position_type)
                    VALUES (?, ?, ?, ?)
                ''', (symbol, quantity, price, trade_type))
        
        # Update cash
        cash_change = -quantity * price if side == 'buy' else quantity * price
        cursor.execute('UPDATE account SET cash = cash + ? WHERE id = 1', (cash_change,))
        
        conn.commit()
        conn.close()

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

def display_options_chain(polygon_api, symbol):
    """Display options chain using Polygon data"""
    st.subheader(f"Options Chain for {symbol}")
    
    # Get current stock price
    try:
        quote_data = polygon_api.get_quote(symbol)
        if quote_data and 'results' in quote_data:
            current_price = quote_data['results']['p']
            st.metric("Current Stock Price", f"${current_price:.2f}")
        else:
            current_price = 100.0  # Fallback
            st.warning("Could not fetch current price, using fallback")
    except:
        current_price = 100.0
        st.warning("Could not fetch current price")
    
    # Get options chain
    with st.spinner("Loading options chain..."):
        options_data = polygon_api.get_options_chain(symbol)
    
    if options_data and 'results' in options_data:
        options = options_data['results']
        
        if not options:
            st.info("No options data available for this symbol")
            return
        
        # Process options data
        calls = []
        puts = []
        
        for option in options[:50]:  # Limit to first 50 for performance
            try:
                contract_type = option.get('contract_type', 'unknown')
                strike = option.get('strike_price', 0)
                expiration = option.get('expiration_date', '')
                
                # Get option quote
                option_ticker = option.get('ticker', '')
                quote = polygon_api.get_options_quotes(option_ticker)
                
                if quote and 'results' in quote:
                    price = quote['results'].get('p', 0)
                    size = quote['results'].get('s', 0)
                else:
                    price = 0
                    size = 0
                
                option_data = {
                    'Strike': f"${strike}",
                    'Expiration': expiration,
                    'Last Price': f"${price:.2f}",
                    'Size': size,
                    'Ticker': option_ticker
                }
                
                if contract_type == 'call':
                    calls.append(option_data)
                elif contract_type == 'put':
                    puts.append(option_data)
                    
            except Exception as e:
                continue
        
        # Display in tabs
        if calls or puts:
            call_tab, put_tab = st.tabs(["📈 Calls", "📉 Puts"])
            
            with call_tab:
                if calls:
                    st.write("**Call Options** (Right to Buy)")
                    df_calls = pd.DataFrame(calls)
                    st.dataframe(df_calls, use_container_width=True)
                else:
                    st.info("No call options data available")
            
            with put_tab:
                if puts:
                    st.write("**Put Options** (Right to Sell)")
                    df_puts = pd.DataFrame(puts)
                    st.dataframe(df_puts, use_container_width=True)
                else:
                    st.info("No put options data available")
        else:
            st.info("No options data could be processed")
    else:
        st.error("Failed to fetch options chain")

def create_payoff_diagram(option_type, strike, premium, current_price, is_buyer=True):
    """Create visual payoff diagram for options"""
    # Price range for x-axis
    price_range = np.linspace(strike * 0.7, strike * 1.3, 100)
    
    if option_type == "Call":
        if is_buyer:
            # Long call payoff
            payoff = np.maximum(price_range - strike, 0) - premium
            title = f"Long Call: Buy right to purchase at ${strike}"
        else:
            # Short call payoff
            payoff = premium - np.maximum(price_range - strike, 0)
            title = f"Short Call: Sell right to purchase at ${strike}"
    else:  # Put
        if is_buyer:
            # Long put payoff
            payoff = np.maximum(strike - price_range, 0) - premium
            title = f"Long Put: Buy right to sell at ${strike}"
        else:
            # Short put payoff
            payoff = premium - np.maximum(strike - price_range, 0)
            title = f"Short Put: Sell right to sell at ${strike}"
    
    # Create the plot
    fig = go.Figure()
    
    # Add payoff line
    fig.add_trace(go.Scatter(
        x=price_range, 
        y=payoff,
        mode='lines',
        name='Profit/Loss',
        line=dict(color='blue', width=3)
    ))
    
    # Add break-even line
    fig.add_hline(y=0, line_dash="dash", line_color="gray", 
                  annotation_text="Break Even")
    
    # Add current price marker
    fig.add_vline(x=current_price, line_dash="dash", line_color="green",
                  annotation_text=f"Current Price: ${current_price:.2f}")
    
    # Add strike price marker
    fig.add_vline(x=strike, line_dash="dash", line_color="red",
                  annotation_text=f"Strike: ${strike}")
    
    # Highlight profit and loss areas
    fig.add_hrect(y0=0, y1=max(payoff), 
                  fillcolor="lightgreen", opacity=0.2,
                  annotation_text="Profit Zone", annotation_position="top right")
    fig.add_hrect(y0=min(payoff), y1=0, 
                  fillcolor="lightcoral", opacity=0.2,
                  annotation_text="Loss Zone", annotation_position="bottom right")
    
    fig.update_layout(
        title=title,
        xaxis_title="Stock Price at Expiration",
        yaxis_title="Profit/Loss per Share",
        hovermode='x unified',
        height=400
    )
    
    return fig

def options_simulator():
    """Interactive options trading simulator"""
    st.subheader("🎮 Options Trading Simulator")
    st.info("Practice options trading with no real money! See how different scenarios affect your profit/loss.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Stock selection
        stock_symbol = st.selectbox(
            "1️⃣ Choose a stock:",
            ["AAPL", "NVDA", "TSLA", "SPY"],
            help="Pick a stock you're familiar with"
        )
        
        # Try to get real price from Polygon
        try:
            quote_data = polygon_api.get_quote(stock_symbol)
            if quote_data and 'results' in quote_data:
                current_price = float(quote_data['results']['p'])
            else:
                raise Exception("No data")
        except:
            # Fallback mock prices
            current_prices = {"AAPL": 185.50, "NVDA": 140.25, "TSLA": 175.80, "SPY": 440.50}
            current_price = current_prices.get(stock_symbol, 100.0)
        
        st.metric("Current Stock Price", f"${current_price:.2f}")
        
        # Market outlook
        outlook = st.radio(
            "2️⃣ What do you think the stock will do?",
            ["📈 Go Up (Bullish)", "📉 Go Down (Bearish)", "➡️ Stay Flat (Neutral)"],
            help="Your market outlook determines which strategy to use"
        )
    
    with col2:
        # Recommend strategy based on outlook
        if "Go Up" in outlook:
            st.success("💡 Recommended: Buy a Call Option")
            st.caption("A call gives you the right to buy shares at a fixed price")
            option_type = "Call"
        elif "Go Down" in outlook:
            st.success("💡 Recommended: Buy a Put Option")
            st.caption("A put gives you the right to sell shares at a fixed price")
            option_type = "Put"
        else:
            st.success("💡 Recommended: Sell Options for Income")
            st.caption("Collect premium by selling options to other traders")
            option_type = st.radio("Option Type:", ["Call", "Put"])
        
        # Strike price selection with guidance
        st.write("3️⃣ Choose your strike price:")
        
        strike = st.slider(
            "Strike Price",
            min_value=int(current_price * 0.9),
            max_value=int(current_price * 1.1),
            value=int(current_price),
            step=1
        )
        
        # Premium calculation (simplified)
        if abs(strike - current_price) < current_price * 0.02:  # ATM
            premium = current_price * 0.03
        elif (option_type == "Call" and strike > current_price) or (option_type == "Put" and strike < current_price):  # OTM
            premium = current_price * 0.015
        else:  # ITM
            premium = current_price * 0.05
        
        premium = round(premium, 2)
        st.metric("Option Premium (Cost)", f"${premium} per share")
        st.caption("Remember: 1 option contract = 100 shares")
    
    # Scenario Analysis
    st.divider()
    st.subheader("4️⃣ See Your Potential Outcomes")
    
    # Create payoff diagram
    fig = create_payoff_diagram(option_type, strike, premium, current_price, is_buyer=True)
    st.plotly_chart(fig, use_container_width=True)

# --- STREAMLIT WEB APPLICATION ---
st.set_page_config(layout="wide", initial_sidebar_state="collapsed")
st.title("Polygon-Based Trading Dashboard - Stocks & Options")

# Initialize APIs
try:
    # Try to get from Streamlit secrets first
    tiingo_key = st.secrets["tiingo"]["api_key"]
    polygon_key = st.secrets["polygon"]["api_key"]
except:
    # Fall back to config file
    config = configparser.ConfigParser()
    config.read('config.ini')
    tiingo_key = config['tiingo']['api_key']
    polygon_key = config['polygon']['api_key']

tiingo_config = {'api_key': tiingo_key, 'session': True}
tiingo_client = TiingoClient(tiingo_config)
polygon_api = PolygonAPI(polygon_key)
paper_db = PaperTradingDB()

# Create tabs for different sections
tab_live, tab_options, tab_backtest = st.tabs(["📊 Live Account", "🎯 Options Trading", "📈 Strategy Backtester"])

with tab_live:
    st.header("Live Account Status")
    
    # Get account info
    account_info = paper_db.get_account_info()
    positions = paper_db.get_positions()
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Cash Balance", f"${account_info['cash']:,.2f}")
    col2.metric("Portfolio Value", f"${account_info['portfolio_value']:,.2f}")
    col3.metric("Account Type", "Live Trading")
    
    st.divider()
    
    # Current positions
    st.subheader("Current Positions")
    if not positions.empty:
        # Get current prices for positions
        position_data = []
        for _, pos in positions.iterrows():
            try:
                quote_data = polygon_api.get_quote(pos['symbol'])
                if quote_data and 'results' in quote_data:
                    current_price = quote_data['results']['p']
                    market_value = pos['quantity'] * current_price
                    unrealized_pnl = (current_price - pos['avg_price']) * pos['quantity']
                else:
                    current_price = pos['avg_price']
                    market_value = pos['quantity'] * current_price
                    unrealized_pnl = 0
                
                position_data.append({
                    'Symbol': pos['symbol'],
                    'Quantity': pos['quantity'],
                    'Avg Price': f"${pos['avg_price']:.2f}",
                    'Current Price': f"${current_price:.2f}",
                    'Market Value': f"${market_value:.2f}",
                    'Unrealized P/L': f"${unrealized_pnl:.2f}",
                    'Type': pos['position_type']
                })
            except:
                position_data.append({
                    'Symbol': pos['symbol'],
                    'Quantity': pos['quantity'],
                    'Avg Price': f"${pos['avg_price']:.2f}",
                    'Current Price': "N/A",
                    'Market Value': "N/A",
                    'Unrealized P/L': "N/A",
                    'Type': pos['position_type']
                })
        
        if position_data:
            df_positions = pd.DataFrame(position_data)
            st.dataframe(df_positions, use_container_width=True)
    else:
        st.info("No open positions")
    
    # Simple trading interface
    st.divider()
    st.subheader("Place Live Trade")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        trade_symbol = st.text_input("Symbol:", "NVDA").upper()
    with col2:
        trade_side = st.selectbox("Side:", ["buy", "sell"])
    with col3:
        trade_qty = st.number_input("Quantity:", min_value=1, value=10)
    
    if st.button("Get Quote & Place Trade"):
        if trade_symbol:
            try:
                quote_data = polygon_api.get_quote(trade_symbol)
                if quote_data and 'results' in quote_data:
                    current_price = quote_data['results']['p']
                    st.success(f"Current price for {trade_symbol}: ${current_price:.2f}")
                    
                    # Place the trade
                    paper_db.add_trade(trade_symbol, trade_qty, current_price, trade_side)
                    
                    if trade_side == "buy":
                        st.success(f"✅ Live trade executed: Bought {trade_qty} shares of {trade_symbol} at ${current_price:.2f}")
                    else:
                        st.success(f"✅ Live trade executed: Sold {trade_qty} shares of {trade_symbol} at ${current_price:.2f}")
                    
                    st.experimental_rerun()
                else:
                    st.error("Could not get quote for this symbol")
            except Exception as e:
                st.error(f"Error placing trade: {e}")

with tab_options:
    st.header("🎯 Options Trading Center")
    
    # User experience level selector
    user_mode = st.radio(
        "Select your experience level:",
        ["👶 Beginner Mode", "🎓 Advanced Mode"],
        horizontal=True,
        key="options_mode"
    )
    
    if "Beginner" in user_mode:
        # Beginner-friendly interface
        tabs = st.tabs([
            "📚 Learn Options", 
            "🎮 Practice Simulator", 
            "📊 Options Chain"
        ])
        
        with tabs[0]:
            # Educational content
            st.subheader("📚 Options Education Center")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("**What are Options?**")
                st.write("Options are contracts that give you the right (but not obligation) to buy or sell a stock at a specific price by a certain date.")
                
                st.success("**Call Options = Movie Tickets** 🎬")
                st.write("• Pay a small fee for the right to buy")
                st.write("• Don't have to use it")
                st.write("• Can be very valuable if price goes up")
                
            with col2:
                st.info("**Why Use Options?**")
                st.write("• **Less Capital**: Control 100 shares for fraction of cost")
                st.write("• **Limited Risk**: Can only lose premium when buying")
                st.write("• **Flexibility**: Multiple strategies for any market")
                
                st.error("**Put Options = Insurance** 🛡️")
                st.write("• Pay premium for protection")
                st.write("• Right to sell at set price")
                st.write("• Profit when stock falls")
        
        with tabs[1]:
            # Simulator
            options_simulator()
        
        with tabs[2]:
            # Options chain
            st.subheader("📊 Live Options Chain (Polygon Data)")
            chain_symbol = st.text_input("Enter symbol:", "NVDA", key="chain_symbol_beginner")
            if st.button("Load Options", key="load_chain_beginner"):
                with st.spinner(f"Loading options for {chain_symbol}..."):
                    try:
                        display_options_chain(polygon_api, chain_symbol)
                    except Exception as e:
                        st.error(f"Error: {e}")
    
    else:
        # Advanced mode
        st.subheader("Advanced Options Trading")
        
        col1, col2 = st.columns([3, 1])
        with col1:
            options_symbol = st.text_input("Enter symbol for options chain:", "NVDA", key="chain_symbol_advanced")
        with col2:
            st.write("")  # Spacer
            st.write("")  # Spacer
            load_btn = st.button("Load Options Chain", key="load_chain_advanced")
        
        if load_btn:
            with st.spinner(f"Loading options for {options_symbol}..."):
                try:
                    display_options_chain(polygon_api, options_symbol)
                except Exception as e:
                    st.error(f"Error loading options chain: {e}")

with tab_backtest:
    st.header("Strategy Backtester")
    st.info("📈 Backtesting still uses Tiingo data (more reliable historical data)")
    
    st.subheader("Adaptive Momentum Strategy Backtest")
    
    with st.expander("ℹ️ About the Strategy"):
        st.markdown("""
        **The Adaptive Momentum Strategy:**
        - Uses 200-day moving average as trend filter
        - Buys when price breaks 2% above MA
        - Sells when price breaks 2% below MA
        - Aims to capture major trends while avoiding whipsaws
        """)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        symbol = st.text_input("Stock Ticker", "NVDA", key="backtest_symbol").upper()
    with col2:
        st.write("")  # Spacer
        st.write("")  # Spacer
        run_backtest = st.button("Run Backtest", type="primary")
    
    if run_backtest:
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
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Buy & Hold Return", results["buy_and_hold"])
                col2.metric("Strategy Return", results["strategy"])
                
                # Calculate outperformance
                bh_return = float(results["buy_and_hold"].strip('%'))
                strat_return = float(results["strategy"].strip('%'))
                outperformance = strat_return - bh_return
                
                col3.metric("Outperformance", f"{outperformance:.2f}%", 
                           delta=f"{outperformance:.2f}%",
                           delta_color="normal" if outperformance > 0 else "inverse")
                
                st.pyplot(fig_or_error)
        else:
            st.warning("Please enter a stock ticker.")

# Footer
st.divider()
st.caption("🔗 **Powered by Polygon.io** - Professional market data with paper trading simulation")