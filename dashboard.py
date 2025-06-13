# dashboard_with_ai_debug.py - Trading Dashboard with AI Debugging Integration

# IMPORTANT: st.set_page_config MUST be the first Streamlit command
import streamlit as st
st.set_page_config(layout="wide", initial_sidebar_state="auto")

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

# Import the ML options strategy module
from ml_options_strategy import create_ml_options_tab

# Import our AI debugging system
from ai_debug_system import AIDebugSystem, create_debug_panel, integrate_ai_debugging, debug_wrapper

# Polygon API wrapper class with debugging
class PolygonAPI:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.polygon.io"
        
    @debug_wrapper
    def get_quote(self, symbol):
        """Get real-time quote with error handling"""
        url = f"{self.base_url}/v2/last/trade/{symbol}"
        params = {"apikey": self.api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Error fetching quote: {response.status_code}")
            return None
    
    @debug_wrapper
    def get_options_chain(self, underlying_symbol, expiration_date=None):
        """Get options chain for a symbol with debugging"""
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
    
    @debug_wrapper
    def get_options_quotes(self, options_ticker):
        """Get options quote with debugging"""
        url = f"{self.base_url}/v3/last/trade/options/{options_ticker}"
        params = {"apikey": self.api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            return None

# Paper Trading Database Manager with debugging
class PaperTradingDB:
    def __init__(self, db_path="paper_trading.db"):
        self.db_path = db_path
        self.init_db()
    
    @debug_wrapper
    def init_db(self):
        """Initialize database tables with error handling"""
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
    
    @debug_wrapper
    def get_positions(self):
        """Get all current positions with error handling"""
        conn = sqlite3.connect(self.db_path)
        df = pd.read_sql_query('SELECT * FROM positions WHERE quantity != 0', conn)
        conn.close()
        return df
    
    @debug_wrapper
    def get_account_info(self):
        """Get account information with error handling"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT cash, portfolio_value FROM account WHERE id = 1')
        result = cursor.fetchone()
        conn.close()
        return {"cash": result[0], "portfolio_value": result[1]} if result else {"cash": 100000, "portfolio_value": 100000}
    
    @debug_wrapper
    def add_trade(self, symbol, quantity, price, side, trade_type='stock'):
        """Add a trade to the database with comprehensive error handling"""
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

@debug_wrapper
def run_backtest_for_dashboard(symbol, start_date, end_date, config, regime_window=200):
    """Backtest function with debugging"""
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

@debug_wrapper
def display_options_chain(polygon_api, symbol):
    """Display options chain with comprehensive error handling"""
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
    except Exception as e:
        current_price = 100.0
        st.warning(f"Could not fetch current price: {e}")
    
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

# --- STREAMLIT WEB APPLICATION WITH AI DEBUGGING ---
st.title("🤖 AI-Enhanced Trading Dashboard - Stocks & Options")

# Initialize AI Debugging System
debug_system = integrate_ai_debugging()

# Add debug panel to sidebar
create_debug_panel()

# Initialize APIs with error handling
try:
    # Try to get from Streamlit secrets first
    tiingo_key = st.secrets["tiingo"]["api_key"]
    polygon_key = st.secrets["polygon"]["api_key"]
except:
    # Fall back to config file
    try:
        config = configparser.ConfigParser()
        config.read('config.ini')
        tiingo_key = config.get('tiingo', 'api_key', fallback='YOUR_TIINGO_API_KEY_HERE')
        polygon_key = config.get('polygon', 'api_key', fallback='YOUR_POLYGON_API_KEY_HERE')
    except Exception as e:
        st.error(f"Configuration error: {e}")
        st.info("Please make sure your config.ini file has the correct format:")
        st.code("""[tiingo]
api_key = your_tiingo_key_here

[polygon]
api_key = your_polygon_key_here""")
        st.stop()

# Check if we have valid API keys
if tiingo_key == 'YOUR_TIINGO_API_KEY_HERE':
    st.error("⚠️ Please update your Tiingo API key in config.ini")
    st.stop()

if polygon_key == 'YOUR_POLYGON_API_KEY_HERE':
    st.error("⚠️ Please update your Polygon API key in config.ini")
    st.stop()

# Initialize API clients
tiingo_config = {'api_key': tiingo_key, 'session': True}
tiingo_client = TiingoClient(tiingo_config)
polygon_api = PolygonAPI(polygon_key)
paper_db = PaperTradingDB()

# Store in session state for debugging context
st.session_state.polygon_api = polygon_api
st.session_state.paper_db = paper_db

# AI Debugging Status Indicator
col1, col2, col3 = st.columns([3, 1, 1])
with col1:
    pass  # Main title space
with col2:
    if debug_system.client:
        st.success("🤖 AI Debug: Online")
    else:
        st.warning("🤖 AI Debug: Offline")
with col3:
    st.info(f"🛠️ Debug Mode: Active")

# Create tabs for different sections
tab_live, tab_options, tab_backtest, tab_ml, tab_debug = st.tabs([
    "📊 Live Account", 
    "🎯 Options Trading", 
    "📈 Strategy Backtester",
    "🧠 ML Strategy Finder",
    "🐛 Debug Console"
])

with tab_live:
    st.header("Live Account Status")
    
    # Get account info with error handling
    try:
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
                except Exception as e:
                    debug_system.logger.error(f"Error fetching price for {pos['symbol']}: {e}")
                    position_data.append({
                        'Symbol': pos['symbol'],
                        'Quantity': pos['quantity'],
                        'Avg Price': f"${pos['avg_price']:.2f}",
                        'Current Price': "Error",
                        'Market Value': "Error",
                        'Unrealized P/L': "Error",
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
                    with debug_system.debug_function("place_trade"):
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
                    # Error will be automatically handled by debug system
                    st.error(f"Error placing trade: {e}")
    
    except Exception as e:
        # Error will be automatically handled by debug system
        st.error(f"Error loading account data: {e}")

with tab_options:
    st.header("🎯 Options Trading Center")
    st.info("🤖 AI debugging active - All options functions monitored")
    
    # Options trading interface with debugging
    options_symbol = st.text_input("Enter symbol for options chain:", "NVDA")
    
    if st.button("Load Options Chain"):
        with st.spinner(f"Loading options for {options_symbol}..."):
            try:
                display_options_chain(polygon_api, options_symbol)
            except Exception as e:
                # Error automatically handled by debug system
                st.error(f"Error loading options: {e}")

with tab_backtest:
    st.header("Strategy Backtester")
    st.info("📈 Backtesting with AI error monitoring")
    
    symbol = st.text_input("Stock Ticker", "NVDA").upper()
    
    if st.button("Run Backtest"):
        if symbol:
            with st.spinner(f"Running backtest for {symbol}..."):
                try:
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
                    else:
                        st.error(f"Backtest failed: {fig_or_error}")
                except Exception as e:
                    # Error automatically handled by debug system
                    st.error(f"Backtest error: {e}")

with tab_ml:
    create_ml_options_tab(polygon_api, tiingo_client)

with tab_debug:
    st.header("🐛 Advanced Debug Console")
    
    # Debug dashboard
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Live Error Monitoring")
        
        # Show recent errors
        try:
            conn = sqlite3.connect("debug_logs.db")
            recent_errors = pd.read_sql_query(
                "SELECT timestamp, error_type, function_name FROM debug_logs ORDER BY timestamp DESC LIMIT 10", 
                conn
            )
            conn.close()
            
            if not recent_errors.empty:
                st.dataframe(recent_errors, use_container_width=True)
            else:
                st.info("No recent errors - system running smoothly! ✅")
        except:
            st.info("Debug database initializing...")
    
    with col2:
        st.subheader("🤖 AI Code Assistant")
        
        code_to_analyze = st.text_area("Paste code for AI analysis:", height=200)
        
        if st.button("🤖 Analyze Code"):
            if code_to_analyze and debug_system.client:
                try:
                    response = debug_system.client.messages.create(
                        model="claude-3-sonnet-20240229",
                        max_tokens=1000,
                        messages=[{
                            "role": "user",
                            "content": f"Analyze this trading dashboard code for bugs, improvements, and best practices:\n\n{code_to_analyze}"
                        }]
                    )
                    
                    analysis = response.content[0].text if hasattr(response.content[0], 'text') else str(response.content[0])
                    st.markdown(analysis)
                except Exception as e:
                    st.error(f"AI analysis failed: {e}")
            elif not debug_system.client:
                st.error("Claude API not available")
            else:
                st.warning("Please enter code to analyze")
    
    # System health metrics
    st.divider()
    st.subheader("📊 System Health")
    
    health_col1, health_col2, health_col3 = st.columns(3)
    
    with health_col1:
        # Database health
        try:
            conn = sqlite3.connect("paper_trading.db")
            trades_count = pd.read_sql_query("SELECT COUNT(*) as count FROM trades", conn).iloc[0]['count']
            conn.close()
            st.metric("Total Trades", trades_count)
        except:
            st.metric("Database Status", "Error")
    
    with health_col2:
        # API health
        try:
            test_quote = polygon_api.get_quote("AAPL")
            if test_quote:
                st.metric("Polygon API", "✅ Online")
            else:
                st.metric("Polygon API", "⚠️ Issues")
        except:
            st.metric("Polygon API", "❌ Offline")
    
    with health_col3:
        # AI Debug health
        if debug_system.client:
            st.metric("AI Debug System", "🤖 Active")
        else:
            st.metric("AI Debug System", "⚠️ Limited")

# Footer with debug info
st.divider()
col1, col2, col3 = st.columns(3)
with col1:
    st.caption("🔗 **Powered by Polygon.io** - Professional market data")
with col2:
    st.caption("🤖 **AI Debug System** - Smart error detection & analysis")
with col3:
    if debug_system.client:
        st.caption("✅ **Claude AI** - Online & monitoring")
    else:
        st.caption("⚠️ **Claude AI** - Add API key for full debugging")