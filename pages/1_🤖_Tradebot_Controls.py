# pages/1_🤖_Tradebot_Controls.py - Updated for Polygon
import streamlit as st
import requests
from tiingo import TiingoClient
import pandas as pd
from datetime import datetime, timedelta
import math
import sqlite3
import configparser

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
            data = response.json()
            if 'results' in data:
                return data['results']['p']  # Return just the price
        return None
    
    def get_options_quote(self, options_ticker):
        """Get options quote"""
        url = f"{self.base_url}/v3/last/trade/options/{options_ticker}"
        params = {"apikey": self.api_key}
        response = requests.get(url, params=params)
        if response.status_code == 200:
            data = response.json()
            if 'results' in data:
                return data['results']['p']
        return None

# Paper Trading Database Manager (same as dashboard)
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
    
    def get_position(self, symbol):
        """Get position for a specific symbol"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT quantity FROM positions WHERE symbol = ? AND quantity != 0', (symbol,))
        result = cursor.fetchone()
        conn.close()
        return result[0] if result else 0
    
    def get_positions(self):
        """Get all current positions with error handling"""
        try:
            conn = sqlite3.connect(self.db_path)
            df = pd.read_sql_query('SELECT * FROM positions WHERE quantity != 0', conn)
            conn.close()
            return df
        except Exception as e:
            # Return empty dataframe if table doesn't exist or other error
            print(f"Error getting positions: {e}")
            return pd.DataFrame(columns=['id', 'symbol', 'quantity', 'avg_price', 'position_type', 'created_at'])
    
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
                new_avg = ((existing_qty * existing_avg) + (quantity * price)) / new_qty if new_qty != 0 else price
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
        
        return True
    
    def get_account_info(self):
        """Get account information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT cash, portfolio_value FROM account WHERE id = 1')
        result = cursor.fetchone()
        conn.close()
        return {"cash": result[0], "portfolio_value": result[1]} if result else {"cash": 100000, "portfolio_value": 100000}

def run_trade_logic():
    """Main trading logic function using Polygon data and paper trading"""
    logs = []
    
    # 1. CONNECT TO APIS
    logs.append("Connecting to APIs...")
    try:
        # Get API keys
        try:
            polygon_key = st.secrets["polygon"]["api_key"]
            tiingo_key = st.secrets["tiingo"]["api_key"]
        except:
            config = configparser.ConfigParser()
            config.read('config.ini')
            polygon_key = config['polygon']['api_key']
            tiingo_key = config['tiingo']['api_key']
        
        polygon_api = PolygonAPI(polygon_key)
        tiingo_config = {'api_key': tiingo_key, 'session': True}
        tiingo_client = TiingoClient(tiingo_config)
        paper_db = PaperTradingDB()
        
        logs.append("✅ Successfully connected to Polygon and Tiingo APIs")
    except Exception as e:
        logs.append(f"❌ ERROR: Failed to connect to APIs. Error: {e}")
        return logs

    # 2. CHECK CURRENT POSITION
    STOCK_TO_TRADE = 'NVDA'
    logs.append(f"Checking current position for {STOCK_TO_TRADE}...")
    try:
        current_qty = paper_db.get_position(STOCK_TO_TRADE)
        logs.append(f"✅ Current position: {current_qty} shares.")
    except Exception as e:
        logs.append(f"❌ ERROR checking position: {e}")
        return logs
    
    # 3. GET LATEST DATA & APPLY LOGIC
    logs.append("Fetching latest market data...")
    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=250)  # Get enough data for 200-day MA
        
        # Use Tiingo for historical data (more reliable)
        data = tiingo_client.get_dataframe(
            STOCK_TO_TRADE, 
            frequency='daily', 
            startDate=start_date.strftime('%Y-%m-%d'),
            endDate=end_date.strftime('%Y-%m-%d')
        )
        
        if len(data) < 200:
            logs.append(f"❌ ERROR: Not enough data. Only {len(data)} days available.")
            return logs
        
        # Get current price from Polygon (real-time)
        current_price = polygon_api.get_quote(STOCK_TO_TRADE)
        if current_price is None:
            # Fallback to Tiingo's latest price
            current_price = data['close'].iloc[-1]
            logs.append(f"Using Tiingo price as fallback: ${current_price:.2f}")
        else:
            logs.append(f"Current price from Polygon: ${current_price:.2f}")

        # Calculate moving average and bands using historical data
        data['regime_ma'] = data['close'].rolling(window=200).mean()
        buffer = 0.02
        data['upper_band'] = data['regime_ma'] * (1 + buffer)
        data['lower_band'] = data['regime_ma'] * (1 - buffer)
        
        latest_ma = data['regime_ma'].iloc[-1]
        latest_upper_band = data['upper_band'].iloc[-1]
        latest_lower_band = data['lower_band'].iloc[-1]
        
        logs.append(f"200-day MA: ${latest_ma:.2f}")
        logs.append(f"Upper band (MA + 2%): ${latest_upper_band:.2f}")
        logs.append(f"Lower band (MA - 2%): ${latest_lower_band:.2f}")
        
    except Exception as e:
        logs.append(f"❌ ERROR fetching market data: {e}")
        return logs
    
    # 4. MAKE A DECISION
    desired_position = None
    if current_price > latest_upper_band:
        desired_position = 1
        logs.append(f"📈 DECISION: Price (${current_price:.2f}) is above upper band (${latest_upper_band:.2f}). Desired position is IN.")
    elif current_price < latest_lower_band:
        desired_position = 0
        logs.append(f"📉 DECISION: Price (${current_price:.2f}) is below lower band (${latest_lower_band:.2f}). Desired position is OUT.")
    else:
        logs.append(f"➡️ DECISION: Price (${current_price:.2f}) is inside buffer zone. Holding current position.")

    # 5. TAKE ACTION (Paper Trading)
    if desired_position is not None:
        if desired_position == 1 and current_qty == 0:
            # BUY
            account_info = paper_db.get_account_info()
            USD_TO_TRADE = min(10000, account_info['cash'])  # Don't trade more than available cash
            qty_to_buy = int(USD_TO_TRADE / current_price)
            
            logs.append(f"💰 ACTION: Current position is 0, desired is IN.")
            logs.append(f"Available cash: ${account_info['cash']:.2f}")
            logs.append(f"Calculating order: ${USD_TO_TRADE} / ${current_price:.2f} = {qty_to_buy} shares")
            
            if qty_to_buy > 0:
                try:
                    success = paper_db.add_trade(STOCK_TO_TRADE, qty_to_buy, current_price, 'buy')
                    if success:
                        logs.append(f"✅ Successfully placed paper BUY order for {qty_to_buy} shares at ${current_price:.2f}")
                    else:
                        logs.append(f"❌ Failed to place paper BUY order")
                except Exception as e:
                    logs.append(f"❌ ERROR placing paper BUY order: {e}")
            else:
                logs.append(f"❌ Cannot buy: Insufficient funds or price too high")
                
        elif desired_position == 0 and current_qty > 0:
            # SELL
            logs.append(f"💸 ACTION: Current position is {current_qty}, desired is OUT.")
            logs.append(f"Placing paper SELL order to liquidate position...")
            
            try:
                success = paper_db.add_trade(STOCK_TO_TRADE, current_qty, current_price, 'sell')
                if success:
                    logs.append(f"✅ Successfully placed paper SELL order for {current_qty} shares at ${current_price:.2f}")
                else:
                    logs.append(f"❌ Failed to place paper SELL order")
            except Exception as e:
                logs.append(f"❌ ERROR placing paper SELL order: {e}")
        else:
            logs.append("✅ ACTION: Desired position matches current position. No trade needed.")
    else:
        logs.append("✅ No action needed - price is in buffer zone.")
            
    logs.append("--- Paper Trading Bot logic finished. ---")
    return logs

def run_options_trade_logic():
    """Options trading logic using Polygon data"""
    logs = []
    
    # 1. CONNECT TO APIS
    logs.append("Connecting to APIs for options trading...")
    try:
        try:
            polygon_key = st.secrets["polygon"]["api_key"]
            tiingo_key = st.secrets["tiingo"]["api_key"]
        except:
            config = configparser.ConfigParser()
            config.read('config.ini')
            polygon_key = config['polygon']['api_key']
            tiingo_key = config['tiingo']['api_key']
        
        polygon_api = PolygonAPI(polygon_key)
        tiingo_config = {'api_key': tiingo_key, 'session': True}
        tiingo_client = TiingoClient(tiingo_config)
        paper_db = PaperTradingDB()
        
        logs.append("✅ Successfully connected to APIs")
    except Exception as e:
        logs.append(f"❌ ERROR: Failed to connect to APIs. Error: {e}")
        return logs

    # 2. CHECK CURRENT OPTIONS POSITIONS
    UNDERLYING = 'NVDA'
    logs.append(f"Checking current options positions for {UNDERLYING}...")
    
    # For this demo, we'll simulate options positions
    # In a real implementation, you'd track options positions separately
    logs.append("✅ No current options positions (demo mode)")
    
    # 3. GET MARKET DATA AND SIGNALS
    logs.append("Analyzing market conditions for options strategy...")
    try:
        # Get current price
        current_price = polygon_api.get_quote(UNDERLYING)
        if current_price is None:
            logs.append("❌ Could not get current price from Polygon")
            return logs
        
        logs.append(f"Current {UNDERLYING} price: ${current_price:.2f}")
        
        # Get historical data for momentum analysis
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=50)
        
        data = tiingo_client.get_dataframe(
            UNDERLYING, 
            frequency='daily', 
            startDate=start_date.strftime('%Y-%m-%d'),
            endDate=end_date.strftime('%Y-%m-%d')
        )
        
        # Simple momentum indicator
        sma_20 = data['close'].rolling(20).mean().iloc[-1]
        price_vs_sma = (current_price - sma_20) / sma_20 * 100
        
        logs.append(f"20-day SMA: ${sma_20:.2f}")
        logs.append(f"Price vs SMA: {price_vs_sma:.1f}%")
        
        # Options strategy decision
        if price_vs_sma > 5:
            logs.append("📈 SIGNAL: Strong bullish momentum detected")
            logs.append("💡 STRATEGY: Consider buying call options")
            
            # Simulate call option trade
            strike_price = round(current_price * 1.05)  # 5% OTM call
            option_premium = current_price * 0.03  # Simplified premium calculation
            
            logs.append(f"📋 Simulated Call Option Trade:")
            logs.append(f"   Strike: ${strike_price}")
            logs.append(f"   Premium: ${option_premium:.2f}")
            logs.append(f"   Breakeven: ${strike_price + option_premium:.2f}")
            
        elif price_vs_sma < -5:
            logs.append("📉 SIGNAL: Strong bearish momentum detected")
            logs.append("💡 STRATEGY: Consider buying put options")
            
            # Simulate put option trade
            strike_price = round(current_price * 0.95)  # 5% OTM put
            option_premium = current_price * 0.03
            
            logs.append(f"📋 Simulated Put Option Trade:")
            logs.append(f"   Strike: ${strike_price}")
            logs.append(f"   Premium: ${option_premium:.2f}")
            logs.append(f"   Breakeven: ${strike_price - option_premium:.2f}")
            
        else:
            logs.append("➡️ SIGNAL: Neutral momentum")
            logs.append("💡 STRATEGY: Consider selling options for income")
        
    except Exception as e:
        logs.append(f"❌ ERROR in options analysis: {e}")
        return logs
    
    logs.append("--- Options analysis complete (demo mode) ---")
    logs.append("📝 Note: Real options trading requires additional risk management")
    
    return logs

# --- STREAMLIT PAGE UI ---
st.set_page_config(initial_sidebar_state="collapsed")
st.title("🤖 Polygon-Based Trading Bot Controls")

# Password protection
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

password_guess = st.text_input("Enter Admin Password", type="password", key="admin_pass")

# Get admin password from secrets or use default for local testing
try:
    admin_password = st.secrets["admin_password"]
except:
    admin_password = "admin123"  # Default for local testing

if password_guess == admin_password and admin_password != "":
    st.session_state.authenticated = True

if st.session_state.authenticated:
    st.success("✅ Access Granted")
    st.info("🔄 **Now using Polygon.io for real-time data + Paper Trading simulation**")
    
    # Choose trading mode
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Paper Trading Bot")
        st.write("Paper trading using Polygon real-time data")
        st.write("• Uses SQLite database for positions")
        st.write("• Polygon API for current prices")
        st.write("• Tiingo for historical data")
        
        if st.button("🏃 Run Stock Paper Trade Check", use_container_width=True):
            with st.spinner("Bot is analyzing stock positions..."):
                returned_logs = run_trade_logic()
                
            st.subheader("Stock Bot Activity Log:")
            log_text = "\n".join(returned_logs)
            st.code(log_text)
            
            st.session_state.last_stock_run = datetime.now()
            st.session_state.last_stock_logs = returned_logs
    
    with col2:
        st.subheader("Options Analysis Bot")
        st.write("Options market analysis using Polygon")
        st.write("• Real-time options data from Polygon")
        st.write("• Momentum-based strategy signals")
        st.write("• Educational/demo mode")
        
        if st.button("🎯 Run Options Analysis", use_container_width=True):
            with st.spinner("Bot is analyzing options strategies..."):
                returned_logs = run_options_trade_logic()
                
            st.subheader("Options Analysis Log:")
            log_text = "\n".join(returned_logs)
            st.code(log_text)
            
            st.session_state.last_options_run = datetime.now()
            st.session_state.last_options_logs = returned_logs
    
    # Show account status
    st.divider()
    st.subheader("📊 Paper Trading Account Status")
    
    try:
        paper_db = PaperTradingDB()
        account_info = paper_db.get_account_info()
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Cash Balance", f"${account_info['cash']:,.2f}")
        col2.metric("Portfolio Value", f"${account_info['portfolio_value']:,.2f}")
        col3.metric("Account Type", "Paper Trading")
        
        # Show positions
        positions = paper_db.get_positions()
        if not positions.empty:
            st.write("**Current Positions:**")
            st.dataframe(positions[['symbol', 'quantity', 'avg_price', 'position_type']], use_container_width=True)
        else:
            st.info("No open positions")
        
    except Exception as e:
        st.error(f"Could not load account info: {e}")
    
    # Show last run info
    st.divider()
    st.subheader("⏰ Bot Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if 'last_stock_run' in st.session_state:
            st.metric("Last Stock Check", st.session_state.last_stock_run.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            st.metric("Last Stock Check", "Never run")
    
    with col2:
        if 'last_options_run' in st.session_state:
            st.metric("Last Options Analysis", st.session_state.last_options_run.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            st.metric("Last Options Analysis", "Never run")

elif password_guess != "":
    st.error("❌ Incorrect password. Access denied.")
else:
    st.info("Please enter the admin password to access trading bot controls.")

# Footer
st.divider()
st.caption("🔗 **Powered by Polygon.io** - Real-time market data with paper trading simulation")
st.caption("⚠️ **Paper Trading Only** - No real money at risk!")