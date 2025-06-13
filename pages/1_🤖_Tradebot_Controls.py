# pages/1_🤖_Tradebot_Controls.py - Enhanced with Volatility Arbitrage
import streamlit as st
import requests
from tiingo import TiingoClient
import pandas as pd
from datetime import datetime, timedelta
import math
import sqlite3
import configparser
import numpy as np

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
    
    def get_options_chain(self, underlying_symbol, expiration_date=None):
        """Get options chain for volatility analysis"""
        url = f"{self.base_url}/v3/reference/options/contracts"
        params = {
            "underlying_ticker": underlying_symbol,
            "apikey": self.api_key,
            "limit": 100
        }
        
        if expiration_date:
            params["expiration_date"] = expiration_date
            
        response = requests.get(url, params=params)
        if response.status_code == 200:
            return response.json()
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

# Enhanced Paper Trading Database Manager
class PaperTradingDB:
    def __init__(self, db_path="paper_trading.db"):
        self.db_path = db_path
        self.init_db()
    
    def init_db(self):
        """Initialize database tables with migration support"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create positions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS positions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                quantity REAL NOT NULL,
                avg_price REAL NOT NULL,
                position_type TEXT NOT NULL, -- 'stock', 'call', 'put', 'vol_arb'
                strategy_type TEXT DEFAULT 'momentum', -- 'momentum', 'vol_arbitrage'
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
                strategy_type TEXT DEFAULT 'momentum',
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Migrate existing tables to add new columns if they don't exist
        try:
            # Check if strategy_type column exists in trades table
            cursor.execute("PRAGMA table_info(trades)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'strategy_type' not in columns:
                cursor.execute('ALTER TABLE trades ADD COLUMN strategy_type TEXT DEFAULT "momentum"')
                print("✅ Added strategy_type column to trades table")
        except Exception as e:
            print(f"Migration warning: {e}")
        
        try:
            # Check if strategy_type column exists in positions table
            cursor.execute("PRAGMA table_info(positions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'strategy_type' not in columns:
                cursor.execute('ALTER TABLE positions ADD COLUMN strategy_type TEXT DEFAULT "momentum"')
                print("✅ Added strategy_type column to positions table")
        except Exception as e:
            print(f"Migration warning: {e}")
        
        # Create volatility tracking table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS volatility_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                date DATE NOT NULL,
                realized_vol REAL,
                implied_vol REAL,
                vol_spread REAL,
                price REAL,
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
            print(f"Error getting positions: {e}")
            return pd.DataFrame(columns=['id', 'symbol', 'quantity', 'avg_price', 'position_type', 'strategy_type', 'created_at'])
    
    def add_trade(self, symbol, quantity, price, side, trade_type='stock', strategy_type='momentum'):
        """Add a trade to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Add trade to trades table
        cursor.execute('''
            INSERT INTO trades (symbol, quantity, price, side, trade_type, strategy_type)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (symbol, quantity, price, side, trade_type, strategy_type))
        
        # Update positions
        cursor.execute('SELECT quantity, avg_price FROM positions WHERE symbol = ? AND strategy_type = ?', (symbol, strategy_type))
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
                UPDATE positions SET quantity = ?, avg_price = ? WHERE symbol = ? AND strategy_type = ?
            ''', (new_qty, new_avg, symbol, strategy_type))
        else:
            if side == 'buy':
                cursor.execute('''
                    INSERT INTO positions (symbol, quantity, avg_price, position_type, strategy_type)
                    VALUES (?, ?, ?, ?, ?)
                ''', (symbol, quantity, price, trade_type, strategy_type))
        
        # Update cash
        cash_change = -quantity * price if side == 'buy' else quantity * price
        cursor.execute('UPDATE account SET cash = cash + ? WHERE id = 1', (cash_change,))
        
        conn.commit()
        conn.close()
        
        return True
    
    def add_volatility_data(self, symbol, realized_vol, implied_vol, price):
        """Track volatility data for analysis"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        vol_spread = implied_vol - realized_vol
        
        cursor.execute('''
            INSERT INTO volatility_data (symbol, date, realized_vol, implied_vol, vol_spread, price)
            VALUES (?, DATE('now'), ?, ?, ?, ?)
        ''', (symbol, realized_vol, implied_vol, vol_spread, price))
        
        conn.commit()
        conn.close()
    
    def get_account_info(self):
        """Get account information"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT cash, portfolio_value FROM account WHERE id = 1')
        result = cursor.fetchone()
        conn.close()
        return {"cash": result[0], "portfolio_value": result[1]} if result else {"cash": 100000, "portfolio_value": 100000}

class VolatilityCalculator:
    """Calculate various volatility metrics"""
    
    @staticmethod
    def realized_volatility(prices, window=30):
        """Calculate realized volatility (historical)"""
        if len(prices) < window:
            return None
        
        returns = prices.pct_change().dropna()
        realized_vol = returns.rolling(window=window).std() * np.sqrt(252)  # Annualized
        return realized_vol.iloc[-1] if not realized_vol.empty else None
    
    @staticmethod
    def parkinson_volatility(high, low, window=30):
        """Parkinson estimator - more efficient than close-to-close"""
        if len(high) < window or len(low) < window:
            return None
        
        hl_ratio = np.log(high / low)
        parkinson_vol = np.sqrt((1/(4*np.log(2))) * (hl_ratio**2).rolling(window=window).mean()) * np.sqrt(252)
        return parkinson_vol.iloc[-1] if not parkinson_vol.empty else None
    
    @staticmethod
    def garch_volatility(returns, window=30):
        """Simple GARCH(1,1) approximation"""
        if len(returns) < window:
            return None
        
        # Simplified GARCH - in practice, you'd use arch package
        squared_returns = returns**2
        garch_vol = np.sqrt(squared_returns.rolling(window=window).mean()) * np.sqrt(252)
        return garch_vol.iloc[-1] if not garch_vol.empty else None
    
    @staticmethod
    def implied_volatility_proxy(option_price, stock_price, strike, time_to_expiry, risk_free_rate=0.05):
        """Simplified Black-Scholes implied volatility approximation"""
        # This is a very simplified approximation - in practice, use scipy.optimize or specialized libraries
        if time_to_expiry <= 0 or option_price <= 0:
            return None
        
        # Approximate IV using Brenner-Subrahmanyam formula
        moneyness = stock_price / strike
        if moneyness <= 0:
            return None
        
        # Simplified approximation
        iv_approx = (option_price / stock_price) * np.sqrt(2 * np.pi / time_to_expiry)
        return min(max(iv_approx, 0.05), 2.0)  # Cap between 5% and 200%

def run_volatility_arbitrage_strategy():
    """Advanced volatility arbitrage strategy"""
    logs = []
    
    # 1. CONNECT TO APIS
    logs.append("🔬 Connecting to APIs for Volatility Arbitrage...")
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
        vol_calc = VolatilityCalculator()
        
        logs.append("✅ Successfully connected to APIs")
    except Exception as e:
        logs.append(f"❌ ERROR: Failed to connect to APIs. Error: {e}")
        return logs

    # 2. ANALYZE VOLATILITY SURFACE
    UNDERLYING = 'NVDA'
    logs.append(f"📊 Analyzing volatility surface for {UNDERLYING}...")
    
    try:
        # Get historical data for realized volatility calculation
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=90)
        
        data = tiingo_client.get_dataframe(
            UNDERLYING,
            frequency='daily',
            startDate=start_date.strftime('%Y-%m-%d'),
            endDate=end_date.strftime('%Y-%m-%d')
        )
        
        if data.empty or len(data) < 30:
            logs.append("❌ Insufficient historical data for volatility analysis")
            return logs
        
        # Get current price
        current_price = polygon_api.get_quote(UNDERLYING)
        if current_price is None:
            current_price = data['close'].iloc[-1]
            logs.append(f"Using historical price: ${current_price:.2f}")
        else:
            logs.append(f"Current price: ${current_price:.2f}")
        
        # Calculate multiple volatility measures
        realized_vol_30 = vol_calc.realized_volatility(data['close'], window=30)
        realized_vol_10 = vol_calc.realized_volatility(data['close'], window=10)
        
        if 'high' in data.columns and 'low' in data.columns:
            parkinson_vol = vol_calc.parkinson_volatility(data['high'], data['low'], window=30)
        else:
            parkinson_vol = None
        
        logs.append(f"📈 Realized Vol (30-day): {realized_vol_30*100:.1f}%" if realized_vol_30 else "❌ Cannot calculate 30-day RV")
        logs.append(f"📈 Realized Vol (10-day): {realized_vol_10*100:.1f}%" if realized_vol_10 else "❌ Cannot calculate 10-day RV")
        if parkinson_vol:
            logs.append(f"📈 Parkinson Vol (30-day): {parkinson_vol*100:.1f}%")
        
    except Exception as e:
        logs.append(f"❌ ERROR in volatility calculation: {e}")
        return logs
    
    # 3. ANALYZE OPTIONS CHAIN FOR IMPLIED VOLATILITY
    logs.append("🎯 Analyzing options chain for implied volatility...")
    
    try:
        options_data = polygon_api.get_options_chain(UNDERLYING)
        
        if not options_data or 'results' not in options_data:
            logs.append("❌ No options data available - simulating implied volatility")
            # Simulate implied volatility for demo
            simulated_iv = realized_vol_30 * (1 + np.random.normal(0, 0.2)) if realized_vol_30 else 0.25
            simulated_iv = max(0.1, min(simulated_iv, 1.0))  # Cap between 10% and 100%
            implied_vol = simulated_iv
            logs.append(f"📊 Simulated Implied Vol: {implied_vol*100:.1f}%")
        else:
            # Analyze real options data
            options = options_data['results'][:10]  # Limit for demo
            iv_estimates = []
            
            for option in options:
                try:
                    strike = option.get('strike_price', current_price)
                    expiry_str = option.get('expiration_date', '')
                    
                    if expiry_str:
                        expiry_date = pd.to_datetime(expiry_str)
                        time_to_expiry = (expiry_date - pd.Timestamp.now()).days / 365.0
                        
                        if time_to_expiry > 0:
                            # Get option price (simplified - would need bid/ask)
                            option_ticker = option.get('ticker', '')
                            option_price = polygon_api.get_options_quote(option_ticker)
                            
                            if option_price and option_price > 0:
                                iv = vol_calc.implied_volatility_proxy(
                                    option_price, current_price, strike, time_to_expiry
                                )
                                if iv:
                                    iv_estimates.append(iv)
                except:
                    continue
            
            if iv_estimates:
                implied_vol = np.median(iv_estimates)
                logs.append(f"📊 Median Implied Vol: {implied_vol*100:.1f}% (from {len(iv_estimates)} options)")
            else:
                # Fallback to simulation
                implied_vol = realized_vol_30 * 1.2 if realized_vol_30 else 0.25
                logs.append(f"📊 Estimated Implied Vol: {implied_vol*100:.1f}%")
        
    except Exception as e:
        logs.append(f"❌ ERROR in options analysis: {e}")
        implied_vol = realized_vol_30 * 1.1 if realized_vol_30 else 0.25
    
    # 4. VOLATILITY ARBITRAGE DECISION
    logs.append("🧠 Making volatility arbitrage decision...")
    
    if realized_vol_30 and implied_vol:
        vol_spread = implied_vol - realized_vol_30
        vol_spread_pct = (vol_spread / realized_vol_30) * 100
        
        logs.append(f"📊 Volatility Analysis:")
        logs.append(f"   Realized Vol: {realized_vol_30*100:.1f}%")
        logs.append(f"   Implied Vol: {implied_vol*100:.1f}%")
        logs.append(f"   Vol Spread: {vol_spread*100:.1f}% ({vol_spread_pct:+.1f}%)")
        
        # Store volatility data
        paper_db.add_volatility_data(UNDERLYING, realized_vol_30, implied_vol, current_price)
        
        # Trading decision logic
        if vol_spread_pct > 15:  # Implied > Realized by 15%+
            logs.append("📈 SIGNAL: SELL VOLATILITY (Implied vol is expensive)")
            logs.append("💡 STRATEGY: Sell straddle/strangle (collect premium)")
            logs.append("🎯 Expected Profit: Market will be less volatile than options imply")
            
            # Simulate vol selling trade
            premium_collected = current_price * 0.05  # Simplified
            logs.append(f"📋 Simulated Volatility Sale:")
            logs.append(f"   Premium Collected: ${premium_collected:.2f}")
            logs.append(f"   Max Profit: ${premium_collected:.2f}")
            logs.append(f"   Breakeven Range: ${current_price - premium_collected:.2f} - ${current_price + premium_collected:.2f}")
            
            # Record trade
            try:
                paper_db.add_trade(
                    f"{UNDERLYING}_VOL_SELL", 1, premium_collected, 'sell', 
                    'vol_arbitrage', 'vol_arbitrage'
                )
                logs.append("✅ Volatility arbitrage trade recorded")
            except Exception as e:
                logs.append(f"❌ Error recording trade: {e}")
                
        elif vol_spread_pct < -10:  # Realized > Implied by 10%+
            logs.append("📉 SIGNAL: BUY VOLATILITY (Implied vol is cheap)")
            logs.append("💡 STRATEGY: Buy straddle/strangle (pay premium)")
            logs.append("🎯 Expected Profit: Market will be more volatile than options imply")
            
            # Simulate vol buying trade
            premium_paid = current_price * 0.04  # Simplified
            logs.append(f"📋 Simulated Volatility Purchase:")
            logs.append(f"   Premium Paid: ${premium_paid:.2f}")
            logs.append(f"   Breakeven Range: ${current_price - premium_paid:.2f} - ${current_price + premium_paid:.2f}")
            
            # Record trade
            try:
                paper_db.add_trade(
                    f"{UNDERLYING}_VOL_BUY", 1, premium_paid, 'buy', 
                    'vol_arbitrage', 'vol_arbitrage'
                )
                logs.append("✅ Volatility arbitrage trade recorded")
            except Exception as e:
                logs.append(f"❌ Error recording trade: {e}")
                
        else:
            logs.append("➡️ SIGNAL: NEUTRAL - Volatility fairly priced")
            logs.append("💡 STRATEGY: Wait for better volatility arbitrage opportunity")
            logs.append(f"🎯 Need vol spread > 15% or < -10% (current: {vol_spread_pct:+.1f}%)")
    
    logs.append("--- Volatility Arbitrage Analysis Complete ---")
    logs.append("📝 Note: This is educational - real vol arbitrage requires delta hedging")
    
    return logs

def run_trade_logic():
    """Original momentum trading logic function"""
    logs = []
    
    # 1. CONNECT TO APIS
    logs.append("Connecting to APIs...")
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
        start_date = end_date - pd.Timedelta(days=250)
        
        data = tiingo_client.get_dataframe(
            STOCK_TO_TRADE, 
            frequency='daily', 
            startDate=start_date.strftime('%Y-%m-%d'),
            endDate=end_date.strftime('%Y-%m-%d')
        )
        
        if len(data) < 200:
            logs.append(f"❌ ERROR: Not enough data. Only {len(data)} days available.")
            return logs
        
        current_price = polygon_api.get_quote(STOCK_TO_TRADE)
        if current_price is None:
            current_price = data['close'].iloc[-1]
            logs.append(f"Using Tiingo price as fallback: ${current_price:.2f}")
        else:
            logs.append(f"Current price from Polygon: ${current_price:.2f}")

        # Calculate moving average and bands
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

    # 5. TAKE ACTION
    if desired_position is not None:
        if desired_position == 1 and current_qty == 0:
            account_info = paper_db.get_account_info()
            USD_TO_TRADE = min(10000, account_info['cash'])
            qty_to_buy = int(USD_TO_TRADE / current_price)
            
            logs.append(f"💰 ACTION: Current position is 0, desired is IN.")
            logs.append(f"Available cash: ${account_info['cash']:.2f}")
            logs.append(f"Calculating order: ${USD_TO_TRADE} / ${current_price:.2f} = {qty_to_buy} shares")
            
            if qty_to_buy > 0:
                try:
                    success = paper_db.add_trade(STOCK_TO_TRADE, qty_to_buy, current_price, 'buy', 'stock', 'momentum')
                    if success:
                        logs.append(f"✅ Successfully placed paper BUY order for {qty_to_buy} shares at ${current_price:.2f}")
                    else:
                        logs.append(f"❌ Failed to place paper BUY order")
                except Exception as e:
                    logs.append(f"❌ ERROR placing paper BUY order: {e}")
            else:
                logs.append(f"❌ Cannot buy: Insufficient funds or price too high")
                
        elif desired_position == 0 and current_qty > 0:
            logs.append(f"💸 ACTION: Current position is {current_qty}, desired is OUT.")
            logs.append(f"Placing paper SELL order to liquidate position...")
            
            try:
                success = paper_db.add_trade(STOCK_TO_TRADE, current_qty, current_price, 'sell', 'stock', 'momentum')
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
            
    logs.append("--- Momentum Trading Bot logic finished. ---")
    return logs

# --- STREAMLIT PAGE UI ---
st.set_page_config(initial_sidebar_state="collapsed")
st.title("🤖 Advanced Trading Bot Controls - Momentum + Volatility Arbitrage")

# Password protection
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

password_guess = st.text_input("Enter Admin Password", type="password", key="admin_pass")

try:
    admin_password = st.secrets["admin_password"]
except:
    admin_password = "admin123"

if password_guess == admin_password and admin_password != "":
    st.session_state.authenticated = True

if st.session_state.authenticated:
    st.success("✅ Access Granted")
    st.info("🔄 **Enhanced with Volatility Arbitrage + Momentum Strategies**")
    
    # Choose trading strategy
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("📈 Momentum Strategy")
        st.write("Classic momentum-based trading")
        st.write("• 200-day moving average regime")
        st.write("• Trend following with buffer zones")
        st.write("• Stock position management")
        
        if st.button("🏃 Run Momentum Strategy", use_container_width=True):
            with st.spinner("Analyzing momentum signals..."):
                returned_logs = run_trade_logic()
                
            st.subheader("Momentum Strategy Log:")
            log_text = "\n".join(returned_logs)
            # Create a much larger text area for better readability
            st.text_area("Strategy Output", log_text, height=400, disabled=True)
            
            st.session_state.last_momentum_run = datetime.now()
            st.session_state.last_momentum_logs = returned_logs
    
    with col2:
        st.subheader("🔬 Volatility Arbitrage")
        st.write("Advanced volatility trading")
        st.write("• Realized vs Implied volatility")
        st.write("• Multi-timeframe vol analysis")
        st.write("• Options premium strategies")
        
        if st.button("🎯 Run Vol Arbitrage", use_container_width=True):
            with st.spinner("Analyzing volatility surface..."):
                returned_logs = run_volatility_arbitrage_strategy()
                
            st.subheader("Volatility Arbitrage Log:")
            log_text = "\n".join(returned_logs)
            # Create a much larger text area for the volatility arbitrage logs
            st.text_area("Volatility Analysis Output", log_text, height=500, disabled=True)
            
            st.session_state.last_vol_arb_run = datetime.now()
            st.session_state.last_vol_arb_logs = returned_logs
    
    with col3:
        st.subheader("🚀 Combined Strategy")
        st.write("Run both strategies together")
        st.write("• Momentum for directional trades")
        st.write("• Vol arbitrage for income")
        st.write("• Portfolio diversification")
        
        if st.button("⚡ Run Both Strategies", use_container_width=True):
            with st.spinner("Running combined analysis..."):
                momentum_logs = run_trade_logic()
                vol_arb_logs = run_volatility_arbitrage_strategy()
                
                combined_logs = ["=== MOMENTUM STRATEGY ==="] + momentum_logs + \
                               ["", "=== VOLATILITY ARBITRAGE ==="] + vol_arb_logs
                
            st.subheader("Combined Strategy Log:")
            log_text = "\n".join(combined_logs)
            # Create an even larger text area for combined logs
            st.text_area("Combined Strategy Output", log_text, height=600, disabled=True)
            
            st.session_state.last_combined_run = datetime.now()
    
    # Account status - Make it wider and more readable
    st.divider()
    st.subheader("📊 Enhanced Paper Trading Account")
    
    try:
        paper_db = PaperTradingDB()
        account_info = paper_db.get_account_info()
        
        # Use wider columns and better formatting
        col1, col2 = st.columns(2)
        with col1:
            st.metric("💰 Cash Balance", f"${account_info['cash']:,.2f}")
            st.metric("📊 Account Type", "Paper Trading")
        with col2:
            st.metric("📈 Portfolio Value", f"${account_info['portfolio_value']:,.2f}")
            st.metric("🎯 Active Strategies", "Momentum + Vol Arbitrage")
        
        # Show P&L
        total_value = account_info['cash'] + account_info['portfolio_value']
        pnl = total_value - 100000  # Starting amount
        pnl_pct = (pnl / 100000) * 100
        
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📊 Total Account Value", f"${total_value:,.2f}")
        with col2:
            st.metric("💹 Unrealized P&L", f"${pnl:,.2f}", f"{pnl_pct:+.2f}%")
        with col3:
            st.metric("🎲 Risk Level", "Moderate")
        
        # Show positions by strategy
        positions = paper_db.get_positions()
        if not positions.empty:
            st.divider()
            st.write("**📋 Current Positions by Strategy:**")
            
            # Group by strategy type
            if 'strategy_type' in positions.columns:
                momentum_pos = positions[positions['strategy_type'] == 'momentum']
                vol_arb_pos = positions[positions['strategy_type'] == 'vol_arbitrage']
                
                if not momentum_pos.empty:
                    st.write("*📈 Momentum Strategy Positions:*")
                    # Make the dataframe wider and more readable
                    display_df = momentum_pos[['symbol', 'quantity', 'avg_price', 'position_type']].copy()
                    display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"${x:.2f}")
                    st.dataframe(display_df, use_container_width=True, height=200)
                
                if not vol_arb_pos.empty:
                    st.write("*🔬 Volatility Arbitrage Positions:*")
                    display_df = vol_arb_pos[['symbol', 'quantity', 'avg_price', 'position_type']].copy()
                    display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"${x:.2f}")
                    st.dataframe(display_df, use_container_width=True, height=200)
            else:
                # Format the display dataframe for better readability
                display_df = positions[['symbol', 'quantity', 'avg_price', 'position_type']].copy()
                display_df['avg_price'] = display_df['avg_price'].apply(lambda x: f"${x:.2f}")
                st.dataframe(display_df, use_container_width=True, height=300)
        else:
            st.info("📝 No open positions - Ready for new opportunities!")
        
    except Exception as e:
        st.error(f"Could not load account info: {e}")
    
    # Strategy status - More organized layout
    st.divider()
    st.subheader("⏰ Strategy Execution Status")
    
    # Create a more organized status display
    status_col1, status_col2, status_col3 = st.columns(3)
    
    with status_col1:
        st.write("**📈 Momentum Strategy**")
        if 'last_momentum_run' in st.session_state:
            last_run = st.session_state.last_momentum_run.strftime("%H:%M:%S")
            st.success(f"✅ Last run: {last_run}")
        else:
            st.info("🔄 Never executed")
    
    with status_col2:
        st.write("**🔬 Volatility Arbitrage**")
        if 'last_vol_arb_run' in st.session_state:
            last_run = st.session_state.last_vol_arb_run.strftime("%H:%M:%S")
            st.success(f"✅ Last run: {last_run}")
        else:
            st.info("🔄 Never executed")
    
    with status_col3:
        st.write("**🚀 Combined Strategy**")
        if 'last_combined_run' in st.session_state:
            last_run = st.session_state.last_combined_run.strftime("%H:%M:%S")
            st.success(f"✅ Last run: {last_run}")
        else:
            st.info("🔄 Never executed")

elif password_guess != "":
    st.error("❌ Incorrect password. Access denied.")
else:
    st.info("Please enter the admin password to access trading bot controls.")

# Footer
st.divider()
st.caption("🔗 **Powered by Polygon.io** - Advanced volatility and momentum strategies")
st.caption("⚠️ **Paper Trading Only** - Educational volatility arbitrage simulation")