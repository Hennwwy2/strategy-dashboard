# pages/1_🤖_Tradebot_Controls.py
import streamlit as st
import alpaca_trade_api as tradeapi
from tiingo import TiingoClient
import pandas as pd
from datetime import datetime, timedelta
import math

def run_trade_logic():
    """Main trading logic function"""
    logs = []
    
    # 1. CONNECT TO APIS
    logs.append("Connecting to APIs...")
    try:
        alpaca_key_id = st.secrets["alpaca"]["api_key_id"]
        alpaca_secret_key = st.secrets["alpaca"]["secret_key"]
        tiingo_key = st.secrets["tiingo"]["api_key"]
        base_url = 'https://paper-api.alpaca.markets'
        api = tradeapi.REST(alpaca_key_id, alpaca_secret_key, base_url, api_version='v2')
        tiingo_config = {'api_key': tiingo_key, 'session': True}
        tiingo_client = TiingoClient(tiingo_config)
        logs.append("✅ Successfully connected to APIs")
    except Exception as e:
        logs.append(f"❌ ERROR: Failed to connect to APIs. Check secrets. Error: {e}")
        return logs

    # 2. CHECK CURRENT POSITION
    STOCK_TO_TRADE = 'NVDA'
    logs.append(f"Checking current position for {STOCK_TO_TRADE}...")
    try:
        position = api.get_position(STOCK_TO_TRADE)
        current_qty = float(position.qty)
        logs.append(f"✅ Current position: {current_qty} shares.")
    except Exception as e:
        # Check if it's just "position not found" error
        if "404" in str(e) or "not found" in str(e).lower():
            current_qty = 0
            logs.append("✅ Current position: 0 shares.")
        else:
            logs.append(f"❌ ERROR checking position: {e}")
            return logs
    
    # 3. GET LATEST DATA & APPLY LOGIC
    logs.append("Fetching latest market data...")
    try:
        end_date = pd.Timestamp.now()
        start_date = end_date - pd.Timedelta(days=250)  # Get enough data for 200-day MA
        
        data = tiingo_client.get_dataframe(
            STOCK_TO_TRADE, 
            frequency='daily', 
            startDate=start_date.strftime('%Y-%m-%d'),
            endDate=end_date.strftime('%Y-%m-%d')
        )
        
        if len(data) < 200:
            logs.append(f"❌ ERROR: Not enough data. Only {len(data)} days available.")
            return logs
            
        latest_price = data['close'].iloc[-1]
        logs.append(f"Latest close price for {STOCK_TO_TRADE}: ${latest_price:.2f}")

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
    if latest_price > latest_upper_band:
        desired_position = 1
        logs.append(f"📈 DECISION: Price (${latest_price:.2f}) is above upper band (${latest_upper_band:.2f}). Desired position is IN.")
    elif latest_price < latest_lower_band:
        desired_position = 0
        logs.append(f"📉 DECISION: Price (${latest_price:.2f}) is below lower band (${latest_lower_band:.2f}). Desired position is OUT.")
    else:
        logs.append(f"➡️ DECISION: Price (${latest_price:.2f}) is inside buffer zone. Holding current position.")

    # 5. TAKE ACTION
    if desired_position is not None:
        if desired_position == 1 and current_qty == 0:
            # BUY
            USD_TO_TRADE = 10000
            qty_to_buy = int(USD_TO_TRADE / latest_price)  # Use int() instead of round()
            
            logs.append(f"💰 ACTION: Current position is 0, desired is IN.")
            logs.append(f"Calculating order: ${USD_TO_TRADE} / ${latest_price:.2f} = {qty_to_buy} shares")
            logs.append(f"Submitting BUY order for {qty_to_buy} shares...")
            
            try:
                order = api.submit_order(
                    symbol=STOCK_TO_TRADE, 
                    qty=qty_to_buy, 
                    side='buy', 
                    type='market', 
                    time_in_force='day'
                )
                logs.append(f"✅ Successfully submitted BUY order. Order ID: {order.id}")
            except Exception as e:
                logs.append(f"❌ ERROR submitting BUY order: {e}")
                
        elif desired_position == 0 and current_qty > 0:
            # SELL
            logs.append(f"💸 ACTION: Current position is {current_qty}, desired is OUT.")
            logs.append(f"Submitting SELL order to liquidate position...")
            
            try:
                order = api.close_position(STOCK_TO_TRADE)
                logs.append(f"✅ Successfully submitted SELL order to close position.")
            except Exception as e:
                logs.append(f"❌ ERROR submitting SELL order: {e}")
        else:
            logs.append("✅ ACTION: Desired position matches current position. No trade needed.")
    else:
        logs.append("✅ No action needed - price is in buffer zone.")
            
    logs.append("--- Bot logic finished. ---")
    return logs

def run_options_trade_logic():
    """Options trading logic"""
    logs = []
    
    # 1. CONNECT TO APIS
    logs.append("Connecting to APIs for options trading...")
    try:
        alpaca_key_id = st.secrets["alpaca"]["api_key_id"]
        alpaca_secret_key = st.secrets["alpaca"]["secret_key"]
        tiingo_key = st.secrets["tiingo"]["api_key"]
        base_url = 'https://paper-api.alpaca.markets'
        api = tradeapi.REST(alpaca_key_id, alpaca_secret_key, base_url, api_version='v2')
        tiingo_config = {'api_key': tiingo_key, 'session': True}
        tiingo_client = TiingoClient(tiingo_config)
        logs.append("✅ Successfully connected to APIs")
    except Exception as e:
        logs.append(f"❌ ERROR: Failed to connect to APIs. Error: {e}")
        return logs

    # 2. CHECK CURRENT OPTIONS POSITIONS
    UNDERLYING = 'NVDA'
    logs.append(f"Checking current options positions for {UNDERLYING}...")
    
    current_option_position = None
    try:
        positions = api.list_positions()
        for position in positions:
            # Check if this is an options position for our underlying
            try:
                asset = api.get_asset(position.symbol)
                # Check if it's an option by looking for underscore in symbol (option format)
                if UNDERLYING in position.symbol and '_' in position.symbol:
                    current_option_position = position
                    logs.append(f"✅ Current position: {position.qty} contracts of {position.symbol}")
                    break
            except:
                continue
    except Exception as e:
        logs.append(f"Warning checking positions: {e}")
    
    if not current_option_position:
        logs.append("✅ No current options position")
    
    # Similar momentum logic as stocks but for options...
    logs.append("Options trading logic would continue here...")
    logs.append("--- Options bot logic finished. ---")
    
    return logs

# --- STREAMLIT PAGE UI ---
st.set_page_config(initial_sidebar_state="collapsed")
st.title("🤖 Tradebot Master Controls")

# Password protection
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

password_guess = st.text_input("Enter Admin Password", type="password", key="admin_pass")

# Get admin password from secrets or use empty string for local testing
try:
    admin_password = st.secrets["admin_password"]
except:
    admin_password = "admin123"  # Default for local testing

if password_guess == admin_password and admin_password != "":
    st.session_state.authenticated = True

if st.session_state.authenticated:
    st.success("✅ Access Granted")
    st.write("Welcome to the control panel. Use the buttons below to manually trigger trading logic.")
    
    # Choose trading mode
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Stock Trading Bot")
        st.write("Traditional stock trading using momentum strategy")
        
        if st.button("🏃 Run Stock Trade Check", use_container_width=True):
            with st.spinner("Bot is analyzing stock positions..."):
                returned_logs = run_trade_logic()
                
            st.subheader("Stock Bot Activity Log:")
            # Display logs in a code block
            log_text = "\n".join(returned_logs)
            st.code(log_text)
            
            # Also save to session state
            st.session_state.last_stock_run = datetime.now()
            st.session_state.last_stock_logs = returned_logs
    
    with col2:
        st.subheader("Options Trading Bot")
        st.write("Options trading using same momentum signals")
        
        if st.button("🎯 Run Options Trade Check", use_container_width=True):
            with st.spinner("Bot is analyzing options positions..."):
                returned_logs = run_options_trade_logic()
                
            st.subheader("Options Bot Activity Log:")
            log_text = "\n".join(returned_logs)
            st.code(log_text)
            
            st.session_state.last_options_run = datetime.now()
            st.session_state.last_options_logs = returned_logs
    
    # Show last run info
    st.divider()
    st.subheader("📊 Bot Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if 'last_stock_run' in st.session_state:
            st.metric("Last Stock Check", st.session_state.last_stock_run.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            st.metric("Last Stock Check", "Never run")
    
    with col2:
        if 'last_options_run' in st.session_state:
            st.metric("Last Options Check", st.session_state.last_options_run.strftime("%Y-%m-%d %H:%M:%S"))
        else:
            st.metric("Last Options Check", "Never run")
    
    # Manual order section
    st.divider()
    with st.expander("🔧 Manual Controls (Advanced)"):
        st.warning("⚠️ Manual orders bypass all safety checks!")
        
        manual_symbol = st.text_input("Symbol:", "NVDA")
        manual_side = st.radio("Side:", ["buy", "sell"], horizontal=True)
        manual_qty = st.number_input("Quantity:", min_value=1, value=10)
        manual_type = st.selectbox("Order Type:", ["market", "limit"])
        
        if manual_type == "limit":
            manual_limit_price = st.number_input("Limit Price:", min_value=0.01, value=100.00)
        
        if st.button("Submit Manual Order", type="secondary"):
            st.info(f"Manual order would be placed here: {manual_side} {manual_qty} {manual_symbol}")

elif password_guess != "":
    st.error("❌ Incorrect password. Access denied.")
else:
    st.info("Please enter the admin password to access tradebot controls.")

# Footer
st.divider()
st.caption("Remember: This bot trades real money (or paper money in test mode). Always monitor its actions!")