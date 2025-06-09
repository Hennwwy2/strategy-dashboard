import streamlit as st
import configparser
import alpaca_trade_api as tradeapi
from tiingo import TiingoClient
import pandas as pd

# --- We move the bot logic into a function that returns logs ---
def run_trade_logic():
    logs = [] # A list to store log messages
    
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
    except Exception as e:
        logs.append(f"❌ ERROR: Failed to connect to APIs. Check secrets. Error: {e}")
        return logs

    # 2. CHECK CURRENT POSITION
    STOCK_TO_TRADE = 'NVDA' # The stock our bot manages
    logs.append(f"Checking current position for {STOCK_TO_TRADE}...")
    try:
        position = api.get_position(STOCK_TO_TRADE)
        current_qty = float(position.qty)
        logs.append(f"✅ Current position: {current_qty} shares.")
    except tradeapi.rest.APIError:
        current_qty = 0
        logs.append("✅ Current position: 0 shares.")
    
    # 3. GET LATEST DATA & APPLY LOGIC
    logs.append("Fetching latest market data...")
    data = tiingo_client.get_dataframe(STOCK_TO_TRADE, frequency='daily', endDate=pd.Timestamp.now())
    latest_price = data['close'].iloc[-1]
    logs.append(f"Latest close price for {STOCK_TO_TRADE}: ${latest_price:.2f}")

    data['regime_ma'] = data['close'].rolling(window=200).mean()
    buffer = 0.02
    data['upper_band'] = data['regime_ma'] * (1 + buffer)
    data['lower_band'] = data['regime_ma'] * (1 - buffer)
    latest_upper_band = data['upper_band'].iloc[-1]
    latest_lower_band = data['lower_band'].iloc[-1]
    
    # 4. MAKE A DECISION
    desired_position = None
    if latest_price > latest_upper_band:
        desired_position = 1
        logs.append(f"DECISION: Price (${latest_price:.2f}) is above upper band (${latest_upper_band:.2f}). Desired position is IN.")
    elif latest_price < latest_lower_band:
        desired_position = 0
        logs.append(f"DECISION: Price (${latest_price:.2f}) is below lower band (${latest_lower_band:.2f}). Desired position is OUT.")
    else:
        logs.append(f"DECISION: Price (${latest_price:.2f}) is inside buffer zone. Holding current position.")

    # 5. TAKE ACTION
    if desired_position is not None:
        if desired_position == 1 and current_qty == 0:
            USD_TO_TRADE = 10000
            qty_to_buy = round(USD_TO_TRADE / latest_price, 4)
            logs.append(f"ACTION: Current position is 0, desired is IN. Submitting BUY for {qty_to_buy} shares.")
            try:
                api.submit_order(symbol=STOCK_TO_TRADE, qty=qty_to_buy, side='buy', type='market', time_in_force='day')
                logs.append("✅ Successfully submitted BUY order.")
            except Exception as e:
                logs.append(f"❌ ERROR submitting BUY order: {e}")
        elif desired_position == 0 and current_qty > 0:
            logs.append(f"ACTION: Current position is {current_qty}, desired is OUT. Submitting SELL to liquidate.")
            try:
                api.close_position(STOCK_TO_TRADE)
                logs.append(f"✅ Successfully submitted SELL order.")
            except Exception as e:
                logs.append(f"❌ ERROR submitting SELL order: {e}")
        else:
            logs.append("ACTION: Desired position matches current position. No trade needed.")
            
    logs.append("--- Bot logic finished. ---")
    return logs

# --- STREAMLIT PAGE UI ---

st.title("Tradebot Master Controls")

# Password protection
password_guess = st.text_input("Enter Admin Password", type="password")
if password_guess == st.secrets["admin_password"]:
    st.success("✅ Access Granted")
    st.write("Welcome to the control panel. Use the button below to manually trigger a run of the trade bot's logic.")
    
    if st.button("Run Trade Check Now"):
        with st.spinner("Bot is running..."):
            # Run the logic and get the logs
            returned_logs = run_trade_logic()
            
            # Display the logs in a code block
            st.subheader("Bot Activity Log:")
            st.code("\n".join(returned_logs))

elif password_guess != "":
    st.error("❌ Incorrect password.")