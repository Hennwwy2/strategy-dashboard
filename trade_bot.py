import configparser
import alpaca_trade_api as tradeapi
from tiingo import TiingoClient
import pandas as pd

# --- CONFIGURATION ---
STOCK_TO_TRADE = 'NVDA' # The stock we want our bot to trade
USD_TO_TRADE = 10000     # The amount of USD to use for a BUY order

def run_trading_bot():
    """
    This function connects to the APIs, gets data, and makes a trade decision.
    """
    print("--- Starting Trading Bot ---")
    
    # 1. CONNECT TO APIS
    # -------------------
    print("Connecting to APIs...")
    config = configparser.ConfigParser()
    config.read('config.ini')

    # Get Alpaca keys
    alpaca_key_id = config['alpaca']['api_key_id']
    alpaca_secret_key = config['alpaca']['secret_key']
    # IMPORTANT: This URL is for paper trading
    base_url = 'https://paper-api.alpaca.markets' 
    
    # Get Tiingo key
    tiingo_key = config['tiingo']['api_key']

    # Instantiate API clients
    alpaca_api = tradeapi.REST(alpaca_key_id, alpaca_secret_key, base_url, api_version='v2')
    tiingo_config = {'api_key': tiingo_key, 'session': True}
    tiingo_client = TiingoClient(tiingo_config)
    
    # 2. CHECK CURRENT POSITION
    # -------------------------
    print(f"Checking current position for {STOCK_TO_TRADE}...")
    try:
        position = alpaca_api.get_position(STOCK_TO_TRADE)
        current_qty = float(position.qty)
        print(f"Current position: {current_qty} shares.")
    except tradeapi.rest.APIError:
        # This error means we have no position in the stock
        current_qty = 0
        print("Current position: 0 shares.")
        
    # 3. GET LATEST DATA
    # -------------------
    print("Fetching latest market data from Tiingo...")
    # Fetching ~1 year of data to ensure we have enough for a 200-day MA
    data = tiingo_client.get_dataframe(STOCK_TO_TRADE, frequency='daily', endDate=pd.Timestamp.now())
    latest_price = data['close'].iloc[-1]
    print(f"Latest close price for {STOCK_TO_TRADE}: ${latest_price:.2f}")

    # 4. APPLY STRATEGY LOGIC
    # ------------------------
    print("Applying strategy logic...")
    data['regime_ma'] = data['close'].rolling(window=200).mean()
    buffer = 0.02
    data['upper_band'] = data['regime_ma'] * (1 + buffer)
    data['lower_band'] = data['regime_ma'] * (1 - buffer)

    # Get the most recent values for our logic
    latest_upper_band = data['upper_band'].iloc[-1]
    latest_lower_band = data['lower_band'].iloc[-1]
    
    # 5. MAKE A DECISION
    # -------------------
    desired_position = None # 1 for "IN", 0 for "OUT"
    if latest_price > latest_upper_band:
        desired_position = 1
        print(f"Decision: Price is above upper band. Desired position is IN.")
    elif latest_price < latest_lower_band:
        desired_position = 0
        print(f"Decision: Price is below lower band. Desired position is OUT.")
    else:
        # Inside the buffer zone, do not change position
        print("Decision: Price is inside buffer zone. Holding current position.")

    # 6. TAKE ACTION (OR NOT)
    # ------------------------
    if desired_position is not None: # Only act if we are not in the buffer zone
        if desired_position == 1 and current_qty == 0:
            print("Action: Current position is 0, desired is IN. Submitting BUY order.")
            qty_to_buy = round(USD_TO_TRADE / latest_price, 4) # Calculate shares to buy
            try:
                alpaca_api.submit_order(
                    symbol=STOCK_TO_TRADE,
                    qty=qty_to_buy,
                    side='buy',
                    type='market',
                    time_in_force='day'
                )
                print(f"Successfully submitted BUY order for {qty_to_buy} shares.")
            except Exception as e:
                print(f"Error submitting BUY order: {e}")

        elif desired_position == 0 and current_qty > 0:
            print(f"Action: Current position is {current_qty}, desired is OUT. Submitting SELL order.")
            try:
                # Liquidate the entire position
                alpaca_api.close_position(STOCK_TO_TRADE)
                print(f"Successfully submitted SELL order for {current_qty} shares.")
            except Exception as e:
                print(f"Error submitting SELL order: {e}")
        else:
            print("Action: Desired position matches current position. No trade needed.")
            
    print("--- Trading Bot Finished ---")


# --- Main execution block ---
if __name__ == '__main__':
    run_trading_bot()