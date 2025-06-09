import streamlit as st
import alpaca_trade_api as tradeapi
from tiingo import TiingoClient
import pandas as pd
from datetime import datetime, timedelta
import math

def find_best_call_option(api, symbol, days_to_expiration=30, moneyness='ATM'):
    """
    Find the best call option to trade based on criteria
    """
    logs = []
    
    # Get current stock price
    quote = api.get_latest_trade(symbol)
    current_price = quote.price
    logs.append(f"Current {symbol} price: ${current_price:.2f}")
    
    # Calculate target expiration date range
    min_expiry = (datetime.now() + timedelta(days=days_to_expiration-10)).strftime('%Y-%m-%d')
    max_expiry = (datetime.now() + timedelta(days=days_to_expiration+10)).strftime('%Y-%m-%d')
    
    # Get available options contracts
    try:
        contracts = api.list_options_contracts(
            underlying_symbols=symbol,
            status='active',
            type='call',
            expiration_date_gte=min_expiry,
            expiration_date_lte=max_expiry
        )
        
        if not contracts:
            logs.append("No suitable contracts found")
            return None, logs
            
        # Filter for desired moneyness
        suitable_contracts = []
        
        for contract in contracts:
            strike = float(contract.strike_price)
            
            if moneyness == 'ATM':
                # At-the-money: strike within 2% of current price
                if abs(strike - current_price) / current_price < 0.02:
                    suitable_contracts.append(contract)
            elif moneyness == 'OTM':
                # Out-of-the-money: strike 2-5% above current price
                if 1.02 <= strike / current_price <= 1.05:
                    suitable_contracts.append(contract)
            elif moneyness == 'ITM':
                # In-the-money: strike 2-5% below current price
                if 0.95 <= strike / current_price <= 0.98:
                    suitable_contracts.append(contract)
        
        if not suitable_contracts:
            logs.append(f"No {moneyness} contracts found")
            return None, logs
            
        # Sort by expiration date and pick the closest to target
        suitable_contracts.sort(key=lambda x: x.expiration_date)
        selected_contract = suitable_contracts[0]
        
        logs.append(f"Selected contract: {selected_contract.symbol}")
        logs.append(f"Strike: ${selected_contract.strike_price}, Expiry: {selected_contract.expiration_date}")
        
        return selected_contract, logs
        
    except Exception as e:
        logs.append(f"Error finding contracts: {e}")
        return None, logs

def run_options_trade_logic():
    """
    Modified trade logic to use call options instead of stocks
    """
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
    except Exception as e:
        logs.append(f"❌ ERROR: Failed to connect to APIs. Error: {e}")
        return logs

    # 2. CHECK CURRENT POSITIONS
    UNDERLYING = 'NVDA'
    logs.append(f"Checking current options positions for {UNDERLYING}...")
    
    current_option_position = None
    try:
        positions = api.list_positions()
        for position in positions:
            # Check if this is an options position for our underlying
            asset = api.get_asset(position.symbol)
            if hasattr(asset, 'underlying_symbol') and asset.underlying_symbol == UNDERLYING:
                current_option_position = position
                logs.append(f"✅ Current position: {position.qty} contracts of {position.symbol}")
                break
    except Exception as e:
        logs.append(f"Error checking positions: {e}")
    
    if not current_option_position:
        logs.append("✅ No current options position")
    
    # 3. GET LATEST DATA & APPLY MOMENTUM LOGIC
    logs.append("Fetching latest market data...")
    data = tiingo_client.get_dataframe(UNDERLYING, frequency='daily', endDate=pd.Timestamp.now())
    latest_price = data['close'].iloc[-1]
    logs.append(f"Latest close price for {UNDERLYING}: ${latest_price:.2f}")

    # Calculate momentum signals (same as before)
    data['regime_ma'] = data['close'].rolling(window=200).mean()
    buffer = 0.02
    data['upper_band'] = data['regime_ma'] * (1 + buffer)
    data['lower_band'] = data['regime_ma'] * (1 - buffer)
    latest_upper_band = data['upper_band'].iloc[-1]
    latest_lower_band = data['lower_band'].iloc[-1]
    
    # 4. MAKE A DECISION
    desired_position = None
    if latest_price > latest_upper_band:
        desired_position = 'LONG'
        logs.append(f"DECISION: Price (${latest_price:.2f}) is above upper band (${latest_upper_band:.2f}). Signal is BULLISH.")
    elif latest_price < latest_lower_band:
        desired_position = 'CASH'
        logs.append(f"DECISION: Price (${latest_price:.2f}) is below lower band (${latest_lower_band:.2f}). Signal is BEARISH.")
    else:
        logs.append(f"DECISION: Price (${latest_price:.2f}) is inside buffer zone. Holding current position.")
    
    # 5. TAKE ACTION WITH OPTIONS
    if desired_position is not None:
        if desired_position == 'LONG' and not current_option_position:
            # Find and buy call options
            logs.append("ACTION: No position, signal is bullish. Looking for call options...")
            
            contract, contract_logs = find_best_call_option(
                api, 
                UNDERLYING, 
                days_to_expiration=45,  # 45-day options
                moneyness='ATM'  # At-the-money for balance of delta and premium
            )
            logs.extend(contract_logs)
            
            if contract:
                try:
                    # Get option quote
                    option_quote = api.get_latest_trade(contract.symbol)
                    option_price = option_quote.price
                    
                    # Calculate position size
                    RISK_AMOUNT = 5000  # Risk $5000 per trade
                    contracts_to_buy = math.floor(RISK_AMOUNT / (option_price * 100))
                    
                    logs.append(f"Option price: ${option_price:.2f}")
                    logs.append(f"Buying {contracts_to_buy} contracts (${contracts_to_buy * option_price * 100:.2f} total)")
                    
                    # Submit order
                    order = api.submit_order(
                        symbol=contract.symbol,
                        qty=contracts_to_buy,
                        side='buy',
                        type='limit',
                        time_in_force='day',
                        limit_price=option_price * 1.02  # Slightly above market for fill
                    )
                    logs.append(f"✅ Successfully submitted BUY order for {contracts_to_buy} call contracts")
                    
                except Exception as e:
                    logs.append(f"❌ ERROR submitting BUY order: {e}")
                    
        elif desired_position == 'CASH' and current_option_position:
            # Sell current call options
            logs.append(f"ACTION: Have {current_option_position.qty} contracts, signal is bearish. Closing position...")
            
            try:
                # Get current bid price
                option_quote = api.get_latest_quote(current_option_position.symbol)
                bid_price = option_quote.bid_price
                
                logs.append(f"Current bid: ${bid_price:.2f}")
                
                # Submit sell order
                order = api.submit_order(
                    symbol=current_option_position.symbol,
                    qty=current_option_position.qty,
                    side='sell',
                    type='limit',
                    time_in_force='day',
                    limit_price=bid_price * 0.98  # Slightly below bid for fill
                )
                logs.append(f"✅ Successfully submitted SELL order to close position")
                
            except Exception as e:
                logs.append(f"❌ ERROR submitting SELL order: {e}")
        else:
            logs.append("ACTION: Desired position matches current position. No trade needed.")
    
    # 6. CHECK FOR EXPIRATION
    if current_option_position:
        # Parse expiration from option symbol (format: NVDA240119C00140000)
        symbol = current_option_position.symbol
        expiry_str = symbol[len(UNDERLYING):len(UNDERLYING)+6]  # Extract YYMMDD
        expiry_date = datetime.strptime('20' + expiry_str, '%Y%m%d')
        days_to_expiry = (expiry_date - datetime.now()).days
        
        logs.append(f"Current position expires in {days_to_expiry} days")
        
        if days_to_expiry <= 5:
            logs.append("⚠️ WARNING: Option expires soon. Consider rolling to next month.")
            
    logs.append("--- Options bot logic finished. ---")
    return logs

# --- ENHANCED DASHBOARD FOR OPTIONS ---
def create_options_dashboard():
    st.header("Options Trading Dashboard")
    
    try:
        alpaca_key_id = st.secrets["alpaca"]["api_key_id"]
        alpaca_secret_key = st.secrets["alpaca"]["secret_key"]
        base_url = 'https://paper-api.alpaca.markets'
        api = tradeapi.REST(alpaca_key_id, alpaca_secret_key, base_url, api_version='v2')
        
        # Display current options positions
        st.subheader("Current Options Positions")
        positions = api.list_positions()
        options_positions = []
        
        for position in positions:
            asset = api.get_asset(position.symbol)
            if hasattr(asset, 'class') and asset.class == 'us_option':
                # Get current quote
                quote = api.get_latest_quote(position.symbol)
                
                # Calculate profit/loss
                avg_cost = float(position.avg_entry_price)
                current_price = float(quote.ask_price)
                pl = (current_price - avg_cost) * float(position.qty) * 100
                pl_percent = ((current_price - avg_cost) / avg_cost) * 100
                
                options_positions.append({
                    'Symbol': position.symbol,
                    'Underlying': asset.underlying_symbol,
                    'Type': 'Call' if 'C' in position.symbol else 'Put',
                    'Contracts': float(position.qty),
                    'Avg Cost': f"${avg_cost:.2f}",
                    'Current Price': f"${current_price:.2f}",
                    'P/L': f"${pl:,.2f}",
                    'P/L %': f"{pl_percent:.1f}%",
                    'Market Value': f"${float(position.market_value):,.2f}"
                })
        
        if options_positions:
            df = pd.DataFrame(options_positions)
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No options positions currently held")
            
        # Greeks display (if available)
        if st.checkbox("Show Greeks Analysis"):
            st.subheader("Position Greeks")
            # Note: Alpaca doesn't provide Greeks directly, but you could calculate them
            # using a library like py_vollib or mibian
            st.info("Greeks calculation requires additional libraries. This is a placeholder for delta, gamma, theta, vega display.")
            
    except Exception as e:
        st.error(f"Error loading options dashboard: {e}")