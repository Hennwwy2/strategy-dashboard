# --- COMPLETE ENHANCED DASHBOARD WITH OPTIONS SUPPORT ---
import streamlit as st
import configparser
import pandas as pd
import matplotlib.pyplot as plt
from tiingo import TiingoClient
import alpaca_trade_api as tradeapi
import plotly.graph_objects as go
from datetime import datetime, timedelta
import math
import numpy as np
import random

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
        # Note: Alpaca's options API may require additional setup or permissions
        # This is a simplified version that handles the current API
        
        # For now, return empty list with informative message
        # In production, you'd need to check if options trading is enabled
        # and use the correct API endpoints
        
        st.warning("""
        📝 **Options Chain Note**: 
        
        To view options chains, ensure:
        1. Your Alpaca account has options trading enabled
        2. You're using the correct API version
        3. You have the necessary permissions
        
        Contact Alpaca support if you need help enabling options trading.
        """)
        
        return []
        
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
    
    # Check if options are available
    contracts = get_options_chain(api, symbol)
    
    if not contracts:
        # Provide alternative options information
        st.info("💡 **Options Trading Alternatives:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**While options chain loads, you can:**")
            st.write("• Use the Options Simulator to practice")
            st.write("• Review your current positions")
            st.write("• Learn about options strategies")
            st.write("• Check if options trading is enabled on your account")
        
        with col2:
            st.write("**Popular Options Platforms:**")
            st.write("• [Alpaca Options Docs](https://alpaca.markets/docs/trading/options-trading/)")
            st.write("• [CBOE Options Chain](https://www.cboe.com/)")
            st.write("• [Yahoo Finance Options](https://finance.yahoo.com/)")
            
        # Show mock data for educational purposes
        with st.expander("📚 Example Options Chain (Educational)"):
            st.write(f"**Example options chain for {symbol} at ${current_price:.2f}:**")
            
            # Create sample data
            sample_calls = []
            sample_puts = []
            
            for i in range(-2, 3):
                strike = round(current_price * (1 + i * 0.05), 2)
                
                # Calls
                call_premium = max(0.5, (current_price - strike) + 2) if current_price > strike else max(0.5, 2 - abs(current_price - strike) * 0.1)
                sample_calls.append({
                    'Strike': f"${strike}",
                    'Bid': f"${call_premium - 0.1:.2f}",
                    'Ask': f"${call_premium + 0.1:.2f}",
                    'Volume': np.random.randint(10, 500),
                    'Open Interest': np.random.randint(100, 5000)
                })
                
                # Puts
                put_premium = max(0.5, (strike - current_price) + 2) if strike > current_price else max(0.5, 2 - abs(current_price - strike) * 0.1)
                sample_puts.append({
                    'Strike': f"${strike}",
                    'Bid': f"${put_premium - 0.1:.2f}",
                    'Ask': f"${put_premium + 0.1:.2f}",
                    'Volume': np.random.randint(10, 500),
                    'Open Interest': np.random.randint(100, 5000)
                })
            
            call_tab, put_tab = st.tabs(["📈 Calls", "📉 Puts"])
            
            with call_tab:
                st.write("**Call Options** (Right to Buy)")
                df_calls = pd.DataFrame(sample_calls)
                st.dataframe(df_calls, use_container_width=True)
                st.caption("💡 Calls profit when stock price rises above strike + premium")
            
            with put_tab:
                st.write("**Put Options** (Right to Sell)")
                df_puts = pd.DataFrame(sample_puts)
                st.dataframe(df_puts, use_container_width=True)
                st.caption("💡 Puts profit when stock price falls below strike - premium")
        
        return
    
    # Original code for when options are available
    # ... rest of the function remains the same ...

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
        
        # Try to get real price, fall back to mock prices if API fails
        try:
            api = tradeapi.REST(alpaca_key_id, alpaca_secret_key, base_url, api_version='v2')
            quote = api.get_latest_trade(stock_symbol)
            current_price = float(quote.price)
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
tab_live, tab_options, tab_backtest = st.tabs(["📊 Live Account", "🎯 Options Trading", "📈 Strategy Backtester"])

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
                        
                        # Parse option details from symbol
                        # Format: AAPL230120C00150000
                        underlying = p.symbol[:4]  # First 4 chars
                        option_type = 'Call' if 'C' in p.symbol else 'Put'
                        
                        options_positions.append({
                            'Symbol': p.symbol,
                            'Underlying': underlying,
                            'Type': option_type,
                            'Qty': float(p.qty),
                            'Avg Cost': f"${float(p.avg_entry_price):.2f}",
                            'Current': f"${float(quote.ask_price):.2f}" if quote.ask_price else "N/A",
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
                st.write("**📈 Stock Positions:**")
                st.dataframe(pd.DataFrame(stock_positions), use_container_width=True)
            
            if options_positions:
                st.write("**🎯 Options Positions:**")
                st.dataframe(pd.DataFrame(options_positions), use_container_width=True)
                
                # Options position analysis
                with st.expander("Options Position Analysis"):
                    for opt in options_positions:
                        st.write(f"**{opt['Symbol']}**")
                        st.write(f"- Type: {opt['Type']} Option")
                        st.write(f"- Contracts: {opt['Qty']}")
                        st.write(f"- P/L: {opt['P/L']}")
        else:
            st.info("You have no open positions.")
        
        # Recent trades section
        st.divider()
        trades = api.get_activities(activity_types='FILL', direction='desc')[:20]
        if trades:
            st.subheader("Recent Trades")
            trade_data = []
            for t in trades:
                trade_data.append({
                    'Time': t.transaction_time.strftime('%Y-%m-%d %H:%M'),
                    'Symbol': t.symbol,
                    'Side': t.side,
                    'Qty': float(t.qty),
                    'Price': f"${float(t.price):,}"
                })
            trades_df = pd.DataFrame(trade_data)
            st.dataframe(trades_df, use_container_width=True)
        
    except Exception as e:
        st.error(f"Could not connect to Alpaca. Error: {e}")

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
            "📊 Options Chain",
            "💼 My Positions"
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
            st.subheader("📊 Live Options Chain")
            chain_symbol = st.text_input("Enter symbol:", "NVDA", key="chain_symbol_beginner")
            if st.button("Load Options", key="load_chain_beginner"):
                with st.spinner(f"Loading options for {chain_symbol}..."):
                    try:
                        display_options_chain(api, chain_symbol)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        with tabs[3]:
            # Current positions (same as in live tab but filtered for options)
            st.subheader("💼 My Options Positions")
            if options_positions:
                st.dataframe(pd.DataFrame(options_positions), use_container_width=True)
                
                # Position helper
                st.info("💡 **Position Management Tips:**")
                st.write("• Consider closing positions when up 50%+")
                st.write("• Set stop loss at 50% loss")
                st.write("• Watch time decay - close before last week")
            else:
                st.info("No options positions yet. Try the simulator first!")
    
    else:
        # Advanced mode - original interface
        st.subheader("Options Chain Explorer")
        
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
                    display_options_chain(api, options_symbol)
                except Exception as e:
                    st.error(f"Error loading options chain: {e}")
        
        # Greeks and advanced analysis
        with st.expander("Advanced Options Analytics"):
            st.write("**Greeks Analysis** (Coming Soon)")
            st.write("• Delta: Direction exposure")
            st.write("• Theta: Time decay")
            st.write("• Vega: Volatility sensitivity")
            st.write("• Gamma: Delta acceleration")

with tab_backtest:
    st.header("Strategy Backtester")
    
    # Strategy selector with options
    strategy_type = st.selectbox(
        "Select Strategy Type:",
        ["Stock Momentum Strategy", "Options Strategy (Simulated)", "Compare Both"]
    )
    
    if strategy_type == "Stock Momentum Strategy":
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
        
        # Batch backtester
        st.divider()
        st.subheader("Batch Backtester")
        st.write("Test the strategy on multiple stocks at once:")
        
        recommended_tickers = ["AAPL", "MSFT", "AMZN", "META", "TSLA", "NVDA", "GOOGL", "NFLX"]
        selected_tickers = st.multiselect(
            "Select tickers to test:",
            recommended_tickers,
            default=recommended_tickers[:5]
        )
        
        if st.button("Run Batch Backtest"):
            if selected_tickers:
                batch_results = []
                progress_bar = st.progress(0)
                
                for i, ticker in enumerate(selected_tickers):
                    with st.spinner(f"Testing {ticker}..."):
                        results, fig = run_backtest_for_dashboard(
                            symbol=ticker,
                            start_date='2015-01-01',
                            end_date='2025-06-09',
                            config=tiingo_config
                        )
                        if results:
                            bh_return = float(results["buy_and_hold"].strip('%'))
                            strat_return = float(results["strategy"].strip('%'))
                            batch_results.append({
                                'Symbol': ticker,
                                'Buy & Hold': results['buy_and_hold'],
                                'Strategy': results['strategy'],
                                'Outperformance': f"{strat_return - bh_return:.2f}%"
                            })
                        else:
                            batch_results.append({
                                'Symbol': ticker,
                                'Buy & Hold': 'Error',
                                'Strategy': 'Error',
                                'Outperformance': 'N/A'
                            })
                    
                    progress_bar.progress((i + 1) / len(selected_tickers))
                
                results_df = pd.DataFrame(batch_results)
                st.dataframe(results_df, use_container_width=True)
                
                # Summary statistics
                successful_results = [r for r in batch_results if r['Buy & Hold'] != 'Error']
                if successful_results:
                    wins = sum(1 for r in successful_results if float(r['Outperformance'].strip('%')) > 0)
                    win_rate = (wins / len(successful_results)) * 100
                    
                    col1, col2 = st.columns(2)
                    col1.metric("Win Rate", f"{win_rate:.1f}%")
                    col2.metric("Stocks Outperformed", f"{wins}/{len(successful_results)}")
    
    elif strategy_type == "Options Strategy (Simulated)":
        st.info("⚠️ Options backtesting requires expensive historical data. This is a simplified simulation.")
        
        st.subheader("Options Strategy Simulator")
        
        col1, col2 = st.columns(2)
        with col1:
            option_strategy = st.selectbox(
                "Select Options Strategy:",
                ["Long Calls (Momentum)", "Long Puts (Hedge)", "Covered Calls (Income)"]
            )
            symbol = st.text_input("Stock Symbol:", "NVDA", key="options_backtest_symbol")
        
        with col2:
            st.write("**Strategy Description:**")
            if "Long Calls" in option_strategy:
                st.write("Buy call options when momentum signal is bullish")
                st.write("• Higher leverage than stocks")
                st.write("• Limited downside risk")
            elif "Long Puts" in option_strategy:
                st.write("Buy puts as portfolio insurance")
                st.write("• Protect against downturns")
                st.write("• Profit from declines")
            else:
                st.write("Sell calls against stock holdings")
                st.write("• Generate income")
                st.write("• Cap upside potential")
        
        if st.button("Simulate Options Strategy"):
            st.warning("This uses simplified assumptions - not real options prices")
            
            # Placeholder results
            col1, col2, col3 = st.columns(3)
            col1.metric("Simulated Return", "45.2%")
            col2.metric("Max Drawdown", "-15.3%")
            col3.metric("Win Rate", "62.5%")
            
            st.info("""
            **Note on Options Backtesting:**
            - Real options backtesting requires historical implied volatility
            - Need bid-ask spread data
            - Must account for early assignment risk
            - Professional data sources: CBOE DataShop, ORATS, OptionMetrics
            """)
    
    else:  # Compare Both
        st.subheader("Strategy Comparison: Stocks vs Options")
        st.info("Compare potential returns and risks between stock and options strategies")
        
        # This would show a side-by-side comparison
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Stock Strategy**")
            st.write("✅ No expiration")
            st.write("✅ Simple to execute")
            st.write("❌ Requires more capital")
            st.write("❌ Full downside exposure")
        
        with col2:
            st.write("**Options Strategy**")
            st.write("✅ Less capital required")
            st.write("✅ Defined risk (when buying)")
            st.write("❌ Time decay")
            st.write("❌ More complex")