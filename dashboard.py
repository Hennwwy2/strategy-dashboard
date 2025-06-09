import streamlit as st
import alpaca_trade_api as tradeapi
from datetime import datetime, timedelta
import pandas as pd

def guided_options_order_form(api):
    """Step-by-step options order form for beginners"""
    st.header("🎯 Place an Options Trade - Guided Mode")
    
    # Initialize session state for multi-step form
    if 'order_step' not in st.session_state:
        st.session_state.order_step = 1
    
    # Progress bar
    progress = st.session_state.order_step / 6
    st.progress(progress)
    st.write(f"Step {st.session_state.order_step} of 6")
    
    # Step 1: Choose Direction
    if st.session_state.order_step == 1:
        st.subheader("Step 1: What's Your Market View? 🤔")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📈 Bullish\n(Stock will go up)", use_container_width=True):
                st.session_state.direction = "bullish"
                st.session_state.order_step = 2
                st.rerun()
        
        with col2:
            if st.button("📉 Bearish\n(Stock will go down)", use_container_width=True):
                st.session_state.direction = "bearish"
                st.session_state.order_step = 2
                st.rerun()
        
        with col3:
            if st.button("😴 Neutral\n(Stock won't move much)", use_container_width=True):
                st.session_state.direction = "neutral"
                st.session_state.order_step = 2
                st.rerun()
        
        st.info("💡 Tip: If you're unsure, start with 'Bullish' and buy a call option on a stock you like")
    
    # Step 2: Choose Stock
    elif st.session_state.order_step == 2:
        st.subheader("Step 2: Choose Your Stock 📊")
        
        # Beginner-friendly stock suggestions
        st.write("**Suggested stocks for beginners (liquid options):**")
        
        popular_stocks = {
            "SPY": "S&P 500 ETF - Entire market",
            "AAPL": "Apple - Tech giant",
            "MSFT": "Microsoft - Stable tech",
            "QQQ": "Nasdaq ETF - Tech focused",
            "IWM": "Russell 2000 - Small caps"
        }
        
        cols = st.columns(len(popular_stocks))
        for i, (symbol, desc) in enumerate(popular_stocks.items()):
            with cols[i]:
                if st.button(f"{symbol}\n{desc}", use_container_width=True):
                    st.session_state.symbol = symbol
                    st.session_state.order_step = 3
                    st.rerun()
        
        st.divider()
        
        # Custom input
        custom_symbol = st.text_input("Or enter any stock symbol:")
        if st.button("Continue with custom symbol"):
            if custom_symbol:
                st.session_state.symbol = custom_symbol.upper()
                st.session_state.order_step = 3
                st.rerun()
        
        if st.button("← Back"):
            st.session_state.order_step = 1
            st.rerun()
    
    # Step 3: Choose Strategy
    elif st.session_state.order_step == 3:
        st.subheader(f"Step 3: Choose Your Strategy for {st.session_state.symbol} 🎲")
        
        # Get current price (mock for demo)
        current_price = 100  # In real app, fetch from API
        
        if st.session_state.direction == "bullish":
            st.success("📈 You're bullish! Here are your options:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Buy Call Option** (Recommended)")
                st.write("✅ Limited risk (can only lose premium)")
                st.write("✅ Unlimited profit potential")
                st.write("✅ Lower capital required")
                if st.button("Choose Call Option", use_container_width=True):
                    st.session_state.option_type = "call"
                    st.session_state.action = "buy"
                    st.session_state.order_step = 4
                    st.rerun()
            
            with col2:
                st.warning("**Buy Stock**")
                st.write("❌ Higher capital required")
                st.write("❌ Can lose entire investment")
                st.write("✅ No expiration")
                if st.button("Buy Stock Instead", use_container_width=True):
                    st.info("Redirecting to stock trading...")
        
        elif st.session_state.direction == "bearish":
            st.error("📉 You're bearish! Here are your options:")
            
            col1, col2 = st.columns(2)
            with col1:
                st.info("**Buy Put Option** (Recommended)")
                st.write("✅ Profit when stock falls")
                st.write("✅ Limited risk")
                st.write("✅ Like insurance for your portfolio")
                if st.button("Choose Put Option", use_container_width=True):
                    st.session_state.option_type = "put"
                    st.session_state.action = "buy"
                    st.session_state.order_step = 4
                    st.rerun()
            
            with col2:
                st.warning("**Short Stock** (Advanced)")
                st.write("❌ Unlimited risk")
                st.write("❌ Requires margin account")
                st.write("❌ Not for beginners")
                st.button("Too Risky!", disabled=True)
        
        else:  # neutral
            st.info("😴 You're neutral! Consider these income strategies:")
            st.warning("⚠️ Selling options is advanced - start with buying first!")
            
        if st.button("← Back"):
            st.session_state.order_step = 2
            st.rerun()
    
    # Step 4: Choose Expiration
    elif st.session_state.order_step == 4:
        st.subheader("Step 4: Choose Expiration Date 📅")
        
        st.info("💡 Tip: Give yourself time to be right! 30-60 days is ideal for beginners")
        
        # Calculate expiration options
        today = datetime.now()
        expiration_options = []
        
        # Weekly expirations for next 8 weeks
        for weeks in [1, 2, 3, 4, 6, 8]:
            exp_date = today + timedelta(weeks=weeks)
            # Options expire on Fridays
            days_until_friday = (4 - exp_date.weekday()) % 7
            exp_date = exp_date + timedelta(days=days_until_friday)
            expiration_options.append({
                'date': exp_date,
                'label': f"{exp_date.strftime('%b %d, %Y')} ({weeks} weeks)",
                'days': (exp_date - today).days
            })
        
        # Display as cards
        cols = st.columns(3)
        for i, exp in enumerate(expiration_options):
            with cols[i % 3]:
                if exp['days'] < 14:
                    card_color = "🔴"
                    risk_level = "High Risk"
                elif exp['days'] < 30:
                    card_color = "🟡"
                    risk_level = "Medium Risk"
                else:
                    card_color = "🟢"
                    risk_level = "Lower Risk"
                
                if st.button(
                    f"{card_color} {exp['label']}\n{risk_level}", 
                    use_container_width=True,
                    key=f"exp_{i}"
                ):
                    st.session_state.expiration = exp['date']
                    st.session_state.order_step = 5
                    st.rerun()
        
        st.warning("⏰ Remember: Options lose value over time (theta decay)")
        
        if st.button("← Back"):
            st.session_state.order_step = 3
            st.rerun()
    
    # Step 5: Choose Strike Price
    elif st.session_state.order_step == 5:
        st.subheader("Step 5: Choose Strike Price 🎯")
        
        # Mock current price
        current_price = 100
        st.metric("Current Stock Price", f"${current_price:.2f}")
        
        # Calculate strike options
        if st.session_state.option_type == "call":
            strike_options = [
                {
                    'strike': current_price * 0.95,
                    'type': 'ITM',
                    'label': 'In the Money',
                    'desc': 'Higher cost, higher chance of profit',
                    'color': 'success'
                },
                {
                    'strike': current_price,
                    'type': 'ATM',
                    'label': 'At the Money',
                    'desc': 'Balanced risk/reward',
                    'color': 'info'
                },
                {
                    'strike': current_price * 1.05,
                    'type': 'OTM',
                    'label': 'Out of Money',
                    'desc': 'Lower cost, needs bigger move',
                    'color': 'warning'
                }
            ]
        else:  # put
            strike_options = [
                {
                    'strike': current_price * 1.05,
                    'type': 'ITM',
                    'label': 'In the Money',
                    'desc': 'Higher cost, higher chance of profit',
                    'color': 'success'
                },
                {
                    'strike': current_price,
                    'type': 'ATM',
                    'label': 'At the Money',
                    'desc': 'Balanced risk/reward',
                    'color': 'info'
                },
                {
                    'strike': current_price * 0.95,
                    'type': 'OTM',
                    'label': 'Out of Money',
                    'desc': 'Lower cost, needs bigger move',
                    'color': 'warning'
                }
            ]
        
        st.info("💡 Beginners: Start with 'At the Money' for balanced risk")
        
        cols = st.columns(3)
        for i, strike_opt in enumerate(strike_options):
            with cols[i]:
                # Mock premium calculation
                if strike_opt['type'] == 'ITM':
                    premium = current_price * 0.05
                elif strike_opt['type'] == 'ATM':
                    premium = current_price * 0.03
                else:
                    premium = current_price * 0.01
                
                if strike_opt['color'] == 'success':
                    st.success(f"**{strike_opt['label']}**")
                elif strike_opt['color'] == 'info':
                    st.info(f"**{strike_opt['label']}**")
                else:
                    st.warning(f"**{strike_opt['label']}**")
                
                st.write(f"Strike: ${strike_opt['strike']:.2f}")
                st.write(f"Cost: ~${premium:.2f}/share")
                st.write(f"Total: ~${premium * 100:.2f}")
                st.caption(strike_opt['desc'])
                
                if st.button(f"Select ${strike_opt['strike']:.2f}", key=f"strike_{i}"):
                    st.session_state.strike = strike_opt['strike']
                    st.session_state.estimated_premium = premium
                    st.session_state.order_step = 6
                    st.rerun()
        
        if st.button("← Back"):
            st.session_state.order_step = 4
            st.rerun()
    
    # Step 6: Review and Confirm
    elif st.session_state.order_step == 6:
        st.subheader("Step 6: Review Your Order 📋")
        
        # Order summary
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Order Details:**")
            st.write(f"Action: BUY")
            st.write(f"Type: {st.session_state.option_type.upper()}")
            st.write(f"Symbol: {st.session_state.symbol}")
            st.write(f"Strike: ${st.session_state.strike:.2f}")
            st.write(f"Expiration: {st.session_state.expiration.strftime('%b %d, %Y')}")
        
        with col2:
            st.write("**Cost Breakdown:**")
            contracts = st.number_input("Number of Contracts:", min_value=1, max_value=10, value=1)
            total_cost = st.session_state.estimated_premium * 100 * contracts
            st.write(f"Premium per share: ${st.session_state.estimated_premium:.2f}")
            st.write(f"Cost per contract: ${st.session_state.estimated_premium * 100:.2f}")
            st.metric("Total Cost:", f"${total_cost:.2f}")
        
        # Risk disclosure
        st.warning("⚠️ **Risk Reminder:**")
        st.write(f"• Maximum loss: ${total_cost:.2f} (your entire investment)")
        st.write("• Options can expire worthless")
        st.write("• This is not a recommendation")
        
        # Profit calculator
        with st.expander("💰 Profit/Loss Calculator"):
            target_price = st.slider(
                "If stock goes to:",
                min_value=float(st.session_state.strike * 0.8),
                max_value=float(st.session_state.strike * 1.2),
                value=float(st.session_state.strike * 1.1)
            )
            
            if st.session_state.option_type == "call":
                if target_price > st.session_state.strike:
                    profit = ((target_price - st.session_state.strike) - st.session_state.estimated_premium) * 100 * contracts
                    st.success(f"Potential Profit: ${profit:.2f}")
                else:
                    st.error(f"Loss: ${total_cost:.2f}")
            else:  # put
                if target_price < st.session_state.strike:
                    profit = ((st.session_state.strike - target_price) - st.session_state.estimated_premium) * 100 * contracts
                    st.success(f"Potential Profit: ${profit:.2f}")
                else:
                    st.error(f"Loss: ${total_cost:.2f}")
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("← Back", use_container_width=True):
                st.session_state.order_step = 5
                st.rerun()
        
        with col2:
            if st.button("❌ Cancel", use_container_width=True):
                for key in ['order_step', 'direction', 'symbol', 'option_type', 
                           'action', 'expiration', 'strike', 'estimated_premium']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        with col3:
            if st.button("✅ Place Order", type="primary", use_container_width=True):
                with st.spinner("Placing order..."):
                    # Here you would actually place the order via Alpaca API
                    st.success("🎉 Order placed successfully!")
                    st.balloons()
                    
                    # Reset form
                    for key in ['order_step', 'direction', 'symbol', 'option_type', 
                               'action', 'expiration', 'strike', 'estimated_premium']:
                        if key in st.session_state:
                            del st.session_state[key]

# Usage in main app
if __name__ == "__main__":
    # Mock API for demo
    api = None
    guided_options_order_form(api)