# ml_options_strategy.py - Enhanced with Position Sizing and P&L Calculator
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
import warnings

warnings.filterwarnings('ignore')

# Simple ML availability check
ML_AVAILABLE = False
try:
    import sklearn
    ML_AVAILABLE = True
    st.success("✅ Machine Learning available")
except ImportError:
    st.info("📊 Running in basic analysis mode (ML libraries not installed)")

class SimpleOptionsAnalyzer:
    def __init__(self, polygon_api, tiingo_client):
        self.polygon_api = polygon_api
        self.tiingo_client = tiingo_client
        
        # Define options strategies with enhanced details
        self.strategies = {
            'long_call': {
                'name': 'Long Call',
                'description': 'Buy call option - bullish strategy',
                'risk': 'Limited to premium paid',
                'reward': 'Unlimited upside potential',
                'best_conditions': 'Strong bullish momentum, low volatility buy',
                'max_loss_multiplier': 1.0,  # 100% of premium
                'breakeven_adjustment': 1.0,  # Premium added to strike
                'profit_factor': 2.5  # Potential profit multiplier
            },
            'long_put': {
                'name': 'Long Put',
                'description': 'Buy put option - bearish strategy',
                'risk': 'Limited to premium paid',
                'reward': 'High profit potential on downside',
                'best_conditions': 'Strong bearish momentum, expect volatility increase',
                'max_loss_multiplier': 1.0,  # 100% of premium
                'breakeven_adjustment': -1.0,  # Premium subtracted from strike
                'profit_factor': 3.0  # Higher profit potential on downside
            },
            'covered_call': {
                'name': 'Covered Call',
                'description': 'Own stock + sell call - income strategy',
                'risk': 'Upside capped at strike price',
                'reward': 'Premium income + dividends',
                'best_conditions': 'Neutral to slightly bullish, high volatility sell',
                'max_loss_multiplier': 0.1,  # Limited loss due to stock ownership
                'breakeven_adjustment': -1.0,  # Premium reduces cost basis
                'profit_factor': 1.5  # Conservative income strategy
            },
            'straddle': {
                'name': 'Long Straddle',
                'description': 'Buy call + put at same strike - volatility play',
                'risk': 'Both premiums paid',
                'reward': 'Profit from large moves either direction',
                'best_conditions': 'Expecting big move but unsure of direction',
                'max_loss_multiplier': 1.0,  # Total premiums paid
                'breakeven_adjustment': 2.0,  # Premium on both sides
                'profit_factor': 4.0  # High volatility profit potential
            }
        }
    
    def analyze_market_conditions(self, data):
        """Enhanced market analysis with volatility metrics"""
        if data.empty:
            return None
        
        # Ensure we have price data
        if 'close' not in data.columns:
            if 'adjClose' in data.columns:
                data['close'] = data['adjClose']
            else:
                return None
        
        latest_price = data['close'].iloc[-1]
        
        # Enhanced technical indicators
        if len(data) >= 20:
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            sma_50 = data['close'].rolling(50).mean().iloc[-1] if len(data) >= 50 else sma_20
            price_vs_sma = (latest_price - sma_20) / sma_20
        else:
            price_vs_sma = 0
            sma_20 = latest_price
            sma_50 = latest_price
        
        # Enhanced volatility calculation
        if len(data) >= 20:
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
            
            # VIX-like volatility index
            high_low_vol = ((data['high'] / data['low']).rolling(20).std().iloc[-1] 
                           if 'high' in data.columns and 'low' in data.columns else volatility)
        else:
            volatility = 0.2
            high_low_vol = 0.2
        
        # RSI calculation
        if len(data) >= 14:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1] if not rsi.empty else 50
        else:
            rsi_value = 50
        
        # Support and resistance levels
        recent_high = data['close'].rolling(20).max().iloc[-1]
        recent_low = data['close'].rolling(20).min().iloc[-1]
        
        return {
            'latest_price': latest_price,
            'price_vs_sma20': price_vs_sma,
            'sma_20': sma_20,
            'sma_50': sma_50,
            'volatility': volatility,
            'high_low_volatility': high_low_vol,
            'rsi': rsi_value,
            'recent_high': recent_high,
            'recent_low': recent_low,
            'price_range': recent_high - recent_low
        }
    
    def recommend_strategy(self, market_conditions):
        """Enhanced strategy recommendation with confidence scoring"""
        if not market_conditions:
            return 'long_call', 'Low', {}
        
        price_trend = market_conditions['price_vs_sma20']
        volatility = market_conditions['volatility']
        rsi = market_conditions['rsi']
        
        # Calculate strategy scores
        strategy_scores = {}
        
        # Long Call scoring
        bullish_score = max(0, price_trend * 100) + max(0, (70 - rsi) / 70 * 50)
        if volatility < 0.3:  # Low vol good for buying
            bullish_score += 25
        strategy_scores['long_call'] = min(100, bullish_score)
        
        # Long Put scoring  
        bearish_score = max(0, -price_trend * 100) + max(0, (rsi - 30) / 70 * 50)
        if volatility < 0.3:
            bearish_score += 25
        strategy_scores['long_put'] = min(100, bearish_score)
        
        # Covered Call scoring
        income_score = 50  # Base income strategy score
        if volatility > 0.4:  # High vol good for selling
            income_score += 30
        if 0 < price_trend < 0.05:  # Slightly bullish
            income_score += 20
        strategy_scores['covered_call'] = min(100, income_score)
        
        # Straddle scoring
        volatility_score = max(0, (volatility - 0.2) * 200)  # Higher vol = better
        uncertainty_score = 50 - abs(rsi - 50)  # Neutral RSI = uncertain direction
        strategy_scores['straddle'] = min(100, volatility_score + uncertainty_score)
        
        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy]
        
        # Determine confidence
        if best_score > 80:
            confidence = 'High'
        elif best_score > 60:
            confidence = 'Medium'
        else:
            confidence = 'Low'
        
        return best_strategy, confidence, strategy_scores
    
    def calculate_position_metrics(self, strategy_key, investment_amount, current_price):
        """Calculate detailed position metrics for the strategy"""
        strategy = self.strategies[strategy_key]
        
        # Estimate option premium (simplified calculation)
        if strategy_key in ['long_call', 'long_put']:
            premium_per_contract = current_price * 0.02  # 2% of stock price
            contracts = int(investment_amount / (premium_per_contract * 100))
            total_premium = contracts * premium_per_contract * 100
        elif strategy_key == 'covered_call':
            shares = int(investment_amount / current_price)
            premium_per_contract = current_price * 0.015  # 1.5% premium collected
            contracts = shares // 100  # One contract per 100 shares
            total_premium = contracts * premium_per_contract * 100
        elif strategy_key == 'straddle':
            # Call + Put premiums
            call_premium = current_price * 0.025
            put_premium = current_price * 0.020
            total_premium_per_straddle = (call_premium + put_premium) * 100
            contracts = int(investment_amount / total_premium_per_straddle)
            total_premium = contracts * total_premium_per_straddle
        
        # Calculate profit/loss scenarios
        max_loss = total_premium * strategy['max_loss_multiplier']
        
        # Calculate potential profits at different price levels
        price_scenarios = []
        for price_change in [-0.3, -0.2, -0.1, 0, 0.1, 0.2, 0.3]:
            future_price = current_price * (1 + price_change)
            
            if strategy_key == 'long_call':
                strike = current_price * 1.05  # 5% OTM
                intrinsic_value = max(0, future_price - strike) * contracts * 100
                profit = intrinsic_value - total_premium
            elif strategy_key == 'long_put':
                strike = current_price * 0.95  # 5% OTM
                intrinsic_value = max(0, strike - future_price) * contracts * 100
                profit = intrinsic_value - total_premium
            elif strategy_key == 'covered_call':
                strike = current_price * 1.1  # 10% OTM covered call
                stock_pnl = (future_price - current_price) * shares
                option_pnl = -max(0, future_price - strike) * contracts * 100
                profit = stock_pnl + option_pnl + total_premium
            elif strategy_key == 'straddle':
                strike = current_price  # ATM straddle
                call_value = max(0, future_price - strike) * contracts * 100
                put_value = max(0, strike - future_price) * contracts * 100
                profit = call_value + put_value - total_premium
            
            price_scenarios.append({
                'price_change': price_change * 100,
                'future_price': future_price,
                'profit_loss': profit
            })
        
        return {
            'contracts': contracts if strategy_key != 'covered_call' else contracts,
            'shares': shares if strategy_key == 'covered_call' else 0,
            'total_premium': total_premium,
            'max_loss': max_loss,
            'max_profit': max(scenario['profit_loss'] for scenario in price_scenarios),
            'price_scenarios': price_scenarios,
            'breakeven_prices': self._calculate_breakeven(strategy_key, current_price, total_premium, contracts)
        }
    
    def _calculate_breakeven(self, strategy_key, current_price, total_premium, contracts):
        """Calculate breakeven price(s) for the strategy"""
        if strategy_key == 'long_call':
            strike = current_price * 1.05
            breakeven = strike + (total_premium / (contracts * 100))
            return [breakeven]
        elif strategy_key == 'long_put':
            strike = current_price * 0.95
            breakeven = strike - (total_premium / (contracts * 100))
            return [breakeven]
        elif strategy_key == 'straddle':
            strike = current_price
            premium_per_share = total_premium / (contracts * 100)
            return [strike - premium_per_share, strike + premium_per_share]
        else:  # covered_call
            return [current_price - (total_premium / (contracts * 100))]

def create_ml_options_tab(polygon_api, tiingo_client):
    """Create the enhanced ML Options Strategy tab with position sizing"""
    st.header("🧠 Enhanced Options Strategy Analyzer")
    st.info("📊 Smart analysis with position sizing and profit/loss calculator")
    
    # Initialize analyzer
    analyzer = SimpleOptionsAnalyzer(polygon_api, tiingo_client)
    
    # Curated stock lists
    stock_categories = {
        "🔥 High Volatility": ["NVDA", "TSLA", "AMD", "SMCI", "PLTR"],
        "⚖️ Medium Volatility": ["AAPL", "MSFT", "GOOGL", "META", "AMZN"],
        "🛡️ Low Volatility": ["SPY", "QQQ", "KO", "JNJ", "PG"]
    }
    
    # User inputs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Stock Selection")
        
        selected_category = st.selectbox(
            "Choose volatility category:",
            list(stock_categories.keys())
        )
        
        symbol = st.selectbox(
            "Select stock:",
            stock_categories[selected_category]
        )
        
        custom_symbol = st.text_input("Or enter custom symbol:")
        if custom_symbol:
            symbol = custom_symbol.upper()
        
        st.success(f"**Selected**: {symbol}")
    
    with col2:
        st.subheader("⚙️ Settings")
        analysis_period = st.selectbox("Analysis Period:", ["6 Months", "1 Year"])
        days = 180 if analysis_period == "6 Months" else 365
    
    # Analysis button
    if st.button("🚀 Analyze Strategy", type="primary"):
        if symbol:
            with st.spinner(f"Analyzing {symbol}..."):
                try:
                    # Get data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    data = tiingo_client.get_dataframe(
                        symbol,
                        frequency='daily',
                        startDate=start_date.strftime('%Y-%m-%d'),
                        endDate=end_date.strftime('%Y-%m-%d')
                    )
                    
                    if data.empty:
                        st.error("No data available")
                        return
                    
                    # Analyze market conditions
                    conditions = analyzer.analyze_market_conditions(data)
                    
                    if conditions:
                        # Display current metrics
                        st.subheader(f"📊 Market Analysis for {symbol}")
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric("Current Price", f"${conditions['latest_price']:.2f}")
                        
                        with col2:
                            trend_pct = conditions['price_vs_sma20'] * 100
                            st.metric("Trend vs 20-day", f"{trend_pct:+.1f}%")
                        
                        with col3:
                            vol_pct = conditions['volatility'] * 100
                            st.metric("Volatility", f"{vol_pct:.1f}%")
                        
                        with col4:
                            st.metric("RSI", f"{conditions['rsi']:.1f}")
                        
                        # Get recommendation
                        strategy, confidence, scores = analyzer.recommend_strategy(conditions)
                        strategy_info = analyzer.strategies[strategy]
                        
                        # Display recommendation
                        st.subheader("🎯 Strategy Recommendation")
                        
                        if confidence == "High":
                            st.success(f"**Recommended**: {strategy_info['name']} (High Confidence)")
                        elif confidence == "Medium":
                            st.info(f"**Recommended**: {strategy_info['name']} (Medium Confidence)")
                        else:
                            st.warning(f"**Recommended**: {strategy_info['name']} (Low Confidence)")
                        
                        # Strategy details
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.write(f"**Description**: {strategy_info['description']}")
                            st.write(f"**Best Conditions**: {strategy_info['best_conditions']}")
                        
                        with col2:
                            st.write(f"**Risk**: {strategy_info['risk']}")
                            st.write(f"**Reward**: {strategy_info['reward']}")
                        
                        # 🎯 POSITION SIZING SLIDER - THE NEW FEATURE!
                        st.divider()
                        st.subheader("💰 Position Sizing & Risk Calculator")
                        
                        # Investment amount slider
                        investment_amount = st.slider(
                            "💵 Investment Amount ($)",
                            min_value=500,
                            max_value=50000,
                            value=5000,
                            step=500,
                            help="Select how much you want to invest in this strategy"
                        )
                        
                        # Calculate position metrics
                        metrics = analyzer.calculate_position_metrics(
                            strategy, investment_amount, conditions['latest_price']
                        )
                        
                        # Display position details
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            if strategy == 'covered_call':
                                st.metric("Shares to Buy", f"{metrics['shares']:,}")
                            else:
                                st.metric("Contracts", f"{metrics['contracts']:,}")
                        
                        with col2:
                            st.metric("Total Premium", f"${metrics['total_premium']:,.0f}")
                        
                        with col3:
                            st.metric("Max Loss", f"${metrics['max_loss']:,.0f}")
                        
                        with col4:
                            max_profit_display = f"${metrics['max_profit']:,.0f}" if metrics['max_profit'] < 999999 else "Unlimited"
                            st.metric("Max Profit", max_profit_display)
                        
                        # Breakeven prices
                        st.write("**Breakeven Price(s):**")
                        for i, price in enumerate(metrics['breakeven_prices']):
                            st.write(f"• Breakeven {i+1}: ${price:.2f}")
                        
                        # Profit/Loss Scenarios Table
                        st.subheader("📊 Profit/Loss Scenarios")
                        
                        scenarios_df = pd.DataFrame(metrics['price_scenarios'])
                        scenarios_df['Price Change'] = scenarios_df['price_change'].apply(lambda x: f"{x:+.0f}%")
                        scenarios_df['Future Price'] = scenarios_df['future_price'].apply(lambda x: f"${x:.2f}")
                        scenarios_df['Profit/Loss'] = scenarios_df['profit_loss'].apply(lambda x: f"${x:+,.0f}")
                        scenarios_df['ROI'] = (scenarios_df['profit_loss'] / investment_amount * 100).apply(lambda x: f"{x:+.1f}%")
                        
                        display_df = scenarios_df[['Price Change', 'Future Price', 'Profit/Loss', 'ROI']]
                        
                        # Color code the dataframe
                        def color_pnl(val):
                            if '+' in val:
                                return 'background-color: #d4edda; color: #155724'
                            elif '-' in val:
                                return 'background-color: #f8d7da; color: #721c24'
                            return ''
                        
                        styled_df = display_df.style.applymap(color_pnl, subset=['Profit/Loss', 'ROI'])
                        st.dataframe(styled_df, use_container_width=True)
                        
                        # Risk Assessment
                        st.subheader("⚠️ Risk Assessment")
                        
                        risk_ratio = metrics['max_loss'] / investment_amount
                        if risk_ratio > 0.5:
                            st.error(f"🚨 **High Risk**: You could lose {risk_ratio*100:.0f}% of your investment")
                        elif risk_ratio > 0.2:
                            st.warning(f"⚠️ **Medium Risk**: You could lose {risk_ratio*100:.0f}% of your investment")
                        else:
                            st.success(f"✅ **Lower Risk**: Maximum loss is {risk_ratio*100:.0f}% of investment")
                        
                        # Quick tips
                        st.info(f"""
                        **💡 Quick Tips for {strategy_info['name']}:**
                        • Start with smaller position sizes to test the strategy
                        • Monitor volatility - it greatly affects option prices
                        • Set profit targets and stop losses before entering
                        • Consider the time decay (theta) for options strategies
                        """)
                        
                        # Simple price chart
                        st.subheader("📈 Price Chart")
                        
                        fig = go.Figure()
                        price_col = 'close' if 'close' in data.columns else 'adjClose'
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data[price_col],
                            mode='lines',
                            name='Price',
                            line=dict(color='blue')
                        ))
                        
                        # Add moving average if we have enough data
                        if len(data) >= 20:
                            ma_20 = data[price_col].rolling(20).mean()
                            fig.add_trace(go.Scatter(
                                x=data.index,
                                y=ma_20,
                                mode='lines',
                                name='20-day MA',
                                line=dict(color='red', dash='dash')
                            ))
                        
                        fig.update_layout(
                            title=f"{symbol} Price Chart",
                            xaxis_title="Date",
                            yaxis_title="Price",
                            height=400
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                        
                    else:
                        st.error("Could not analyze market conditions")
                
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    st.write("**Debug info**: Check if the symbol exists and has sufficient data")
        
        else:
            st.warning("Please select a symbol")
    
    # Educational section
    with st.expander("📚 Strategy Guide & Tips"):
        st.markdown("""
        ### Options Strategy Overview
        
        **🔥 High Volatility Stocks**
        - Best for: Straddles, selling premium strategies
        - Risk: Higher option premiums, faster time decay
        - Tip: Consider shorter expiration dates
        
        **⚖️ Medium Volatility Stocks**  
        - Best for: Directional plays, covered calls
        - Good balance of risk/reward
        - Tip: Most versatile for various strategies
        
        **🛡️ Low Volatility Stocks**
        - Best for: Income strategies, buying cheap options
        - Risk: Lower profit potential, less movement
        - Tip: Focus on longer-term strategies
        
        ### Position Sizing Guidelines
        - **Conservative**: Risk 1-2% of portfolio per trade
        - **Moderate**: Risk 3-5% of portfolio per trade  
        - **Aggressive**: Risk 5-10% of portfolio per trade
        
        ### Key Metrics Explained
        - **Breakeven**: Stock price where strategy breaks even
        - **Max Loss**: Worst-case scenario loss
        - **ROI**: Return on investment percentage
        - **Time Decay**: Options lose value as expiration approaches
        """)

# Keep the integration function
def add_ml_tab_to_dashboard():
    """Integration function for the main dashboard"""
    pass