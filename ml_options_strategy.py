# ml_options_strategy.py - Simplified Version to Fix Crash
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
        
        # Define options strategies
        self.strategies = {
            'long_call': {
                'name': 'Long Call',
                'description': 'Buy call option - bullish strategy',
                'risk': 'Limited to premium paid',
                'reward': 'Unlimited upside potential',
                'best_conditions': 'Strong bullish momentum, low volatility buy'
            },
            'long_put': {
                'name': 'Long Put',
                'description': 'Buy put option - bearish strategy',
                'risk': 'Limited to premium paid',
                'reward': 'High profit potential on downside',
                'best_conditions': 'Strong bearish momentum, expect volatility increase'
            },
            'covered_call': {
                'name': 'Covered Call',
                'description': 'Own stock + sell call - income strategy',
                'risk': 'Upside capped at strike price',
                'reward': 'Premium income + dividends',
                'best_conditions': 'Neutral to slightly bullish, high volatility sell'
            },
            'straddle': {
                'name': 'Long Straddle',
                'description': 'Buy call + put at same strike - volatility play',
                'risk': 'Both premiums paid',
                'reward': 'Profit from large moves either direction',
                'best_conditions': 'Expecting big move but unsure of direction'
            }
        }
    
    def analyze_market_conditions(self, data):
        """Simple market analysis"""
        if data.empty:
            return None
        
        # Ensure we have price data
        if 'close' not in data.columns:
            if 'adjClose' in data.columns:
                data['close'] = data['adjClose']
            else:
                return None
        
        latest_price = data['close'].iloc[-1]
        
        # Simple moving averages
        if len(data) >= 20:
            sma_20 = data['close'].rolling(20).mean().iloc[-1]
            price_vs_sma = (latest_price - sma_20) / sma_20
        else:
            price_vs_sma = 0
        
        # Simple volatility
        if len(data) >= 20:
            returns = data['close'].pct_change()
            volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        else:
            volatility = 0.2
        
        # Simple RSI
        if len(data) >= 14:
            delta = data['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            rsi_value = rsi.iloc[-1] if not rsi.empty else 50
        else:
            rsi_value = 50
        
        return {
            'latest_price': latest_price,
            'price_vs_sma20': price_vs_sma,
            'volatility': volatility,
            'rsi': rsi_value
        }
    
    def recommend_strategy(self, market_conditions):
        """Simple rule-based strategy recommendation"""
        if not market_conditions:
            return 'long_call', 'Low'
        
        price_trend = market_conditions['price_vs_sma20']
        volatility = market_conditions['volatility']
        rsi = market_conditions['rsi']
        
        # Simple decision tree
        if price_trend > 0.05 and rsi < 70:
            return 'long_call', 'High'
        elif price_trend < -0.05 and rsi > 30:
            return 'long_put', 'High'
        elif volatility > 0.4:
            return 'straddle', 'Medium'
        else:
            return 'covered_call', 'Medium'

def create_ml_options_tab(polygon_api, tiingo_client):
    """Create the ML Options Strategy tab"""
    st.header("🧠 Options Strategy Analyzer")
    st.info("📊 Smart analysis to find optimal options strategies")
    
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
        analysis_period = st.selectbox("Period:", ["6 Months", "1 Year"])
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
                        st.subheader(f"📊 Current Analysis for {symbol}")
                        
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
                        strategy, confidence = analyzer.recommend_strategy(conditions)
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
                        
                        # Simple price chart
                        st.subheader("📈 Price Chart")
                        
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(
                            x=data.index,
                            y=data['close'] if 'close' in data.columns else data['adjClose'],
                            mode='lines',
                            name='Price',
                            line=dict(color='blue')
                        ))
                        
                        # Add moving average if we have enough data
                        if len(data) >= 20:
                            ma_20 = data['close'].rolling(20).mean() if 'close' in data.columns else data['adjClose'].rolling(20).mean()
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
    with st.expander("📚 Strategy Guide"):
        st.markdown("""
        ### Options Strategy Overview
        
        **🔥 High Volatility Stocks**
        - Best for: Straddles, selling premium
        - Avoid: Buying expensive options
        
        **⚖️ Medium Volatility Stocks**  
        - Best for: Directional plays, covered calls
        - Good balance of risk/reward
        
        **🛡️ Low Volatility Stocks**
        - Best for: Income strategies, buying cheap options
        - Focus on consistent returns
        
        ### Key Metrics
        - **Trend vs 20-day**: Shows momentum direction
        - **Volatility**: Higher = more expensive options
        - **RSI**: >70 overbought, <30 oversold
        """)

# Keep the integration function
def add_ml_tab_to_dashboard():
    """Integration function for the main dashboard"""
    pass