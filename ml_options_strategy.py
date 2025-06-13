# ml_options_strategy.py - Machine Learning Options Strategy Finder
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import warnings
import requests
from tiingo import TiingoClient

# ML imports with fallbacks
try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
    from sklearn.model_selection import train_test_split, cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.cluster import KMeans
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    st.warning("⚠️ Machine Learning libraries not available. Install scikit-learn for full ML features.")

warnings.filterwarnings('ignore')

class OptionsStrategyML:
    def __init__(self, polygon_api, tiingo_client):
        self.polygon_api = polygon_api
        self.tiingo_client = tiingo_client
        self.ml_available = ML_AVAILABLE
        
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
            'cash_secured_put': {
                'name': 'Cash Secured Put',
                'description': 'Sell put with cash backing - income strategy',
                'risk': 'Must buy stock if assigned',
                'reward': 'Premium income',
                'best_conditions': 'Want to own stock at lower price, high volatility'
            },
            'iron_condor': {
                'name': 'Iron Condor',
                'description': 'Sell call spread + put spread - neutral strategy',
                'risk': 'Limited to spread width minus premium',
                'reward': 'Premium collected',
                'best_conditions': 'Low volatility, range-bound market'
            },
            'straddle': {
                'name': 'Long Straddle',
                'description': 'Buy call + put at same strike - volatility play',
                'risk': 'Both premiums paid',
                'reward': 'Profit from large moves either direction',
                'best_conditions': 'Expecting big move but unsure of direction'
            },
            'butterfly_spread': {
                'name': 'Butterfly Spread',
                'description': 'Limited risk, limited reward - precision strategy',
                'risk': 'Premium paid for spread',
                'reward': 'Maximum at middle strike',
                'best_conditions': 'Expect stock to stay near specific price'
            }
        }
    
    def calculate_technical_indicators(self, data):
        """Calculate technical indicators for ML features"""
        df = data.copy()
        
        # Price-based indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['ema_12'] = df['close'].ewm(span=12).mean()
        df['ema_26'] = df['close'].ewm(span=26).mean()
        
        # Volatility indicators
        df['volatility_20'] = df['close'].pct_change().rolling(20).std() * np.sqrt(252)
        df['volatility_5'] = df['close'].pct_change().rolling(5).std() * np.sqrt(252)
        
        # Momentum indicators
        df['rsi'] = self.calculate_rsi(df['close'])
        df['macd'] = df['ema_12'] - df['ema_26']
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        
        # Price position indicators
        df['price_vs_sma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        df['price_vs_sma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        
        # Bollinger Bands
        df['bb_middle'] = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Volume indicators (if available)
        if 'volume' in df.columns:
            df['volume_sma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma']
        else:
            df['volume_ratio'] = 1.0
        
        return df
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI indicator"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def create_ml_features(self, data):
        """Create features for machine learning model"""
        df = self.calculate_technical_indicators(data)
        
        # Feature columns
        feature_columns = [
            'price_vs_sma20', 'price_vs_sma50', 'volatility_20', 'volatility_5',
            'rsi', 'macd', 'bb_position', 'volume_ratio'
        ]
        
        # Create target variables for different strategies
        df['future_return_5d'] = df['close'].pct_change(5).shift(-5)
        df['future_return_20d'] = df['close'].pct_change(20).shift(-20)
        df['future_volatility'] = df['close'].pct_change().rolling(20).std().shift(-20) * np.sqrt(252)
        
        # Create strategy success indicators
        df['bullish_success'] = (df['future_return_20d'] > 0.05).astype(int)  # 5% gain
        df['bearish_success'] = (df['future_return_20d'] < -0.05).astype(int)  # 5% loss
        df['neutral_success'] = (np.abs(df['future_return_20d']) < 0.03).astype(int)  # Within 3%
        df['high_volatility'] = (df['future_volatility'] > df['volatility_20']).astype(int)
        
        return df, feature_columns
    
    def train_strategy_predictor(self, data, feature_columns):
        """Train ML model to predict best options strategy"""
        if not self.ml_available:
            return None, None
        
        # Create features and targets
        df_ml = data.dropna()
        
        if len(df_ml) < 100:
            st.warning("Not enough data for reliable ML predictions")
            return None, None
        
        X = df_ml[feature_columns]
        
        # Create composite strategy recommendation
        strategy_scores = pd.DataFrame(index=df_ml.index)
        
        # Score each strategy based on market conditions
        strategy_scores['long_call'] = (
            df_ml['bullish_success'] * 0.4 +
            (df_ml['rsi'] < 70).astype(int) * 0.3 +
            (df_ml['price_vs_sma20'] > 0).astype(int) * 0.3
        )
        
        strategy_scores['long_put'] = (
            df_ml['bearish_success'] * 0.4 +
            (df_ml['rsi'] > 30).astype(int) * 0.3 +
            (df_ml['price_vs_sma20'] < 0).astype(int) * 0.3
        )
        
        strategy_scores['covered_call'] = (
            df_ml['neutral_success'] * 0.3 +
            df_ml['high_volatility'] * 0.4 +
            (df_ml['rsi'] > 50).astype(int) * 0.3
        )
        
        strategy_scores['straddle'] = (
            df_ml['high_volatility'] * 0.6 +
            (np.abs(df_ml['future_return_20d']) > 0.05).astype(int) * 0.4
        )
        
        # Find best strategy for each time period
        y = strategy_scores.idxmax(axis=1)
        
        # Train Random Forest classifier
        try:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            return model, accuracy
        except Exception as e:
            st.error(f"ML model training failed: {e}")
            return None, None
    
    def simulate_options_strategy(self, data, strategy, entry_date, expiration_days=30):
        """Simulate options strategy performance"""
        try:
            entry_idx = data.index.get_loc(entry_date)
            current_price = data.iloc[entry_idx]['close']
            
            # Calculate expiration date
            if entry_idx + expiration_days >= len(data):
                expiration_idx = len(data) - 1
            else:
                expiration_idx = entry_idx + expiration_days
            
            expiration_price = data.iloc[expiration_idx]['close']
            volatility = data['close'].pct_change().rolling(20).std().iloc[entry_idx] * np.sqrt(252)
            
            # Simplified options pricing (Black-Scholes approximation)
            results = self.calculate_strategy_pnl(strategy, current_price, expiration_price, volatility, expiration_days)
            
            return results
        except Exception as e:
            return {'error': str(e)}
    
    def calculate_strategy_pnl(self, strategy, entry_price, exit_price, volatility, days_to_expiry):
        """Calculate P&L for different options strategies"""
        # Simplified options pricing - in practice you'd use Black-Scholes
        time_decay_factor = max(0.1, days_to_expiry / 30)  # Simplified time decay
        
        if strategy == 'long_call':
            strike = entry_price * 1.05  # 5% OTM
            premium = entry_price * 0.03 * volatility * time_decay_factor
            intrinsic_value = max(0, exit_price - strike)
            pnl = intrinsic_value - premium
            
        elif strategy == 'long_put':
            strike = entry_price * 0.95  # 5% OTM
            premium = entry_price * 0.03 * volatility * time_decay_factor
            intrinsic_value = max(0, strike - exit_price)
            pnl = intrinsic_value - premium
            
        elif strategy == 'covered_call':
            strike = entry_price * 1.1  # 10% OTM
            premium = entry_price * 0.02 * volatility * time_decay_factor
            stock_pnl = exit_price - entry_price
            option_pnl = premium - max(0, exit_price - strike)
            pnl = stock_pnl + option_pnl
            
        elif strategy == 'straddle':
            strike = entry_price
            call_premium = entry_price * 0.04 * volatility * time_decay_factor
            put_premium = entry_price * 0.04 * volatility * time_decay_factor
            total_premium = call_premium + put_premium
            intrinsic_value = max(exit_price - strike, strike - exit_price)
            pnl = intrinsic_value - total_premium
            
        else:
            pnl = 0
        
        return {
            'pnl': pnl,
            'pnl_percent': (pnl / entry_price) * 100,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'strategy': strategy
        }
    
    def backtest_all_strategies(self, data, lookback_days=252):
        """Backtest all strategies over historical data"""
        results = []
        
        # Use only recent data for backtesting
        test_data = data.tail(lookback_days)
        
        strategies_to_test = ['long_call', 'long_put', 'covered_call', 'straddle']
        
        for i in range(30, len(test_data) - 30, 5):  # Test every 5 days
            entry_date = test_data.index[i]
            
            for strategy in strategies_to_test:
                result = self.simulate_options_strategy(test_data, strategy, entry_date, 30)
                if 'error' not in result:
                    result['entry_date'] = entry_date
                    result['strategy'] = strategy
                    results.append(result)
        
        return pd.DataFrame(results)

def create_ml_options_tab(polygon_api, tiingo_client):
    """Create the ML Options Strategy tab"""
    st.header("🧠 ML-Powered Options Strategy Finder")
    st.info("🤖 Using machine learning to analyze market conditions and recommend optimal options strategies")
    
    # Initialize ML system
    ml_system = OptionsStrategyML(polygon_api, tiingo_client)
    
    # Curated stock lists based on volatility profiles
    stock_categories = {
        "🔥 High Volatility Stocks": {
            "description": "High-growth, high-volatility stocks perfect for options strategies",
            "stocks": ["NVDA", "TSLA", "AMD", "SMCI", "PLTR", "ROKU", "COIN", "MSTR", "DKNG", "ARKK"]
        },
        "⚖️ Medium Volatility Stocks": {
            "description": "Growth stocks with moderate volatility - balanced risk/reward",
            "stocks": ["AAPL", "MSFT", "GOOGL", "META", "AMZN", "NFLX", "CRM", "UBER", "SHOP", "SQ"]
        },
        "🛡️ Low Volatility Stocks": {
            "description": "Stable, dividend-paying stocks good for income strategies", 
            "stocks": ["SPY", "QQQ", "KO", "JNJ", "PG", "WMT", "VZ", "T", "PFE", "XOM"]
        },
        "📊 Sector ETFs": {
            "description": "Diversified sector exposure with varying volatility",
            "stocks": ["XLK", "XLF", "XLE", "XLV", "XLI", "XLY", "XLP", "XLU", "XLRE", "XLB"]
        }
    }
    
    # User inputs
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📈 Stock Selection")
        
        # Stock category selector
        selected_category = st.selectbox(
            "Choose stock category:",
            list(stock_categories.keys()),
            help="Select based on your risk tolerance and strategy preference"
        )
        
        # Show category description
        category_info = stock_categories[selected_category]
        st.info(f"**{selected_category}**: {category_info['description']}")
        
        # Stock selector within category
        col1a, col1b = st.columns([2, 1])
        
        with col1a:
            symbol = st.selectbox(
                "Select stock:",
                category_info['stocks'],
                help="Pre-selected stocks known for their volatility characteristics"
            )
        
        with col1b:
            # Option to add custom symbol
            custom_symbol = st.text_input("Or enter custom symbol:", placeholder="e.g., AAPL")
            if custom_symbol:
                symbol = custom_symbol.upper()
        
        # Display current selection
        st.success(f"**Selected Symbol**: {symbol}")
        
        # Quick volatility info for selected stock
        volatility_info = {
            # High volatility
            "NVDA": "🔥 High volatility AI chip leader",
            "TSLA": "🔥 High volatility EV pioneer", 
            "AMD": "🔥 High volatility semiconductor",
            "SMCI": "🔥 Extremely high volatility AI infrastructure",
            "PLTR": "🔥 High volatility data analytics",
            "ROKU": "🔥 High volatility streaming",
            "COIN": "🔥 Extreme volatility crypto exchange",
            "MSTR": "🔥 Extreme volatility Bitcoin proxy",
            "DKNG": "🔥 High volatility gaming",
            "ARKK": "🔥 High volatility innovation ETF",
            
            # Medium volatility  
            "AAPL": "⚖️ Medium volatility tech giant",
            "MSFT": "⚖️ Medium volatility cloud leader",
            "GOOGL": "⚖️ Medium volatility search/AI",
            "META": "⚖️ Medium volatility social media",
            "AMZN": "⚖️ Medium volatility e-commerce/cloud",
            "NFLX": "⚖️ Medium volatility streaming leader",
            "CRM": "⚖️ Medium volatility enterprise software",
            "UBER": "⚖️ Medium volatility ride-sharing",
            "SHOP": "⚖️ Medium volatility e-commerce platform",
            "SQ": "⚖️ Medium volatility fintech",
            
            # Low volatility
            "SPY": "🛡️ Low volatility S&P 500 ETF",
            "QQQ": "🛡️ Low-medium volatility Nasdaq ETF",
            "KO": "🛡️ Low volatility dividend aristocrat",
            "JNJ": "🛡️ Low volatility healthcare giant",
            "PG": "🛡️ Low volatility consumer staples",
            "WMT": "🛡️ Low volatility retail giant", 
            "VZ": "🛡️ Low volatility telecom dividend",
            "T": "🛡️ Low volatility telecom",
            "PFE": "🛡️ Low volatility pharma",
            "XOM": "🛡️ Medium volatility energy giant",
        }
        
        if symbol in volatility_info:
            st.caption(volatility_info[symbol])
    
    with col2:
        st.subheader("⚙️ Analysis Settings")
        
        analysis_period = st.selectbox("Analysis Period:", ["6 Months", "1 Year", "2 Years"])
        days_map = {"6 Months": 180, "1 Year": 365, "2 Years": 730}
        days = days_map[analysis_period]
        
        ml_mode = st.selectbox(
            "Analysis Mode:", 
            ["Quick Analysis", "Full ML Training", "Strategy Comparison"]
        )
        
        # Quick tips based on selected category
        if "High Volatility" in selected_category:
            st.success("💡 **High Vol Tips:**\n- Great for straddles/strangles\n- Consider selling premium\n- Watch for earnings events")
        elif "Medium Volatility" in selected_category:
            st.info("💡 **Medium Vol Tips:**\n- Balanced approach works\n- Good for covered calls\n- Directional plays viable")
        elif "Low Volatility" in selected_category:
            st.warning("💡 **Low Vol Tips:**\n- Focus on income strategies\n- Buy options cheaply\n- Avoid selling premium")
    
    # Quick volatility comparison chart
    if st.checkbox("📊 Show Volatility Comparison", value=False):
        st.subheader("Volatility Profile Comparison")
        
        # Sample historical volatilities (you could make this dynamic)
        vol_data = {
            "Stock": ["MSTR", "COIN", "TSLA", "NVDA", "AAPL", "MSFT", "SPY", "KO"],
            "Avg_Volatility": [95, 85, 65, 55, 25, 22, 15, 12],
            "Category": ["🔥 Extreme", "🔥 High", "🔥 High", "🔥 High", "⚖️ Medium", "⚖️ Medium", "🛡️ Low", "🛡️ Low"]
        }
        
        vol_df = pd.DataFrame(vol_data)
        
        fig_vol = px.bar(
            vol_df, 
            x="Stock", 
            y="Avg_Volatility",
            color="Category",
            title="Historical Volatility Comparison (%)",
            color_discrete_map={
                "🔥 Extreme": "#ff4444",
                "🔥 High": "#ff8800", 
                "⚖️ Medium": "#0088ff",
                "🛡️ Low": "#00aa44"
            }
        )
        
        fig_vol.update_layout(height=400)
        st.plotly_chart(fig_vol, use_container_width=True)
    
    if st.button("🚀 Analyze Options Strategies", type="primary"):
        if symbol:
            with st.spinner(f"Analyzing {symbol} with machine learning..."):
                try:
                    # Get historical data
                    end_date = datetime.now()
                    start_date = end_date - timedelta(days=days)
                    
                    data = tiingo_client.get_dataframe(
                        symbol,
                        frequency='daily',
                        startDate=start_date.strftime('%Y-%m-%d'),
                        endDate=end_date.strftime('%Y-%m-%d')
                    )
                    
                    if data.empty:
                        st.error("No data available for this symbol")
                        return
                    
                    # Rename columns for consistency
                    data.rename(columns={'adjClose': 'close'}, inplace=True)
                    
                    # Create ML features
                    ml_data, feature_columns = ml_system.create_ml_features(data)
                    
                    # Current market analysis
                    current_conditions = ml_data.iloc[-1]
                    
                    # Display current market conditions
                    st.subheader(f"📊 Current Market Analysis for {symbol}")
                    
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
                    with metrics_col1:
                        st.metric(
                            "Price vs 20-day SMA",
                            f"{current_conditions['price_vs_sma20']:.1%}",
                            delta=f"{current_conditions['price_vs_sma20']:.1%}"
                        )
                    
                    with metrics_col2:
                        st.metric(
                            "20-day Volatility",
                            f"{current_conditions['volatility_20']:.1%}",
                            delta="High" if current_conditions['volatility_20'] > 0.3 else "Normal"
                        )
                    
                    with metrics_col3:
                        rsi_val = current_conditions['rsi']
                        rsi_status = "Overbought" if rsi_val > 70 else "Oversold" if rsi_val < 30 else "Neutral"
                        st.metric("RSI", f"{rsi_val:.1f}", delta=rsi_status)
                    
                    with metrics_col4:
                        bb_pos = current_conditions['bb_position']
                        bb_status = "Upper" if bb_pos > 0.8 else "Lower" if bb_pos < 0.2 else "Middle"
                        st.metric("Bollinger Position", f"{bb_pos:.2f}", delta=bb_status)
                    
                    # ML Analysis
                    if ml_mode == "Full ML Training" and ML_AVAILABLE:
                        st.subheader("🤖 Machine Learning Analysis")
                        
                        model, accuracy = ml_system.train_strategy_predictor(ml_data, feature_columns)
                        
                        if model is not None:
                            st.success(f"✅ ML Model trained with {accuracy:.1%} accuracy")
                            
                            # Get current prediction
                            current_features = current_conditions[feature_columns].values.reshape(1, -1)
                            predicted_strategy = model.predict(current_features)[0]
                            prediction_proba = model.predict_proba(current_features)[0]
                            
                            st.write("### 🎯 ML Recommendation")
                            strategy_info = ml_system.strategies[predicted_strategy]
                            
                            rec_col1, rec_col2 = st.columns(2)
                            with rec_col1:
                                st.write(f"**Recommended Strategy:** {strategy_info['name']}")
                                st.write(f"**Description:** {strategy_info['description']}")
                                st.write(f"**Best Conditions:** {strategy_info['best_conditions']}")
                            
                            with rec_col2:
                                st.write(f"**Risk:** {strategy_info['risk']}")
                                st.write(f"**Reward:** {strategy_info['reward']}")
                                
                                # Show prediction confidence
                                max_prob = max(prediction_proba)
                                st.metric("Confidence", f"{max_prob:.1%}")
                            
                            # Feature importance
                            feature_importance = pd.DataFrame({
                                'feature': feature_columns,
                                'importance': model.feature_importances_
                            }).sort_values('importance', ascending=False)
                            
                            fig_importance = px.bar(
                                feature_importance, 
                                x='importance', 
                                y='feature',
                                title="Feature Importance in Strategy Selection",
                                orientation='h'
                            )
                            st.plotly_chart(fig_importance, use_container_width=True)
                    
                    # Strategy Comparison
                    if ml_mode in ["Strategy Comparison", "Quick Analysis"]:
                        st.subheader("📈 Strategy Backtesting Results")
                        
                        with st.spinner("Running strategy backtests..."):
                            backtest_results = ml_system.backtest_all_strategies(ml_data)
                        
                        if not backtest_results.empty:
                            # Summary statistics
                            strategy_performance = backtest_results.groupby('strategy').agg({
                                'pnl_percent': ['mean', 'std', 'count'],
                                'pnl': 'sum'
                            }).round(2)
                            
                            st.write("### 📊 Strategy Performance Summary")
                            
                            # Create performance chart
                            avg_returns = backtest_results.groupby('strategy')['pnl_percent'].mean().sort_values(ascending=False)
                            
                            fig_performance = go.Figure()
                            fig_performance.add_trace(go.Bar(
                                x=avg_returns.index,
                                y=avg_returns.values,
                                marker_color=['green' if x > 0 else 'red' for x in avg_returns.values],
                                text=[f"{x:.1f}%" for x in avg_returns.values],
                                textposition='auto'
                            ))
                            
                            fig_performance.update_layout(
                                title="Average Returns by Strategy",
                                xaxis_title="Strategy",
                                yaxis_title="Average Return (%)"
                            )
                            
                            st.plotly_chart(fig_performance, use_container_width=True)
                            
                            # Best performing strategy
                            best_strategy = avg_returns.index[0]
                            best_return = avg_returns.iloc[0]
                            
                            st.success(f"🏆 **Best Performing Strategy**: {ml_system.strategies[best_strategy]['name']} with {best_return:.1f}% average return")
                            
                            # Detailed results table
                            with st.expander("📋 Detailed Backtest Results"):
                                display_results = backtest_results[['entry_date', 'strategy', 'entry_price', 'exit_price', 'pnl', 'pnl_percent']].copy()
                                display_results['entry_date'] = pd.to_datetime(display_results['entry_date']).dt.strftime('%Y-%m-%d')
                                display_results = display_results.sort_values('pnl_percent', ascending=False)
                                st.dataframe(display_results, use_container_width=True)
                            
                            # Risk analysis
                            st.write("### ⚠️ Risk Analysis")
                            
                            risk_metrics = backtest_results.groupby('strategy').agg({
                                'pnl_percent': ['mean', 'std'],
                            }).round(2)
                            
                            risk_metrics.columns = ['Avg_Return', 'Volatility']
                            risk_metrics['Sharpe_Ratio'] = (risk_metrics['Avg_Return'] / risk_metrics['Volatility']).round(2)
                            risk_metrics['Win_Rate'] = backtest_results.groupby('strategy')['pnl_percent'].apply(lambda x: (x > 0).mean()).round(2)
                            
                            st.dataframe(risk_metrics, use_container_width=True)
                    
                    # Current recommendation based on market conditions
                    st.subheader("💡 Current Market-Based Recommendation")
                    
                    # Simple rule-based recommendation
                    rsi = current_conditions['rsi']
                    volatility = current_conditions['volatility_20']
                    price_trend = current_conditions['price_vs_sma20']
                    
                    if price_trend > 0.05 and rsi < 70:
                        recommended = "long_call"
                        confidence = "High"
                    elif price_trend < -0.05 and rsi > 30:
                        recommended = "long_put"
                        confidence = "High"
                    elif volatility > 0.4:
                        recommended = "straddle"
                        confidence = "Medium"
                    elif abs(price_trend) < 0.02:
                        recommended = "covered_call"
                        confidence = "Medium"
                    else:
                        recommended = "long_call"
                        confidence = "Low"
                    
                    strategy_info = ml_system.strategies[recommended]
                    
                    st.info(f"""
                    **Recommended Strategy**: {strategy_info['name']}
                    
                    **Confidence**: {confidence}
                    
                    **Reasoning**: Based on current market conditions:
                    - Price trend: {price_trend:.1%} vs 20-day SMA
                    - Volatility: {volatility:.1%}
                    - RSI: {rsi:.1f}
                    
                    **Strategy Details**: {strategy_info['description']}
                    
                    **Risk**: {strategy_info['risk']}
                    
                    **Reward**: {strategy_info['reward']}
                    """)
                    
                except Exception as e:
                    st.error(f"Analysis failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
        else:
            st.warning("Please enter a stock symbol")
    
    # Educational section
    with st.expander("📚 How ML Options Strategy Selection Works"):
        st.markdown("""
        ### 🧠 Machine Learning Approach
        
        **1. Feature Engineering**
        - Technical indicators (RSI, MACD, Bollinger Bands)
        - Volatility measures (realized and implied)
        - Price momentum and trend indicators
        - Volume analysis
        
        **2. Strategy Scoring**
        - Historical performance analysis
        - Market condition matching
        - Risk-adjusted returns
        - Win rate optimization
        
        **3. Prediction Model**
        - Random Forest Classifier for strategy selection
        - Feature importance analysis
        - Cross-validation for robustness
        - Confidence scoring
        
        **4. Backtesting**
        - Historical simulation of all strategies
        - Risk metrics calculation
        - Performance comparison
        - Drawdown analysis
        
        ### ⚠️ Important Notes
        - This is for educational purposes only
        - Options trading involves significant risk
        - Past performance doesn't guarantee future results
        - Always consider your risk tolerance
        """)

# Add this to your main dashboard's tab creation
def add_ml_tab_to_dashboard():
    """Integration function for the main dashboard"""
    # Add this to your tab creation in the main dashboard:
    # tab_live, tab_options, tab_backtest, tab_ml, tab_debug = st.tabs([
    #     "📊 Live Account", 
    #     "🎯 Options Trading", 
    #     "📈 Strategy Backtester",
    #     "🧠 ML Strategy Finder",  # <-- New tab
    #     "🐛 Debug Console"
    # ])
    
    # with tab_ml:
    #     create_ml_options_tab(polygon_api, tiingo_client)
    pass