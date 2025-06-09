import configparser
import pandas as pd
import matplotlib.pyplot as plt
from tiingo import TiingoClient

def run_adaptive_momentum_backtest(symbol, start_date, end_date, config, regime_window=200):
    """
    Runs a backtest for the Improved Adaptive Momentum strategy with a Buffer Zone.
    """
    # 1. Download Historical Data from Tiingo
    print(f"Downloading data for {symbol} from Tiingo...")
    try:
        client = TiingoClient(config)
        data = client.get_dataframe(symbol,
                                      frequency='daily',
                                      startDate=start_date,
                                      endDate=end_date)
        data.rename(columns={'adjClose': 'Adj Close'}, inplace=True)
        if data.empty:
            print(f"No data found for {symbol}. Exiting.")
            return
    except Exception as e:
        print(f"An error occurred during download: {e}")
        return

    # 2. Calculate the Regime Filter (Moving Average)
    data['regime_ma'] = data['Adj Close'].rolling(window=regime_window).mean()

    # 3. Generate Trading Signals with a Buffer Zone
    # Define the buffer as a percentage (2% = 0.02)
    buffer = 0.02
    data['upper_band'] = data['regime_ma'] * (1 + buffer)
    data['lower_band'] = data['regime_ma'] * (1 - buffer)

    # Start with no position
    data['signal'] = 0

    # Use a loop to apply the new logic
    # We start after the moving average has been calculated
    for i in range(regime_window, len(data)):
        
        # If price crosses ABOVE the upper band, signal a BUY
        if data['Adj Close'][i] > data['upper_band'][i]:
            data['signal'][i] = 1
            
        # If price crosses BELOW the lower band, signal a SELL
        elif data['Adj Close'][i] < data['lower_band'][i]:
            data['signal'][i] = 0
            
        # If price is inside the buffer zone, DO NOTHING.
        # Carry forward the previous day's signal.
        else:
            data['signal'][i] = data['signal'][i-1]


    # To avoid lookahead bias, signals are for the NEXT day's position.
    data['signal'] = data['signal'].shift(1)

    # 4. Calculate Strategy Returns
    data['daily_return'] = data['Adj Close'].pct_change()
    data['strategy_return'] = data['daily_return'] * data['signal']

    # 5. Calculate Cumulative Returns
    data['buy_hold_cumulative'] = (1 + data['daily_return']).cumprod()
    data['strategy_cumulative'] = (1 + data['strategy_return']).cumprod()
    data.dropna(inplace=True)

    # 6. Print Performance Metrics
    print(f"\n--- Backtest Results for IMPROVED Strategy ---")
    buy_hold_return = (data['buy_hold_cumulative'].iloc[-1] - 1) * 100
    strategy_return = (data['strategy_cumulative'].iloc[-1] - 1) * 100
    print(f"For {symbol}:")
    print(f"Buy & Hold Total Return:   {buy_hold_return:.2f}%")
    print(f"Strategy Total Return:     {strategy_return:.2f}%")
    print("----------------------------------------------------\n")

    # 7. Plotting the Results
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True, gridspec_kw={'height_ratios': [3, 1]})
    ax1.plot(data.index, data['buy_hold_cumulative'], label='Buy & Hold', color='black', linestyle='--')
    ax1.plot(data.index, data['strategy_cumulative'], label='IMPROVED Strategy', color='blue', linewidth=2)
    ax1.set_title(f'IMPROVED Adaptive Momentum Strategy vs. Buy & Hold for {symbol}')
    ax1.legend()
    ax2.plot(data.index, data['signal'], label='Market Regime (1=Risk-On, 0=Risk-Off)', color='green', marker='.', linestyle='none')
    ax2.set_yticks([0, 1])
    ax2.set_yticklabels(['Risk-Off', 'Risk-On'])
    ax2.set_ylim(-0.1, 1.1)
    ax2.legend()
    plt.tight_layout()
    plt.show()

# --- MAIN EXECUTION BLOCK (No changes needed here) ---
if __name__ == '__main__':
    
    config_file = 'config.ini'
    config = configparser.ConfigParser()
    config.read(config_file)
    tiingo_key = config['tiingo']['api_key']
    
    if "YOUR_TIINGO_API_KEY_HERE" in tiingo_key:
        print("!!! PLEASE EDIT your config.ini file and add your actual Tiingo API key. !!!")
    else:
        tiingo_config = {'api_key': tiingo_key, 'session': True}

        run_adaptive_momentum_backtest(
            symbol='NVDA',
            start_date='2015-01-01',
            end_date='2025-06-08',
            config=tiingo_config
        )

        run_adaptive_momentum_backtest(
            symbol='KO',
            start_date='2015-01-01',
            end_date='2025-06-08',
            config=tiingo_config
        )