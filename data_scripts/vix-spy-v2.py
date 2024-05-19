import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
import logging
import schedule
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(filename='vix_spy.log', 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the tickers and the range (SPY, VIX, SH)
tickers = ["SPY", "^VIX", "SH"]
start_date = "2006-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')

# Set the path for the SQLite database
db_path = 'financial_data.db'
engine = create_engine(f'sqlite:///{db_path}')

# Load strategy from CSV
strategy_path = '/Users/jordanukawoko/Library/Mobile Documents/com~apple~CloudDocs/Auto-Adjusting/Auto-Adjusting/data_scripts/strat-1.csv'
strategy_df = pd.read_csv(strategy_path)
strategy_df.columns = ['VIX Level', 'SPY Allocation', 'SH Allocation']

# Handle "36+" VIX Level (1-36+)
strategy_df['VIX Level'] = strategy_df['VIX Level'].replace('36+', '36').astype(int)

# Convert strategy allocations to numeric to make them parsable
strategy_df['SPY Allocation'] = strategy_df['SPY Allocation'].str.rstrip('%').astype('float') / 100.0 * 100
strategy_df['SH Allocation'] = strategy_df['SH Allocation'].str.rstrip('%').astype('float') / 100.0 * 100

def download_and_save_data(ticker, start, end, engine):
    try:
        print(f"Downloading data for {ticker} from {start} to {end}...")
        df = yf.download(ticker, start=start, end=end)
        df.to_sql(ticker, engine, if_exists='replace')
        print(f"Data for {ticker} saved to database.")
        logging.info(f"Data for {ticker} saved to database.")
    except Exception as e:
        print(f"Error downloading or saving data for {ticker}: {e}")
        logging.error(f"Error downloading or saving data for {ticker}: {e}")

def update_data(ticker, engine):
    try:
        with engine.connect() as conn:
            result = conn.execute(text(f"SELECT MAX(Date) FROM '{ticker}'"))
            last_date = result.fetchone()[0]
            if last_date is None:
                last_date = "2006-01-01"
            else:
                last_date = datetime.strptime(str(last_date)[:10], '%Y-%m-%d') + timedelta(days=1)
                last_date = last_date.strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        print(f"Updating data for {ticker} from {last_date} to {end_date}...")
        df = yf.download(ticker, start=last_date, end=end_date)
        df.to_sql(ticker, engine, if_exists='append')
        print(f"Data for {ticker} updated in the database.")
        logging.info(f"Data for {ticker} updated in the database.")
    except Exception as e:
        print(f"Error updating data for {ticker}: {e}")
        logging.error(f"Error updating data for {ticker}: {e}")

def load_data(engine, ticker):
    print(f"Loading data for {ticker} from the database...")
    query = text(f'SELECT * FROM "{ticker}"')
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

def calculate_daily_returns(df, column='Adj Close'):
    df['Daily Return'] = df[column].pct_change()
    return df

def prepare_data(engine):
    spy_df = load_data(engine, "SPY")
    vix_df = load_data(engine, "^VIX")
    
    print("Aligning SPY and VIX data...")
    spy_df, vix_df = spy_df.align(vix_df, join='inner', axis=0)
    
    print("Filling missing data if any...")
    spy_df.fillna(method='ffill', inplace=True)
    vix_df.fillna(method='ffill', inplace=True)
    
    spy_df = calculate_daily_returns(spy_df)
    
    return spy_df, vix_df

def calculate_sharpe_ratio(returns, risk_free_rate):
    avg_return = returns.mean()
    std_dev = returns.std()
    sharpe_ratio = (avg_return - risk_free_rate) / std_dev
    return sharpe_ratio

def implement_strategy(spy_df, vix_df, strategy):
    print("Implementing strategy...")
    
    spy_df['SPY Allocation'] = 100  # Default allocation
    spy_df['SH Allocation'] = 0     # Default allocation
    
    spy_df['VIX Level'] = 0  # Initialize VIX Level column
    
    for i, allocation in enumerate(strategy):
        if i == 36:
            spy_df.loc[vix_df['Adj Close'] >= i, 'SPY Allocation'] = allocation
            spy_df.loc[vix_df['Adj Close'] >= i, 'SH Allocation'] = 100 - allocation
            spy_df.loc[vix_df['Adj Close'] >= i, 'VIX Level'] = i
        else:
            spy_df.loc[vix_df['Adj Close'] == i, 'SPY Allocation'] = allocation
            spy_df.loc[vix_df['Adj Close'] == i, 'SH Allocation'] = 100 - allocation
            spy_df.loc[vix_df['Adj Close'] == i, 'VIX Level'] = i

    spy_df['Portfolio Return'] = (spy_df['Daily Return'] * spy_df['SPY Allocation'] / 100) - (spy_df['Daily Return'] * spy_df['SH Allocation'] / 100)

    return spy_df

def objective_function(strategy, spy_df, vix_df, risk_free_rate):
    spy_df = implement_strategy(spy_df.copy(), vix_df, strategy)
    sharpe_ratio = calculate_sharpe_ratio(spy_df['Portfolio Return'].dropna(), risk_free_rate)
    return -sharpe_ratio  # We negate because we want to maximize Sharpe ratio

def plot_total_returns(spy_df, start_date, end_date, title):
    filtered_df = spy_df.loc[start_date:end_date]
    filtered_df.loc[:, 'Cumulative Return'] = (1 + filtered_df['Portfolio Return']).cumprod()
    filtered_df.loc[:, 'SPY Cumulative Return'] = (1 + filtered_df['Daily Return']).cumprod()
    
    plt.figure(figsize=(14, 7))
    plt.plot(filtered_df['Cumulative Return'], label='Portfolio Cumulative Return')
    plt.plot(filtered_df['SPY Cumulative Return'], label='SPY Cumulative Return')
    plt.title(title)
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_optimized_ratios(strategy):
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(strategy)), strategy, color='blue', alpha=0.7)
    plt.xlabel('VIX Level')
    plt.ylabel('SPY Allocation (%)')
    plt.title('Optimized SPY Allocation Ratios')
    plt.grid(True)
    plt.show()

def plot_returns_build_up(spy_df):
    plt.figure(figsize=(10, 6))
    spy_df.loc[:, 'Cumulative Return'] = (1 + spy_df['Portfolio Return']).cumprod()
    for level in range(37):
        level_df = spy_df[spy_df['VIX Level'] == level]
        plt.plot(level_df.index, level_df['Cumulative Return'], label=f'VIX Level {level}')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Cumulative Returns Build-up for Different VIX Levels')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_comparison_before_after(spy_df, vix_df, initial_strategy, optimal_strategy):
    initial_df = implement_strategy(spy_df.copy(), vix_df, initial_strategy)
    optimal_df = implement_strategy(spy_df.copy(), vix_df, optimal_strategy)
    
    plt.figure(figsize=(14, 7))
    initial_df.loc[:, 'Cumulative Return'] = (1 + initial_df['Portfolio Return']).cumprod()
    optimal_df.loc[:, 'Cumulative Return'] = (1 + optimal_df['Portfolio Return']).cumprod()
    
    plt.plot(initial_df.index, initial_df['Cumulative Return'], label='Before Optimization')
    plt.plot(optimal_df.index, optimal_df['Cumulative Return'], label='After Optimization')
    
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.title('Comparison of Cumulative Returns Before and After Optimization')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    for ticker in tickers:
        download_and_save_data(ticker, start_date, end_date, engine)
        update_data(ticker, engine)
    
    spy_df, vix_df = prepare_data(engine)
    
    initial_strategy = strategy_df['SPY Allocation'].values
    bounds = [(0, 100) for _ in range(len(initial_strategy))]
    
    risk_free_rate = 0.03
    
    result = minimize(objective_function, initial_strategy, args=(spy_df, vix_df, risk_free_rate),
                      bounds=bounds, method='SLSQP')
    
    if not result.success:
        print("Optimization failed:", result.message)
        return
    
    optimal_strategy = result.x
    
    print("Optimal Strategy Allocations:")
    for level, allocation in enumerate(optimal_strategy):
        print(f"VIX Level {level}: SPY Allocation = {allocation:.2f}%, SH Allocation = {100 - allocation:.2f}%")
    
    spy_df = implement_strategy(spy_df, vix_df, optimal_strategy)
    
    print("\nSPY Daily Returns for the past 30 days:")
    print(spy_df[['Adj Close', 'Daily Return', 'SPY Allocation', 'SH Allocation', 'Portfolio Return']].tail(30))
    
    spy_df.to_csv('prepared_SPY.csv')
    vix_df.to_csv('prepared_VIX.csv')
    
    print("Data update and preparation complete. Prepared data saved to CSV files.")
    logging.info("Data update and preparation complete. Prepared data saved to CSV files.")
    
    sharpe_ratio = calculate_sharpe_ratio(spy_df['Portfolio Return'].dropna(), risk_free_rate)
    print(f"\nSharpe Ratio for the optimized strategy: {sharpe_ratio}")
    logging.info(f"Sharpe Ratio for the optimized strategy: {sharpe_ratio}")

    plt.figure(figsize=(14, 7))
    plt.plot(spy_df['Adj Close'], label='SPY Adjusted Close Price')
    plt.title('SPY Adjusted Close Price (2006 - Present)')
    plt.xlabel('Date')
    plt.ylabel('Adjusted Close Price')
    plt.legend()
    plt.grid(True)
    plt.show()

    spy_df.loc[:, 'Cumulative Return'] = (1 + spy_df['Portfolio Return']).cumprod()
    spy_df.loc[:, 'SPY Cumulative Return'] = (1 + spy_df['Daily Return']).cumprod()
    plt.figure(figsize=(14, 7))
    plt.plot(spy_df['Cumulative Return'], label='Portfolio Cumulative Return')
    plt.plot(spy_df['SPY Cumulative Return'], label='SPY Cumulative Return')
    plt.title('Cumulative Returns of Portfolio vs SPY')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Return')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.bar(['SPY', 'Optimised Strategy'], [calculate_sharpe_ratio(spy_df['Daily Return'].dropna(), risk_free_rate), sharpe_ratio])
    plt.title('Sharpe Ratio Comparison')
    plt.ylabel('Sharpe Ratio')
    plt.grid(True)
    plt.show()

    # Plot total returns from June 3, 2006, to present day
    plot_total_returns(spy_df, '2006-06-03', end_date, 'Total Returns from June 3, 2006 to Present Day')

    # Plot total returns from June 3, 2006, to June 2023
    plot_total_returns(spy_df, '2006-06-03', '2023-06-30', 'Total Returns from June 3, 2006 to June 2023')

    # New plots
    plot_optimized_ratios(optimal_strategy)
    plot_returns_build_up(spy_df)
    plot_comparison_before_after(spy_df, vix_df, initial_strategy, optimal_strategy)

def fetch_and_update():
    for ticker in tickers:
        update_data(ticker, engine)

schedule.every().day.at("09:00").do(lambda: schedule.every(1).minute.until("15:30").do(fetch_and_update))
schedule.every().day.at("15:30").do(schedule.clear)

if __name__ == "__main__":
    main()
    while True:
        schedule.run_pending()
        time.sleep(1)
