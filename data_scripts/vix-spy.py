import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
import logging
from scipy.optimize import minimize

# Set up logging
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), 'vix_spy.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Define the tickers and the date range
tickers = ["SPY", "^VIX"]
start_date = "2006-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')

# Set the path for the SQLite database
db_path = os.path.join(os.path.dirname(__file__), 'financial_data.db')
engine = create_engine(f'sqlite:///{db_path}')

# Load strategy from CSV
strategy_path = '/Users/jordanukawoko/Library/Mobile Documents/com~apple~CloudDocs/Auto-Adjusting/Auto-Adjusting/data_scripts/strat-1.csv'
strategy_df = pd.read_csv(strategy_path)
strategy_df.columns = ['VIX Level', 'SPY Allocation', 'SH Allocation']

# Handle "36+" VIX Level
strategy_df['VIX Level'] = strategy_df['VIX Level'].replace('36+', '36').astype(int)

# Convert strategy allocations to numeric
strategy_df['SPY Allocation'] = strategy_df['SPY Allocation'].str.rstrip('%').astype('float') / 100.0 * 100
strategy_df['SH Allocation'] = strategy_df['SH Allocation'].str.rstrip('%').astype('float') / 100.0 * 100

# Function to download and save data to SQLite database
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

# Function to update data in the SQLite database
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

# Load Data from SQLite database
def load_data(engine, ticker):
    print(f"Loading data for {ticker} from the database...")
    query = text(f'SELECT * FROM "{ticker}"')
    with engine.connect() as conn:
        df = pd.read_sql(query, conn, parse_dates=['Date'])
    df.set_index('Date', inplace=True)
    return df

# Calculate daily returns
def calculate_daily_returns(df, column='Adj Close'):
    df['Daily Return'] = df[column].pct_change()
    return df

# Preparing the data
def prepare_data(engine):
    # Load data
    spy_df = load_data(engine, "SPY")
    vix_df = load_data(engine, "^VIX")
    
    # Ensure the dates align
    print("Aligning SPY and VIX data...")
    spy_df, vix_df = spy_df.align(vix_df, join='inner', axis=0)
    
    # Check for missing data
    print("Filling missing data if any...")
    spy_df.fillna(method='ffill', inplace=True)
    vix_df.fillna(method='ffill', inplace=True)
    
    # Calculate daily returns for SPY
    spy_df = calculate_daily_returns(spy_df)
    
    return spy_df, vix_df

# Calculate Sharpe ratio
def calculate_sharpe_ratio(returns, risk_free_rate):
    avg_return = returns.mean()
    std_dev = returns.std()
    sharpe_ratio = (avg_return - risk_free_rate) / std_dev
    return sharpe_ratio

# Implement strategy
def implement_strategy(spy_df, vix_df, strategy):
    print("Implementing strategy...")
    
    spy_df['SPY Allocation'] = 100  # Default allocation
    spy_df['SH Allocation'] = 0     # Default allocation
    
    for i, allocation in enumerate(strategy):
        if i == 36:
            spy_df.loc[vix_df['Adj Close'] >= i, 'SPY Allocation'] = allocation
            spy_df.loc[vix_df['Adj Close'] >= i, 'SH Allocation'] = 100 - allocation
        else:
            spy_df.loc[vix_df['Adj Close'] == i, 'SPY Allocation'] = allocation
            spy_df.loc[vix_df['Adj Close'] == i, 'SH Allocation'] = 100 - allocation

    spy_df['Portfolio Return'] = (spy_df['Daily Return'] * spy_df['SPY Allocation'] / 100)

    return spy_df

# Objective function for optimization
def objective_function(strategy, spy_df, vix_df, risk_free_rate):
    spy_df = implement_strategy(spy_df.copy(), vix_df, strategy)
    sharpe_ratio = calculate_sharpe_ratio(spy_df['Portfolio Return'].dropna(), risk_free_rate)
    return -sharpe_ratio  # We negate because we want to maximize Sharpe ratio

def main():
    for ticker in tickers:
        download_and_save_data(ticker, start_date, end_date, engine)
        update_data(ticker, engine)
    
    spy_df, vix_df = prepare_data(engine)
    
    # Define initial strategy and bounds
    initial_strategy = strategy_df['SPY Allocation'].values
    bounds = [(0, 100) for _ in range(len(initial_strategy))]
    
    # Define risk-free rate
    risk_free_rate = 0.0001  # Example risk-free rate
    
    # Perform optimization
    result = minimize(objective_function, initial_strategy, args=(spy_df, vix_df, risk_free_rate),
                      bounds=bounds, method='SLSQP')
    
    if not result.success:
        print("Optimization failed:", result.message)
        return
    
    optimal_strategy = result.x
    
    print("Optimal Strategy Allocations:")
    for level, allocation in enumerate(optimal_strategy):
        print(f"VIX Level {level}: SPY Allocation = {allocation:.2f}%, SH Allocation = {100 - allocation:.2f}%")
    
    # Implement the optimal strategy
    spy_df = implement_strategy(spy_df, vix_df, optimal_strategy)
    
    # Print daily returns calculations
    print("\nSPY Daily Returns:")
    print(spy_df[['Adj Close', 'Daily Return', 'SPY Allocation', 'SH Allocation', 'Portfolio Return']].tail(10))
    
    # Save the prepared data to CSV files
    spy_df.to_csv(os.path.join(os.path.dirname(__file__), 'prepared_SPY.csv'))
    vix_df.to_csv(os.path.join(os.path.dirname(__file__), 'prepared_VIX.csv'))
    
    print("Data update and preparation complete. Prepared data saved to CSV files.")
    logging.info("Data update and preparation complete. Prepared data saved to CSV files.")
    
    # Calculate Sharpe ratio for the optimized strategy
    sharpe_ratio = calculate_sharpe_ratio(spy_df['Portfolio Return'].dropna(), risk_free_rate)
    print(f"\nSharpe Ratio for the optimized strategy: {sharpe_ratio}")
    logging.info(f"Sharpe Ratio for the optimized strategy: {sharpe_ratio}")

if __name__ == "__main__":
    main()
