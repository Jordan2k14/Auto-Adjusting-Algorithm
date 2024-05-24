import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import logging
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np
import time

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
strategy_path = 'strat-1-v2.csv'
strategy_df = pd.read_csv(strategy_path)
strategy_df.columns = ['VIX Level', 'SPY Allocation', 'SH Allocation']

# Handle "36+" VIX Level (1-36+)
strategy_df['VIX Level'] = strategy_df['VIX Level'].replace('36+', '36').astype(int)

# Convert strategy allocations to numeric to make them parsable
strategy_df['SPY Allocation'] = strategy_df['SPY Allocation'].str.rstrip('%').astype('float') / 100.0
strategy_df['SH Allocation'] = strategy_df['SH Allocation'].str.rstrip('%').astype('float') / 100.0

def download_and_save_data(ticker, start, end, engine):
    try:
        df = yf.download(ticker, start=start, end=end)
        df.to_sql(ticker, engine, if_exists='replace')
        logging.info(f"Data for {ticker} saved to database.")
    except Exception as e:
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
        df = yf.download(ticker, start=last_date, end=end_date)
        if not df.empty:
            df.to_sql(ticker, engine, if_exists='append')
        logging.info(f"Data for {ticker} updated in the database.")
    except Exception as e:
        logging.error(f"Error updating data for {ticker}: {e}")

def load_data(engine, ticker):
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
    
    spy_df, vix_df = spy_df.align(vix_df, join='inner', axis=0)
    
    spy_df.fillna(method='ffill', inplace=True)
    vix_df.fillna(method='ffill', inplace=True)
    
    spy_df = calculate_daily_returns(spy_df)
    
    return spy_df, vix_df

def calculate_monthly_return(returns):
    cumulative_return = (1 + returns).prod()
    num_months = len(returns) / 21  # Approximate number of trading days in a month
    monthly_return = cumulative_return ** (1 / num_months) - 1
    return monthly_return

def implement_strategy(spy_df, vix_df, strategy):
    spy_df['SPY Allocation'] = 1.0  # Default allocation as a fraction (100%)
    spy_df['SH Allocation'] = 0.0   # Default allocation as a fraction (0%)
    
    spy_df['VIX Level'] = 0  # Initialize VIX Level column
    
    for i in range(37):
        allocation = strategy[i]
        if i == 36:
            spy_df.loc[vix_df['Adj Close'] >= i, 'SPY Allocation'] = allocation
            spy_df.loc[vix_df['Adj Close'] >= i, 'SH Allocation'] = 1 - allocation
            spy_df.loc[vix_df['Adj Close'] >= i, 'VIX Level'] = i
        else:
            spy_df.loc[vix_df['Adj Close'] == i, 'SPY Allocation'] = allocation
            spy_df.loc[vix_df['Adj Close'] == i, 'SH Allocation'] = 1 - allocation
            spy_df.loc[vix_df['Adj Close'] == i, 'VIX Level'] = i

    spy_df['Portfolio Return'] = (spy_df['Daily Return'] * spy_df['SPY Allocation']) - (spy_df['Daily Return'] * spy_df['SH Allocation'])

    return spy_df

def objective_function_monthly_return(strategy, spy_df, vix_df):
    strategy = np.clip(strategy, 0.01, 0.99)  # Ensure strategy values are within bounds
    spy_df = implement_strategy(spy_df.copy(), vix_df, strategy)
    monthly_return = calculate_monthly_return(spy_df['Portfolio Return'].dropna())
    return -monthly_return  # We negate because we want to maximize monthly return

def optimize_strategy(spy_df, vix_df, end_date):
    initial_strategy = strategy_df['SPY Allocation'].values
    bounds = [(0.01, 0.99) for _ in range(len(initial_strategy))]  # Adding constraints to avoid extreme allocations
    
    spy_df_period = spy_df.loc[:end_date]
    vix_df_period = vix_df.loc[:end_date]
    
    result = minimize(objective_function_monthly_return, initial_strategy, args=(spy_df_period, vix_df_period),
                      bounds=bounds, method='SLSQP')
    
    if not result.success:
        logging.error(f"Optimization failed for period ending {end_date}: {result.message}")
        return initial_strategy
    
    return result.x

def plot_best_monthly_returns(spy_df, vix_df, start_date, end_dates):
    monthly_returns = []
    months = []
    
    for end in end_dates:
        optimal_strategy = optimize_strategy(spy_df, vix_df, end)
        spy_df_optimized = implement_strategy(spy_df.copy(), vix_df, optimal_strategy)
        monthly_return = calculate_monthly_return(spy_df_optimized['Portfolio Return'].dropna())
        monthly_returns.append(monthly_return * 100)  # Convert to percentage
        months.append(end[:7])  # Extract year and month
    
    plt.figure(figsize=(14, 7))
    plt.bar(months, monthly_returns, color='skyblue')
    plt.xlabel('Month')
    plt.ylabel('Best Monthly Return (%)')
    plt.title('Best Monthly Returns')
    plt.xticks(rotation=90)
    plt.grid(True)
    
    for i, v in enumerate(monthly_returns):
        plt.text(i, v + 0.2, f"{v:.2f}%", ha='center', va='bottom')
    
    plt.show()

def main():
    start_time = time.time()
    
    for ticker in tickers:
        download_and_save_data(ticker, start_date, end_date, engine)
        update_data(ticker, engine)
    
    spy_df, vix_df = prepare_data(engine)
    
    # Generate a list of end dates for each month from start_date to end_date
    end_dates = pd.date_range(start=start_date, end=end_date, freq='M').strftime('%Y-%m-%d').tolist()
    
    plot_best_monthly_returns(spy_df, vix_df, start_date, end_dates)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total time taken: {elapsed_time:.2f} seconds")

if __name__ == "__main__":
    main()
