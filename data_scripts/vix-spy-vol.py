import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
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

def calculate_annual_return(returns):
    cumulative_return = (1 + returns).prod()
    num_years = len(returns) / 252
    annual_return = cumulative_return ** (1 / num_years) - 1
    return annual_return

def calculate_annual_volatility(returns):
    return returns.resample('Y').std() * (252 ** 0.5)

def calculate_average_annual_volatility(returns):
    annual_volatility = calculate_annual_volatility(returns)
    return annual_volatility.mean()

def implement_strategy(spy_df, vix_df, strategy):
    print("Implementing strategy...")
    
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

def objective_function_annual_return(strategy, spy_df, vix_df):
    spy_df = implement_strategy(spy_df.copy(), vix_df, strategy)
    annual_return = calculate_annual_return(spy_df['Portfolio Return'].dropna())
    return -annual_return  # We negate because we want to maximize annual return

def objective_function_annual_volatility(strategy, spy_df, vix_df):
    spy_df = implement_strategy(spy_df.copy(), vix_df, strategy)
    annual_volatility = calculate_annual_volatility(spy_df['Portfolio Return'].dropna())
    return annual_volatility.mean()  # We want to minimize average annual volatility

# Function to plot yearly volatility after optimization
def plot_yearly_volatility(spy_df, vix_df, optimal_strategy):
    spy_df = implement_strategy(spy_df.copy(), vix_df, optimal_strategy)
    yearly_volatility = calculate_annual_volatility(spy_df['Portfolio Return'].dropna())
    
    plt.figure(figsize=(14, 7))
    plt.plot(yearly_volatility.index, yearly_volatility, label='Yearly Volatility')
    plt.xlabel('Year')
    plt.ylabel('Volatility')
    plt.title('Yearly Volatility After Optimization')
    plt.legend()
    plt.grid(True)
    for x, y in zip(yearly_volatility.index, yearly_volatility):
        plt.text(x, y, f'{y:.2%}', ha='center')
    plt.show()

# Function to plot average yearly volatility after optimization
def plot_average_yearly_volatility(spy_df, vix_df, optimal_strategy):
    spy_df = implement_strategy(spy_df.copy(), vix_df, optimal_strategy)
    avg_annual_volatility = calculate_average_annual_volatility(spy_df['Portfolio Return'].dropna())
    
    plt.figure(figsize=(14, 7))
    plt.bar(['Average Annual Volatility'], [avg_annual_volatility], color='blue', alpha=0.7)
    plt.ylabel('Volatility')
    plt.title('Average Yearly Volatility After Optimization')
    for i, val in enumerate([avg_annual_volatility]):
        plt.text(i, val, f'{val:.2%}', ha='center', va='bottom')
    plt.show()

# Function to compare yearly average volatility vs optimized yearly average volatility
def plot_comparison_yearly_volatility(spy_df, vix_df, initial_strategy, optimal_strategy):
    initial_df = implement_strategy(spy_df.copy(), vix_df, initial_strategy)
    optimal_df = implement_strategy(spy_df.copy(), vix_df, optimal_strategy)
    
    initial_volatility = calculate_average_annual_volatility(initial_df['Portfolio Return'].dropna())
    optimized_volatility = calculate_average_annual_volatility(optimal_df['Portfolio Return'].dropna())
    
    plt.figure(figsize=(14, 7))
    plt.bar(['Initial', 'Optimized'], [initial_volatility, optimized_volatility], color=['red', 'green'], alpha=0.7)
    plt.ylabel('Volatility')
    plt.title('Yearly Average Volatility: Initial vs Optimized')
    for i, val in enumerate([initial_volatility, optimized_volatility]):
        plt.text(i, val, f'{val:.2%}', ha='center', va='bottom')
    plt.show()

def main():
    for ticker in tickers:
        download_and_save_data(ticker, start_date, end_date, engine)
        update_data(ticker, engine)
    
    spy_df, vix_df = prepare_data(engine)
    
    initial_strategy = strategy_df['SPY Allocation'].values
    bounds = [(0.01, 0.99) for _ in range(len(initial_strategy))]  # Adding constraints to avoid extreme allocations
    
    result_annual = minimize(objective_function_annual_return, initial_strategy, args=(spy_df, vix_df),
                      bounds=bounds, method='SLSQP')
    
    if not result_annual.success:
        print("Optimization for average annual return failed:", result_annual.message)
        return
    
    optimal_strategy_annual = result_annual.x
    
    result_annual_volatility = minimize(objective_function_annual_volatility, initial_strategy, args=(spy_df, vix_df),
                      bounds=bounds, method='SLSQP')
    
    if not result_annual_volatility.success:
        print("Optimization for annual volatility failed:", result_annual_volatility.message)
        return
    
    optimal_strategy_annual_volatility = result_annual_volatility.x
    
    print("Optimal Strategy Allocations for Average Annual Return:")
    for level, allocation in enumerate(optimal_strategy_annual):
        print(f"VIX Level {level}: SPY Allocation = {allocation:.2f}, SH Allocation = {1 - allocation:.2f}")
    
    print("Optimal Strategy Allocations for Annual Volatility:")
    for level, allocation in enumerate(optimal_strategy_annual_volatility):
        print(f"VIX Level {level}: SPY Allocation = {allocation:.2f}, SH Allocation = {1 - allocation:.2f}")
    
    spy_df_annual = implement_strategy(spy_df.copy(), vix_df, optimal_strategy_annual)
    spy_df_annual_volatility = implement_strategy(spy_df.copy(), vix_df, optimal_strategy_annual_volatility)
    
    print("\nSPY Daily Returns for the past 30 days:")
    print(spy_df_annual[['Adj Close', 'Daily Return', 'SPY Allocation', 'SH Allocation', 'Portfolio Return']].tail(30))
    
    spy_df_annual.to_csv('prepared_SPY_annual.csv')
    spy_df_annual_volatility.to_csv('prepared_SPY_annual_volatility.csv')
    vix_df.to_csv('prepared_VIX.csv')
    
    print("Data update and preparation complete. Prepared data saved to CSV files.")
    logging.info("Data update and preparation complete. Prepared data saved to CSV files.")
    
    # New plots
    plot_yearly_volatility(spy_df, vix_df, optimal_strategy_annual_volatility)
    plot_average_yearly_volatility(spy_df, vix_df, optimal_strategy_annual_volatility)
    plot_comparison_yearly_volatility(spy_df, vix_df, initial_strategy, optimal_strategy_annual_volatility)

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
