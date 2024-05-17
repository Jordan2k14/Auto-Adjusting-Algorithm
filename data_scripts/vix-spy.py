import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
import logging

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
    
    # Now we ensure the dates align
    print("Aligning SPY and VIX data...")
    spy_df, vix_df = spy_df.align(vix_df, join='inner', axis=0)
    
    # Always important to check for missing data
    print("Filling missing data if any...")
    spy_df.fillna(method='ffill', inplace=True)
    vix_df.fillna(method='ffill', inplace=True)
    
    # Calculating daily returns for SPY 
    spy_df = calculate_daily_returns(spy_df)
    
    return spy_df, vix_df

def main():
    for ticker in tickers:
        download_and_save_data(ticker, start_date, end_date, engine)
        update_data(ticker, engine)
    
    spy_df, vix_df = prepare_data(engine)
    
    # Print daily returns calculations
    print("\nSPY Daily Returns:")
    print(spy_df[['Adj Close', 'Daily Return']].tail())
    
    print("\nVIX Data:")
    print(vix_df[['Adj Close']].tail())
    
    # Save the prepared data to CSV files
    spy_df.to_csv(os.path.join(os.path.dirname(__file__), 'prepared_SPY.csv'))
    vix_df.to_csv(os.path.join(os.path.dirname(__file__), 'prepared_VIX.csv'))
    
    print("Data update and preparation complete. Prepared data saved to CSV files.")
    logging.info("Data update and preparation complete. Prepared data saved to CSV files.")

if __name__ == "__main__":
    main()
