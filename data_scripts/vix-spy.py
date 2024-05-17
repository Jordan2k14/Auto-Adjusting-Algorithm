import yfinance as yf
import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import os
import logging

# We are Setting up logging
logging.basicConfig(filename=os.path.join(os.path.dirname(__file__), 'vix_spy.log'), 
                    level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# We are defining both the SPY and VIX from 2006 -> Onwards
tickers = ["SPY", "^VIX"]
start_date = "2006-01-01"
end_date = datetime.now().strftime('%Y-%m-%d')

# We are setting the database path
db_path = os.path.join(os.path.dirname(__file__), 'financial_data.db')
engine = create_engine(f'sqlite:///{db_path}')

# Function to download and save data to SQLite database
def download_and_save_data(ticker, start, end, engine):
    try:
        df = yf.download(ticker, start=start, end=end)
        df.to_sql(ticker, engine, if_exists='replace')
        logging.info(f"Data for {ticker} saved to database.")
    except Exception as e:
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
                last_date = (datetime.strptime(last_date, '%Y-%m-%d') + timedelta(days=1)).strftime('%Y-%m-%d')
        
        end_date = datetime.now().strftime('%Y-%m-%d')
        df = yf.download(ticker, start=last_date, end=end_date)
        df.to_sql(ticker, engine, if_exists='append')
        logging.info(f"Data for {ticker} updated in the database.")
    except Exception as e:
        logging.error(f"Error updating data for {ticker}: {e}")

# Main function to download or update data every midnight
def main():
    for ticker in tickers:
        download_and_save_data(ticker, start_date, end_date, engine)
        update_data(ticker, engine)

if __name__ == "__main__":
    main()
