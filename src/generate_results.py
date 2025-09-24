import pandas as pd
import pandas_datareader as pdr
import yfinance as yf
from datetime import datetime

def collect_economic_data():
    """Get FRED economic indicators"""
    start_date = '2024-01-01'
    end_date = '2025-07-1'
    
    fuel_prices = pdr.get_data_fred('DJAAFUELJAN', start_date, end_date)
    consumer_conf = pdr.get_data_fred('UMCSENT', start_date, end_date)
    gdp = pdr.get_data_fred('GDP', start_date, end_date)
    
    return fuel_prices, consumer_conf, gdp

def collect_stock_data():
    """Get airline stock prices and earnings"""
    tickers = ['AAL', 'DAL', 'UAL', 'LUV']
    stock_data = {}
    
    for ticker in tickers:
        stock = yf.Ticker(ticker)
        stock_data[ticker] = {
            'prices': stock.history(start='2023-01-01'),
            'earnings': stock.quarterly_earnings
        }
    
    return stock_data

def load_manual_data():
    """Load your manually created CSVs"""
    airline_data = pd.read_csv('airline_quarterly_data.csv')
    tsa_data = pd.read_csv('tsa_daily_total.csv')
    
    return airline_data, tsa_data

if __name__ == "__main__":
    # Run all data collection
    print("Collecting economic data...")
    fuel, confidence, gdp = collect_economic_data()
    
    print("Collecting stock data...")
    stocks = collect_stock_data()
    
    print("Loading manual data...")
    airlines, tsa = load_manual_data()
    
    print("All data collected!")