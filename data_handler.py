import os
import pandas as pd
import numpy as np
import yfinance as yf


def fetch_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close']


def save_data_to_file(df, ticker):
    if not os.path.exists('input'):
        os.makedirs('input')
    file_path = os.path.join('input', f'{ticker}.csv')
    df.to_csv(file_path)


def load_data_from_file(ticker):
    file_path = os.path.join('input', f'{ticker}.csv')
    if os.path.exists(file_path):
        return pd.read_csv(file_path, index_col=0, parse_dates=True)
    else:
        return None


def get_data(ticker, start_date, end_date):
    df_data = load_data_from_file(ticker)
    if df_data is None:
        df_data = fetch_data(ticker, start_date, end_date)
        save_data_to_file(df_data, ticker)
    return df_data