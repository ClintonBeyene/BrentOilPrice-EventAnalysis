# price_analysis.py

import pandas as pd
import numpy as np
from datetime import timedelta
from scipy import stats
import os

data_path = 'C:/Users/Hp/Desktop/Kifiya/week10/BrentOilPrice-EventAnalysis/data/Copy of BrentOilPrices.csv'

def load_price_data():
    data = pd.read_csv(data_path, parse_dates=['Date'])
    data['Date'] = pd.to_datetime(data['Date'], format='mixed')
    data.set_index('Date', inplace=True)
    return data

def get_prices_around_event(event_date, data, days_before=30, days_after=30):
    before_date = event_date - timedelta(days=days_before)
    after_date = event_date + timedelta(days=days_after)
    return data[(data.index >= before_date) & (data.index <= after_date)]

def calculate_analysis_metrics(data):
    volatility = np.std(data['Price']) / np.mean(data['Price'])
    avg_price_change = data['Price'].diff().abs().mean()
    min_price = data['Price'].min()
    max_price = data['Price'].max()
    total_price_change = data['Price'].iloc[-1] - data['Price'].iloc[0]
    correlation = data['Price'].corr(data['Date'].apply(lambda x: x.toordinal()))
    
    return {
        "volatility": round(volatility, 4),
        "average_price_change": round(avg_price_change, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "total_price_change": round(total_price_change, 2),
        "correlation": round(correlation, 4),
        "model_accuracy": {
            "RMSE": 2.3,  # Placeholder
            "MAE": 1.5    # Placeholder
        }
    }

def calculate_event_impact(event, date, price_data):
    event_date = pd.to_datetime(date)
    prices_around_event = get_prices_around_event(event_date, price_data, days_before=180, days_after=180)
    change_1m, change_3m, change_6m = None, None, None

    try:
        price_before_1m = price_data.loc[event_date - timedelta(days=30), 'Price']
        price_after_1m = price_data.loc[event_date + timedelta(days=30), 'Price']
        change_1m = ((price_after_1m - price_before_1m) / price_before_1m) * 100
    except KeyError:
        change_1m = None

    try:
        price_before_3m = price_data.loc[event_date - timedelta(days=90), 'Price']
        price_after_3m = price_data.loc[event_date + timedelta(days=90), 'Price']
        change_3m = ((price_after_3m - price_before_3m) / price_before_3m) * 100
    except KeyError:
        change_3m = None

    try:
        price_before_6m = price_data.loc[event_date - timedelta(days=180), 'Price']
        price_after_6m = price_data.loc[event_date + timedelta(days=180), 'Price']
        change_6m = ((price_after_6m - price_before_6m) / price_before_6m) * 100
    except KeyError:
        change_6m = None

    cum_return_before = prices_around_event.loc[:event_date]['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
    cum_return_after = prices_around_event.loc[event_date:]['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
    before_prices = prices_around_event.loc[:event_date]['Price']
    after_prices = prices_around_event.loc[event_date:]['Price']
    t_stat, p_val = stats.ttest_ind(before_prices, after_prices, nan_policy='omit')

    return {
        "Event": event,
        "Date": date,
        "Change_1M": change_1m,
        "Change_3M": change_3m,
        "Change_6M": change_6m,
        "Cumulative Return Before": float(cum_return_before),
        "Cumulative Return After": float(cum_return_after),
        "T-Statistic": float(t_stat),
        "P-Value": float(p_val)
    }

def calculate_price_trends(data):
    return {
        'prices': data['Price'].tolist(),
        'dates': data.index.strftime('%Y').tolist()  # Format dates as year strings
    }


def calculate_price_distribution(data, bin_size=5):
    # Calculate the minimum and maximum prices to determine the range
    min_price = data['Price'].min()
    max_price = data['Price'].max()

    # Create bins from min to max price
    bins = list(range(int(min_price), int(max_price) + bin_size, bin_size))
    
    # Use pd.cut to create a Series that indicates which bin each price belongs to
    data['PriceRange'] = pd.cut(data['Price'], bins=bins, right=False)

    # Group by the PriceRange and count the occurrences
    distribution = data.groupby('PriceRange').size().reset_index(name='Frequency')

    # Format the distribution for visualization
    distribution['PriceRange'] = distribution['PriceRange'].astype(str)  # Convert bins to string for better visualization
    return distribution.to_dict(orient='records')  # Return as a list of dictionaries


def calculate_yearly_average_price(data):
    yearly_avg = data['Price'].resample('YE').mean()
    return yearly_avg.reset_index().rename(columns={'Date': 'Year', 'Price': 'Average_Price'}).to_dict(orient='records')
