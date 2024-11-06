import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose

class OilPriceVisualizer:
    def __init__(self, df):
        self.df = df
        self.df['Year'] = self.df.index.year

    def plot_price_trend(self):
        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Price'], label='Brent Oil Price')
        plt.title('Brent Oil Price Over Time')
        plt.xlabel('Date')
        plt.ylabel('Price (USD per barrel)')
        plt.legend()
        plt.show()

    def plot_yearly_average(self):
        aggregated_df = self.df.groupby('Year')['Price'].mean().reset_index()
        plt.figure(figsize=(12, 6))
        sns.barplot(x='Year', y='Price', data=aggregated_df, palette='viridis')
        plt.title('Average Yearly Brent Oil Prices', fontsize=16)
        plt.xlabel('Year', fontsize=14)
        plt.ylabel('Average Price (USD per barrel)', fontsize=14)
        plt.xticks(rotation=45)
        plt.grid(axis='y')
        plt.tight_layout()
        plt.show()

    def plot_with_events(self):
        significant_events = {
            '1990-08-02': 'Start-Gulf War',
            '1991-02-28': 'End-Gulf War',
            '2001-09-11': '9/11 Terrorist Attacks',
            '2003-03-20': 'Invasion of Iraq',
            '2005-07-07': 'London Terrorist Attack',
            '2010-12-18': 'Start-Arab Spring',
            '2011-02-17': 'Civil War in Libya Start',
            '2015-11-13': 'Paris Terrorist Attacks',
            '2019-12-31': 'Attack on US Embassy in Iraq',
            '2022-02-24': 'Russian Invasion of Ukraine',
        }

        plt.figure(figsize=(14, 7))
        plt.plot(self.df['Price'], label='Brent Oil Price')
        for date, event in significant_events.items():
            plt.axvline(pd.to_datetime(date), color='r', linestyle='--', label=f'{event} ({date})')
        plt.title('Brent Oil Price Over Time with Event Markers')
        plt.xlabel('Date')
        plt.ylabel('Price (USD per barrel)')
        plt.legend(loc='best')
        plt.show()

    def plot_rolling_volatility(self, window=30):
        self.df['Rolling_Volatility'] = self.df['Price'].rolling(window=window).std()
        plt.figure(figsize=(12, 6))
        plt.plot(self.df.index, self.df['Rolling_Volatility'], label=f'{window}-Day Rolling Volatility', color='blue')
        plt.title(f'{window}-Day Rolling Volatility of Brent Oil Prices', fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Rolling Volatility', fontsize=14)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def plot_seasonal_decomposition(self):
        decomposition = seasonal_decompose(self.df['Price'], model='additive', period=365)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid

        plt.figure(figsize=(14, 10))

        plt.subplot(411)
        sns.lineplot(x=self.df.index, y=self.df['Price'], label='Original', color='blue')
        plt.legend(loc='best')
        plt.title('Brent Oil Prices - Seasonal Decomposition')

        plt.subplot(412)
        sns.lineplot(x=trend.index, y=trend, label='Trend', color='green')
        plt.legend(loc='best')

        plt.subplot(413)
        sns.lineplot(x=seasonal.index, y=seasonal, label='Seasonal', color='red')
        plt.legend(loc='best')

        plt.subplot(414)
        sns.lineplot(x=residual.index, y=residual, label='Residual', color='purple')
        plt.legend(loc='best')

        plt.tight_layout()
        plt.show()