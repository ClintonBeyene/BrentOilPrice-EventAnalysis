import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import pymc as pm
import arviz as az
from scipy import stats
from datetime import timedelta

class OilPriceAnalyzer:
    def __init__(self, df):
        self.df = df
        self.significant_events = {
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

    def test_stationarity(self, series, title, label, alpha=0.05):
        adf_result = adfuller(series)
        
        print('ADF Statistic:', adf_result[0])
        print('p-value:', adf_result[1])
        print('Critical Values:')
        for key, value in adf_result[4].items():
            print(f'   {key}: {value}')
        
        if adf_result[1] < alpha:
            print("The ADF test suggests the series is stationary.")
        else:
            print("The ADF test suggests the series is not stationary.")

        plt.figure(figsize=(10, 6))
        plt.plot(series, label=label)
        plt.title(title)
        plt.xlabel('Time')
        plt.ylabel('Price')
        plt.legend()
        plt.grid()
        plt.show()
        
        return adf_result[0], adf_result[1]

    def analyze_stationarity(self):
        data = self.df['Price']
        data_diff = data.diff().dropna()
        log_data = np.log(data)
        log_data_diff = log_data.diff().dropna()

        print("First Differencing:")
        self.test_stationarity(data_diff, title='First Differenced Brent Oil Prices', label='First Differenced Series')
        
        print("\nLog Differencing:")
        self.test_stationarity(log_data_diff, title='Log Differenced Brent Oil Prices', label='Log Differenced Series')

    def plot_acf_pacf(self):
        data_diff = self.df['Price'].diff().dropna()
        log_data_diff = np.log(self.df['Price']).diff().dropna()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plot_acf(data_diff, lags=40, ax=plt.gca())
        plt.title('ACF Plot for First Differenced Series')
        plt.subplot(1, 2, 2)
        plot_pacf(data_diff, lags=40, ax=plt.gca())
        plt.title('PACF Plot for First Differenced Series')
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plot_acf(log_data_diff, lags=40, ax=plt.gca())
        plt.title('ACF Plot for Log Differenced Series')
        plt.subplot(1, 2, 2)
        plot_pacf(log_data_diff, lags=40, ax=plt.gca())
        plt.title('PACF Plot for Log Differenced Series')
        plt.tight_layout()
        plt.show()

    def cusum_analysis(self):
        mean_price = self.df['Price'].mean()
        cusum = np.cumsum(self.df['Price'] - mean_price)
        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, cusum, label='CUSUM Price Deviations')
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Date')
        plt.ylabel('CUSUM Value')
        plt.title('CUSUM Analysis')
        plt.legend()
        plt.show()

    def bayesian_change_point(self):
        data = self.df['Price'].values
        prior_mu1 = np.mean(data)
        prior_mu2 = np.mean(data)

        # Define the Bayesian model
        with pm.Model() as model:
            # Define the change point
            change_point = pm.DiscreteUniform('change_point', lower=0, upper=len(data) - 1)

            # Segment-specific means and standard deviations
            mu1 = pm.Normal('mean_prior1', mu=prior_mu1, sigma=5)  # Mean for segment 1
            mu2 = pm.Normal('mean_prior2', mu=prior_mu2, sigma=5)  # Mean for segment 2
            sigma1 = pm.HalfNormal('sigma1', sigma=5)  # Std deviation for segment 1
            sigma2 = pm.HalfNormal('sigma2', sigma=5)  # Std deviation for segment 2

            # Likelihood with switching behavior at change point
            mu = pm.math.switch(change_point >= np.arange(len(data)), mu1, mu2)
            sigma = pm.math.switch(change_point >= np.arange(len(data)), sigma1, sigma2)
            likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=data)

            # Sample from the posterior
            trace = pm.sample(4000, tune=2000, chains=4, random_seed=42)


        # Plot convergence diagnostics
        az.plot_trace(trace)
        plt.show()

        # Plot posterior distributions
        az.plot_posterior(trace)
        plt.show()

        # Analyze the change point
        s_posterior = trace.posterior['change_point'].values.flatten()
        estimated_change_point = int(np.median(s_posterior))

        # Map change point index back to date
        change_point_date =self.df.index[estimated_change_point]
        print(f"Estimated Change Point Date: {change_point_date}")

        # Calculate 95% HDI for the change point
        hdi = az.hdi(s_posterior, hdi_prob=0.95)
        print(f"95% HDI for the Change Point: {hdi}")

        plt.figure(figsize=(14, 7))
        plt.plot(self.df.index, data, label='Brent Oil Price', color='blue')
        plt.axvline(change_point_date, color='red', linestyle='--', label=f'Change Point ({change_point_date.year})')
        plt.fill_betweenx([np.min(data), np.max(data)], self.df.index[int(hdi[0])], self.df.index[int(hdi[1])], color='red', alpha=0.2, label='95% HDI')
        plt.title('Bayesian Change Point Detection with PyMC')
        plt.xlabel('Date')
        plt.ylabel('Price (USD)')
        plt.legend()
        plt.grid()
        plt.show()


    def get_prices_around_event(self, event_date, days_before=30, days_after=30):
        before_date = event_date - timedelta(days=days_before)
        after_date = event_date + timedelta(days=days_after)
        prices_around_event = self.df[(self.df.index >= before_date) & (self.df.index <= after_date)]
        return prices_around_event

    def analyze_events(self):
        results = []
        for date_str, event_name in self.significant_events.items():
            event_date = pd.to_datetime(date_str)
            prices_around_event = self.get_prices_around_event(event_date, days_before=180, days_after=180)
            
            try:
                nearest_before_1m = self.df.index[self.df.index <= event_date - timedelta(days=30)][-1]
                nearest_after_1m = self.df.index[self.df.index >= event_date + timedelta(days=30)][0]
                price_before_1m = self.df.loc[nearest_before_1m, 'Price']
                price_after_1m = self.df.loc[nearest_after_1m, 'Price']
                change_1m = ((price_after_1m - price_before_1m) / price_before_1m) * 100
            except (IndexError, KeyError):
                change_1m = None
            
            try:
                nearest_before_3m = self.df.index[self.df.index <= event_date - timedelta(days=90)][-1]
                nearest_after_3m = self.df.index[self.df.index >= event_date + timedelta(days=90)][0]
                price_before_3m = self.df.loc[nearest_before_3m, 'Price']
                price_after_3m = self.df.loc[nearest_after_3m, 'Price']
                change_3m = ((price_after_3m - price_before_3m) / price_before_3m) * 100
            except (IndexError, KeyError):
                change_3m = None

            try:
                nearest_before_6m = self.df.index[self.df.index <= event_date - timedelta(days=180)][-1]
                nearest_after_6m = self.df.index[self.df.index >= event_date + timedelta(days=180)][0]
                price_before_6m = self.df.loc[nearest_before_6m, 'Price']
                price_after_6m = self.df.loc[nearest_after_6m, 'Price']
                change_6m = ((price_after_6m - price_before_6m) / price_before_6m) * 100
            except (IndexError, KeyError):
                change_6m = None
            
            if not prices_around_event.empty:
                try:
                    prices_before = prices_around_event.loc[:event_date]
                    prices_after = prices_around_event.loc[event_date:]
                    
                    cum_return_before = prices_before['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
                    cum_return_after = prices_after['Price'].pct_change().add(1).cumprod().iloc[-1] - 1
                except:
                    cum_return_before = None
                    cum_return_after = None
            else:
                cum_return_before = None
                cum_return_after = None
            
            results.append({
                "Event": event_name,
                "Date": date_str,
                "Change_1M": change_1m,
                "Change_3M": change_3m,
                "Change_6M": change_6m,
                "Cumulative Return Before": cum_return_before,
                "Cumulative Return After": cum_return_after
            })

        event_impact_df = pd.DataFrame(results)
        print("\nEvent Impact Analysis:")
        print(event_impact_df)

        self.visualize_event_impact(event_impact_df)

    def visualize_event_impact(self, event_impact_df):
        plt.figure(figsize=(14, 8))
        for date_str, event_name in self.significant_events.items():
            event_date = pd.to_datetime(date_str)
            prices_around_event = self.get_prices_around_event(event_date, days_before=180, days_after=180)
            
            if not prices_around_event.empty:
                plt.plot(prices_around_event.index, prices_around_event['Price'], label=f"{event_name} ({date_str})")
                plt.axvline(event_date, color='red', linestyle='--', linewidth=0.8)
                plt.text(event_date, prices_around_event['Price'].max(), event_name, 
                         rotation=90, verticalalignment='bottom', fontsize=8)

        plt.title("Brent Oil Price Trends Around Key Events")
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()

        changes_data = event_impact_df.melt(id_vars=["Event", "Date"], 
                                              value_vars=["Change_1M", "Change_3M", "Change_6M"])
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Bar plot for percentage changes
        sns.barplot(data=changes_data, x="Event", y="value", hue="variable", ax=axes[0])
        axes[0].set_title("Percentage Change in Brent Oil Prices Before and After Events")
        axes[0].set_ylabel("Percentage Change")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=45, ha='right')
        axes[0].legend(title="Change Period")

        returns_data = event_impact_df.melt(id_vars=["Event", "Date"], 
                                              value_vars=["Cumulative Return Before", "Cumulative Return After"])
        sns.barplot(data=returns_data, x="Event", y="value", hue="variable", ax=axes[1])
        axes[1].set_title("Cumulative Returns Before and After Events")
        axes[1].set_ylabel("Cumulative Return")
        axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45, ha='right')
        axes[1].legend(title="Return Period")

        plt.tight_layout()
        plt.show()

        self.statistical_analysis()

    def statistical_analysis(self):
        t_test_results = {}
        for date_str, event_name in self.significant_events.items():
            event_date = pd.to_datetime(date_str)
            prices_around = self.get_prices_around_event(event_date, days_before=180, days_after=180)
            
            if not prices_around.empty:
                before_prices = prices_around.loc[:event_date]['Price']
                after_prices = prices_around.loc[event_date:]['Price']
                
                if len(before_prices) > 1 and len(after_prices) > 1:
                    t_stat, p_val = stats.ttest_ind(before_prices, after_prices, nan_policy='omit')
                    t_test_results[event_name] = {"t-statistic": t_stat, "p-value": p_val}
                else:
                    t_test_results[event_name] = {"t-statistic": None, "p-value": None}

        t_test_df = pd.DataFrame(t_test_results).T
        print("\nT-Test Results:")
        print(t_test_df)