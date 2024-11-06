import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import grangercausalitytests
from sklearn.preprocessing import StandardScaler

try:
    import seaborn as sns
except ImportError:
    print("Warning: Seaborn not found. Some plot styles may be affected.")
    sns = None

class OilPriceAnalyzer:
    def __init__(self, oil_data):
        self.oil_data = oil_data
        self.setup_plot_style()
    
    def setup_plot_style(self):
        plt.style.use('default')  # Use default Matplotlib style
        if sns is not None and sns.color_palette() is not None:
            sns.set_palette("husl")
        else:
            print("Warning: Seaborn color palette not available. Using default colors.")
        
    def merge_and_clean_data(self, indicator_data, indicator_name):
        merged_data = pd.merge(indicator_data, self.oil_data.reset_index(), on='Date')
        merged_data.dropna(inplace=True)
        
        Q1 = merged_data[indicator_name].quantile(0.25)
        Q3 = merged_data[indicator_name].quantile(0.75)
        IQR = Q3 - Q1
        merged_data = merged_data[
            (merged_data[indicator_name] >= Q1 - 1.5 * IQR) & 
            (merged_data[indicator_name] <= Q3 + 1.5 * IQR)
        ]
        
        return merged_data
    
    def calculate_statistics(self, merged_data, indicator_name):
        stats_dict = {}
        
        correlation, p_value = stats.pearsonr(
            merged_data[indicator_name],
            merged_data['Price']
        )
        
        stats_dict['correlation'] = correlation
        stats_dict['p_value'] = p_value
        stats_dict['r_squared'] = correlation ** 2
        
        merged_data['rolling_corr'] = merged_data[indicator_name].rolling(
            window=180
        ).corr(merged_data['Price'])
        
        return stats_dict, merged_data
    
    def create_visualization(self, merged_data, indicator_name, x_label, stats_dict):
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'Analysis of {indicator_name} vs Oil Prices', fontsize=16)
        
        # Scatter plot with regression line
        axes[0, 0].scatter(merged_data[indicator_name], merged_data['Price'], alpha=0.5)
        z = np.polyfit(merged_data[indicator_name], merged_data['Price'], 1)
        p = np.poly1d(z)
        axes[0, 0].plot(merged_data[indicator_name], p(merged_data[indicator_name]), "r--")
        axes[0, 0].set_title('Scatter Plot with Regression Line')
        axes[0, 0].set_xlabel(x_label)
        axes[0, 0].set_ylabel('Oil Price ($)')
        
        # Rolling correlation plot
        merged_data['rolling_corr'].plot(ax=axes[0, 1])
        axes[0, 1].set_title('6-Month Rolling Correlation')
        axes[0, 1].set_xlabel('Date')
        axes[0, 1].set_ylabel('Correlation Coefficient')
        
        # Joint distribution plot (simplified)
        axes[1, 0].hist2d(merged_data[indicator_name], merged_data['Price'], bins=50)
        axes[1, 0].set_title('Joint Distribution')
        axes[1, 0].set_xlabel(x_label)
        axes[1, 0].set_ylabel('Oil Price ($)')
        
        # Time series plot
        ax2 = axes[1, 1].twinx()
        merged_data[indicator_name].plot(ax=axes[1, 1], color='blue', label=indicator_name)
        merged_data['Price'].plot(ax=ax2, color='red', label='Oil Price')
        axes[1, 1].set_title('Time Series Comparison')
        axes[1, 1].set_xlabel('Date')
        axes[1, 1].set_ylabel(x_label, color='blue')
        ax2.set_ylabel('Oil Price ($)', color='red')
        
        # Add statistics text box
        stats_text = (
            f'Correlation: {stats_dict["correlation"]:.3f}\n'
            f'P-value: {stats_dict["p_value"]:.3e}\n'
            f'R-squared: {stats_dict["r_squared"]:.3f}'
        )
        fig.text(0.02, 0.02, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.show()
    
    def analyze_granger_causality(self, merged_data, indicator_name, max_lag=12):
        data = pd.DataFrame({
            'indicator': merged_data[indicator_name],
            'oil_price': merged_data['Price']
        })
        
        scaler = StandardScaler()
        data_scaled = pd.DataFrame(
            scaler.fit_transform(data),
            columns=data.columns,
            index=data.index
        )
        
        print(f"\nGranger Causality Test Results for {indicator_name}:")
        print("\nTesting if indicator Granger-causes oil prices:")
        grangercausalitytests(data_scaled[['indicator', 'oil_price']], maxlag=max_lag, verbose=False)
        print("\nTesting if oil prices Granger-cause indicator:")
        grangercausalitytests(data_scaled[['oil_price', 'indicator']], maxlag=max_lag, verbose=False)
    
    def analyze_indicator(self, indicator_data, indicator_name, x_label):
        print(f"\nAnalyzing relationship between {indicator_name} and oil prices:")
        
        merged_data = self.merge_and_clean_data(indicator_data, indicator_name)
        stats_dict, merged_data = self.calculate_statistics(merged_data, indicator_name)
        self.create_visualization(merged_data, indicator_name, x_label, stats_dict)
        self.analyze_granger_causality(merged_data, indicator_name)

def analyze_indicators(gdp_data, inflation_data, unemployment_data, exchange_rate_data, oil_data):
    """
    Analyze relationships between all economic indicators and oil prices.
    """
    analyzer = OilPriceAnalyzer(oil_data)
    
    # Analyze each indicator
    analyzer.analyze_indicator(gdp_data, 'GDP', 'GDP (current US$)')
    analyzer.analyze_indicator(inflation_data, 'CPI', 'Inflation Rate (%)')
    analyzer.analyze_indicator(unemployment_data, 'Unemployment_Rate', 'Unemployment Rate (%)')
    analyzer.analyze_indicator(exchange_rate_data, 'Exchange_Rate', 'Exchange Rate (USD to Local Currency)')