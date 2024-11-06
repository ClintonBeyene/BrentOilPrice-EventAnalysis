import logging
import pandas as pd
import numpy as np
import wbdata
from pathlib import Path

class WorldBankDataFetcher:
    """A class to handle fetching and processing World Bank data."""
    
    def __init__(self, start_date, end_date):
        """Initialize with date range for data collection."""
        self.start_date = start_date
        self.end_date = end_date
        self.setup_logging()
        
    def setup_logging(self):
        """Configure logging settings."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('worldbank_data_fetch.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

    def fetch_indicator_data(self, indicator_code, indicator_name, country='WLD'):
        """
        Fetch data for a specific indicator from World Bank.
        
        Args:
            indicator_code (str): World Bank indicator code
            indicator_name (str): Name for the indicator
            country (str): Country code (default: 'WLD' for World)
            
        Returns:
            pandas.DataFrame: Fetched and processed data
        """
        try:
            self.logger.info(f"Fetching {indicator_name} data...")
            data = wbdata.get_dataframe(
                {indicator_code: indicator_name},
                country=country,
                date=(self.start_date, self.end_date)
            )
            self.logger.info(f"Successfully fetched {indicator_name} data")
            return data
        except Exception as e:
            self.logger.error(f"Error fetching {indicator_name} data: {str(e)}")
            return None

    def process_data(self, df, indicator_name):
        """
        Process and clean the fetched data.
        
        Args:
            df (pandas.DataFrame): Input DataFrame
            indicator_name (str): Name of the indicator
            
        Returns:
            pandas.DataFrame: Processed DataFrame
        """
        try:
            if df is None or df.empty:
                self.logger.warning(f"No data available for {indicator_name}")
                return pd.DataFrame()

            # Clean and process the data
            df = df.reset_index()
            df.columns = ['date', indicator_name]
            df['date'] = pd.to_datetime(df['date'])
            
            # Handle missing values
            df[indicator_name] = df[indicator_name].replace([np.inf, -np.inf], np.nan)
            df.dropna(inplace=True)
            
            # Convert to daily frequency
            full_index = pd.date_range(start=df['date'].min(), end=df['date'].max(), freq='D')
            df_daily = df.set_index('date').reindex(full_index)
            df_daily.interpolate(method='cubic', inplace=True)
            df_daily.reset_index(inplace=True)
            df_daily.rename(columns={'index': 'Date'}, inplace=True)
            
            self.logger.info(f"Successfully processed {indicator_name} data")
            return df_daily
            
        except Exception as e:
            self.logger.error(f"Error processing {indicator_name} data: {str(e)}")
            return pd.DataFrame()

    def save_data(self, df, indicator_name):
        """
        Save processed data to CSV file.
        
        Args:
            df (pandas.DataFrame): DataFrame to save
            indicator_name (str): Name of the indicator
        """
        try:
            # Create data directory if it doesn't exist
            output_dir = Path("../data")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save file
            output_path = output_dir / f"{indicator_name}_cleaned_data_daily.csv"
            df.to_csv(output_path, index=False)
            self.logger.info(f"Successfully saved {indicator_name} data to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving {indicator_name} data: {str(e)}")

def main():
    # Define indicators
    indicators = {
        'NY.GDP.MKTP.CD': {'name': 'GDP', 'country': 'WLD'},
        'FP.CPI.TOTL.ZG': {'name': 'CPI', 'country': 'WLD'},
        'SL.UEM.TOTL.ZS': {'name': 'Unemployment_Rate', 'country': 'WLD'},
        'PA.NUS.FCRF': {'name': 'Exchange_Rate', 'country': 'EMU'}
    }
    
    # Initialize data fetcher
    fetcher = WorldBankDataFetcher('1987-05-20', '2022-11-14')
    
    # Process each indicator
    for indicator_code, info in indicators.items():
        # Fetch data
        raw_data = fetcher.fetch_indicator_data(
            indicator_code,
            info['name'],
            info['country']
        )
        
        # Process data
        processed_data = fetcher.process_data(raw_data, info['name'])
        
        # Save data
        if not processed_data.empty:
            fetcher.save_data(processed_data, info['name'])

if __name__ == "__main__":
    main()