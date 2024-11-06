# Import necessary libraries
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.data_loading import DataLoader
import logging
import os
import sys

# Ensure the logs directory exists
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'data_handling.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DataHandler:
    def __init__(self, csv_file_path):
        """
        Initialize the DataHandler with the path to the CSV file.
        
        :param csv_file_path: Path to the CSV file containing the data.
        """
        self.csv_file_path = csv_file_path
        self.data_loader = DataLoader(csv_file_path)
        self.df = None

    def load_and_prepare_data(self):
        """
        Load the data, load it into the database, and query it back.
        """
        try:
            logging.info('Loading and preparing data')
            self.data_loader.load_data()
            self.data_loader.load_data_to_database()
            self.df = self.data_loader.query_data_from_database()
            if self.df is not None:
                logging.info('Data loaded from database')
                print("Data from the database is loaded")
            else:
                logging.error('Failed to load data from database')
                print("Failed to load data from database")
        except Exception as e:
            logging.error(f"Error loading and preparing data: {e}")

    def perform_eda(self):
        """
        Perform exploratory data analysis (EDA) on the DataFrame.
        """
        if self.df is None:
            logging.error('DataFrame is None. Cannot perform EDA.')
            print("DataFrame is None. Cannot perform EDA.")
            return

        try:
            logging.info('Performing exploratory data analysis')

            # View the head of the dataset
            print("Head of the dataset:")
            print(self.df.head())

            # Shape of the dataset
            print(f"Shape of the dataset: {self.df.shape}")

            # Information about the dataset
            print("Information about the dataset:")
            print(self.df.info())

            
            # Check for missing values
            print("Missing values in the dataset:")
            print(self.df.isnull().sum())

            # Descriptive statistics
            print("Descriptive statistics:")
            print(self.df['Price'].describe())

            # Check how symmetrical the price feature is
            skewness = self.df['Price'].skew()
            print(f"Skewness of the Price feature: {skewness}")
            if skewness > 0:
                print("The feature price is moderately skewed in a positive direction.")
            elif skewness < 0:
                print("The feature price is moderately skewed in a negative direction.")
            else:
                print("The feature price is symmetrically distributed.")

            # Checking outliers
            kurtosis = self.df['Price'].kurtosis()
            print(f"Kurtosis of the Price feature: {kurtosis}")
            if kurtosis > 0:
                print("The kurtosis value indicates heavier tails, suggesting the distribution has more extreme (outliers) values.")
            elif kurtosis < 0:
                print("The kurtosis value indicates lighter tails, suggesting the distribution has fewer extreme (outliers) values.")
            else:
                print("The kurtosis value is close to zero, indicating a normal distribution.")

            # Density plot
            plt.figure(figsize=(16, 10))
            self.df['Price'].plot(kind='density', figsize=(16, 10))
            plt.title("Density Plot of Price")
            plt.show()

            # Distribution of Price data
            plt.figure(figsize=(10, 6))
            sns.boxplot(x='Price', data=self.df)
            plt.title("Box Plot of Price")
            plt.show()

            # Set 'Date' as the index
            self.df.set_index('Date', inplace=True)

            # Ensure the index is a DatetimeIndex
            self.df.index = pd.to_datetime(self.df.index)

            # Ensure the index is a DatetimeIndex with daily frequency
            self.df = self.df.asfreq('D')

            # Sort the data by date
            self.df = self.df.sort_values('Date')

            # Check for missing values after setting the index
            print("Missing values after setting the index:")
            print(self.df.isnull().sum())

            # Interpolate missing values using time interpolation
            self.df['Price'] = self.df['Price'].interpolate(method='time')

            # Display the first few rows after interpolating missing values
            print("First few rows after interpolating missing values:")
            print(self.df.head())

            # Summary of price distribution
            price_range = (self.df['Price'] >= 0) & (self.df['Price'] <= 50)
            price_count = price_range.sum()
            total_count = len(self.df)
            percentage = (price_count / total_count) * 100
            print(f"Price distribution shows that {percentage:.2f}% of the data are in the 0 to 50 price range.")

            logging.info('Exploratory data analysis completed')
        except Exception as e:
            logging.error(f"Error performing exploratory data analysis: {e}")

    
    def get_dataframe(self):
        """
        Return the DataFrame for further analysis.
        
        :return: DataFrame containing the data.
        """
        return self.df
    