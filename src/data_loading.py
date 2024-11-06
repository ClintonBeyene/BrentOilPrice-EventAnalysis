# Import necessary libraries
import pandas as pd
from sqlalchemy import create_engine
import os
import sys
from dotenv import load_dotenv
import warnings
import logging

# Add the parent directory to the system path
sys.path.append(os.path.join(os.path.abspath('../')))

# Ensure the logs directory exists
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs"))
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Set up logging
logging.basicConfig(
    filename=os.path.join(log_dir, 'data_loading.log'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

# Database configuration
DATABASE_URI = os.getenv('DATABASE_URI')

# Create the engine
engine = create_engine(DATABASE_URI)

class DataLoader:
    def __init__(self, csv_file_path):
        """
        Initialize the DataLoader with the path to the CSV file.
        
        :param csv_file_path: Path to the CSV file containing the data.
        """
        self.csv_file_path = csv_file_path
        self.df = None

    def load_data(self):
        """
        Load the data from the CSV file into a DataFrame.
        """
        try:
            logging.info('Loading data from CSV file')
            self.df = pd.read_csv(self.csv_file_path)
            logging.info('Data loaded from CSV file')
            print("Data successfully loaded from CSV file")
        except Exception as e:
            logging.error(f"Error loading data from CSV file: {e}")

    def load_data_to_database(self):
        """
        Load the DataFrame into the database.
        """
        try:
            logging.info('Loading the data into the database')
            table_name = 'BrentOilPrices'
            self.df.to_sql(table_name, con=engine, if_exists='replace', index=False)
            logging.info('Data loaded into database')
            print(f"Data successfully loaded into table: {table_name}")
        except Exception as e:
            logging.error(f"Error loading data into the database: {e}")

    def query_data_from_database(self):
        """
        Query the data from the database and return it as a DataFrame.
        
        :return: DataFrame containing the data from the database.
        """
        try:
            logging.info('Loading the data from the database')
            table_name = 'BrentOilPrices'
            query = f'SELECT * FROM public."{table_name}"'
            df = pd.read_sql(query, con=engine)
            logging.info('Data loaded from database')
            return df
        except Exception as e:
            logging.error(f"Error querying data from the database: {e}")
            return None