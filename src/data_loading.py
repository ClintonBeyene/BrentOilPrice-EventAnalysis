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

# Load the CSV file
csv_file_path = '../data/Copy of BrentOilPrices.csv'
df = pd.read_csv(csv_file_path)

def data_loading(df):
    try:
        logging.info('Loading the data into database')
        # Load the data into the database
        table_name = 'BrentOilPrices'
        df.to_sql(table_name, con=engine, if_exists='replace', index=False)
        logging.info('Data loaded into database')
        print(f"Data successfully loaded into table: {table_name}")
    except Exception as e:
        logging.error(f"Error loading data into the database: {e}")

def query_data():
    try:
        logging.info('Loading the data from database')
        table_name = 'BrentOilPrices'
        query = f'SELECT * FROM {table_name}'

        # Read the data into a DataFrame
        df = pd.read_sql(query, con=engine)
        return df
    except Exception as e:
        logging.error(f"Error querying data from the database: {e}")
        return None