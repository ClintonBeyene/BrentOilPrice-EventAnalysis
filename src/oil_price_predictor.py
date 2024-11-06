# oil_price_predictor.py

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam

class OilPricePredictor:
    def __init__(self):
        self.merged_data = None
        self.feature_data = None
        self.X = None
        self.y = None
        
    def load_data(self, data_path="../data"):
        """Load and prepare the data from CSV files."""
        try:
            # Load individual datasets
            gdp_data_daily = pd.read_csv(f"{data_path}/GDP_cleaned_data_daily.csv")
            gdp_data_daily['Date'] = pd.to_datetime(gdp_data_daily['Date'])
            gdp_data_daily.set_index('Date', inplace=True)
            
            cpi_data_daily = pd.read_csv(f"{data_path}/CPI_cleaned_data_daily.csv")
            cpi_data_daily['Date'] = pd.to_datetime(cpi_data_daily['Date'])
            cpi_data_daily.set_index('Date', inplace=True)
            
            exchange_rate_data_daily = pd.read_csv(f"{data_path}/Exchange_Rate_cleaned_data_daily.csv")
            exchange_rate_data_daily['Date'] = pd.to_datetime(exchange_rate_data_daily['Date'])
            exchange_rate_data_daily.set_index('Date', inplace=True)
            
            oil_data_daily = pd.read_csv(f"{data_path}/Copy of BrentOilPrices.csv")
            oil_data_daily['Date'] = pd.to_datetime(oil_data_daily['Date'])
            oil_data_daily.set_index('Date', inplace=True)
            
            print("Data loaded successfully!")
            
            # Merge data
            self.merged_data = self.merge_data(oil_data_daily, gdp_data_daily, 
                                             cpi_data_daily, exchange_rate_data_daily)
            self.feature_data = self.create_features(self.merged_data)
            
            # Prepare features and target
            self.X = self.feature_data[['GDP', 'CPI', 'Exchange_Rate', 'Price_Pct_Change',
                                      'GDP_Pct_Change', 'CPI_Pct_Change', 'Exchange_Rate_Pct_Change',
                                      'Price_MA7', 'Price_MA30', 'Price_Volatility']]
            self.y = self.feature_data['Price']
            
            return True
            
        except FileNotFoundError as e:
            print(f"Error loading data: {e}")
            print("Please ensure all required CSV files are in the correct directory.")
            return False

    def merge_data(self, oil_data, gdp_data, cpi_data, exchange_rate_data):
        """Merge all datasets."""
        merged_data = pd.concat([oil_data, gdp_data, cpi_data, exchange_rate_data], 
                              axis=1, join='inner')
        merged_data.columns = ['Price', 'GDP', 'CPI', 'Exchange_Rate']
        return merged_data.dropna()

    def create_features(self, data):
        """Create features from the merged data."""
        feature_data = data.copy()
        
        # Calculate percentage changes
        feature_data['Price_Pct_Change'] = feature_data['Price'].pct_change()
        feature_data['GDP_Pct_Change'] = feature_data['GDP'].pct_change()
        feature_data['CPI_Pct_Change'] = feature_data['CPI'].pct_change()
        feature_data['Exchange_Rate_Pct_Change'] = feature_data['Exchange_Rate'].pct_change()
        
        # Moving averages
        feature_data['Price_MA7'] = feature_data['Price'].rolling(window=7).mean()
        feature_data['Price_MA30'] = feature_data['Price'].rolling(window=30).mean()
        
        # Volatility
        feature_data['Price_Volatility'] = feature_data['Price'].rolling(window=30).std()
        
        return feature_data.dropna()

    def build_var_model(self, train_data, maxlags=5):
        """Build and train VAR model."""
        try:
            scaler = MinMaxScaler()
            train_data_scaled = scaler.fit_transform(train_data)
            model = VAR(train_data_scaled)
            results = model.fit(maxlags=maxlags, ic='aic')
            return results, scaler
        except Exception as e:
            print(f"Error in VAR model fitting: {str(e)}")
            return None, None

    def build_arima_model(self, data, order=(1,1,1)):
        """Build and train ARIMA model."""
        try:
            model = ARIMA(data, order=order)
            results = model.fit()
            return results
        except Exception as e:
            print(f"Error in ARIMA model fitting: {str(e)}")
            return None

    def build_lstm_model(self, X_train, y_train):
        """Build and train LSTM model."""
        try:
            X_scaler = MinMaxScaler()
            y_scaler = MinMaxScaler()
            
            X_scaled = X_scaler.fit_transform(X_train)
            y_scaled = y_scaler.fit_transform(y_train.values.reshape(-1, 1))
            
            X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
            
            model = Sequential([
                LSTM(50, activation='relu', input_shape=(1, X_train.shape[1])),
                Dense(1)
            ])
            
            model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
            model.fit(X_lstm, y_scaled, epochs=100, batch_size=32, verbose=0)
            
            return model, X_scaler, y_scaler
        except Exception as e:
            print(f"Error in LSTM model building: {str(e)}")
            return None, None, None

    def train_and_evaluate(self, n_splits=5):
        """Train and evaluate all models using time series cross-validation."""
        if self.X is None or self.y is None:
            print("Data not loaded. Please load data first using load_data().")
            return None

        tscv = TimeSeriesSplit(n_splits=n_splits)
        var_scores, arima_scores, lstm_scores = [], [], []

        for train_index, test_index in tscv.split(self.X):
            X_train, X_test = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train, y_test = self.y.iloc[train_index], self.y.iloc[test_index]

            # VAR Model
            var_results, var_scaler = self.build_var_model(X_train)
            if var_results is not None:
                var_forecast = var_results.forecast(var_scaler.transform(X_train.values), 
                                                  steps=len(X_test))
                var_forecast = var_scaler.inverse_transform(var_forecast)[:, 0]
                var_scores.append(mean_squared_error(y_test, var_forecast))

            # ARIMA Model
            arima_results = self.build_arima_model(y_train) 
            if arima_results is not None:
                arima_forecast = arima_results.forecast(steps=len(y_test))
                arima_scores.append(mean_squared_error(y_test, arima_forecast))

            # LSTM Model
            lstm_model, X_scaler, y_scaler = self.build_lstm_model(X_train, y_train)
            if lstm_model is not None:
                X_test_scaled = X_scaler.transform(X_test)
                X_test_lstm = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))
                lstm_forecast = lstm_model.predict(X_test_lstm)
                lstm_forecast = y_scaler.inverse_transform(lstm_forecast)
                lstm_scores.append(mean_squared_error(y_test, lstm_forecast.flatten()))

        # Calculate average scores
        avg_var_mse = np.mean(var_scores) if var_scores else None
        avg_arima_mse = np.mean(arima_scores) if arima_scores else None
        avg_lstm_mse = np.mean(lstm_scores) if lstm_scores else None

        # Print evaluation results
        print("\nCross-Validation Model Evaluation Results:")
        if avg_var_mse is not None:
            print(f"VAR - Average MSE: {avg_var_mse:.4f}")
        if avg_arima_mse is not None:
            print(f"ARIMA - Average MSE: {avg_arima_mse:.4f}")
        if avg_lstm_mse is not None:
            print(f"LSTM - Average MSE: {avg_lstm_mse:.4f}")

 
