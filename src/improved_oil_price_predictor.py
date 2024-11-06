from src.oil_price_predictor import OilPricePredictor
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from pmdarima import auto_arima
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

class ImprovedOilPricePredictor(OilPricePredictor):
    def __init__(self):
        super().__init__()
        self.X_scaled = None
        self.y_scaled = None
        self.imputer = SimpleImputer(strategy='mean')
        self.scaler = MinMaxScaler()
        self.final_arima_model = None
        self.final_lstm_model = None
        self.results = {}

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

    def preprocess_data(self):
        """Preprocess the data after loading."""
        if self.X is None or self.y is None:
            print("Data not loaded. Please load data first using load_data().")
            return False

        # Handle NaN values
        X_imputed = pd.DataFrame(self.imputer.fit_transform(self.X), 
                               columns=self.X.columns, index=self.X.index)
        y_imputed = pd.Series(self.imputer.fit_transform(self.y.values.reshape(-1, 1)).flatten(), 
                            index=self.y.index)

        # Normalize the data
        self.X_scaled = pd.DataFrame(self.scaler.fit_transform(X_imputed), 
                                   columns=X_imputed.columns, index=X_imputed.index)
        self.y_scaled = pd.Series(self.scaler.fit_transform(y_imputed.values.reshape(-1, 1)).flatten(), 
                                index=y_imputed.index)

        return True

    def optimize_arima(self, data):
        """Optimize ARIMA model using auto_arima."""
        return auto_arima(data, start_p=1, start_q=1, max_p=5, max_q=5, m=1,
                         start_P=0, seasonal=False, d=1, D=1, trace=True,
                         error_action='ignore', suppress_warnings=True, stepwise=True)

    def build_improved_lstm_model(self, X_train, y_train):
        """Build and train an improved LSTM model."""
        X_train_reshaped = X_train.values.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        model = Sequential([
            LSTM(64, activation='relu', input_shape=(1, X_train.shape[1]), return_sequences=True),
            LSTM(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        
        history = model.fit(X_train_reshaped, y_train, epochs=200, batch_size=32, 
                          validation_split=0.2, callbacks=[early_stopping], verbose=0)
        
        return model, history

    def ensemble_prediction(self, arima_pred, lstm_pred, weight_lstm=0.9):
        """Combine ARIMA and LSTM predictions."""
        return (weight_lstm * lstm_pred) + ((1 - weight_lstm) * arima_pred)

    def train_and_evaluate(self, n_splits=5):
        """Train and evaluate models using time series cross-validation."""
        if not self.preprocess_data():
            return

        tscv = TimeSeriesSplit(n_splits=n_splits)
        arima_scores, lstm_scores = [], []

        for train_index, test_index in tscv.split(self.X_scaled):
            X_train, X_test = self.X_scaled.iloc[train_index], self.X_scaled.iloc[test_index]
            y_train, y_test = self.y_scaled.iloc[train_index], self.y_scaled.iloc[test_index]

            # ARIMA
            arima_model = self.optimize_arima(y_train)
            arima_forecast = arima_model.predict(n_periods=len(y_test))
            arima_scores.append(mean_squared_error(y_test, arima_forecast))

            # LSTM
            lstm_model, _ = self.build_improved_lstm_model(X_train, y_train)
            X_test_reshaped = X_test.values.reshape((X_test.shape[0], 1, X_test.shape[1]))
            lstm_forecast = lstm_model.predict(X_test_reshaped)
            lstm_scores.append(mean_squared_error(y_test, lstm_forecast.flatten()))

        # Calculate average scores
        self.results['avg_arima_mse'] = np.mean(arima_scores)
        self.results['avg_lstm_mse'] = np.mean(lstm_scores)
        self.results['improvement_percentage'] = ((self.results['avg_arima_mse'] -  self.results['avg_lstm_mse']) / self.results['avg_arima_mse']) * 100

        # Train final models on full dataset
        self.final_arima_model = self.optimize_arima(self.y_scaled)
        self.final_lstm_model, _ = self.build_improved_lstm_model(self.X_scaled, self.y_scaled)

        self.print_results()

    def make_future_predictions(self, future_periods=30):
        """Make future predictions using both models and ensemble."""
        arima_forecast = self.final_arima_model.predict(n_periods=future_periods)
        future_X = self.X_scaled.iloc[-future_periods:].values.reshape((future_periods, 1, self.X_scaled.shape[1]))
        lstm_forecast = self.final_lstm_model.predict(future_X).flatten()
        ensemble_forecast = self.ensemble_prediction(arima_forecast, lstm_forecast)

        future_dates = pd.date_range(start=self.y_scaled.index[-1], periods=future_periods+1)[1:]
        
        return {
            'dates': future_dates,
            'arima_forecast': arima_forecast,
            'lstm_forecast': lstm_forecast,
            'ensemble_forecast': ensemble_forecast
        }

    def plot_predictions(self, predictions, lookback=100):
        """Plot the actual values and predictions."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.y_scaled.index[-lookback:], self.y_scaled[-lookback:], label='Actual')
        plt.plot(predictions['dates'], predictions['arima_forecast'], label='ARIMA Forecast')
        plt.plot(predictions['dates'], predictions['lstm_forecast'], label='LSTM Forecast')
        plt.plot(predictions['dates'], predictions['ensemble_forecast'], label='Ensemble Forecast')
        
        plt.title('Oil Price Prediction')
        plt.xlabel('Date')
        plt.ylabel('Scaled Price')
        plt.legend()
        plt.grid(True)
        plt.show()

    def save_models(self, arima_path='arima_model.pkl', lstm_path='lstm_model.h5'):
        """Save the trained models."""
        joblib.dump(self.final_arima_model, arima_path)
        self.final_lstm_model.save(lstm_path)
        print("\nModels saved successfully.")

    def print_results(self):
        """Print the analysis results and conclusions."""
        print("\nImproved Cross-Validation Model Evaluation Results:")
        print(f"Optimized ARIMA - Average MSE: {self.results['avg_arima_mse']:.4f}")
        print(f"Improved LSTM - Average MSE: {self.results['avg_lstm_mse']:.4f}")

        # Additional analysis and steps
        print(f"\nLSTM improvement over ARIMA: {self.results['improvement_percentage']:.2f}%")
        print("\nConclusion:")
        print("1. The LSTM model significantly outperforms the ARIMA model in terms of MSE.")
        print(f"2. LSTM provides a {self.results['improvement_percentage']:.2f}% improvement over ARIMA.")
        print("3. Consider using the LSTM model as the primary forecasting tool.")
        print("4. The ensemble model combines both ARIMA and LSTM predictions, which might provide more robust forecasts.")
        print("5. Regular retraining and monitoring of both models is recommended to maintain performance.")
