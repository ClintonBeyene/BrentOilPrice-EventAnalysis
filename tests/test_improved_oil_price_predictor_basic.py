import unittest
import pandas as pd
import numpy as np
from src.improved_oil_price_predictor import ImprovedOilPricePredictor

class TestImprovedOilPricePredictorBasic(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures before each test method."""
        self.predictor = ImprovedOilPricePredictor()

    def test_initialization(self):
        """Test if the predictor initializes with correct default values."""
        self.assertIsNone(self.predictor.X_scaled)
        self.assertIsNone(self.predictor.y_scaled)
        self.assertIsNone(self.predictor.final_arima_model)
        self.assertIsNone(self.predictor.final_lstm_model)
        self.assertEqual(self.predictor.results, {})

    def test_ensemble_prediction(self):
        """Test if ensemble prediction calculation is correct."""
        arima_pred = np.array([1.0, 2.0, 3.0])
        lstm_pred = np.array([2.0, 3.0, 4.0])
        weight_lstm = 0.8

        result = self.predictor.ensemble_prediction(arima_pred, lstm_pred, weight_lstm)
        
        # Manual calculation
        expected = (weight_lstm * lstm_pred) + ((1 - weight_lstm) * arima_pred)
        np.testing.assert_array_almost_equal(result, expected)

if __name__ == '__main__':
    unittest.main()