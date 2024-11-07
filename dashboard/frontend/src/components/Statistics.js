// frontend/src/components/Statistics.js

import React from 'react';
import './Dashboard.css';  // Optional: For styling the component

const Statistics = ({ analysis }) => {
    return (
        <div className="statistics-container border p-4 mb-4">
            <h2 className="text-center mb-4">Key Analysis Metrics</h2>

            <div className="stat-section mb-4">
                <h4>Volatility & Price Change</h4>
                <p><strong>Volatility:</strong> {analysis.volatility ?? 'N/A'}</p>
                <p><strong>Average Daily Price Change:</strong> {analysis.average_price_change ?? 'N/A'}</p>
                <p><strong>Price Range:</strong> {analysis.min_price ?? 'N/A'} - {analysis.max_price ?? 'N/A'}</p>
                <p><strong>Total Price Change:</strong> {analysis.total_price_change ?? 'N/A'}</p>
            </div>

            <div className="stat-section mb-4">
                <h4>Correlation Metrics</h4>
                <p><strong>Correlation with Time:</strong> {analysis.correlation ?? 'N/A'}</p>
            </div>

            <div className="stat-section">
                <h4>Model Accuracy Metrics</h4>
                <p><strong>RMSE:</strong> {analysis.model_accuracy?.RMSE ?? 'N/A'}</p>
                <p><strong>MAE:</strong> {analysis.model_accuracy?.MAE ?? 'N/A'}</p>
            </div>
        </div>
    );
}

export default Statistics;
