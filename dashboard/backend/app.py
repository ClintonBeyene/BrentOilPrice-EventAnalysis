# app.py

from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from models.price_analysis import (
    load_price_data,
    calculate_price_trends, 
    calculate_yearly_average_price, 
    calculate_analysis_metrics,
    calculate_price_distribution, 
    calculate_event_impact, 
    get_prices_around_event
)

app = Flask(__name__)
CORS(app)

# Load data once when the application starts
price_data = load_price_data()

# Define significant events as a constant
SIGNIFICANT_EVENTS = {
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

def handle_error(func):
    """Decorator to handle errors in routes"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    wrapper.__name__ = func.__name__
    return wrapper

@app.route('/api/price-trends', methods=['GET'])
@handle_error
def get_price_trends():
    """Get price trends around significant events"""
    trends_data = []
    for event_date, event_name in SIGNIFICANT_EVENTS.items():
        event_date = pd.to_datetime(event_date)
        prices_around_event = get_prices_around_event(
            event_date, 
            price_data, 
            days_before=180, 
            days_after=180
        )
        trends_data.append({
            'event': event_name,
            'date': event_date.strftime('%Y-%m-%d'),
            'prices': prices_around_event['Price'].tolist(),
            'dates': [date.strftime('%Y-%m-%d') for date in prices_around_event.index]
        })
    return jsonify(trends_data)

@app.route('/api/event-impact', methods=['GET'])
@handle_error
def get_event_impact():
    """Calculate impact of significant events on oil prices"""
    results = []
    for event_date, event_name in SIGNIFICANT_EVENTS.items():
        impact_data = calculate_event_impact(event_name, event_date, price_data)
        results.append(impact_data)
    return jsonify(results)

@app.route('/api/analysis-metrics', methods=['GET'])
@handle_error
def get_analysis():
    """Get general analysis metrics"""
    analysis_results = calculate_analysis_metrics(price_data.reset_index())
    return jsonify(analysis_results)

@app.route('/api/prices', methods=['GET'])
@handle_error
def get_price_trend():
    """Get complete price trends with optional date filtering"""
    start_date = request.args.get('startDate')
    end_date = request.args.get('endDate')

    # Convert start_date and end_date to datetime objects
    if start_date and end_date:
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # Filter the price_data DataFrame based on the date range
        filtered_price_data = price_data[(price_data.index >= start_date) & (price_data.index <= end_date)]
    else:
        filtered_price_data = price_data

    # Calculate price trends for the filtered data
    price_data_dict = calculate_price_trends(filtered_price_data)
    return jsonify(price_data_dict)

@app.route('/api/average-yearly-price', methods=['GET'])
@handle_error
def get_yearly_average():
    """Get yearly average prices"""
    analysis_results = calculate_yearly_average_price(price_data)
    return jsonify(analysis_results)

@app.route('/api/price-distribution', methods=['GET'])
@handle_error
def get_distribution():
    """Get price distribution analysis"""
    analysis_results = calculate_price_distribution(price_data)
    return jsonify(analysis_results)

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': pd.Timestamp.now().isoformat()
    })

if __name__ == '__main__':
    app.run(debug=True)