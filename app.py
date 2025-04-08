from flask import Flask, request, jsonify
from prophet import Prophet
import yfinance as yf
import pandas as pd

app = Flask(__name__)

@app.route('/predict', methods=['GET'])
def predict_stock():
    symbol = request.args.get('stock')

    if not symbol:
        return jsonify({"error": "Please provide a stock symbol, e.g., ?stock=AAPL"}), 400

    try:
        # Download last 2 years of daily data
        df = yf.download(symbol, period="2y")

        # Make sure data is not empty
        if df.empty:
            return jsonify({"error": f"No data found for symbol {symbol}"}), 404

        # Reset index and prepare for Prophet
        df.reset_index(inplace=True)
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']

        # Drop any rows with missing values
        df.dropna(inplace=True)

        # Prophet model
        model = Prophet()
        model.fit(df)

        # Predict next 7 days
        future = model.make_future_dataframe(periods=7)
        forecast = model.predict(future)

        # Get the next 7 days' predictions
        prediction = forecast[['ds', 'yhat']].tail(7)
        prediction['ds'] = prediction['ds'].astype(str)  # Convert datetime to string

        return jsonify(prediction.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
