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
    

@app.route("/history")
def get_history():
    stock = request.args.get("stock")
    range_period = request.args.get("range", "1y")  # default to 1 year

    if not stock:
        return jsonify({"error": "No stock symbol provided"}), 400

    try:
        ticker = yf.Ticker(stock.strip().upper())  # ensure clean symbol
        hist = ticker.history(period=range_period, interval="1d")  # add interval

        if hist.empty:
            return jsonify({"error": f"No data found for symbol: {stock}"}), 404

        data = [{"ds": str(date.date()), "close": round(row["Close"], 2)} for date, row in hist.iterrows()]
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
