from flask import Flask, request, jsonify
from flask_cors import CORS
from prophet import Prophet
import yfinance as yf
import pandas as pd
import os
from google.cloud import firestore

app = Flask(__name__)
CORS(app, origins=["https://stock-predictor-ivory.vercel.app"])  # ✅ Restrict to your frontend domain
# Set the path to your JSON key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "firebase-key.json"

# Initialize Firestore client
db = firestore.Client()

# Example usage
doc_ref = db.collection("test_collection").document("sample_doc")
doc_ref.set({
    "message": "Hello from Python!",
    "timestamp": firestore.SERVER_TIMESTAMP
})

# @app.route('/test-firestore')
# def test_firestore():
#     try:
#         doc_ref = db.collection("test").document("ping")
#         doc_ref.set({
#             "message": "Hello from Flask!",
#             "timestamp": firestore.SERVER_TIMESTAMP
#         })
#         return jsonify({"success": True, "message": "Data written to Firestore!"})
#     except Exception as e:
#         return jsonify({"success": False, "error": str(e)}), 500


# ----------- Prediction Route -------------
@app.route('/predict', methods=['GET'])
def predict_stock():
    symbol = request.args.get('stock')
    days = int(request.args.get('days', 7))  # default to 7 days

    if not symbol:
        return jsonify({"error": "Please provide a stock symbol, e.g., ?stock=AAPL"}), 400

    try:
        # Download last 2 years of daily data
        df = yf.download(symbol, period="2y")

        if df.empty:
            return jsonify({"error": f"No data found for symbol {symbol}"}), 404

        # Prepare data for Prophet
        df.reset_index(inplace=True)
        df = df[['Date', 'Close']]
        df.columns = ['ds', 'y']
        df.dropna(inplace=True)

        # Fit Prophet model
        model = Prophet()
        model.fit(df)

        # Predict future
        future = model.make_future_dataframe(periods=days)
        forecast = model.predict(future)

        # Get predictions
        prediction = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(days)
        prediction = prediction.round(2)
        prediction['ds'] = prediction['ds'].astype(str)  # Convert to string

        # Save each prediction as a separate Firestore document
        for row in prediction.to_dict(orient='records'):
            doc_ref = db.collection("predictions").document(f"{symbol.upper()}_{row['ds']}")
            doc_ref.set({
                "symbol": symbol.upper(),
                "target_date": row["ds"],
                "predicted_price": row["yhat"],
                "yhat_lower": row["yhat_lower"],
                "yhat_upper": row["yhat_upper"],
                "predicted_on": str(pd.Timestamp.now().date()),
                "created_at": firestore.SERVER_TIMESTAMP
            })

        return jsonify(prediction.to_dict(orient='records'))

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------- comparison -------------


@app.route('/get-comparisons', methods=['GET'])
def get_comparisons():
    symbol = request.args.get('stock')
    if not symbol:
        return jsonify({"error": "Missing stock symbol"}), 400

    try:
        comparisons = db.collection('comparisons') \
            .where('symbol', '==', symbol.upper()) \
            .order_by('target_date', direction=firestore.Query.DESCENDING) \
            .stream()

        data = [doc.to_dict() for doc in comparisons]

        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500



# ----------- Historical Data Route -------------
@app.route("/history")
def get_history():
    stock = request.args.get("stock")
    range_period = request.args.get("range", "1y")  # default to 1 year

    if not stock:
        return jsonify({"error": "No stock symbol provided"}), 400

    try:
        ticker = yf.Ticker(stock.strip().upper())
        hist = ticker.history(period=range_period, interval="1d")

        if hist.empty:
            return jsonify({"error": f"No data found for symbol: {stock}"}), 404

        data = [
            {"ds": str(date.date()), "close": round(row["Close"], 2)}
            for date, row in hist.iterrows()
        ]
        return jsonify(data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ----------- Main App Runner -------------
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=10000)
