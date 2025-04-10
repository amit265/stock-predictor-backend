import firebase_admin
from firebase_admin import credentials, firestore
import yfinance as yf
from datetime import datetime, timedelta
import pytz
import pandas as pd

# Initialize Firebase
cred = credentials.Certificate("firebase-key.json")  # your service account file
firebase_admin.initialize_app(cred)
db = firestore.client()

# Today's date (in IST)
IST = pytz.timezone('Asia/Kolkata')
today = datetime.now(IST).date().isoformat()

def fetch_actual_price(symbol, target_date):
    try:
        target = datetime.fromisoformat(target_date)
        next_day = (target + timedelta(days=1)).date().isoformat()
        data = yf.download(symbol, start=target_date, end=next_day, interval="1d")

        if data.empty:
            print(f"No data for {symbol} on {target_date}")
            return None

        price = data['Close'].iloc[0]
        return round(float(price), 2)

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None


def compare_predictions():
    predictions_ref = db.collection("predictions")
    predictions = predictions_ref.where("target_date", "==", today).stream()

    for doc in predictions:
        data = doc.to_dict()
        symbol = data["symbol"]
        predicted = data["predicted_price"]
        target_date = data["target_date"]

        actual = fetch_actual_price(symbol, target_date)

        if actual is None:
            print(f"No actual data for {symbol} on {target_date}")
            continue

        error = round(abs(actual - predicted), 2)
        accuracy = round((1 - error / actual) * 100, 2)

        comparison_data = {
            "symbol": symbol,
            "target_date": target_date,
            "predicted_price": predicted,
            "actual_price": actual,
            "error": error,
            "accuracy_percent": accuracy,
            "comparison_done_at": datetime.now().isoformat()
        }

        db.collection("comparisons").add(comparison_data)
        print(f"✅ Compared {symbol} — Accuracy: {accuracy}%")

if __name__ == "__main__":
    compare_predictions()
