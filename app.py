from flask import Flask, render_template, request, jsonify, redirect, url_for
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
import os

from get_historical_data import get_historical_data
from data_preparation import load_and_clean_data
from feature_engineering import add_technical_indicators, add_time_features
from model_training import prepare_data, split_data
from prediction import predict_future_price

app = Flask(__name__)

# Define available cryptocurrencies
crypto_symbols = {"BTC": "Bitcoin", "ETH": "Ethereum", "DOGE": "Dogecoin"}
selected_crypto = "BTC"

# Create a cache for models to avoid reloading
app.config["MODELS"] = {}
app.config["SCALERS"] = {}

# Helper function to encode Matplotlib images
def plot_to_base64():
    img = io.BytesIO()
    plt.savefig(img, format="png")
    img.seek(0)
    return base64.b64encode(img.getvalue()).decode()

# Ensure required directories exist
if not os.path.exists("data"):
    os.makedirs("data")
if not os.path.exists("models"):
    os.makedirs("models")

# Homepage - Select Cryptocurrency
@app.route("/", methods=["GET", "POST"])
def index():
    global selected_crypto
    if request.method == "POST":
        selected_crypto = request.form["crypto"]
    return render_template("index.html", selected_crypto=selected_crypto, crypto_symbols=crypto_symbols)

# View Historical Data
@app.route("/historical")
def historical():
    file_path = f"data/{selected_crypto}_historical_data.csv"
    
    if not os.path.exists(file_path):
        df = get_historical_data(selected_crypto)
        df.to_csv(file_path, index=False)
    else:
        df = load_and_clean_data(file_path)

    plt.figure(figsize=(12,6))
    plt.plot(df["time"], df["close"], label="Closing Price", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{crypto_symbols[selected_crypto]} Historical Prices")
    plt.legend()
    plt.grid(True)

    image_base64 = plot_to_base64()
    plt.close()

    return render_template("historical.html", image_base64=image_base64, selected_crypto=selected_crypto)

# Compare Historical vs Predicted Data
@app.route("/compare")
def compare():
    file_path = f"data/{selected_crypto}_historical_data.csv"
    model_path = f"models/trained_model_{selected_crypto}.h5"

    if not os.path.exists(file_path) or not os.path.exists(model_path):
        return jsonify({"error": "Missing data or trained model. Train the model first."}), 400

    df = load_and_clean_data(file_path)
    df = add_technical_indicators(df)
    df = add_time_features(df)

    X_scaled, y_scaled, scaler_x, scaler_y = prepare_data(df, selected_crypto)
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    # Load model if not already loaded
    if selected_crypto not in app.config["MODELS"]:
        model = tf.keras.models.load_model(model_path)
        app.config["MODELS"][selected_crypto] = model
    else:
        model = app.config["MODELS"][selected_crypto]

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)

    plt.figure(figsize=(12,6))
    plt.plot(df["time"][-len(y_pred):], y_actual, label="Actual Price", color="green")
    plt.plot(df["time"][-len(y_pred):], y_pred, label="Predicted Price", linestyle="dashed", color="red")
    plt.xlabel("Date")
    plt.ylabel("Price (USD)")
    plt.title(f"{crypto_symbols[selected_crypto]} Actual vs Predicted Prices")
    plt.legend()
    plt.grid(True)

    image_base64 = plot_to_base64()
    plt.close()

    return render_template("compare.html", image_base64=image_base64, selected_crypto=selected_crypto)

# Predict Future Prices
@app.route("/predict", methods=["GET", "POST"])
def predict():
    days_ahead = 1
    if request.method == "POST":
        days_ahead = int(request.form["days"])

    model_path = f"models/trained_model_{selected_crypto}.h5"
    scaler_x_path = f"models/scaler_x_{selected_crypto}.save"
    scaler_y_path = f"models/scaler_y_{selected_crypto}.save"
    data_path = f"data/processed_data_{selected_crypto}.csv"

    if not os.path.exists(model_path) or not os.path.exists(scaler_x_path) or not os.path.exists(scaler_y_path) or not os.path.exists(data_path):
        return jsonify({"error": "Missing required files. Train the model first."}), 400

    # Load the model if not already loaded
    if selected_crypto not in app.config["MODELS"]:
        model = tf.keras.models.load_model(model_path)
        app.config["MODELS"][selected_crypto] = model
    else:
        model = app.config["MODELS"][selected_crypto]

    # Load scalers
    if selected_crypto not in app.config["SCALERS"]:
        scaler_x = joblib.load(scaler_x_path)
        scaler_y = joblib.load(scaler_y_path)
        app.config["SCALERS"][selected_crypto] = (scaler_x, scaler_y)
    else:
        scaler_x, scaler_y = app.config["SCALERS"][selected_crypto]

    df = pd.read_csv(data_path)
    features = ["open", "high", "low", "close", "volumefrom", "volumeto",
                "MA7", "MA21", "EMA", "RSI", "day_of_week", "month"]

    recent_data = df.tail(1).copy()
    predictions = []
    dates = []

    for i in range(days_ahead):
        X = recent_data[features].values
        X_scaled = scaler_x.transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predictions.append(y_pred.flatten()[0])

        last_date = pd.to_datetime(recent_data["time"].values[0]) + pd.Timedelta(days=1)
        dates.append(last_date)

        recent_data["close"] = y_pred.flatten()[0]
        recent_data["time"] = last_date

    plt.figure(figsize=(12,6))
    plt.plot(dates, predictions, marker="o", linestyle="-", color="blue")
    plt.xlabel("Date")
    plt.ylabel("Predicted Price (USD)")
    plt.title(f"{crypto_symbols[selected_crypto]} Price Prediction")
    plt.grid(True)

    image_base64 = plot_to_base64()
    plt.close()

    return render_template("predict.html", image_base64=image_base64, selected_crypto=selected_crypto)

if __name__ == "__main__":
    app.run(debug=True, threaded=False)