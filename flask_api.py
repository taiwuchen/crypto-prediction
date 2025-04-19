from flask import Flask, request, jsonify, g # Added g
from flask_cors import CORS
import tensorflow as tf
import joblib
import pandas as pd
import numpy as np # Added numpy
import os
from src.get_historical_data import get_historical_data
from src.data_preparation import load_and_clean_data
from src.model_training import prepare_data, split_data, build_model, train_model
from src.features import add_technical_indicators, add_time_features
import logging # Added logging

# Setup logging
logging.basicConfig(level=logging.INFO)

# --- Helper Functions ---

def get_data_paths(symbol):
    """Returns paths for data and model files."""
    base_path = os.path.dirname(__file__) # Ensure paths are relative to flask_api.py
    return {
        "model": os.path.join(base_path, f'models/trained_model_{symbol}.h5'),
        "scaler_x": os.path.join(base_path, f'models/scaler_x_{symbol}.save'),
        "scaler_y": os.path.join(base_path, f'models/scaler_y_{symbol}.save'),
        "processed_data": os.path.join(base_path, f'data/processed_data_{symbol}.csv'),
        "historical_data": os.path.join(base_path, f'data/{symbol}_historical_data.csv')
    }

def check_files_exist(paths):
    """Checks if all files in the paths dictionary exist."""
    for key, path in paths.items():
        if not os.path.exists(path):
            logging.error(f"File not found: {path} (for key: {key})")
            return False
    return True

def filter_dataframe_by_timeframe(df, time_column, time_frame):
    """Filters a DataFrame based on a specified time frame."""
    df[time_column] = pd.to_datetime(df[time_column]) # Ensure datetime format
    if time_frame == 'week':
        days = 7
    elif time_frame == 'month':
        days = 30
    elif time_frame == 'year':
        days = 365
    elif time_frame == 'all':
        days = None
    else: # Default to week
        days = 7

    if days is not None:
        end_date = df[time_column].max()
        start_date = end_date - pd.Timedelta(days=days)
        df_filtered = df[df[time_column] >= start_date].copy() # Use .copy() to avoid SettingWithCopyWarning
    else:
        df_filtered = df.copy()

    # Convert time back to string for JSON serialization
    df_filtered[time_column] = df_filtered[time_column].dt.strftime('%Y-%m-%d')
    return df_filtered

# --- API Logic Functions ---

def init_data_api(symbol):
    """Initializes all necessary data files for a given cryptocurrency.
    This function will:
    1. Fetch and save historical data
    2. Process the data (add technical indicators and time features)
    3. Train the model and save it
    4. Save the scalers for future use
    """
    try:
        # Create necessary directories
        if not os.path.exists('data'):
            os.makedirs('data')
        if not os.path.exists('models'):
            os.makedirs('models')
            
        paths = get_data_paths(symbol)
        
        # Step 1: Fetch and save historical data
        logging.info(f"Fetching historical data for {symbol}")
        df = get_historical_data(symbol)
        df.to_csv(paths['historical_data'], index=False)
        
        # Step 2: Process the data
        logging.info(f"Processing data for {symbol}")
        df = load_and_clean_data(paths['historical_data'])
        df = add_technical_indicators(df)
        df = add_time_features(df)
        df.to_csv(paths['processed_data'], index=False)
        
        # Step 3: Prepare data for model training
        logging.info(f"Preparing data for model training for {symbol}")
        X_scaled, y_scaled, scaler_x, scaler_y = prepare_data(df, symbol)
        joblib.dump(scaler_x, paths['scaler_x'])
        joblib.dump(scaler_y, paths['scaler_y'])
        
        # Step 4: Split data and reshape for LSTM
        X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
        X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
        
        # Step 5: Build and train model
        logging.info(f"Building and training model for {symbol}")
        model = build_model((X_train.shape[1], X_train.shape[2]))
        history = train_model(model, X_train, y_train, X_test, y_test, symbol)
        
        return {
            "success": True,
            "message": f"Successfully initialized data for {symbol}",
            "files_created": list(paths.values())
        }, None
        
    except Exception as e:
        logging.exception(f"Error during initialization for {symbol}: {e}")
        return None, str(e)

def predict_future_prices_api(symbol, days_ahead):
    paths = get_data_paths(symbol)
    required_paths = {k: paths[k] for k in ['model', 'scaler_x', 'scaler_y', 'processed_data']}
    if not check_files_exist(required_paths):
         logging.error(f"Missing files for prediction for symbol {symbol}")
         return None, "Model or data files not found."

    try:
        model = tf.keras.models.load_model(paths['model'])
        scaler_x = joblib.load(paths['scaler_x'])
        scaler_y = joblib.load(paths['scaler_y'])
        df = pd.read_csv(paths['processed_data'])
        features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto',
                    'MA7', 'MA21', 'EMA', 'RSI', 'day_of_week', 'month']

        # Ensure all required features are present
        if not all(feature in df.columns for feature in features):
             missing = [f for f in features if f not in df.columns]
             logging.error(f"Missing features in processed data for {symbol}: {missing}")
             return None, f"Missing features in processed data: {missing}"

        recent_data = df.tail(1).copy()
        predictions = []
        dates = []

        for i in range(days_ahead):
            X = recent_data[features].values
            X_scaled = scaler_x.transform(X)
            X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

            # Use context manager for TensorFlow session if needed, or ensure eager execution
            y_pred_scaled = model.predict(X_scaled)
            y_pred = scaler_y.inverse_transform(y_pred_scaled)
            predictions.append(float(y_pred.flatten()[0]))

            last_date = pd.to_datetime(recent_data['time'].values[0]) + pd.Timedelta(days=1)
            dates.append(str(last_date.date()))

            # Update recent_data with predicted values for multi-step prediction
            new_row = recent_data.iloc[-1].copy()
            predicted_price = y_pred.flatten()[0]
            new_row['open'] = new_row['close'] # Use previous close as new open
            new_row['high'] = max(new_row['high'], predicted_price) # Simple high estimate
            new_row['low'] = min(new_row['low'], predicted_price) # Simple low estimate
            new_row['close'] = predicted_price
            # Keep volume features constant or use a simple average if needed
            # new_row['volumefrom'] = new_row['volumefrom']
            # new_row['volumeto'] = new_row['volumeto']
            new_row['time'] = last_date
            # Recalculate time features
            new_row['day_of_week'] = last_date.dayofweek
            new_row['month'] = last_date.month
            # Note: Technical indicators (MA, EMA, RSI) are NOT recalculated here.
            # This is a simplification. For more accuracy, they should be recalculated
            # based on the growing series including the new prediction.
            # new_row['MA7'] = ...
            # new_row['MA21'] = ...
            # new_row['EMA'] = ...
            # new_row['RSI'] = ...

            recent_data = pd.DataFrame([new_row]) # Use pd.DataFrame constructor

        return {"dates": dates, "predictions": predictions}, None
    except Exception as e:
        logging.exception(f"Error during prediction for {symbol}: {e}")
        return None, str(e)


def get_historical_data_api(symbol, time_frame):
    paths = get_data_paths(symbol)
    data_file = paths['historical_data']
    try:
        if not os.path.exists(data_file):
            logging.info(f"Historical data file not found for {symbol}. Fetching...")
            df = get_historical_data(symbol)
            # Ensure 'time' column is datetime before saving
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.to_csv(data_file, index=False)
            logging.info(f"Saved historical data for {symbol} to {data_file}")
        else:
             logging.info(f"Loading historical data for {symbol} from {data_file}")

        df = load_and_clean_data(data_file) # This should handle reading the CSV and parsing dates
        df_filtered = filter_dataframe_by_timeframe(df, 'time', time_frame)

        # Select and rename columns for consistency
        df_filtered = df_filtered[['time', 'close']].rename(columns={'time': 'date', 'close': 'price'})
        return df_filtered.to_dict(orient='records'), None
    except Exception as e:
        logging.exception(f"Error fetching historical data for {symbol}: {e}")
        return None, str(e)

def get_comparison_data_api(symbol, time_frame):
    """Generates comparison data (actual vs predicted) for the test set."""
    paths = get_data_paths(symbol)
    required_paths = {k: paths[k] for k in ['model', 'scaler_x', 'scaler_y', 'historical_data']}
    if not check_files_exist(required_paths):
         logging.error(f"Missing files for comparison for symbol {symbol}")
         return None, "Model or data files not found for comparison."

    try:
        # 1. Load and process data
        logging.info(f"Loading and processing data for comparison: {symbol}")
        df = load_and_clean_data(paths['historical_data'])
        df = add_technical_indicators(df)
        df = add_time_features(df)
        # Save processed data if it wasn't already (optional, could be done by main.py)
        # df.to_csv(paths['processed_data'], index=False)

        # 2. Prepare data for model
        logging.info(f"Preparing data for model: {symbol}")
        # Ensure 'prepare_data' uses the correct features and handles potential missing columns
        X_scaled, y_scaled, scaler_x_loaded, scaler_y_loaded = prepare_data(df, symbol)

        # 3. Split data (we only need the test set for comparison)
        logging.info(f"Splitting data: {symbol}")
        _, X_test, _, y_test = split_data(X_scaled, y_scaled)
        X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        # 4. Load model
        logging.info(f"Loading model: {symbol}")
        model = tf.keras.models.load_model(paths['model'])

        # 5. Predict on test set
        logging.info(f"Predicting on test set: {symbol}")
        y_pred_scaled = model.predict(X_test)

        # 6. Inverse transform
        logging.info(f"Inverse transforming results: {symbol}")
        y_pred = scaler_y_loaded.inverse_transform(y_pred_scaled)
        y_actual = scaler_y_loaded.inverse_transform(y_test)

        # 7. Get corresponding dates from original df
        # The test set corresponds to the latter part of the dataframe
        test_set_size = len(y_actual)
        dates = df['time'][-test_set_size:].reset_index(drop=True)

        # 8. Create result DataFrame
        result_df = pd.DataFrame({
            'date': dates,
            'actual': y_actual.flatten(),
            'predicted': y_pred.flatten()
        })

        # 9. Filter by time frame
        logging.info(f"Filtering comparison data by time frame '{time_frame}': {symbol}")
        result_df_filtered = filter_dataframe_by_timeframe(result_df, 'date', time_frame)

        return result_df_filtered.to_dict(orient='records'), None

    except FileNotFoundError as e:
        logging.error(f"File not found during comparison for {symbol}: {e}")
        return None, f"Missing required file: {e.filename}"
    except KeyError as e:
         logging.error(f"Missing column/key during comparison for {symbol}: {e}")
         return None, f"Data processing error: Missing expected column '{e}'"
    except Exception as e:
        logging.exception(f"Error during comparison data generation for {symbol}: {e}")
        return None, str(e)


# --- Flask App Setup ---
app = Flask(__name__)
# Configure CORS to allow all origins, methods, and headers
CORS(app, resources={r"/*": {"origins": "*", "methods": ["GET", "POST", "OPTIONS"], "allow_headers": ["Content-Type", "Accept"]}})

@app.before_request
def log_request_info():
    app.logger.info('Headers: %s', request.headers)
    app.logger.info('Body: %s', request.get_data())

@app.after_request
def after_request(response):
    # Log response status
    app.logger.info(f"Response Status: {response.status}")
    # To log response data (be cautious with large responses):
    # try:
    #     response_data = response.get_json() if response.is_json else response.get_data(as_text=True)
    #     app.logger.info(f"Response Data: {response_data[:500]}...") # Log first 500 chars
    # except Exception as e:
    #     app.logger.error(f"Error logging response data: {e}")
    return response

@app.errorhandler(Exception)
def handle_exception(e):
    app.logger.error(f"Unhandled Exception: {e}")
    return jsonify({"error": str(e)}), 500

# --- API Routes ---

@app.route("/")
def home():
    return "Crypto Prediction API is running! Use /predict, /historical, /compare endpoints."

@app.route("/init", methods=["GET"])
def init_data():
    symbol = request.args.get("symbol", "BTC").upper()
    app.logger.info(f"Received /init request for {symbol}")
    
    result, error_msg = init_data_api(symbol)
    if error_msg:
        app.logger.error(f"Error in /init for {symbol}: {error_msg}")
        return jsonify({"error": error_msg}), 500
    
    app.logger.info(f"Successfully initialized data for {symbol}")
    return jsonify(result)

@app.route("/predict", methods=["GET"])
def predict():
    symbol = request.args.get("symbol", "BTC").upper()
    period = request.args.get("period", "day")
    app.logger.info(f"Received /predict request for {symbol}, period {period}")
    if period == "day":
        days_ahead = 1
    elif period == "week":
        days_ahead = 7
    elif period == "month":
        days_ahead = 30
    else:
        app.logger.warning(f"Invalid period '{period}' requested for /predict")
        return jsonify({"error": "Invalid period. Use 'day', 'week', or 'month'."}), 400

    result, error_msg = predict_future_prices_api(symbol, days_ahead)
    if error_msg:
        status_code = 404 if "not found" in error_msg.lower() else 500
        app.logger.error(f"Error in /predict for {symbol}: {error_msg}")
        return jsonify({"error": error_msg}), status_code
    app.logger.info(f"Successfully generated prediction for {symbol}")
    return jsonify(result)

@app.route("/historical", methods=["GET"])
def historical():
    symbol = request.args.get("symbol", "BTC").upper()
    time_frame = request.args.get("time_frame", "week") # e.g., week, month, year, all
    app.logger.info(f"Received /historical request for {symbol}, time_frame {time_frame}")
    if time_frame not in ['week', 'month', 'year', 'all']:
         app.logger.warning(f"Invalid time_frame '{time_frame}' requested for /historical")
         return jsonify({"error": "Invalid time_frame. Use 'week', 'month', 'year', or 'all'."}), 400

    data, error_msg = get_historical_data_api(symbol, time_frame)
    if error_msg:
        app.logger.error(f"Error in /historical for {symbol}: {error_msg}")
        return jsonify({"error": error_msg}), 500
    app.logger.info(f"Successfully retrieved historical data for {symbol}")
    return jsonify({"data": data})

@app.route("/compare", methods=["GET"])
def compare():
    symbol = request.args.get("symbol", "BTC").upper()
    time_frame = request.args.get("time_frame", "week") # e.g., week, month, year, all
    app.logger.info(f"Received /compare request for {symbol}, time_frame {time_frame}")
    if time_frame not in ['week', 'month', 'year', 'all']:
         app.logger.warning(f"Invalid time_frame '{time_frame}' requested for /compare")
         return jsonify({"error": "Invalid time_frame. Use 'week', 'month', 'year', or 'all'."}), 400

    data, error_msg = get_comparison_data_api(symbol, time_frame)
    if error_msg:
        status_code = 404 if "not found" in error_msg.lower() else 500
        app.logger.error(f"Error in /compare for {symbol}: {error_msg}")
        return jsonify({"error": error_msg}), status_code
    app.logger.info(f"Successfully generated comparison data for {symbol}")
    return jsonify({"data": data})


if __name__ == "__main__":
    # Ensure necessary directories exist before starting
    if not os.path.exists('data'):
        os.makedirs('data')
    if not os.path.exists('models'):
        os.makedirs('models')
    app.run(debug=True, port=5001)