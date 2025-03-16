import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.get_historical_data import get_historical_data
from src.data_preparation import load_and_clean_data
from src.features import add_technical_indicators, add_time_features
from src.model_training import prepare_data, split_data, build_model, train_model
from src.model_evaluation import evaluate_model
from src.prediction import predict_future_price
import joblib
import tensorflow as tf
import os

# Create necessary directories if they don't exist
def create_directories():
    directories = ['data', 'models']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

def select_cryptocurrency():
    print("\nSelect a cryptocurrency:")
    print("1. Bitcoin (BTC)")
    print("2. Ethereum (ETH)")
    print("3. Dogecoin (DOGE)")
    crypto_choice = input("Enter your choice (1-3): ")

    if crypto_choice == '1':
        symbol = 'BTC'
        crypto_name = 'Bitcoin'
    elif crypto_choice == '2':
        symbol = 'ETH'
        crypto_name = 'Ethereum'
    elif crypto_choice == '3':
        symbol = 'DOGE'
        crypto_name = 'Dogecoin'
    else:
        print("Invalid choice. Defaulting to Bitcoin.")
        symbol = 'BTC'
        crypto_name = 'Bitcoin'
    return symbol, crypto_name

def main_menu(symbol, crypto_name):
    print(f"\nWelcome to the {crypto_name} Price Prediction Application!")
    print("Please select an option:")
    print("1. View Historical Data")
    print("2. Compare Historical and Predicted Data")
    print("3. Predict Future Prices")
    print("4. Change Cryptocurrency")
    print("5. Exit")
    choice = input("Enter your choice (1-5): ")
    return choice

def view_historical_data(symbol, crypto_name):
    print(f"\nFetching and displaying historical data for {crypto_name}...")
    df = get_historical_data(symbol)
    data_file = f'data/{symbol}_historical_data.csv'
    df.to_csv(data_file, index=False)
    df = load_and_clean_data(data_file)

    print("\nSelect the time frame to view historical data:")
    print("1. Last Week")
    print("2. Last Month")
    print("3. Last Year")
    print("4. All Time")
    time_choice = input("Enter your choice (1-4): ")

    if time_choice == '1':
        days = 7
    elif time_choice == '2':
        days = 30
    elif time_choice == '3':
        days = 365
    elif time_choice == '4':
        days = None  # Represents All Time
    else:
        print("Invalid choice. Defaulting to Last Week.")
        days = 7

    if days is not None:
        end_date = df['time'].max()
        start_date = end_date - pd.Timedelta(days=days)
        df_filtered = df[df['time'] >= start_date]
        title_time_frame = f"Last {days} Days"
    else:
        df_filtered = df
        title_time_frame = "All Time"

    plt.figure(figsize=(12,6))
    plt.plot(df_filtered['time'], df_filtered['close'])
    plt.title(f'{crypto_name} Historical Close Prices - {title_time_frame}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()
    input("Press Enter to return to the main menu...")



def compare_historical_predicted_data(symbol, crypto_name):
    print(f"\nComparing historical data with predicted data for {crypto_name}...")
    print("\nSelect the time frame to compare data:")
    print("1. Last Week")
    print("2. Last Month")
    print("3. Last Year")
    print("4. All Time")
    time_choice = input("Enter your choice (1-4): ")

    if time_choice == '1':
        days = 7
    elif time_choice == '2':
        days = 30
    elif time_choice == '3':
        days = 365
    elif time_choice == '4':
        days = None
    else:
        print("Invalid choice. Defaulting to Last Week.")
        days = 7


    data_file = f'data/{symbol}_historical_data.csv'
    df = load_and_clean_data(data_file)
    df = add_technical_indicators(df)
    df = add_time_features(df)
    processed_data_file = f'data/processed_data_{symbol}.csv'
    df.to_csv(processed_data_file, index=False)
    X_scaled, y_scaled, scaler_x, scaler_y = prepare_data(df, symbol)
    joblib.dump(scaler_x, f'models/scaler_x_{symbol}.save')
    joblib.dump(scaler_y, f'models/scaler_y_{symbol}.save')
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = train_model(model, X_train, y_train, X_test, y_test, symbol)

    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)

    dates = df['time'][-len(y_pred):].reset_index(drop=True)
    result_df = pd.DataFrame({
        'Date': dates,
        'Actual': y_actual.flatten(),
        'Predicted': y_pred.flatten()
    })


    if days is not None:
        end_date = result_df['Date'].max()
        start_date = end_date - pd.Timedelta(days=days)
        result_df_filtered = result_df[result_df['Date'] >= start_date]
        title_time_frame = f"Last {days} Days"
    else:
        result_df_filtered = result_df
        title_time_frame = "All Time"

    plt.figure(figsize=(12,6))
    plt.plot(result_df_filtered['Date'], result_df_filtered['Actual'], label='Actual Price')
    plt.plot(result_df_filtered['Date'], result_df_filtered['Predicted'], label='Predicted Price')
    plt.title(f'{crypto_name} Actual vs. Predicted Prices - {title_time_frame}')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
    input("Press Enter to return to the main menu...")



def predict_future_prices(symbol, crypto_name):
    print(f"\nPredicting future prices for {crypto_name}...")
    print("Choose a prediction period:")
    print("1. Next Day")
    print("2. Next Week")
    print("3. Next Month")
    period_choice = input("Enter your choice (1-3): ")

    if period_choice == '1':
        days_ahead = 1
    elif period_choice == '2':
        days_ahead = 7
    elif period_choice == '3':
        days_ahead = 30
    else:
        print("Invalid choice. Defaulting to Next Day prediction.")
        days_ahead = 1

    model = tf.keras.models.load_model(f'models/trained_model_{symbol}.h5')
    scaler_x = joblib.load(f'models/scaler_x_{symbol}.save')
    scaler_y = joblib.load(f'models/scaler_y_{symbol}.save')

    processed_data_file = f'data/processed_data_{symbol}.csv'
    df = pd.read_csv(processed_data_file)
    features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto',
                'MA7', 'MA21', 'EMA', 'RSI', 'day_of_week', 'month']

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

        last_date = pd.to_datetime(recent_data['time'].values[0]) + pd.Timedelta(days=1)
        dates.append(last_date)

        # Update recent_data with predicted values
        new_row = recent_data.iloc[-1].copy()
        new_row['open'] = new_row['close']
        new_row['high'] = new_row['close']
        new_row['low'] = new_row['close']
        new_row['close'] = y_pred.flatten()[0]
        new_row['volumefrom'] = new_row['volumefrom']
        new_row['volumeto'] = new_row['volumeto']
        new_row['time'] = last_date
        new_row['day_of_week'] = last_date.dayofweek
        new_row['month'] = last_date.month
        new_row['MA7'] = new_row['MA7']
        new_row['MA21'] = new_row['MA21']
        new_row['EMA'] = new_row['EMA']
        new_row['RSI'] = new_row['RSI']

        recent_data = pd.DataFrame([new_row])

    # Plot predicted prices
    plt.figure(figsize=(12,6))
    plt.plot(dates, predictions, marker='o', linestyle='-')
    plt.title(f'{crypto_name} Predicted Prices for the Next {days_ahead} Day(s)')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True)

    if days_ahead > 1:
        plt.xlim(dates[0], dates[-1])
        ymin = min(predictions) * 0.995
        ymax = max(predictions) * 1.005
        plt.ylim(ymin, ymax)

    plt.show()

    if days_ahead == 1:
        print(f"Predicted price for {dates[0].date()} is: ${predictions[0]:.2f}")

    input("Press Enter to return to the main menu...")

if __name__ == "__main__":
    create_directories()
    symbol, crypto_name = select_cryptocurrency()
    while True:
        choice = main_menu(symbol, crypto_name)
        if choice == '1':
            view_historical_data(symbol, crypto_name)
        elif choice == '2':
            compare_historical_predicted_data(symbol, crypto_name)
        elif choice == '3':
            predict_future_prices(symbol, crypto_name)
        elif choice == '4':
            symbol, crypto_name = select_cryptocurrency()
        elif choice == '5':
            print("Exiting the application. Goodbye!")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")
