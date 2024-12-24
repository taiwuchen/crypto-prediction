import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from get_historical_data import get_historical_data
from data_preparation import load_and_clean_data
from feature_engineering import add_technical_indicators, add_time_features
from model_training import prepare_data, split_data, build_model, train_model
from model_evaluation import evaluate_model
from prediction import predict_future_price
import joblib
import tensorflow as tf

def main_menu():
    print("\nWelcome to the Bitcoin Price Prediction Application!")
    print("Please select an option:")
    print("1. View Historical Data")
    print("2. Compare Historical and Predicted Data")
    print("3. Predict Future Prices")
    print("4. Exit")
    choice = input("Enter your choice (1-4): ")
    return choice

def view_historical_data():
    print("\nFetching and displaying historical data...")
    # Ask user for the time frame
    print("Select the time frame to view historical data:")
    print("1. Last Week")
    print("2. Last Month")
    print("3. Last Year")
    time_choice = input("Enter your choice (1-3): ")

    # Fetch data
    df = get_historical_data()
    df.to_csv('bitcoin_historical_data.csv', index=False)
    df = load_and_clean_data('bitcoin_historical_data.csv')

    # Determine the time frame
    if time_choice == '1':
        days = 7
    elif time_choice == '2':
        days = 30
    elif time_choice == '3':
        days = 365
    else:
        print("Invalid choice. Defaulting to Last Week.")
        days = 7

    # Filter data for the selected time frame
    end_date = df['time'].max()
    start_date = end_date - pd.Timedelta(days=days)
    df_filtered = df[df['time'] >= start_date]

    # Plot historical data
    plt.figure(figsize=(12,6))
    plt.plot(df_filtered['time'], df_filtered['close'])
    plt.title(f'Bitcoin Historical Close Prices - Last {days} Days')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.grid(True)
    plt.show()
    input("Press Enter to return to the main menu...")


def compare_historical_predicted_data():
    print("\nComparing historical data with predicted data...")
    # Ask user for the time frame
    print("Select the time frame to compare data:")
    print("1. Last Week")
    print("2. Last Month")
    print("3. Last Year")
    time_choice = input("Enter your choice (1-3): ")

    # Prepare data
    df = load_and_clean_data('bitcoin_historical_data.csv')
    df = add_technical_indicators(df)
    df = add_time_features(df)
    df.to_csv('data/processed_data.csv', index=False)
    X_scaled, y_scaled, scaler_x, scaler_y = prepare_data(df)
    joblib.dump(scaler_x, 'scaler_x.save')
    joblib.dump(scaler_y, 'scaler_y.save')
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
    # Reshape data
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    # Build and train model
    model = build_model((X_train.shape[1], X_train.shape[2]))
    history = train_model(model, X_train, y_train, X_test, y_test)
    # Evaluate model
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)

    # Convert y_pred and y_actual to DataFrame for easier handling
    dates = df['time'][-len(y_pred):].reset_index(drop=True)
    result_df = pd.DataFrame({'Date': dates, 'Actual': y_actual.flatten(), 'Predicted': y_pred.flatten()})

    # Determine the time frame
    if time_choice == '1':
        days = 7
    elif time_choice == '2':
        days = 30
    elif time_choice == '3':
        days = 365
    else:
        print("Invalid choice. Defaulting to Last Week.")
        days = 7

    # Filter data for the selected time frame
    end_date = result_df['Date'].max()
    start_date = end_date - pd.Timedelta(days=days)
    result_df_filtered = result_df[result_df['Date'] >= start_date]

    # Plot actual vs. predicted prices
    plt.figure(figsize=(12,6))
    plt.plot(result_df_filtered['Date'], result_df_filtered['Actual'], label='Actual Price')
    plt.plot(result_df_filtered['Date'], result_df_filtered['Predicted'], label='Predicted Price')
    plt.title(f'Actual vs. Predicted Prices - Last {days} Days')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.grid(True)
    plt.show()
    input("Press Enter to return to the main menu...")


def predict_future_prices():
    print("\nPredicting future prices...")
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

    # Load the model and scalers
    model = tf.keras.models.load_model('trained_model.h5')
    scaler_x = joblib.load('scaler_x.save')
    scaler_y = joblib.load('scaler_y.save')

    # Load the data
    df = pd.read_csv('data/processed_data.csv')
    features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto',
                'MA7', 'MA21', 'EMA', 'RSI', 'day_of_week', 'month']

    # Get the most recent data point
    recent_data = df.tail(1).copy()
    predictions = []
    dates = []

    for i in range(days_ahead):
        X = recent_data[features].values
        X_scaled = scaler_x.transform(X)
        X_scaled = X_scaled.reshape(X_scaled.shape[0], 1, X_scaled.shape[1])

        # Predict
        y_pred_scaled = model.predict(X_scaled)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)
        predictions.append(y_pred.flatten()[0])

        # Prepare for next prediction
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

    # Plot future predictions
    plt.figure(figsize=(12,6))
    plt.plot(dates, predictions, marker='o', linestyle='-')
    plt.title(f'Bitcoin Predicted Prices for the Next {days_ahead} Day(s)')
    plt.xlabel('Date')
    plt.ylabel('Predicted Price (USD)')
    plt.grid(True)

    if days_ahead > 1:
        # Zoom in to see the changes more clearly
        plt.xlim(dates[0], dates[-1])
        ymin = min(predictions) * 0.995
        ymax = max(predictions) * 1.005
        plt.ylim(ymin, ymax)

    plt.show()

    if days_ahead == 1:
        print(f"Predicted price for {dates[0].date()} is: ${predictions[0]:.2f}")

    input("Press Enter to return to the main menu...")


if __name__ == "__main__":
    while True:
        choice = main_menu()
        if choice == '1':
            view_historical_data()
        elif choice == '2':
            compare_historical_predicted_data()
        elif choice == '3':
            predict_future_prices()
        elif choice == '4':
            print("Exiting the application. Goodbye!")
            sys.exit()
        else:
            print("Invalid choice. Please try again.")
