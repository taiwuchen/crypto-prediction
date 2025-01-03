import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

def evaluate_model(model, X_test, y_test, scaler_y, symbol, crypto_name):
    y_pred_scaled = model.predict(X_test)

    # Inverse transform predictions and actual values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)

    print("Predicted values:")
    print(y_pred.flatten())
    print("\nActual values:")
    print(y_actual.flatten())

    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)

    print(f"\nMSE: {mse}")
    print(f"MAE: {mae}")

    # Plot actual vs. predicted prices
    plt.figure(figsize=(12,6))
    plt.plot(y_actual, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price')
    plt.title(f'{crypto_name} Actual vs. Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    symbol = 'BTC'  # Replace with desired symbol or pass as argument
    crypto_name = 'Bitcoin'  # Corresponding cryptocurrency name

    model = tf.keras.models.load_model(f'trained_model_{symbol}.h5')
    df = pd.read_csv(f'data/processed_data_{symbol}.csv')
    scaler_x = joblib.load(f'models/scaler_x_{symbol}.save')
    scaler_y = joblib.load(f'models/scaler_y_{symbol}.save')
    features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto',
                'MA7', 'MA21', 'EMA', 'RSI', 'day_of_week', 'month']
    target = 'close'

    df['target'] = df[target].shift(-1)
    df = df.dropna().reset_index(drop=True)

    X = df[features].values
    y = df['target'].values.reshape(-1, 1)

    X_scaled = scaler_x.transform(X)
    y_scaled = scaler_y.transform(y)

    train_size = int(len(X_scaled) * 0.8)
    X_test = X_scaled[train_size:]
    y_test = y_scaled[train_size:]

    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    evaluate_model(model, X_test, y_test, scaler_y, symbol, crypto_name)