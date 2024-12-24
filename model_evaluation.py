import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

def evaluate_model(model, X_test, y_test, scaler_y):
    # Make predictions
    y_pred_scaled = model.predict(X_test)

    # Inverse transform predictions and actual values
    y_pred = scaler_y.inverse_transform(y_pred_scaled)
    y_actual = scaler_y.inverse_transform(y_test)

    # Print predicted and actual values
    print("Predicted values:")
    print(y_pred.flatten())
    print("\nActual values:")
    print(y_actual.flatten())

    # Calculate metrics
    mse = mean_squared_error(y_actual, y_pred)
    mae = mean_absolute_error(y_actual, y_pred)

    print(f"\nMSE: {mse}")
    print(f"MAE: {mae}")

    # Plot actual vs. predicted prices
    plt.figure(figsize=(12,6))
    plt.plot(y_actual, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price')
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load the model
    model = tf.keras.models.load_model('trained_model.h5')

    # Load the data
    df = pd.read_csv('data/processed_data.csv')

    # Load the scalers
    scaler_x = joblib.load('scaler_x.save')
    scaler_y = joblib.load('scaler_y.save')

    # Prepare the data
    features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto',
                'MA7', 'MA21', 'EMA', 'RSI', 'day_of_week', 'month']
    target = 'close'

    df['target'] = df[target].shift(-1)
    df = df.dropna().reset_index(drop=True)

    X = df[features].values
    y = df['target'].values.reshape(-1, 1)

    # Scale features and target
    X_scaled = scaler_x.transform(X)
    y_scaled = scaler_y.transform(y)

    # Split the data
    train_size = int(len(X_scaled) * 0.8)
    X_test = X_scaled[train_size:]
    y_test = y_scaled[train_size:]

    # Reshape input
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Evaluate the model
    evaluate_model(model, X_test, y_test, scaler_y)