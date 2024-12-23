import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf

def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    print(f"MSE: {mse}")
    print(f"MAE: {mae}")

    # Plot actual vs. predicted prices
    plt.figure(figsize=(12,6))
    plt.plot(y_test, label='Actual Price')
    plt.plot(y_pred, label='Predicted Price')
    plt.title('Actual vs. Predicted Prices')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # Load the model using tf.keras
    model = tf.keras.models.load_model('trained_model.h5')

    # Load and prepare data
    df = pd.read_csv('data/processed_data.csv')

    from model_training import prepare_data, split_data
    X_scaled, y, scaler = prepare_data(df)
    _, X_test, _, y_test = split_data(X_scaled, y)

    # Reshape input for LSTM
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    evaluate_model(model, X_test, y_test)
