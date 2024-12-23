import numpy as np
import pandas as pd
import tensorflow as tf
from model_training import prepare_data
from sklearn.preprocessing import MinMaxScaler

def predict_future_price(model, scaler, recent_data):
    # Prepare the data
    X = recent_data.values
    X_scaled = scaler.transform(X)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    # Predict
    predictions = model.predict(X_scaled)
    return predictions

if __name__ == "__main__":
    # Load the model using tf.keras
    model = tf.keras.models.load_model('trained_model.h5')

    # Load and prepare data
    df = pd.read_csv('data/processed_data.csv')

    # Reuse the scaler from training
    _, _, scaler = prepare_data(df)

    # Define the features used in training
    features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto',
                'MA7', 'MA21', 'EMA', 'RSI', 'day_of_week', 'month']

    # Get the most recent data point
    recent_data = df[features].tail(1)

    # Make prediction
    predicted_price = predict_future_price(model, scaler, recent_data)
    print(f"Predicted Next Day Price: {predicted_price[0][0]}")
