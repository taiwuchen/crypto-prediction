import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

def predict_future_price(model, scaler_x, scaler_y, recent_data):
    # Prepare the data
    X = recent_data.values
    X_scaled = scaler_x.transform(X)
    X_scaled = np.reshape(X_scaled, (X_scaled.shape[0], 1, X_scaled.shape[1]))

    y_pred_scaled = model.predict(X_scaled)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    return y_pred

if __name__ == "__main__":
    symbol = 'BTC'
    model = tf.keras.models.load_model(f'trained_model_{symbol}.h5')
    scaler_x = joblib.load(f'models/scaler_x_{symbol}.save')
    scaler_y = joblib.load(f'models/scaler_y_{symbol}.save')

    df = pd.read_csv(f'data/processed_data_{symbol}.csv')

    features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto',
                'MA7', 'MA21', 'EMA', 'RSI', 'day_of_week', 'month']

    recent_data = df[features].tail(1)

    predicted_price = predict_future_price(model, scaler_x, scaler_y, recent_data)
    print(f"Predicted Next Day Price: {predicted_price[0][0]}")