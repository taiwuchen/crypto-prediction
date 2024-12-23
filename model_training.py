import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf


def prepare_data(df):
    # Select features and target variable
    features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto',
                'MA7', 'MA21', 'EMA', 'RSI', 'day_of_week', 'month']
    target = 'close'

    # Shift target variable to predict the next day's price
    df['target'] = df[target].shift(-1)
    df = df.dropna()

    X = df[features].values
    y = df['target'].values

    # Feature Scaling
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler

def split_data(X, y):
    # Split data into training and testing sets without shuffling
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]
    return X_train, X_test, y_train, y_test

def build_model(input_shape):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(units=50))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_model(model, X_train, y_train, X_test, y_test):
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=1
    )
    # Save the trained model
    model.save('trained_model.h5')
    print("Model trained and saved.")
    return history

if __name__ == "__main__":
    # Load the processed data
    df = pd.read_csv('data/processed_data.csv')

    # Prepare the data
    X_scaled, y, scaler = prepare_data(df)

    # Split the data
    X_train, X_test, y_train, y_test = split_data(X_scaled, y)

    # Reshape data for LSTM input
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Build the model
    model = build_model((X_train.shape[1], X_train.shape[2]))

    # Train the model
    history = train_model(model, X_train, y_train, X_test, y_test)
