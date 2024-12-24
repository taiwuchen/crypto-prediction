import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import joblib

def prepare_data(df):
    # Select features and target variable
    features = ['open', 'high', 'low', 'close', 'volumefrom', 'volumeto',
                'MA7', 'MA21', 'EMA', 'RSI', 'day_of_week', 'month']
    target = 'close'

    df['target'] = df[target].shift(-1)
    df = df.dropna().reset_index(drop=True)

    X = df[features].values
    y = df['target'].values.reshape(-1, 1)

    scaler_x = MinMaxScaler(feature_range=(0, 1))
    X_scaled = scaler_x.fit_transform(X)
    scaler_y = MinMaxScaler(feature_range=(0, 1))
    y_scaled = scaler_y.fit_transform(y)

    return X_scaled, y_scaled, scaler_x, scaler_y

def split_data(X_scaled, y_scaled):
    train_size = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
    y_train, y_test = y_scaled[:train_size], y_scaled[train_size:]
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
    model.save('trained_model.h5')
    print("Model trained and saved.")

    # Plot the loss
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12,6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    return history

if __name__ == "__main__":
    df = pd.read_csv('data/processed_data.csv')

    X_scaled, y_scaled, scaler_x, scaler_y = prepare_data(df)
    joblib.dump(scaler_x, 'scaler_x.save')
    joblib.dump(scaler_y, 'scaler_y.save')
    X_train, X_test, y_train, y_test = split_data(X_scaled, y_scaled)
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    model = build_model((X_train.shape[1], X_train.shape[2]))
    model.summary()  # Print model summary
    history = train_model(model, X_train, y_train, X_test, y_test)