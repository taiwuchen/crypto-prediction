import pandas as pd
import numpy as np

def add_technical_indicators(df):
    # Moving Average (MA)
    df['MA7'] = df['close'].rolling(window=7).mean()
    df['MA21'] = df['close'].rolling(window=21).mean()

    # Exponential Moving Average (EMA)
    df['EMA'] = df['close'].ewm(span=20, adjust=False).mean()

    # Relative Strength Index (RSI)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Drop rows with NaN values resulting from calculations
    df = df.dropna()

    return df

def add_time_features(df):
    # Day of week and month as features
    df['day_of_week'] = df['time'].dt.dayofweek
    df['month'] = df['time'].dt.month

    return df

if __name__ == "__main__":
    from src.data_preparation import load_and_clean_data
    data_file = 'bitcoin_historical_data.csv'
    df = load_and_clean_data(data_file)
    df = add_technical_indicators(df)
    df = add_time_features(df)
    # Save the processed data
    df.to_csv('data/processed_data.csv', index=False)
    print(df.head())
