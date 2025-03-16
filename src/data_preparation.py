import pandas as pd

def load_and_clean_data(file_path):
    # Load the CSV file
    df = pd.read_csv(file_path)

    # Convert 'time' column to datetime
    df['time'] = pd.to_datetime(df['time'])

    # Sort by time to maintain chronological order
    df = df.sort_values('time').reset_index(drop=True)

    # Drop unnecessary columns
    df = df[['time', 'open', 'high', 'low', 'close', 'volumefrom', 'volumeto']]

    # Check for missing values
    if df.isnull().values.any():
        df = df.dropna()

    return df

if __name__ == "__main__":
    data_file = 'bitcoin_historical_data.csv'
    df = load_and_clean_data(data_file)
    print(df.head())
