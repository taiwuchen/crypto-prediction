import requests
import pandas as pd

def get_historical_data(symbol='BTC', comparison_symbol='USD', limit=2000, aggregate=1, exchange='Coinbase'):
    url = 'https://min-api.cryptocompare.com/data/v2/histoday'
    params = {
        'fsym': symbol,
        'tsym': comparison_symbol,
        'limit': limit,
        'aggregate': aggregate,
        'e': exchange
    }
    response = requests.get(url, params)
    data = response.json()['Data']['Data']
    df = pd.DataFrame(data)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    return df

if __name__ == "__main__":
    df = get_historical_data()
    df.to_csv('bitcoin_historical_data.csv', index=False)

