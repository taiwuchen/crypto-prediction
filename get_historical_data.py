import requests
import pandas as pd

def get_historical_data(symbol, comparison_symbol, limit, aggregate, exchange):
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

# Example usage:
df = get_historical_data('BTC', 'USD', 2000, 1, 'Coinbase')
df.to_csv('bitcoin_historical_data.csv', index=False)
