import json

from binance.client import Client

import quantsuite.trader.binance as binance_funcs

with open('secret.json') as f:
    keys = json.load(f)
client = Client(keys['binance_api_key'], keys['binance_api_secret'])

# all_symbols, count = get_number_of_coins(client)


isolated_margin_USDT = binance_funcs.get_all_isolated_margin_symbols(client)
isolated_margin_USDT = [i + 'USDT' for i in isolated_margin_USDT]

args = [{'interval': client.KLINE_INTERVAL_1MINUTE, 'output': 'data/crypto_1m.pkl'},
        {'interval': client.KLINE_INTERVAL_3MINUTE, 'output': 'data/crypto_3m.pkl'},
        {'interval': client.KLINE_INTERVAL_5MINUTE, 'output': 'data/crypto_5m.pkl'},
        {'interval': client.KLINE_INTERVAL_15MINUTE, 'output': 'data/crypto_15m.pkl'},
        {'interval': client.KLINE_INTERVAL_30MINUTE, 'output': 'data/crypto_30m.pkl'},
        {'interval': client.KLINE_INTERVAL_1HOUR, 'output': 'data/crypto_1h.pkl'},
        {'interval': client.KLINE_INTERVAL_2HOUR, 'output': 'data/crypto_2h.pkl'},
        {'interval': client.KLINE_INTERVAL_4HOUR, 'output': 'data/crypto_4h.pkl'},
        {'interval': client.KLINE_INTERVAL_6HOUR, 'output': 'data/crypto_6h.pkl'},
        {'interval': client.KLINE_INTERVAL_8HOUR, 'output': 'data/crypto_8h.pkl'},
        {'interval': client.KLINE_INTERVAL_12HOUR, 'output': 'data/crypto_12h.pkl'},
        {'interval': client.KLINE_INTERVAL_1DAY, 'output': 'data/crypto_1d.pkl'},
        {'interval': client.KLINE_INTERVAL_3DAY, 'output': 'data/crypto_3d.pkl'},
        {'interval': client.KLINE_INTERVAL_1WEEK, 'output': 'data/crypto_1w.pkl'},
        {'interval': client.KLINE_INTERVAL_1MONTH, 'output': 'data/crypto_1M.pkl'}]
args.reverse()
for arg in args:
    print('working on', arg)
    hist = binance_funcs.download_hist(binance_client=client, symbols=isolated_margin_USDT, interval=arg['interval'])
    hist.to_pickle(arg['output'])
