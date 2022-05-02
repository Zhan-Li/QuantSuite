import datetime
import re
from typing import List

import pandas as pd
from binance.client import Client


# [
#   [
#     1499040000000,      // Open time
#     "0.01634790",       // Open
#     "0.80000000",       // High
#     "0.01575800",       // Low
#     "0.01577100",       // Close
#     "148976.11427815",  // Volume
#     1499644799999,      // Close time
#     "2434.19055334",    // Quote asset volume
#     308,                // Number of trades
#     "1756.87402397",    // Taker buy base asset volume
#     "28.46694368",      // Taker buy quote asset volume
#     "17928899.62484339" // Ignore.
#   ]
# ]


#
def convert_to_dataframe(klines):
    price_data = [{
        'open_time': datetime.datetime.fromtimestamp(kline[0] / 1000),
        'open': float(kline[1]),
        'high': float(kline[2]),
        'low': float(kline[3]),
        'close': float(kline[4]),
        'volume': float(kline[5]),
        'close_time': datetime.datetime.fromtimestamp(kline[6] / 1000),
        'quote_asset_volume': float(kline[7]),
        'number_of_trades': float(kline[8]),
        'taker_buy_base_asset_volume': float(kline[9]),
        'taker_buy_quote_asset_volume': float(kline[10]),
        'ignore': kline[11]}
        for kline in klines]
    return pd.DataFrame(price_data)


def download_symbols(binance_client: Client, denomination='USDT'):
    pattern = '[A-Z]{2,}' + denomination
    prices = binance_client.get_all_tickers()
    symbols = []
    for i in prices:
        symbol_search = re.search(pattern, i['symbol'])
        if symbol_search:
            symbol = symbol_search.group()
            if 'DOWN' not in symbol and 'UP' not in symbol and 'BULL' not in symbol and 'BEAR' not in symbol:
                symbols.append(symbol)
    if 'INCH' + denomination in symbols:
        symbols.remove('INCH' + denomination)
        symbols.append('1INCH' + denomination)
    FIAT_mkt = ['USDT', 'BKRW', 'BUSD', 'EUR', 'TUSD', 'USDC', 'TRY', 'PAX', 'AUD', 'BIDR', 'BRL', 'DAI', 'GBP',
                'IDRT',
                'NGN', 'RUB', 'ZAR', 'UAH', 'BVND']
    FIAT_symbols = [i + 'USDT' for i in FIAT_mkt]
    return [symbol for symbol in symbols if symbol not in FIAT_symbols]


def get_number_of_coins(binance_client: Client):
    BNB_mkt = ['BNB']
    BTC_mkt = ['BTC']
    ALTS_mkt = ['ETH', 'TRX', 'XRP']
    FIAT_mkt = ['USDT', 'BKRW', 'BUSD', 'EUR', 'TUSD', 'USDC', 'TRY', 'PAX', 'AUD', 'BIDR', 'BRL', 'DAI', 'GBP', 'IDRT',
                'NGN', 'RUB', 'ZAR', 'UAH', 'BVND']
    denominations = []
    counts = []
    clean_symbols = []
    for denomination in BNB_mkt + BTC_mkt + ALTS_mkt + FIAT_mkt:
        symbols = download_symbols(binance_client, denomination)
        denomination_removed = [re.search('(.*)' + denomination, symbol).group(1) for symbol in symbols]
        clean_symbols.extend(denomination_removed)
        denominations.append(denomination)
        counts.append(len(symbols))
    return list(set(clean_symbols)), \
           pd.DataFrame({'denomination': denominations, 'count': counts}).sort_values('count', ascending=False)


def download_hist(binance_client: Client, symbols: List[str], interval=Client.KLINE_INTERVAL_12HOUR,
                  start_str="1 Jan, 2009", end_str=None, output='data/crypto.parquet'):
    data = pd.DataFrame()
    for symbol in symbols:
        print(f'Downloading data for {symbol}')
        klines = binance_client.get_historical_klines(symbol, interval, start_str, end_str, limit=1000)
        if klines is not None:
            klines_df = convert_to_dataframe(klines)
            klines_df = klines_df.drop_duplicates('open_time', keep='last')
            klines_df['symbol'] = symbol
            klines_df['r'] = klines_df['open'].pct_change()
            col_arranged = ['symbol', 'open_time', 'close_time', 'r']
            klines_df = klines_df.reindex(col_arranged + [i for i in klines_df.columns if i not in col_arranged],
                                          axis=1)
            data = data.append(klines_df)
    return data.to_parquet(output, index=False)
