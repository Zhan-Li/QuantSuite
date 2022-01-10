import math
from typing import List

import pandas as pd
from binance.client import Client
from binance.enums import *
from binance.exceptions import BinanceAPIException
from pandas import DataFrame as PandasDataFrame


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
def convert_klines_to_dataframe(klines):
    keys = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_asset_volume',
            'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore']
    klines_df = pd.DataFrame([dict(zip(keys, i)) for i in klines])
    cols = klines_df.columns.tolist()
    for col in [i for i in cols if 'time' in i]:
        klines_df[col] = pd.to_datetime(klines_df[col], unit='ms')
    for col in [i for i in cols if 'time' not in i]:
        klines_df[col] = klines_df[col].astype(float)
    return pd.DataFrame(klines_df)


def download_hist(binance_client: Client, symbols: List[str], interval=Client.KLINE_INTERVAL_12HOUR,
                  start_str="1 Jan, 2009", end_str=None):
    data = pd.DataFrame()
    for symbol in symbols:
        klines = binance_client.get_historical_klines(symbol, interval, start_str, end_str, limit=1000)
        if klines is not None:
            klines_df = convert_klines_to_dataframe(klines)
            klines_df = klines_df.drop_duplicates('open_time', keep='last')
            klines_df['symbol'] = symbol
            data = data.append(klines_df)
    data['r'] = data['open'].pct_change()
    col_arranged = ['symbol', 'open_time', 'close_time', 'r']
    return data.reindex(col_arranged + [i for i in data.columns if i not in col_arranged], axis=1)


def get_spot_USDT_balances(binance_client: Client, quote='USDT'):
    spot_balance = pd.DataFrame(binance_client.get_account()['balances'])
    spot_balance['free'] = spot_balance['free'].astype(float)
    spot_balance['locked'] = spot_balance['locked'].astype(float)
    return spot_balance.loc[spot_balance['asset'] == quote].iloc[0]['free']


def get_isolated_margin_account(binance_client, quote='USDT'):
    assets = binance_client.get_isolated_margin_account()['assets']
    symbol = pd.DataFrame([asset['symbol'] for asset in assets], columns=['symbol'])

    asset_cols = ['asset', 'free', 'locked', 'totalAsset', 'borrowed', 'interest', 'netAsset']
    base_asset = pd.DataFrame([asset['baseAsset'] for asset in assets])[asset_cols]
    base_asset.columns = ['base_' + i for i in base_asset.columns]
    quote_asset = pd.DataFrame([asset['quoteAsset'] for asset in assets])[asset_cols]
    quote_asset.columns = ['quote_' + i for i in quote_asset.columns]
    account = pd.concat([symbol, base_asset, quote_asset], axis=1)

    prices = pd.DataFrame(binance_client.get_all_tickers())
    account = account.merge(prices, on='symbol')

    num_cols = [i for i in account.columns if i not in ['symbol', 'base_asset', 'quote_asset']]
    for col in num_cols:
        account[col] = account[col].astype(float)

    account['base_netAssetOfQuote'] = account['base_netAsset'] * account['price']
    account['totalEquity'] = account['base_netAssetOfQuote'] + account['quote_netAsset']
    return account.loc[account['quote_asset'] == quote]


def get_symbol_filters(binance_client, quote='USDT'):
    """
    get ticksize and stepsize for symbols
    """
    bases = []
    quotes = []
    tickSizes = []
    stepSizes = []
    marginAllowed = []
    for info in binance_client.get_exchange_info()['symbols']:
        bases.append(info['baseAsset'])
        quotes.append(info['quoteAsset'])
        marginAllowed.append(info['isMarginTradingAllowed'])
        filters = info['filters']
        tickSizes.append(filters[0]['tickSize'])
        stepSizes.append(filters[2]['stepSize'])

    symbolInfo = pd.DataFrame({'base': bases, 'quote': quotes, 'marginAllowed': marginAllowed,
                               'tickSize': tickSizes, 'stepSize': stepSizes})
    symbolInfo['tickPrecision'] = symbolInfo['tickSize'].str.extract('0.(.*1)', expand=False).str.len()
    symbolInfo['tickPrecision'] = symbolInfo['tickPrecision'].fillna(0)
    symbolInfo['tickPrecision'] = symbolInfo['tickPrecision'].astype(int)
    symbolInfo['stepPrecision'] = symbolInfo['stepSize'].str.extract('0.(.*1)', expand=False).str.len()
    symbolInfo['stepPrecision'] = symbolInfo['stepPrecision'].fillna(0)
    symbolInfo['stepPrecision'] = symbolInfo['stepPrecision'].astype(int)
    return symbolInfo.loc[symbolInfo['quote'] == quote]


def get_bid_ask_spread(binance_client, base_symbols: List[str], quote='USDT'):
    spreads = pd.DataFrame()
    for base_symbol in base_symbols:
        orderBook = binance_client.get_order_book(symbol=base_symbol + quote)
        bids = pd.DataFrame(orderBook['bids'], columns=['bid', 'bidQuantity'])
        asks = pd.DataFrame(orderBook['asks'], columns=['ask', 'askQuantity'])
        df = pd.concat([bids, asks], axis=1)
        df['symbol'] = base_symbol + quote
        df['bid'] = df['bid'].astype(float)
        df['ask'] = df['ask'].astype(float)
        df['bidQuantity'] = df['bidQuantity'].astype(float)
        df['askQuantity'] = df['askQuantity'].astype(float)
        df['spread'] = (df['ask'] - df['bid']) / ((df['ask'] + df['bid']) / 2)
        df['bidValue'] = df['bid'] * df['bidQuantity']
        df['askValue'] = df['ask'] * df['askQuantity']
        spreads = spreads.append(df.iloc[0])
    return spreads


def get_all_isolated_margin_symbols(binance_client, quote='USDT') -> List:
    fiat_symbols = ['USDT', 'BKRW', 'BUSD', 'EUR', 'TUSD', 'USDC', 'TRY', 'PAX', 'AUD', 'BIDR', 'BRL', 'DAI', 'GBP',
                    'IDRT', 'NGN', 'RUB', 'ZAR', 'UAH', 'BVND']
    retired_symbol = ['LEND']
    filters = get_symbol_filters(binance_client, quote)
    nomargin_symbols = filters.loc[filters['marginAllowed'] == False]['base'].tolist()
    excluded_symbols = fiat_symbols + retired_symbol + nomargin_symbols
    symbols = pd.DataFrame(binance_client.get_all_isolated_margin_symbols())
    base_symbols = symbols.loc[(symbols['quote'] == quote)]['base'].tolist()
    return [i for i in base_symbols if i not in excluded_symbols]


def get_cross_margin_symbols(binance_client) -> List:
    fiat_symbols = ['USDT', 'BKRW', 'BUSD', 'EUR', 'TUSD', 'USDC', 'TRY', 'PAX', 'AUD', 'BIDR', 'BRL', 'DAI', 'GBP',
                    'IDRT', 'NGN', 'RUB', 'ZAR', 'UAH', 'BVND']
    retired_symbol = ['LEND']
    excluded_symbols = fiat_symbols + retired_symbol
    account = pd.DataFrame(binance_client.get_margin_account()['userAssets'])
    return [i for i in account['asset'] if i not in excluded_symbols]


def create_all_isolated_margin_account(binance_client, quote='USDT'):
    symbols_USDT = get_all_isolated_margin_symbols(binance_client, quote)
    for base_symbol in symbols_USDT:
        try:
            binance_client.create_isolated_margin_account(base=base_symbol, quote='USDT')
        except BinanceAPIException as e:
            print(base_symbol, e)


def round_down(num, precision):
    return math.floor(num * 10 ** precision) / (10 ** precision)


def round_up(num, precision):
    return math.ceil(num * 10 ** precision) / (10 ** precision)


def order(binance_client: Client, base: str, targetQtyQuote: float, filters: PandasDataFrame,
          shortMarginLevel=1.51, longMarginLevel=2, quote='USDT', tradeFee=0.001):
    def replay_loan():
        account = get_isolated_margin_account(binance_client, quote)
        BaseQuote = account.loc[account['base_asset'] == base].iloc[0]
        currentQuantityHeld = BaseQuote['base_free']
        currentQuantityBorrowed = BaseQuote['base_borrowed'] + BaseQuote['base_interest']
        if min(currentQuantityBorrowed, currentQuantityHeld) > 0:
            print('Repaying loans', base + quote)
            binance_client.repay_margin_loan(asset=base, amount=min(currentQuantityBorrowed, currentQuantityHeld),
                                             symbol=base + quote,
                                             isIsolated=True)

        QuoteHeld = BaseQuote['quote_free']
        QuoteBorrowed = BaseQuote['quote_borrowed'] + BaseQuote['quote_interest']
        QuoteBorrowedNeeded = targetQtyQuote / longMarginLevel if targetQtyQuote > 0 else \
            abs(targetQtyQuote) * (shortMarginLevel - 1)
        QuoteTransfer = QuoteBorrowed - QuoteBorrowedNeeded
        if min(QuoteHeld, QuoteTransfer) > 0:
            print('Repaying loans', base + quote)
            binance_client.repay_margin_loan(asset=quote, amount=min(QuoteHeld, QuoteTransfer),
                                             symbol=base + quote,
                                             isIsolated=True)

    def transfer_isolated_margin_to_spot():
        account = get_isolated_margin_account(binance_client, quote)
        BaseQuote = account.loc[account['base_asset'] == base].iloc[0]
        debt = (BaseQuote['base_borrowed'] + BaseQuote['base_interest']) * BaseQuote['price'] + \
               BaseQuote['quote_borrowed'] + BaseQuote['quote_interest']
        asset = BaseQuote['base_free'] * BaseQuote['price']
        marginLevelMin = 2.01  # can only transfer if marginal >2
        quoteRequired = marginLevelMin * debt - asset

        quoteTransfer = BaseQuote['quote_free'] - quoteRequired
        if quoteTransfer > 0:
            binance_client.transfer_isolated_margin_to_spot(asset=quote, symbol=base + quote,
                                                            amount=min(quoteTransfer, BaseQuote['quote_free']))

    def transfer_spot_to_isolated_margin(targetQtyQuote):
        account = get_isolated_margin_account(binance_client, quote)
        moneyHave = account.loc[account['base_asset'] == base].iloc[0]['totalEquity']
        moneyNeed = abs(targetQtyQuote) / longMarginLevel - moneyHave if targetQtyQuote > 0 \
            else abs(targetQtyQuote) * (shortMarginLevel - 1) - moneyHave
        if moneyNeed > 0:
            binance_client.transfer_spot_to_isolated_margin(asset=quote, symbol=base + quote, amount=moneyNeed)
        if moneyNeed < 0:
            transfer_isolated_margin_to_spot()

    def reduce_position(targetQtyQuote):
        print('Reducing position:', base + quote)
        account = get_isolated_margin_account(binance_client, quote)
        BaseQuote = account.loc[account['base_asset'] == base].iloc[0]
        currentQtyQuote = BaseQuote['base_netAssetOfQuote']
        QtyHeld = BaseQuote['base_free']
        QtyBorrowed = BaseQuote['base_borrowed'] + BaseQuote['base_interest']
        my_quantity = QtyHeld if currentQtyQuote > 0 else QtyBorrowed
        try:
            filledQtyQuote = 0
            counter = 0
            while abs(currentQtyQuote - targetQtyQuote) - filledQtyQuote > 10:
                depth = binance_client.get_order_book(symbol=base + quote)
                my_bid = float(depth['bids'][0][0]) + 1 / 10 ** tickPrecision * counter
                my_ask = float(depth['asks'][0][0]) - 1 / 10 ** tickPrecision * counter
                my_price = my_ask if currentQtyQuote > 0 else my_bid
                order = binance_client.create_margin_order(
                    symbol=base + quote,
                    quantity=min(round_down((abs(currentQtyQuote - targetQtyQuote) - filledQtyQuote) / my_price,
                                            quantityPrecision),
                                 round_down(my_quantity, quantityPrecision)),
                    side=SIDE_SELL if currentQtyQuote > 0 else SIDE_BUY,
                    price=round_down(my_price, tickPrecision),
                    sideEffectType='NO_SIDE_EFFECT',
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_IOC,
                    isIsolated='TRUE')
                filledQtyQuote = filledQtyQuote + float(order['cummulativeQuoteQty'])
                counter = counter + 1
        except BinanceAPIException as e:
            print(e)
        replay_loan()
        transfer_isolated_margin_to_spot()

    def increase_position(targetQtyQuote):
        transfer_spot_to_isolated_margin(targetQtyQuote)
        print('Increase position:', base + quote)
        filledQtyQuote = 0
        counter = 0
        account = get_isolated_margin_account(binance_client, quote)
        currentQtyQuote = account.loc[account['base_asset'] == base].iloc[0]['base_netAssetOfQuote']
        try:
            while abs(targetQtyQuote - currentQtyQuote) - filledQtyQuote > 10:
                depth = binance_client.get_order_book(symbol=base + quote)
                highest_bid = float(depth['bids'][0][0])
                lowest_ask = float(depth['asks'][0][0])
                my_bid = highest_bid + 1 / 10 ** tickPrecision * counter
                my_ask = lowest_ask - 1 / 10 ** tickPrecision * counter
                my_price = my_bid if targetQtyQuote - currentQtyQuote > 0 else my_ask
                order = binance_client.create_margin_order(
                    symbol=base + quote,
                    price=round_down(my_price, tickPrecision),
                    quantity=round_down((abs(targetQtyQuote - currentQtyQuote) - filledQtyQuote) / my_price,
                                        quantityPrecision),
                    side=SIDE_BUY if targetQtyQuote > 0 else SIDE_SELL,
                    sideEffectType='MARGIN_BUY',
                    type=ORDER_TYPE_LIMIT,
                    timeInForce=TIME_IN_FORCE_IOC,
                    isIsolated='TRUE')
                filledQtyQuote = filledQtyQuote + float(order['cummulativeQuoteQty'])
                counter = counter + 1
        except BinanceAPIException as e:
            print(e)

    myAccount = get_isolated_margin_account(binance_client, quote)
    myCurrentQtyQuote = myAccount.loc[myAccount['base_asset'] == base].iloc[0]['base_netAssetOfQuote']
    tickPrecision = filters.loc[(filters['base'] == base) & (filters['quote'] == quote)].iloc[0]['tickPrecision']
    quantityPrecision = filters.loc[(filters['base'] == base) & (filters['quote'] == quote)].iloc[0][
        'stepPrecision']
    # from long to short or short to long
    replay_loan()
    if myCurrentQtyQuote * targetQtyQuote < 0:
        if abs(myCurrentQtyQuote) >= 10:
            reduce_position(0)
        if abs(targetQtyQuote) >= 10:
            increase_position(targetQtyQuote)
    if myCurrentQtyQuote * targetQtyQuote >= 0:
        # reduce or close short or long position
        if abs(myCurrentQtyQuote) - abs(targetQtyQuote) >= 10:
            reduce_position(targetQtyQuote)
        # increases or open short or long position
        if abs(targetQtyQuote) - abs(myCurrentQtyQuote) >= 10:
            increase_position(targetQtyQuote)
