from datetime import datetime
import time
import utilities as utils
import pandas as pd
from requests import Request, Session
from requests.exceptions import ConnectionError, Timeout, TooManyRedirects
import json
from cryptocmd import CmcScraper

with open('../quantsuite/pairs_trading/secret.json') as f:
    secret = json.loads(f.read())
# pandas options
pd.set_option('display.max_columns', 100)
pd.set_option('display.width', 300)

# get crypto tickers from coinmarketcap
url = 'https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest'
parameters = {'start':'1', 'limit':'5000','convert':'USD'}
headers = {'Accepts': 'application/json', 'X-CMC_PRO_API_KEY': secret['cmc_api']}
session = Session()
session.headers.update(headers)
response = session.get(url, params=parameters)
cmc_listings = json.loads(response.text)
cmc_df = pd.DataFrame(cmc_listings['data'])
cmc_df['market_cap']= cmc_df['quote'].apply(lambda x: x['USD']['market_cap'])
cmc_df['volume_24h']= cmc_df['quote'].apply(lambda x: x['USD']['volume_24h'])
cmc_df = cmc_df[['symbol', 'market_cap']].sort_values('market_cap', ascending = False).head(100)
crypto_tickers = cmc_df['symbol'].to_list()
# download cryptos price data from Coinmarketcap
crypto_hists = []
for crypto in crypto_tickers:
    print(f'Downloading {crypto}...')
    scraper = CmcScraper(crypto)
    crypto_hist = scraper.get_dataframe()
    crypto_hist['symbol'] = crypto
    crypto_hists.append(crypto_hist)
    time.sleep(10)
pd.concat(crypto_hists,axis=0).to_pickle('../quantsuite/pairs_trading/crypto_cmc_daily.pkl')






