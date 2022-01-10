import json

import nasdaqdatalink
import pandas as pd

# pandas options
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
# download all tickers
with open('../quantsuite/pairs_trading/secret.json') as f:
    keys = json.load(f)
    # download all tickers for US firms
nasdaqdatalink.ApiConfig.api_key = keys['quandl_api']
nasdaqdatalink.get_table('SHARADAR/TICKERS', table='SEP', paginate=True).to_pickle('sharadar_tickers.pkl')
# download stock data
tickers_meta = pd.read_pickle('sharadar_tickers.pkl')
tickers = tickers_meta['ticker'].drop_duplicates().to_list()
n1 = 0
hists = []
while n1 <= len(tickers) - 1:
    # quandl api has 1 million row limit. Sharadar has ten year data. Thus, 350 * 252*10 = 0.9 million
    print(f'Downloading historical daily stock data...')
    n2 = n1 + 350
    hist = nasdaqdatalink.get_table('SHARADAR/SEP', ticker=tickers[n1: n2], paginate=True)
    hists.append(hist)
    n1 = n2

pd.concat(hists, axis=0).to_pickle('hists.pkl')
# match with tickers meta data
hists_df = pd.read_pickle('hists.pkl')
hists_df = hists_df.loc[hists_df['date'] >= '2006-01-01']
tickers_meta = pd.read_pickle('sharadar_tickers.pkl') \
    [['ticker', 'exchange', 'isdelisted', 'sicindustry', 'famaindustry', 'sector', 'industry', 'scalemarketcap']]
stock = hists_df.merge(tickers_meta, on='ticker')
# selected based on size
size_selected = (
        (stock['scalemarketcap'] == '6 - Mega') |
        (stock['scalemarketcap'] == '5 - Large') |
        (stock['scalemarketcap'] == '4 - Mid'))
exchange_selected = stock['exchange'] != 'OTC'
# filtering this will inflate performance due to survival bias. However, the database does not have delisting value.
listed_today = stock['isdelisted'] == 'N'
stock.loc[size_selected & exchange_selected & listed_today].to_pickle('stock.pkl')
