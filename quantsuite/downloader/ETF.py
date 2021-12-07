import re

import requests
from bs4 import BeautifulSoup
import pandas as pd
from pandas import DataFrame
import yfinance as yf
import time
import numpy as np
from tqdm import tqdm
from typing import List


class CountryETFDownloader:

    def __init__(self):
        pass

    def _get_country_exchanges(self, country: str = 'US') -> DataFrame:
        "US, CA, MX"
        # download exhange names and exchange codes
        html = requests.get('https://www.interactivebrokers.com/en/index.php?f=1563&p=etf').text
        soup = BeautifulSoup(html, 'lxml')
        rows = soup.find('tbody').find_all('a')
        exchanges = [[row.text, re.findall('exch=(.*)', row['href'])[0], row['href']] for row in rows]
        exchanges = pd.DataFrame(exchanges, columns=['exch_name', 'exch_code', 'url'])
        # add country
        exchanges['country'] = 'US'
        exchanges.loc[exchanges['exch_code'].str.contains('chix_caomega|tse'), 'country'] = 'CA'
        exchanges.loc[exchanges['exch_code'].str.contains('mexi'), 'country'] = 'MX'
        return exchanges.loc[exchanges['country'] == country]

    def _get_exchange_ETF_tickers(self, exchange: str) -> DataFrame:
        """
        Scrapes ETF tickers for an exhange from IB product listing webpage
        """

        tables = pd.read_html(f'https://www.interactivebrokers.com/en/index.php?f=567&exch={exchange}')
        return tables[2]

    def get_country_ETF_tickers(self, country: str = 'US') -> DataFrame:
        """
        Get ETF info for a country by combinding info from several exchanges
        exchanges: list of exchanges in a country. Valid exchange codes are given in
        https://www.interactivebrokers.com/en/index.php?f=1562&p=north_america
        """
        exchanges = self._get_country_exchanges(country)['exch_code']
        print(f'Scraping tickers for ETFs listed in {country}...')
        ETFs = pd.DataFrame()
        for exchange in exchanges:
            ETFs = ETFs.append(self._get_exchange_ETF_tickers(exchange))
        ETFs = ETFs.rename(columns={'Fund Description (Click link for more details)': 'fund_desp', 'Symbol': 'ticker'}) \
            .drop(columns=['IB Symbol', 'Currency'])
        return ETFs.drop_duplicates(subset='ticker')

    def _download_ETF_hist(self, ticker, auto_adjust=True) -> DataFrame:
        """
        download singa ETF history
        """
        ETF = yf.Ticker(ticker)
        try:
            ETF_hist = ETF.history(period='max', auto_adjust=auto_adjust).reset_index()
            ETF_hist = ETF_hist.assign(Ticker=ticker)
            return ETF_hist
        except Exception as e:
            print(e)

    def download_country_ETFs(self, country: str = 'US', auto_adjust=True, pause = 1):
        tickers = self.get_country_ETF_tickers(country)['ticker'].to_list()
        hists = pd.DataFrame()
        print(f'Downloading {country} ETF data')
        for ticker in tqdm(tickers):
            hist = self._download_ETF_hist(ticker, auto_adjust)
            hists = hists.append(hist)
            time.sleep(pause)
        print(f'Downloading {country} ETFs has finished')
        return hists

    def get_ETF_holdings(ticker: str) -> str:
        """
        This function scraps ETF holdings  from Yahoo finance
        """
        print(f'Downloading {ticker} holdings...')
        html = requests.get(f'https://finance.yahoo.com/quote/{ticker}/holdings?p={ticker}').text
        soup = BeautifulSoup(html, 'lxml')
        try:
            table_rows = soup.find('div', attrs={'data-test': 'top-holdings'}).find('table').find('tbody').find_all(
                'tr')
            ETF_holdings = '|'.join([row.find('td').text for row in table_rows])
        except:
            ETF_holdings = np.nan
        return ETF_holdings

    def get_ETF_sum_AUM(ticker: str) -> list:
        """
        This function downloads  ETF summary and AUM using yfinance get_info method.
        """
        print(f'Downloading {ticker} summary and AUM...')
        ETF = yf.Ticker(ticker)
        try:
            ETF_info = ETF.get_info()
            ETF_summary = ETF_info['longBusinessSummary']
            ETF_mktcap = ETF_info['totalAssets']
        except:
            ETF_summary = np.nan
            ETF_mktcap = np.nan
        return [ETF_summary, ETF_mktcap]
