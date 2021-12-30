
from polygon import RESTClient
import nasdaqdatalink
import json
import pandas as pd
from datetime import date, timedelta
import pandas_market_calendars as  mcal
import numpy as np
from ib_insync import util, Stock, Order
import yagmail
import datetime
import time


class Algo:
    def __init__(self, ib):
        self.ib = ib
        self.best_params = {'look_back': -69, 'return_min': 0.036, 'n': 33, 'volc_min': 100000}
        with open('/Volumes/Data/secrets/live_trading_secret.json') as f:
            self.keys = json.load(f)
        self.polygon_client = RESTClient(self.keys['polygon_key'])

    @staticmethod
    def send_email(email_title, email_body, receiver_email,
                   sender_email, sender_email_pass, sender_host='smtp.aol.com', sender_host_port=465):
        yag = yagmail.SMTP(sender_email, sender_email_pass, host=sender_host, port=sender_host_port)
        yag.send(receiver_email, email_title, email_body)

    def download_all_tickers_info(self):
        # download all tickers for US firms
        nasdaqdatalink.ApiConfig.api_key = self.keys['quandl_key']
        tickers = nasdaqdatalink.get_table('SHARADAR/TICKERS', table='SEP', paginate=True)
        tickers = tickers.loc[tickers['isdelisted'] == 'N']
        return tickers

    def download_hist_data(self, tickers_list=None):
        """
        This function downloads stock volumes from -lookback days to yesterday. Then it calculate the past average volume.
        tickers: list
        :return: dataframe

        """
        if tickers_list is None:
            tickers_list = self.download_all_tickers_info()['ticker']
        lookback = self.best_params['look_back']
        hists = pd.DataFrame()
        for ticker in tickers_list:
            try:
                hist = pd.DataFrame(self.polygon_client.stocks_equities_aggregates(
                    ticker, 1, 'day',
                    from_=datetime.date.today() + timedelta(days=np.round(lookback / 5 * 8)),
                    to=datetime.date.today() + timedelta(days=-1),
                    unadjusted=False,
                    limit=50000).results)
                hist['ticker'] = ticker
                if len(hist) >= abs(lookback):  # remove partial history.
                    hists = hists.append(hist)
            except Exception as e:
                print(ticker, e)
        hists['t'] = pd.to_datetime(hists['t'], unit='ms')
        hists_selected = hists.sort_values('t').groupby('ticker').apply(lambda x: x.iloc[lookback:]) \
            .reset_index(drop=True)
        return hists_selected

    def save_avg_hist_volume(self, tickers_list=None):
        if tickers_list is None:
            tickers_list = self.download_all_tickers_info()['ticker']
        hists = self.download_hist_data(tickers_list)
        t_max = hists.groupby('ticker')['t'].max().to_frame()
        v_avg = hists.groupby('ticker')['v'].mean().to_frame()
        merged = t_max.merge(v_avg, on=['ticker']).reset_index().rename(columns = {'t': 't_hist', 'v': 'v_hist'})
        merged = merged.loc[merged['t_hist'] == merged['t_hist'].max()]
        merged.to_pickle('hists_v.pkl')
        # send email report
        num_tickers = len(merged['ticker'].unique())
        date = merged['t_hist'].unique()
        body = f'Download {num_tickers} stocks. Date updated at {date}'
        self.send_email('Stock data download report', body,
                        self.keys['email'], self.keys['email'], self.keys['email_password'], self.keys["smtp"], 465)

    def download_live_data(self):
        market = pd.DataFrame(self.polygon_client.stocks_equities_snapshot_all_tickers().tickers)
        market['h_today'] = market['day'].apply(lambda x: x['h'])
        market['l_today'] = market['day'].apply(lambda x: x['l'])
        market['o_today'] = market['day'].apply(lambda x: x['o'])
        market['v_today'] = market['day'].apply(lambda x: x['v'])
        market['last_price'] = market['lastTrade'].apply(lambda x: x['p'])
        market['ystday_close'] = market['prevDay'].apply(lambda x: x['c'])
        market['r_hc'] = market['h_today']/market['ystday_close'] - 1
        market['r_lc'] = market['l_today'] / market['ystday_close'] - 1
        market['r_oc'] = market['o_today'] / market['ystday_close'] - 1
        market['r_cc'] = market['last_price'] / market['ystday_close'] - 1
        market['r_avg'] = (market['r_hc'].abs() + market['r_lc'].abs() + market['r_oc'].abs() + market['r_cc'].abs())/4
        market['v_d_today'] = market['v_today'] * market['last_price']
        market['updated_today'] = pd.to_datetime(market['updated'], unit='ns')
        market = market.loc[market['updated_today'].dt.date == date.today()]
        market = market.loc[~market['ticker'].str.contains('\.')]
        market = market[['updated_today', 'ticker', 'last_price', 'v_today', 'v_d_today', 'r_avg']]
        return market

    def get_v_ratio(self, check_hists=True):
        hist_avg = pd.read_pickle('hists_v.pkl')
        if check_hists is True:
            nyse = mcal.get_calendar('NYSE')
            trading_days = nyse.schedule(start_date=date.today() + timedelta(-10), end_date=date.today())
            last_trading_day = trading_days['market_close'].iloc[-2].date()
            if len(hist_avg['t_hist'].unique()) != 1 or \
                    hist_avg['t_hist'].min().date() != last_trading_day or \
                    hist_avg['v_hist'].min() < 0 or\
                    len(hist_avg) < 5000:
                raise ValueError('History data check failed')
            else:
                print('Data stamp:', hist_avg['t_hist'].min(), 'Total number of tickers:', len(hist_avg['ticker'].unique()))
        market = self.download_live_data()
        merged = market.merge(hist_avg, on='ticker')
        merged = merged\
            .loc[merged['v_d_today'] >= self.best_params['volc_min']]\
            .loc[merged['r_avg'].abs() <= self.best_params['return_min']]
        merged['v_ratio'] = merged['v_today']/merged['v_hist']
        return merged

    def trade_stock(self, ticker, account, action, quantity, Tif:str):
        """
        :param ticker:
        :param account:
        :param action:
        :param quantity:
        :param Tif: DAY, GTC, IOC, GTD, OPG, FOK, DTC
        :return:
        """
        ticker = ticker
        contract = Stock(ticker, exchange='SMART', currency='USD')
        self.ib.qualifyContracts(contract)
        order = Order()
        order.account = account
        order.action = action
        order.orderType = "REL"
        order.UsePriceMgmtAlgo = True
        order.totalQuantity = quantity
        order.auxPrice = 0.01
        order.Tif = Tif
        self.ib.placeOrder(contract, order)

    def sell_current_positions(self):
        self.ib.reqPositions() # you need this to update your positions after trading
        current_positions = util.df(self.ib.positions())
        util.df(self.ib.reqPositions())
        current_positions['ticker'] = current_positions['contract'].apply(lambda x: x.symbol)
        current_positions['current_position'] = current_positions['position']
        current_positions = current_positions[['account', 'ticker', 'current_position']]
        current_positions = current_positions.loc[current_positions['ticker'] != 'FLY']
        if len(current_positions) > 0:
            for i in range(len(current_positions)):
                row = current_positions.iloc[i]
                if row['current_position'] > 0:
                    action = 'SELL'
                if row['current_position'] < 0:
                    action = 'BUY'
                print(action, row['ticker'])
                self.trade_stock(ticker=row['ticker'], account=row['account'], action=action,
                                 quantity=np.abs(int(row['current_position'])), Tif='DTC')
            time.sleep(1)

    def get_target_positions(self, account_pct = 0.95):
        account_values = util.df(self.ib.accountValues())
        account_values = account_values.loc[(account_values['tag'] == 'ExcessLiquidity')][['account', 'value']]
        account_values['value'] = account_values['value'].astype(float)
        total_position = account_values['value'].sum() * account_pct
        per_position_value = total_position / self.best_params['n']  # only invest a certain account percentage
        account_values['num_positions'] = (account_values['value'] / per_position_value).apply(np.floor).astype(int)
        accounts = []
        for i in range(len(account_values)):
            accounts.extend([account_values.iloc[i]['account']] * account_values.iloc[i]['num_positions'])
        # get target position
        n = len(accounts)
        v_ratio = self.get_v_ratio()
        target_position = v_ratio.nlargest(n, 'v_ratio')
        target_position['target_value'] =  total_position/len(target_position)
        target_position['target_position'] = (target_position['target_value']/target_position['last_price'])\
            .apply(np.floor).astype(int)
        target_position['account'] = accounts
        return target_position[['account', 'ticker', 'target_value', 'target_position', 'last_price', 'v_ratio', 'r_avg', 'v_d_today']]\
            .sort_values('v_ratio',ascending=False)

    def buy_target_positions(self):
        target_position = self.get_target_positions()
        print(target_position)
        target_position.to_pickle('target_positions.pkl')
        # order
        for i in range(len(target_position)):
            row = target_position.iloc[i]
            ticker = row['ticker']
            if row['target_position'] > 0:
                action = 'BUY'
            elif row['target_position'] < 0:
                action = 'SELL'
            self.trade_stock(ticker, account=row['account'], action=action,
                       quantity=np.abs(int(row['target_position'])), Tif='DTC')
            time.sleep(1)

    def report(self):
        report = util.df(self.ib.fills())
        #report = util.df(ib.fills())
        #report = util.df(ib.trades())
        report['ticker'] = report['contract'].apply(lambda x: x.symbol)
        report['shares'] = report['execution'].apply(lambda x: x.shares)
        report['avg_price'] = report['execution'].apply(lambda x: x.avgPrice)
        report['commission'] = report['commissionReport'].apply(lambda x: x.commission)
        report['realizedPNL'] = report['commissionReport'].apply(lambda x: x.realizedPNL)
        def aggregate(df):
            d = {}
            d['quantity'] = df['shares'].sum()
            d['avg_price'] = (df['avg_price']*df['shares']).sum()/d['quantity']
            d['commission'] = df['commission'].sum()
            d['realizedPNL'] = df['realizedPNL'].sum()
            return pd.Series(d)
        report= report.groupby('ticker').apply(aggregate)
        print('Total comission:', report['commission'].sum())
        print('Total PnL:', report['realizedPNL'].sum())
        return report

