import time

import pandas as pd
import yfinance as yf


def scrape_options(ticker: str, min_exp_date=None, max_exp_date=None, pause=1):
    """
    Download option date from yahoo finance
    pause: pause between downloading options of different expiration dates
    """
    stock = yf.Ticker(ticker)
    try:
        options = pd.DataFrame()
        for exp_date in stock.options:
            if min_exp_date is not None and exp_date < min_exp_date:
                break
            if max_exp_date is not None and exp_date > max_exp_date:
                break
            print(f'Downloading {ticker} options with expiration date {exp_date}')
            opt_chain = stock.option_chain(exp_date)
            puts = opt_chain.puts
            puts['type'] = 'put'
            puts['expiration'] = exp_date
            options = options.append(puts)
            calls = opt_chain.calls
            calls['type'] = 'call'
            calls['expiration'] = exp_date
            options = options.append(calls)
            time.sleep(pause)
        return options
    except IndexError as e:
        print(e, ticker, 'has no options')
