# This script download data from mysql database
import json

import pandas as pd
from sqlalchemy import create_engine

# global params
with open('secret.json') as myfile:
    secrets = json.load(myfile)
usr = secrets['mysql_usr']
pin = secrets['mysql_password']
start = '1996-01-01'
# import option signals from mysql
optionsig_connection = create_engine(f'mysql+pymysql://{usr}:{pin}@localhost/option_sig')
dfs = []
for table in optionsig_connection.table_names():
    df = pd.read_sql(f"SELECT * FROM {table} WHERE date >='{start}'", con=optionsig_connection)
    dfs.append(df)
# import crsp from mysql
crsp_connection = create_engine(f'mysql+pymysql://{usr}:{pin}@localhost/crsp')
crsp = pd.read_sql(f"SELECT date, cusip, mktcap, ret FROM stock WHERE date >='{start}'", con=crsp_connection)
# combine data
crsp = crsp.sort_values('date')
stock = crsp
for i in dfs:
    stock = stock.merge(i, on=['date', 'cusip'])
stock.to_pickle('data/return_prediction.pkl')
