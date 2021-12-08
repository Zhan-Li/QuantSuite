import importlib
from quantsuite import forcaster; importlib.reload(forcaster)
import seaborn as sns
import pandas as pd
import json
from sqlalchemy import create_engine, insert, Table, MetaData
sns.set_theme()
# params
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
# global params
with open('secret.json') as myfile:
    secrets = json.load(myfile)
usr = secrets['mysql_usr']
pin= secrets['mysql_password']
# import option signals from mysql
optionsig_connection = create_engine(f'mysql+pymysql://{usr}:{pin}@localhost/option_sig')
dfs = []
for table in optionsig_connection.table_names():
    df = pd.read_sql(f"SELECT * FROM {table}", con=optionsig_connection)
    dfs.append(df)
# import crsp from mysql
crsp_connection = create_engine(f'mysql+pymysql://{usr}:{pin}@localhost/crsp')
crsp = pd.read_sql(f"SELECT date, cusip, ret FROM stock", con=crsp_connection)
# combine data
crsp = crsp.sort_values('date')
stock = crsp
for i in dfs:
    stock = stock.merge(i, on = ['date', 'cusip'])
stock.to_pickle('data/return_prediction.pkl')
