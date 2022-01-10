import pandas as pd
from algo import Algo
from ib_insync import IB, IBC
import datetime

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
pd.set_option('display.expand_frame_repr', False)
# start IB

ibc = IBC(983, tradingMode='live')
ibc.start()

# connect IB-----------------------------------------
ib = IB()
ib.connect('127.0.0.1', 1983, clientId=1)
# download hist_volume ------------------------------
algo = Algo(ib)
#algo.save_avg_hist_volume()
# trade ------------------------------
now = datetime.datetime.now()
start_time = now.replace(hour=14, minute=45, second=0, microsecond=0)
open_orders = True
while bool(open_orders) is True and now > start_time:
    algo.sell_current_positions()
    open_orders = ib.openOrders()
    if len(open_orders) == 0:
        algo.buy_target_positions()

# report --------------
algo.report()

