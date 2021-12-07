import tabula
import pandas as pd
from functools import reduce

def read_holdings(url_key, url_value):
    usr_agent = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_1) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Safari/605.1.15'
    weight_col = 'weight_'+url_key
    df2 = tabula.read_pdf(url_value, user_agent=usr_agent,pages='all')
    df=pd.DataFrame(df2[0]).dropna(how='all',axis=1 )
    df.columns = ['counter', 'company', 'ticker', 'cusip','shares', 'mktcap', weight_col]
    return df[['ticker', weight_col]]
def download_ARK():
    ARKK = 'https://ark-funds.com/wp-content/fundsiteliterature/holdings/ARK_INNOVATION_ETF_ARKK_HOLDINGS.pdf'
    ARKQ = 'https://ark-funds.com/wp-content/fundsiteliterature/holdings/ARK_AUTONOMOUS_TECHNOLOGY_&_ROBOTICS_ETF_ARKQ_HOLDINGS.pdf'
    ARKW = 'https://ark-funds.com/wp-content/fundsiteliterature/holdings/ARK_NEXT_GENERATION_INTERNET_ETF_ARKW_HOLDINGS.pdf'
    ARKG = 'https://ark-funds.com/wp-content/fundsiteliterature/holdings/ARK_GENOMIC_REVOLUTION_MULTISECTOR_ETF_ARKG_HOLDINGS.pdf'
    ARKF = 'https://ark-funds.com/wp-content/fundsiteliterature/holdings/ARK_FINTECH_INNOVATION_ETF_ARKF_HOLDINGS.pdf'
    urls = {'ARKK': ARKK, 'ARKQ': ARKQ, 'ARKW': ARKW, 'ARKG': ARKG, 'ARKF': ARKF}
    df_list = [read_holdings(key, value) for key, value in urls.items()]
    df_merged = reduce(lambda x,y: pd.merge(x,y,on=['ticker'], how='outer'), df_list)
    df_merged.loc[df_merged['ticker'].notnull()].to_csv('ARK.csv')

