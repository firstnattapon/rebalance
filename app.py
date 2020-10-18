import ccxt
import datetime  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
import pandas as pd
import pytz
import matplotlib.patches as mpatches 
from pypfopt import HRPOpt, CovarianceShrinkage , plotting , risk_models , DiscreteAllocation
from pypfopt import expected_returns

pd.set_option("display.precision", 6)
st.set_option('deprecation.showPyplotGlobalUse', False)
st.beta_set_page_config(
  page_title="rebalance",
#   layout="wide",
  initial_sidebar_state="expanded")
# sns.set_style("whitegrid")

timeframe = st.sidebar.selectbox('timeframe',('1d' , '15m' ,'1h' , '4h'))
limit =  st.sidebar.selectbox('limit',(180 , 90 , 270 , 365 , 1000 ,2500 , 5000))
shift_d   = st.sidebar.number_input('shift_d', 1)
n_changepoints =  st.sidebar.number_input('n_changepoints',min_value=0,value=25,step=1)

asset_0 = st.sidebar.text_input('asset_1', 'ADA-PERP')
asset_1 = st.sidebar.text_input('asset_2', 'BNB-PERP')
asset_2 = st.sidebar.text_input('asset_3', 'LINK-PERP')
asset_3 = st.sidebar.text_input('asset_4', 'BSV-PERP')
asset_4 = st.sidebar.text_input('asset_5', 'OKB-PERP')
asset_5 = st.sidebar.text_input('asset_6', 'None')
asset_6 = st.sidebar.text_input('asset_7', 'None')
asset_7 = st.sidebar.text_input('asset_8', 'None')

pair = [asset_0 ,asset_1 ,asset_2 ,asset_3 ,asset_4, asset_5, asset_6, asset_7]
pair = [i for i in pair if i != 'None']

data_ = pd.DataFrame()
for i in pair :
    exchange = ccxt.ftx({'apiKey': 'ngR2rWcJjdZr-pRlcZjuhz3pAfFcWKSMqu2xVj6N','secret':  'OL_aQBcwMelSKmkZn57RkMzys21yyAZN9H6CzZ_3' ,'enableRateLimit': True }) 
    ohlcv =  exchange.fetch_ohlcv(  i  , timeframe , limit=limit )
    ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
    df =  pd.DataFrame(ohlcv)
    df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()
    df =  df.set_index(df['t']) ; df = df.drop(['t'] , axis= 1 )
    dataset = df  ; dataset = dataset.dropna()
    data_[i] = dataset.c
data_.dropna(axis=1 ,inplace=True)

returns = risk_models.returns_from_prices(data_ , log_returns=True)
S = CovarianceShrinkage(data_ ,frequency=365).ledoit_wolf()
hrp = HRPOpt(returns , cov_matrix=S)
# weights = hrp.optimize()
# hrp.portfolio_performance(verbose=True);

weights = {w:1 /len(pair) for w in pair}

hrp.set_weights(weights)
w = hrp.portfolio_performance(verbose=True,frequency=365)
st.write("Expected annual return: {:.2f}%".format(w[0]*100))
st.write("Annual volatility: {:.2f}%".format(w[1]*100))
st.write("Sharpe Ratio: {:.2f}".format(w[2]))
    
returns = risk_models.returns_from_prices(data_ , log_returns=True)
returns["sum"] = returns.sum(axis=1)
returns["cum"] = returns['sum'].cumsum(axis=0)
returns = returns.reset_index()
plt.figure(figsize=(12,6))
plt.plot(returns.cum)
st.pyplot()

shift_d = shift_d
Prop = returns
Prop['ds'] = Prop['t'] 
Prop['y'] =  Prop['cum'] 
Prop = Prop.iloc[ : , -2:]

m = Prophet( n_changepoints = n_changepoints )
m.fit(Prop) 
future = m.make_future_dataframe(periods=shift_d)
forecast = m.predict(future)
fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
st.pyplot() 

prices = returns.set_index('ds')
prices = prices.y
peeks = prices.cummax()
drowdown = (prices - peeks)/peeks
plt.plot(drowdown)
st.pyplot() 
st.write(drowdown.min())

# 'ALTBULL/USD',#0
# 'BCHBULL/USD',#0
# 'BSVBULL/USD',#0
# 'EOSBULL/USD',#0
# 'ETCBULL/USD',#0
# 'LTCBULL/USD',#0
# 'MIDBULL/USD',#0
# 'OKBBULL/USD',#0
# 'TRXBULL/USD',#0
# 'XRPBULL/USD',#0

# 'ALGOHEDGE/USD',#1
# 'ALTHEDGE/USD',#1
# 'BCHHEDGE/USD',#1
# 'BNBHEDGE/USD',#1
# 'EOSHEDGE/USD',#1
# 'ETHHEDGE/USD',#1
# 'HEDGE/USD',#1
# 'HEDGESHIT/USD',#1
# 'LINKHEDGE/USD',#1
# 'LTCHEDGE/USD',#1
# 'MIDHEDGE/USD',#1
# 'TRXHEDGE/USD',#1
# 'USDTBULL/USD',#1
# 'XRPHEDGE/USD',#1

# 'ADA-PERP',#2
# 'ETH-PERP',#2
# 'ETH/USD',#2
# 'TRX-PERP',#2
# 'LEOBULL/USD',#2

# 'ALTBEAR/USD',#3
# 'BEAR/USD',#3
# 'BEARSHIT/USD',#3
# 'BNBBEAR/USD',#3
# 'BSVHEDGE/USD',#3
# 'EOSBEAR/USD',#3
# 'ETHBEAR/USD',#3
# 'EXCHBEAR/USD',#3
# 'HTBEAR/USD',#3
# 'HTHEDGE/USD',#3
# 'LTCBEAR/USD',#3
# 'MIDBEAR/USD',#3
# 'OKBHEDGE/USD',#3
# 'TRXBEAR/USD',#3
# 'XRPBEAR/USD',#3

# 'ALT-PERP',#4
# 'BNB-PERP',#4
# 'DOGE-PERP',#4
# 'MATIC-PERP',#4
# 'MID-PERP',#4
# 'SHIT-PERP',#4
# 'XRP-PERP',#4

# 'LEOBEAR/USD',#5
# 'LEOHEDGE/USD',#5
# 'USDTBEAR/USD',#5
# 'USDTHEDGE/USD',#5

# 'ALGO-PERP',#6
# 'LINK-PERP',#6
# 'USDT-PERP',#6
# 'USDT/USD',#6
# 'LINKBULL/USD',#6

# 'BCH-PERP',#7
# 'BSV-PERP',#7
# 'EOS-PERP',#7
# 'ETC-PERP',#7
# 'LTC-PERP',#7
# 'ALGOBULL/USD',#7
# 'BNBBULL/USD',#7
# 'BULL/USD',#7
# 'BULLSHIT/USD',#7
# 'DOGEBULL/USD',#7
# 'ETHBULL/USD',#7
# 'EXCHBULL/USD',#7
# 'HTBULL/USD',#7

# 'BTC-PERP',#8
# 'BTC/USD',#8
# 'EXCH-PERP',#8
# 'FTT/BTC',#8
# 'FTT/USD',#8
# 'FTT/USDT',#8
# 'HT-PERP',#8
# 'LEO-PERP',#8
# 'OKB-PERP',#8

# 'ALGOBEAR/USD',#9
# 'BCHBEAR/USD',#9
# 'BSVBEAR/USD',#9
# 'ETCBEAR/USD',#9
# 'LINKBEAR/USD',#9
# 'OKBBEAR/USD',#9
