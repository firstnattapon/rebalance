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

asset_0 = st.sidebar.text_input('asset_1', 'LINK/USDT')
asset_1 = st.sidebar.text_input('asset_2', 'ADA/USDT')
asset_2 = st.sidebar.text_input('asset_3', 'ETH/USDT')
asset_3 = st.sidebar.text_input('asset_4', 'DOGE/USDT')
asset_4 = st.sidebar.text_input('asset_5', 'OMG/USDT')
asset_5 = st.sidebar.text_input('asset_6', 'THETA/USDT')
asset_6 = st.sidebar.text_input('asset_7', 'None')
asset_7 = st.sidebar.text_input('asset_8', 'None')
asset_8 = st.sidebar.text_input('asset_9', 'None')
asset_9 = st.sidebar.text_input('asset_10', 'None')

pair = [asset_0 ,asset_1 ,asset_2 ,asset_3 ,asset_4, asset_5, asset_6, asset_7, asset_8, asset_9]
pair = [i for i in pair if i != 'None']

data_ = pd.DataFrame()
for i in pair :
    exchange = ccxt.binance({'apiKey': '','secret':  '' ,'enableRateLimit': True }) 
    ohlcv =  exchange.fetch_ohlcv(  i  , timeframe , limit=limit )
    ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
    df =  pd.DataFrame(ohlcv)
    df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()
    df =  df.set_index(df['t']) ; df = df.drop(['t'] , axis= 1 )
    dataset = df  ; dataset = dataset.dropna()
    data_[i] = dataset.c
data_.dropna(axis=1 ,inplace=True)

returns = risk_models.returns_from_prices(data_ , log_returns=True)
S = CovarianceShrinkage(data_ ,frequency=180).ledoit_wolf()
hrp = HRPOpt(returns , cov_matrix=S)
# weights = hrp.optimize()
# hrp.portfolio_performance(verbose=True);

weights = {w: 1/len(pair) for w in pair}
hrp.set_weights(weights)
w = hrp.portfolio_performance(verbose=True,frequency=180)

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
