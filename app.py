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
  layout="wide",
  initial_sidebar_state="expanded")
# sns.set_style("whitegrid")

asset_1 = st.sidebar.text_input('asset_1', 'ADA-PERP')
asset_2 = st.sidebar.text_input('asset_2', 'BNB-PERP')
asset_3 = st.sidebar.text_input('asset_3', 'LINK-PERP')
asset_4 = st.sidebar.text_input('asset_4', 'BSV-PERP')
asset_5 = st.sidebar.text_input('asset_5', 'OKB-PERP')

pair = [asset_1 ,asset_2 ,asset_3 ,asset_4 ,asset_5]
timeframe = st.sidebar.selectbox('coin',('1d' , '15m' ,'1h' , '4h'))
limit =  st.sidebar.selectbox('limit',(180 , 270 , 365))

data_ = pd.DataFrame()
for i in pair :
    exchange = ccxt.ftx({'apiKey': 'ngR2rWcJjdZr-pRlcZjuhz3pAfFcWKSMqu2xVj6N','secret':  'OL_aQBcwMelSKmkZn57RkMzys21yyAZN9H6CzZ_3' ,'enableRateLimit': True }) 
    ohlcv =  exchange.fetch_ohlcv(  i  , timeframe , limit=limit )
    ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
    df =  pd.DataFrame(ohlcv)
    df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x , pytz.timezone('Asia/Bangkok') ))
    df =  df.set_index(df['t']) ; df = df.drop(['t'] , axis= 1 )
    dataset = df  ; dataset = dataset.dropna()
    data_[i] = dataset.c

data_.dropna(axis=1 ,inplace=True)
returns = risk_models.returns_from_prices(data_ , log_returns=True)

returns["sum"] = returns.sum(axis=1)
returns["cum"] = returns['sum'].cumsum(axis=0)
plt.figure(figsize=(16,8))
plt.plot(returns.cum)
st.pyplot()
# returns.cum.plot();




# exchange = ccxt.binance({'apiKey': ''   ,'secret':  ''  , 'enableRateLimit': True }) 
# e = exchange.load_markets()

# filter 	  =  st.sidebar.text_input('filter','T')
# pair 		   = [i for i in e if i[-1] == filter]
# coin      = st.sidebar.selectbox('coin',tuple(pair))
# timeframe =  st.sidebar.selectbox('coin',('1d' , '15m' ,'1h' , '4h'))
# # limit     =   st.sidebar.selectbox('limit',(180 , 270 , 365))
# n_changepoints =  st.sidebar.number_input('n_changepoints',min_value=0,value=25,step=1)
# shift_d   = st.sidebar.number_input('shift_d', 1)

# def A ():
#   global shift_d
#   ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=90 )
#   ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
#   df =  pd.DataFrame(ohlcv)
#   df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

#   shift_d = shift_d
#   Prop = df
#   Prop['ds'] = Prop['t'] 
#   Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
#   Prop = Prop.iloc[ : , -2:]

#   m = Prophet( n_changepoints = n_changepoints )
#   m.fit(Prop) 
#   future = m.make_future_dataframe(periods=shift_d)
#   forecast = m.predict(future)
#   fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
#   st.pyplot() ; #st.write(Prop.tail(1))

# def B ():
#   global shift_d
#   ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=180 )
#   ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
#   df =  pd.DataFrame(ohlcv)
#   df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

#   shift_d = shift_d
#   Prop = df
#   Prop['ds'] = Prop['t'] 
#   Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
#   Prop = Prop.iloc[ : , -2:]

#   m = Prophet( n_changepoints = n_changepoints )
#   m.fit(Prop) 
#   future = m.make_future_dataframe(periods=shift_d)
#   forecast = m.predict(future)
#   fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
#   st.pyplot() ; #st.write(Prop.tail(1))
  
  
# def C ():
#   global shift_d
#   ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=270 )
#   ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
#   df =  pd.DataFrame(ohlcv)
#   df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

#   shift_d = shift_d
#   Prop = df
#   Prop['ds'] = Prop['t'] 
#   Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
#   Prop = Prop.iloc[ : , -2:]

#   m = Prophet( n_changepoints = n_changepoints )
#   m.fit(Prop) 
#   future = m.make_future_dataframe(periods=shift_d)
#   forecast = m.predict(future)
#   fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
#   st.pyplot() ; #st.write(Prop.tail(1)) 
  
  
# def D ():
#   global shift_d
#   ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=365 )
#   ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
#   df =  pd.DataFrame(ohlcv)
#   df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

#   shift_d = shift_d
#   Prop = df
#   Prop['ds'] = Prop['t'] 
#   Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
#   Prop = Prop.iloc[ : , -2:]

#   m = Prophet( n_changepoints = n_changepoints )
#   m.fit(Prop) 
#   future = m.make_future_dataframe(periods=shift_d)
#   forecast = m.predict(future)
#   fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
#   st.pyplot() ; #st.write(Prop.tail(1))   
  
# col1, col2 = st.beta_columns(2)
# col3, col4 = st.beta_columns(2)

# with col1:
#   col1.text("90")
#   _ = A()

# with col2:
#   col2.text("180")
#   _ = B()

# with col3:
#   col3.text("270")
#   _ = C()

# with col4:
#   col4.text("365")
#   _ = D() 
 

