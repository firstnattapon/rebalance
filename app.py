import ccxt
import datetime  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from fbprophet import Prophet
from fbprophet.plot import add_changepoints_to_plot
pd.set_option("display.precision", 6)
# sns.set_style("whitegrid")

exchange = ccxt.binance({'apiKey': ''   ,'secret':  ''  , 'enableRateLimit': True }) 
e = exchange.load_markets()

filter 	  =  st.sidebar.text_input('filter','T')
pair 		   = [i for i in e if i[-1] == filter]
coin      = st.sidebar.selectbox('coin',tuple(pair))
timeframe =  st.sidebar.selectbox('coin',('1d' , '15m' ,'1h' , '4h'))
limit     =   st.sidebar.number_input('limit',value=180)
n_changepoints =  st.sidebar.number_input('n_changepoints',25)
shift_d   = st.sidebar.number_input('shift_d', 1)

ohlcv =  exchange.fetch_ohlcv( coin  , timeframe , limit=limit )
ohlcv = exchange.convert_ohlcv_to_trading_view(ohlcv)
df =  pd.DataFrame(ohlcv)
df.t = df.t.apply(lambda  x :  datetime.datetime.fromtimestamp(x)) ; df = df.dropna()

shift_d = shift_d
Prop = df
Prop['ds'] = Prop['t'] 
Prop['y'] =  (Prop['o']  + Prop['h']  +Prop['l']  +Prop['c'] ) / 4
Prop = Prop.iloc[ : , -2:]

m = Prophet( n_changepoints = n_changepoints )
m.fit(Prop) 
future = m.make_future_dataframe(periods=shift_d)
forecast = m.predict(future)
fig = add_changepoints_to_plot((m.plot(forecast)).gca(), m, forecast)
st.pyplot() ; st.write(Prop.tail(1))

pct = pd.DataFrame()
pct['y'] = Prop.y.pct_change()
pct['cf_buy'] =  pct.y.map( lambda  x : np.where (x > 0 , x  , 0 ))  
pct['sum_buy'] = pct.cf_buy.cumsum()    
pct['cf_sell'] =  pct.y.map( lambda  x : np.where (x < 0 , abs(x)  , 0) )  
pct['sum_sell'] = pct.cf_sell.cumsum() 
pct['cf_all'] =  pct.y.map( lambda  x : abs(x) )  
pct['sum_all'] = pct.cf_all.cumsum() 
st.write(pct.tail(1))

f, (ax1, ax2) = plt.subplots(2  , figsize=(15,15) )
ax1.plot(pct.sum_all)
ax2.plot(pct.sum_buy)
ax2.plot(pct.sum_sell)
st.pyplot() 





st.button("Show", key=1)
st.button("Show", key=2)
st.write('<style>div.Widget.row-widget.stRadio > div{flex-direction:row;}</style>', unsafe_allow_html=True)

