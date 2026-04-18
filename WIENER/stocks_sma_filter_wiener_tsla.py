#Friend Link: https://medium.com/@alexzap922/backtesting-tesla-crossover-strategies-noise-resilient-wiener-filter-ma-vs-sma-2b9ab0fe2b63?sk=7da5c50b334844ed2151994cba80a6e1
#https://medium.com/@alexzap922/backtesting-tesla-crossover-strategies-noise-resilient-wiener-filter-ma-vs-sma-2b9ab0fe2b63
import os
os.chdir('YOURPATH')    # Set working directory
os. getcwd() 
#IMPORTS
import pandas as pd 
import matplotlib.pyplot as plt 
import requests
import math
from termcolor import colored as cl 
import numpy as np
import os

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (14, 6)

#!pip install scipy
from scipy.signal import wiener

# EXTRACTING STOCKS DATA

def get_historical_data(symbol, start_date, end_date):
    api_key = 'a07d718849d64be78e8a7d5669e4e3af'
    api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
    raw_df = requests.get(api_url).json()
#    print (raw_df)
    df = pd.DataFrame(raw_df['values']).iloc[::-1].set_index('datetime').astype(float)
    df = df[df.index >= start_date]
    df = df[df.index <= end_date]
    df.index = pd.to_datetime(df.index)
    return df

start='2020-01-01'
end='2024-07-22'

aapl = get_historical_data('TSLA', start,end)
aapl.tail()

sp = get_historical_data('SPY', start,end)
sp.tail()

aapl['daily_ret']=aapl['close'].pct_change(1)
sp['daily_ret']=sp['close'].pct_change(1)
aapl['daily_ret'].plot(label='TSLA')
sp['daily_ret'].plot(label='SPY')
plt.grid(color='black')
plt.title('Daily Returns')
plt.legend()

# Calculate cumulative returns
aapl['cum_ret'] = (1 + aapl['daily_ret']).cumprod()
sp['cum_ret'] = (1 + sp['daily_ret']).cumprod()

aapl['cum_ret'].plot(label='TSLA')
sp['cum_ret'].plot(label='SPY')
plt.grid(color='black')
plt.title('Cumulative Returns')
plt.legend()
plt.show()

#functions
def implement_ma_strategy(data, short_window, long_window):
    sma1 = short_window
    sma2 = long_window
    buy_price = []
    sell_price = []
    sma_signal = []
    signal = 0
    
    for i in range(len(data)):
        if sma1.iloc[i] > sma2.iloc[i]:
            if signal != 1:
                buy_price.append(data.iloc[i])
                sell_price.append(np.nan)
                signal = 1
                sma_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                sma_signal.append(0)
        elif sma2.iloc[i] > sma1.iloc[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data.iloc[i])
                signal = -1
                sma_signal.append(-1)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                sma_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            sma_signal.append(0)
            
    return buy_price, sell_price, sma_signal

def get_position(data, signal):
    position = []
    for i in range(len(signal)):
        if signal[i] > 1:
            position.append(0)
        else:
            position.append(1)
    for i in range(len(aapl['close'])):
        if signal[i] == 1:
            position[i] = 1
        elif signal[i] == -1:
            position[i] = 0
        else:
            position[i] = position[i-1]
    return position

def SMA(ohlc, period= 41, column="close"):
        """
        Simple moving average - rolling mean in pandas lingo. Also known as 'MA'.
        The simple moving average (SMA) is the most basic of the moving averages used for trading.
        """

        return pd.Series(
            ohlc[column].rolling(window=period).mean(),
            name="{0} period SMA".format(period),
        )

n = [5, 10]
for i in n:
    aapl[f'sma_{i}'] = SMA(aapl, period= i, column="close")

aapl.tail()

plt.plot(aapl['close'], label = 'TSLA', linewidth = 5, alpha = 0.3)
plt.plot(aapl['sma_5'], label = 'SMA 5')
plt.plot(aapl['sma_10'], label = 'SMA 10')
plt.title('TSLA Simple Moving Averages (5, 10)')
plt.legend(loc = 'upper left')
plt.xlabel('Date')
plt.ylabel('Price USD')
plt.show()

def implement_sma_strategy(data, short_window, long_window):
    sma1 = short_window
    sma2 = long_window
    buy_price = []
    sell_price = []
    sma_signal = []
    signal = 0
    
    for i in range(len(data)):
        if sma1.iloc[i] > sma2.iloc[i]:
            if signal != 1:
                buy_price.append(data.iloc[i])
                sell_price.append(np.nan)
                signal = 1
                sma_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                sma_signal.append(0)
        elif sma2.iloc[i] > sma1.iloc[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data.iloc[i])
                signal = -1
                sma_signal.append(-1)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                sma_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            sma_signal.append(0)
            
    return buy_price, sell_price, sma_signal

sma_20 = aapl['sma_5']
sma_50 = aapl['sma_10']

buy_price, sell_price, signal = implement_sma_strategy(aapl['close'], sma_20, sma_50)

plt.plot(aapl['close'], alpha = 0.5, label = 'TSLA')
plt.plot(sma_20, alpha = 0.5, label = 'SMA 5')
plt.plot(sma_50, alpha = 0.5, label = 'SMA 10')
plt.scatter(aapl.index, buy_price, marker = '^', s = 200, color = 'darkgreen', label = 'BUY SIGNAL')
plt.scatter(aapl.index, sell_price, marker = 'v', s = 200, color = 'crimson', label = 'SELL SIGNAL')
plt.legend(loc = 'upper left')
plt.title('TSLA SMA CROSSOVER TRADING SIGNALS')
plt.show()

position=get_position(aapl['close'], signal)

sma_20 = pd.DataFrame(sma_20).rename(columns = {0:'sma_5'})
sma_50 = pd.DataFrame(sma_50).rename(columns = {0:'sma_10'}) 
signal = pd.DataFrame(signal).rename(columns = {0:'sma_signal'}).set_index(aapl.index)
position = pd.DataFrame(position).rename(columns = {0:'sma_position'}).set_index(aapl.index)

frames = [sma_20, sma_50, signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)

rets = aapl.close.pct_change().dropna()
strat_rets = strategy.sma_position[1:]*rets

plt.title('Daily Returns')
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='Buy-Hold')
strat_rets.plot(color = 'r', linewidth = 1,label='Strategy')
plt.legend()
plt.show()

rets_cum = (1 + rets).cumprod() - 1 
strat_cum = (1 + strat_rets).cumprod() - 1

plt.title('Cumulative Returns')
rets_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7,label='Buy-Hold')
strat_cum.plot(color = 'r', linewidth = 2,label='Strategy')
plt.legend()
plt.show()

n = [5, 10]

for i in n:
        aapl[f'sma_{i}'] = wiener(aapl['close'], i)


sma_20 = aapl['sma_5']
sma_50 = aapl['sma_10']

buy_price, sell_price, signal = implement_ma_strategy(aapl['close'], sma_20, sma_50)
position=get_position(aapl['close'], signal)

sma_20 = pd.DataFrame(sma_20).rename(columns = {0:'sma_5'})
sma_50 = pd.DataFrame(sma_50).rename(columns = {0:'sma_10'}) 
signal = pd.DataFrame(signal).rename(columns = {0:'sma_signal'}).set_index(aapl.index)
position = pd.DataFrame(position).rename(columns = {0:'sma_position'}).set_index(aapl.index)

frames = [sma_20, sma_50, signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)

strategy

rets = aapl.close.pct_change().dropna()
strat_rets = strategy.sma_position[1:]*rets

plt.title('Daily Returns')
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7)
strat_rets.plot(color = 'r', linewidth = 1)
plt.show()

rets_cum = (1 + rets).cumprod() - 1 
strat_cum = (1 + strat_rets).cumprod() - 1

plt.title('Cumulative Returns')
rets_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7)
strat_cum.plot(color = 'r', linewidth = 2)
plt.show()

plt.plot(aapl['close'], label = 'TSLA', linewidth = 5, alpha = 0.3)
plt.plot(aapl['sma_5'], label = 'WFMA 5')
plt.plot(aapl['sma_10'], label = 'WFMA 10')
plt.title('TSLA WFMA (5, 10)')
plt.legend(loc = 'upper left')
plt.show()

plt.plot(aapl['close'], alpha = 0.3, label = 'TSLA')
plt.plot(sma_20, alpha = 0.6, label = 'WFMA 5')
plt.plot(sma_50, alpha = 0.6, label = 'WFMA 10')
plt.scatter(aapl.index, buy_price, marker = '^', s = 200, color = 'darkblue', label = 'BUY SIGNAL')
plt.scatter(aapl.index, sell_price, marker = 'v', s = 200, color = 'crimson', label = 'SELL SIGNAL')
plt.legend(loc = 'upper left')
plt.title('TSLA WFMA CROSSOVER TRADING SIGNALS')
plt.show()

plt.title('Daily Returns')
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7)
strat_rets.plot(color = 'r', linewidth = 1)
plt.show()

rets_cum = (1 + rets).cumprod() - 1 
strat_cum = (1 + strat_rets).cumprod() - 1

plt.title('Cumulative Returns')
rets_cum.plot(color = 'blue', alpha = 0.3, linewidth = 7)
strat_cum.plot(color = 'r', linewidth = 2)
plt.show()
