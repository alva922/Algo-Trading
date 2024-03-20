#Volatility Indicators
#https://github.com/Nikhil-Adithyan/Algorithmic-Trading-with-Python/blob/main/Volatility/Bollinger_Bands.py
import os
os.chdir('YOURPATH')    # Set working directory
os. getcwd()
import pandas as pd 
import matplotlib.pyplot as plt 
import requests
import math
from termcolor import colored as cl 
import numpy as np

plt.style.use('fivethirtyeight')
plt.rcParams['figure.figsize'] = (20, 10)

# EXTRACTING STOCK DATA

def get_historical_data(symbol, start_date):
    api_key = 'your_api_key'
    api_url = f'https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day&outputsize=5000&apikey={api_key}'
    raw_df = requests.get(api_url).json()
    df = pd.DataFrame(raw_df['values']).iloc[::-1].set_index('datetime').astype(float)
    df = df[df.index >= start_date]
    df.index = pd.to_datetime(df.index)
    return df

aapl = get_historical_data('NVDA', '2023-01-03')
aapl.tail()
#fig1
plt.plot(aapl.index, aapl['close'])
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.title('Stock Prices')
plt.show()
#fig2
def sma(data, window):
    sma = data.rolling(window = window).mean()
    return sma

aapl['sma_20'] = sma(aapl['close'], 20)
aapl.tail(3)

aapl['close'].plot(label = 'CLOSE', alpha = 0.6)
aapl['sma_20'].plot(label = 'SMA 20', linewidth = 2)
plt.xlabel('Date')
plt.ylabel('Closing Prices')
plt.legend(loc = 'upper left')
plt.show()
#fig3
def bb(data, sma, window):
    std = data.rolling(window = window).std()
    upper_bb = sma + std * 2
    lower_bb = sma - std * 2
    return upper_bb, lower_bb

aapl['upper_bb'], aapl['lower_bb'] = bb(aapl['close'], aapl['sma_20'], 20)
aapl.tail()

aapl['close'].plot(label = 'CLOSE PRICES', color = 'skyblue')
aapl['upper_bb'].plot(label = 'UPPER BB 20', linestyle = '--', linewidth = 1, color = 'black')
aapl['sma_20'].plot(label = 'MIDDLE BB 20', linestyle = '--', linewidth = 1.2, color = 'grey')
aapl['lower_bb'].plot(label = 'LOWER BB 20', linestyle = '--', linewidth = 1, color = 'black')
plt.legend(loc = 'upper left')
plt.title('BOLLINGER BANDS')
plt.show()
#fig4
def implement_bb_strategy(data, lower_bb, upper_bb):
    buy_price = []
    sell_price = []
    bb_signal = []
    signal = 0
    
    for i in range(len(data)):
        if data[i-1] > lower_bb[i-1] and data[i] < lower_bb[i]:
            if signal != 1:
                buy_price.append(data[i])
                sell_price.append(np.nan)
                signal = 1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        elif data[i-1] < upper_bb[i-1] and data[i] > upper_bb[i]:
            if signal != -1:
                buy_price.append(np.nan)
                sell_price.append(data[i])
                signal = -1
                bb_signal.append(signal)
            else:
                buy_price.append(np.nan)
                sell_price.append(np.nan)
                bb_signal.append(0)
        else:
            buy_price.append(np.nan)
            sell_price.append(np.nan)
            bb_signal.append(0)
            
    return buy_price, sell_price, bb_signal

buy_price, sell_price, bb_signal = implement_bb_strategy(aapl['close'], aapl['lower_bb'], aapl['upper_bb'])
aapl['close'].plot(label = 'CLOSE PRICES', alpha = 0.3)
aapl['upper_bb'].plot(label = 'UPPER BB', linestyle = '--', linewidth = 1, color = 'black')
aapl['sma_20'].plot(label = 'MIDDLE BB', linestyle = '--', linewidth = 1.2, color = 'grey')
aapl['lower_bb'].plot(label = 'LOWER BB', linestyle = '--', linewidth = 1, color = 'black')
plt.scatter(aapl.index, buy_price, marker = '^', color = 'green', label = 'BUY', s = 200)
plt.scatter(aapl.index, sell_price, marker = 'v', color = 'red', label = 'SELL', s = 200)
plt.title('BB STRATEGY TRADING SIGNALS')
plt.legend(loc = 'upper left')
plt.show()
#fig6
position = []
for i in range(len(bb_signal)):
    if bb_signal[i] > 1:
        position.append(0)
    else:
        position.append(1)
        
for i in range(len(aapl['close'])):
    if bb_signal[i] == 1:
        position[i] = 1
    elif bb_signal[i] == -1:
        position[i] = 0
    else:
        position[i] = position[i-1]
        
upper_bb = aapl['upper_bb']
lower_bb = aapl['lower_bb']
close_price = aapl['close']
bb_signal = pd.DataFrame(bb_signal).rename(columns = {0:'bb_signal'}).set_index(aapl.index)
position = pd.DataFrame(position).rename(columns = {0:'bb_position'}).set_index(aapl.index)

frames = [close_price, upper_bb, lower_bb, bb_signal, position]
strategy = pd.concat(frames, join = 'inner', axis = 1)

#strategy.tail(10)
rets = aapl.close.pct_change().dropna()
strat_rets = strategy.bb_position[1:]*rets

plt.title('Daily Returns')
rets.plot(color = 'blue', alpha = 0.3, linewidth = 7)
strat_rets.plot(color = 'r', linewidth = 1)
plt.show()
#fig7
# BACKTESTING
import math

aapl_ret = pd.DataFrame(np.diff(aapl['close'])).rename(columns = {0:'returns'})
tsi_strategy_ret = []

for i in range(len(aapl_ret)):
    returns = aapl_ret['returns'][i]*strategy['bb_position'][i]
    tsi_strategy_ret.append(returns)
    
tsi_strategy_ret_df = pd.DataFrame(tsi_strategy_ret).rename(columns = {0:'bb_returns'})
investment_value = 10000
number_of_stocks = math.floor(investment_value/aapl['close'][0])
tsi_investment_ret = []

for i in range(len(tsi_strategy_ret_df['bb_returns'])):
    returns = number_of_stocks*tsi_strategy_ret_df['bb_returns'][i]
    tsi_investment_ret.append(returns)

tsi_investment_ret_df = pd.DataFrame(tsi_investment_ret).rename(columns = {0:'investment_returns'})
total_investment_ret = round(sum(tsi_investment_ret_df['investment_returns']), 2)
profit_percentage = math.floor((total_investment_ret/investment_value)*100)
print(cl('Profit gained from the BB strategy by investing $10k : {}'.format(total_investment_ret), attrs = ['bold']))
print(cl('Profit percentage of the BB strategy : {}%'.format(profit_percentage), attrs = ['bold']))
#Profit gained from the BB strategy by investing $10k : 19797.48
#Profit percentage of the BB strategy : 197%
# SPY ETF COMPARISON

def get_benchmark(start_date, investment_value):
    spy = get_historical_data('SPY', start_date)['close']
    benchmark = pd.DataFrame(np.diff(spy)).rename(columns = {0:'benchmark_returns'})
    
    investment_value = investment_value
    number_of_stocks = math.floor(investment_value/spy[-1])
    benchmark_investment_ret = []
    
    for i in range(len(benchmark['benchmark_returns'])):
        returns = number_of_stocks*benchmark['benchmark_returns'][i]
        benchmark_investment_ret.append(returns)

    benchmark_investment_ret_df = pd.DataFrame(benchmark_investment_ret).rename(columns = {0:'investment_returns'})
    return benchmark_investment_ret_df

benchmark = get_benchmark('2023-01-03', 10000)
investment_value = 10000
total_benchmark_investment_ret = round(sum(benchmark['investment_returns']), 2)
benchmark_profit_percentage = math.floor((total_benchmark_investment_ret/investment_value)*100)
print(cl('Benchmark profit by investing $10k : {}'.format(total_benchmark_investment_ret), attrs = ['bold']))
print(cl('Benchmark Profit percentage : {}%'.format(benchmark_profit_percentage), attrs = ['bold']))
print(cl('BB Strategy profit is {}% higher than the Benchmark Profit'.format(profit_percentage - benchmark_profit_percentage), attrs = ['bold']))

#Benchmark profit by investing $10k : 2562.91
#Benchmark Profit percentage : 25%
#BB Strategy profit is 172% higher than the Benchmark Profit
#https://github.com/Nikhil-Adithyan/Algorithmic-Trading-with-Python/blob/main/Volatility/Keltner_Channel.py
# KELTNER CHANNEL CALCULATION

def get_kc(high, low, close, kc_lookback, multiplier, atr_lookback):
    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift()))
    tr3 = pd.DataFrame(abs(low - close.shift()))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
    atr = tr.ewm(alpha = 1/atr_lookback).mean()
    
    kc_middle = close.ewm(kc_lookback).mean()
    kc_upper = close.ewm(kc_lookback).mean() + multiplier * atr
    kc_lower = close.ewm(kc_lookback).mean() - multiplier * atr
    
    return kc_middle, kc_upper, kc_lower
    
aapl = aapl.iloc[:,:4]
aapl['kc_middle'], aapl['kc_upper'], aapl['kc_lower'] = get_kc(aapl['high'], aapl['low'], aapl['close'], 20, 2, 10)
#aapl.tail()

# KELTNER CHANNEL PLOT

plt.plot(aapl['close'], linewidth = 2, label = 'NVDA')
plt.plot(aapl['kc_upper'], linewidth = 2, color = 'orange', linestyle = '--', label = 'KC UPPER 20')
plt.plot(aapl['kc_middle'], linewidth = 1.5, color = 'grey', label = 'KC MIDDLE 20')
plt.plot(aapl['kc_lower'], linewidth = 2, color = 'orange', linestyle = '--', label = 'KC LOWER 20')
plt.legend(loc = 'lower right', fontsize = 15)
plt.title('KELTNER CHANNEL 20')
plt.show()
#fig8
