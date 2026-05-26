#Backtesting Moving Averages Across Multiple Asset Classes — Part 3: EMA
#Python-Based EMA Backtesting Across Asset Classes: Out-of-Sample Performance, Volatility, and Drawdowns
#https://medium.com/insiderfinance/backtesting-moving-averages-across-multiple-asset-classes-part-3-ema-d73cdaf52d9b?sk=a42a43edcbcda4aea7019bc5056ee697

import requests
import pandas as pd

API_KEY = "YOUR API KEY"
def get_eodhd_data(symbol, exchange):
    url = f"https://eodhistoricaldata.com/api/eod/{symbol}.{exchange}"
    
    params = {
        "api_token": API_KEY,
        "period": "d",
        "from": "2016-04-24",
        "to": "2026-04-24",
        "fmt": "json"
    }
    
    r = requests.get(url, params=params)
    data = r.json()
    
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    
    return df

spy = get_eodhd_data("SPY", "US")
gld = get_eodhd_data("GLD", "US")
btc = get_eodhd_data("BTC-USD", "CC")

close = pd.DataFrame({
    "SPY": spy["close"],
    "GLD": gld["close"],
    "BTC": btc["close"]
})

close = close.dropna()

print(close.tail())

import pandas as pd
import numpy as np

# --- Your input (already prepared) ---
# close = pd.DataFrame({
#     "SPY": spy["close"],
#     "GLD": gld["close"],
#     "BTC": btc["close"]
# }).dropna()

# --- Parameters ---
fast_span = 10
slow_span = 100

# --- 1. Compute EMAs ---
ema_fast = close.ewm(span=fast_span, adjust=False).mean()
ema_slow = close.ewm(span=slow_span, adjust=False).mean()

# --- 2. Generate signals (per asset) ---
signal = (ema_fast > ema_slow).astype(int)

# Shift to avoid lookahead bias (trade next day)
signal = signal.shift(1).fillna(0)


# --- 3. Compute returns ---
returns = close.pct_change().fillna(0)

# --- 4. Strategy returns per asset ---
strategy_returns = signal * returns

# --- 5. Equity curves per asset ---
equity_curves = (1 + strategy_returns).cumprod()

# --- 6. Buy & Hold benchmark ---
buy_hold = (1 + returns).cumprod()

# --- 7. Summary stats function ---
def performance_stats(ret):
    ann_ret = (1 + ret.mean())**252 - 1
    ann_vol = ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else np.nan
    max_dd = (ret.add(1).cumprod() / ret.add(1).cumprod().cummax() - 1).min()
    return pd.Series({
        "Annual Return": ann_ret,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd
    })

# --- 8. Compute stats per asset ---
stats = pd.concat(
    {col: performance_stats(strategy_returns[col]) for col in close.columns},
    axis=1
).T

# --- 9. Compare with Buy & Hold ---
bh_stats = pd.concat(
    {col: performance_stats(returns[col]) for col in close.columns},
    axis=1
).T

print("Strategy Stats:\n", stats)
print("\nBuy & Hold Stats:\n", bh_stats)

import matplotlib.pyplot as plt

for col in close.columns:
    plt.figure(figsize=(10, 5))
    
    plt.plot(equity_curves.index, equity_curves[col], label="EMA Strategy")
    plt.plot(buy_hold.index, buy_hold[col], label="Buy & Hold", linestyle="--")
    
    plt.title(f"{col} — EMA Strategy vs Buy & Hold")
    plt.xlabel("Date")
    plt.ylabel("Equity (Growth of $1)")
    plt.legend()
    plt.grid(True)
    
    plt.show()

fig, axes = plt.subplots(len(close.columns), 1, figsize=(10, 12), sharex=True)

for i, col in enumerate(close.columns):
    axes[i].plot(equity_curves.index, equity_curves[col], label="EMA Strategy")
    axes[i].plot(buy_hold.index, buy_hold[col], label="Buy & Hold", linestyle="--")
    
    axes[i].set_title(col)
    axes[i].legend()
    axes[i].grid(True)

plt.xlabel("Date")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt

# --- Parameters ---
window = 126  # ~6 months of trading days

# --- Rolling Sharpe function ---
def rolling_sharpe(returns, window):
    roll_mean = returns.rolling(window).mean()
    roll_std = returns.rolling(window).std()
    sharpe = (roll_mean / roll_std) * np.sqrt(252)
    return sharpe

# --- Compute rolling Sharpe ---
rolling_sharpe_strategy = strategy_returns.apply(lambda x: rolling_sharpe(x, window))
rolling_sharpe_bh = returns.apply(lambda x: rolling_sharpe(x, window))

# --- Plot per asset ---
for col in close.columns:
    plt.figure(figsize=(10, 5))
    
    plt.plot(rolling_sharpe_strategy.index, rolling_sharpe_strategy[col],
             label="EMA Strategy")
    plt.plot(rolling_sharpe_bh.index, rolling_sharpe_bh[col],
             label="Buy & Hold", linestyle="--")
    
    #plt.axhline(0)  # zero Sharpe reference
    
    plt.title(f"{col} — 6M Rolling Sharpe")
    plt.xlabel("Date")
    plt.ylabel("Sharpe Ratio")
    plt.legend()
    plt.grid(True)
    
    plt.show()

# --- Drawdown function ---
def drawdown(equity):
    peak = equity.cummax()
    dd = (equity / peak) - 1
    return dd

# --- Compute drawdowns ---
dd_strategy = equity_curves.apply(drawdown)
dd_bh = buy_hold.apply(drawdown)

import matplotlib.pyplot as plt

for col in close.columns:
    plt.figure(figsize=(10, 5))
    
    plt.plot(dd_strategy.index, dd_strategy[col], label="EMA Strategy")
    plt.plot(dd_bh.index, dd_bh[col], label="Buy & Hold", linestyle="--")
    
    plt.axhline(0)  # peak line
    
    plt.title(f"{col} — Drawdown")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.legend()
    plt.grid(True)
    
    plt.show()

# Monthly returns
monthly_returns = close.resample("M").last().pct_change().dropna()

strategy_returns = strategy_returns.fillna(0)

# Strategy monthly returns
strategy_monthly = (1 + strategy_returns).resample("M").prod() - 1

monthly_bh = (1 + monthly_returns).prod() - 1

fast = close.ewm(span=10, adjust=False).mean()
slow = close.ewm(span=100, adjust=False).mean()

signal = (fast > slow).astype(int)

# entries / exits
buy_signals = (signal.diff() == 1)
sell_signals = (signal.diff() == -1)

import matplotlib.pyplot as plt

for col in close.columns:
    
    fig, ax = plt.subplots(figsize=(12, 5))
    
    # Price
    ax.plot(close.index, close[col], label="Price", linewidth=1.5)
    
    # Buy signals
    ax.scatter(close.index[buy_signals[col]],
               close[col][buy_signals[col]],
               marker="^", color="green", label="Buy Signal", s=120)
    
    # Sell signals
    ax.scatter(close.index[sell_signals[col]],
               close[col][sell_signals[col]],
               marker="v", color="red", label="Sell Signal", s=120)
    
    # Buy & Hold entry (first day only)
    ax.scatter(close.index[0], close[col].iloc[0],
               marker="o", color="blue", label="Buy & Hold Entry", s=120)
    
    ax.set_title(f"{col} — Price with EMA Trading Signals")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True)
    
    plt.tight_layout()
    plt.show()

import numpy as np
import pandas as pd

returns = close.pct_change().fillna(0)

results = []

fast_range = range(5, 51, 5)
slow_range = range(20, 201, 10)

for col in close.columns:
    
    asset_returns = returns[col]
    
    best_sharpe = -np.inf
    best_params = None
    
    for fast in fast_range:
        for slow in slow_range:
            
            if fast >= slow:
                continue
            
            ema_fast = close[col].ewm(span=fast, adjust=False).mean()
            ema_slow = close[col].ewm(span=slow, adjust=False).mean()
            
            signal = (ema_fast > ema_slow).astype(int).shift(1).fillna(0)
            
            strat_ret = signal * asset_returns
            
            ann_ret = strat_ret.mean() * 252
            ann_vol = strat_ret.std() * np.sqrt(252)
            
            sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_params = (fast, slow)
    
    results.append({
        "asset": col,
        "best_fast": best_params[0],
        "best_slow": best_params[1],
        "best_sharpe": best_sharpe
    })

results_df = pd.DataFrame(results)
print(results_df)

optimal_equity = {}

for _, row in results_df.iterrows():
    
    col = row["asset"]
    fast = int(row["best_fast"])
    slow = int(row["best_slow"])
    
    ema_fast = close[col].ewm(span=fast, adjust=False).mean()
    ema_slow = close[col].ewm(span=slow, adjust=False).mean()
    
    signal = (ema_fast > ema_slow).astype(int).shift(1).fillna(0)
    
    strat_ret = signal * returns[col]
    
    equity = (1 + strat_ret).cumprod()
    
    optimal_equity[col] = equity

import matplotlib.pyplot as plt

for col in close.columns:
    
    plt.figure(figsize=(10, 5))
    
    plt.plot(optimal_equity[col], label="Optimized EMA Strategy")
    plt.plot((1 + returns[col]).cumprod(), "--", label="Buy & Hold")
    
    plt.title(f"{col} — Optimized EMA vs Buy & Hold")
    plt.legend()
    #plt.grid()
    plt.xlabel('Date')
    
    plt.show()

split_idx = int(len(close) * 0.8)

train_close = close.iloc[:split_idx]
test_close  = close.iloc[split_idx:]

train_ret = train_close.pct_change().fillna(0)
test_ret  = test_close.pct_change().fillna(0)

fast_range = range(5, 51, 5)
slow_range = range(20, 201, 10)

best_params = {}

for col in close.columns:
    
    best_sharpe = -np.inf
    best_pair = None
    
    for fast in fast_range:
        for slow in slow_range:
            if fast >= slow:
                continue
            
            ema_fast = train_close[col].ewm(span=fast, adjust=False).mean()
            ema_slow = train_close[col].ewm(span=slow, adjust=False).mean()
            
            signal = (ema_fast > ema_slow).astype(int).shift(1).fillna(0)
            strat_ret = signal * train_ret[col]
            
            ann_ret = strat_ret.mean() * 252
            ann_vol = strat_ret.std() * np.sqrt(252)
            sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
            
            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_pair = (fast, slow)
    
    best_params[col] = {
        "fast": best_pair[0],
        "slow": best_pair[1],
        "train_sharpe": best_sharpe
    }

best_params

oos_equity = {}

for col in close.columns:
    
    fast = best_params[col]["fast"]
    slow = best_params[col]["slow"]
    
    ema_fast = test_close[col].ewm(span=fast, adjust=False).mean()
    ema_slow = test_close[col].ewm(span=slow, adjust=False).mean()
    
    signal = (ema_fast > ema_slow).astype(int).shift(1).fillna(0)
    
    strat_ret = signal * test_ret[col]
    
    equity = (1 + strat_ret).cumprod()
    oos_equity[col] = equity

import matplotlib.pyplot as plt

for col in close.columns:
    
    plt.figure(figsize=(10,5))
    
    bh = (1 + test_ret[col]).cumprod()
    
    plt.plot(oos_equity[col], label="EMA (OOS)")
    plt.plot(bh, "--", label="Buy & Hold (OOS)")
    
    plt.title(f"{col} — Out-of-Sample Performance")
    plt.legend()
    plt.grid()
    
    plt.show()

import pandas as pd
import numpy as np

oos_equity = {}
oos_returns = {}

for col in close.columns:
    
    fast = best_params[col]["fast"]
    slow = best_params[col]["slow"]
    
    ema_fast = test_close[col].ewm(span=fast, adjust=False).mean()
    ema_slow = test_close[col].ewm(span=slow, adjust=False).mean()
    
    signal = (ema_fast > ema_slow).astype(int).shift(1).fillna(0)
    
    strat_ret = signal * test_ret[col]
    oos_returns[col] = strat_ret
    
    equity = (1 + strat_ret).cumprod()
    oos_equity[col] = equity

# Convert to DataFrames
oos_equity = pd.DataFrame(oos_equity)
oos_returns = pd.DataFrame(oos_returns)

# ---- Performance Metrics ---- #

def performance_stats(returns, equity, freq=252):
    stats = {}
    
    total_return = equity.iloc[-1] - 1
    ann_return = (equity.iloc[-1])**(freq / len(equity)) - 1
    ann_vol = returns.std() * np.sqrt(freq)
    
    sharpe = ann_return / ann_vol if ann_vol != 0 else np.nan
    
    # Drawdown
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    
    stats["Total Return"] = total_return
    stats["Annual Return"] = ann_return
    stats["Annual Vol"] = ann_vol
    stats["Sharpe"] = sharpe
    stats["Max Drawdown"] = max_dd
    
    return stats

# Compute stats per asset
results = {}

for col in oos_equity.columns:
    results[col] = performance_stats(oos_returns[col], oos_equity[col])

results_df = pd.DataFrame(results).T

print(results_df)

import numpy as np
import pandas as pd

returns = close.pct_change().fillna(0)

fast_range = range(5, 51, 5)
slow_range = range(20, 201, 10)

def compute_sharpe_grid(asset):
    
    sharpe_grid = pd.DataFrame(
        index=fast_range,
        columns=slow_range,
        dtype=float
    )
    
    for f in fast_range:
        for s in slow_range:
            if f >= s:
                continue
            
            ema_fast = close[asset].ewm(span=f, adjust=False).mean()
            ema_slow = close[asset].ewm(span=s, adjust=False).mean()
            
            signal = (ema_fast > ema_slow).astype(int).shift(1).fillna(0)
            strat = signal * returns[asset]
            
            mu = strat.mean() * 252
            vol = strat.std() * np.sqrt(252)
            
            sharpe = mu / vol if vol != 0 else np.nan
            
            sharpe_grid.loc[f, s] = sharpe
    
    return sharpe_grid

import matplotlib.pyplot as plt

for asset in close.columns:
    
    grid = compute_sharpe_grid(asset)
    
    plt.figure(figsize=(10, 6))
    
    plt.imshow(
        grid.values,
        aspect="auto",
        origin="lower"
    )
    
    plt.colorbar(label="Sharpe Ratio")
    
    plt.xticks(
        ticks=np.arange(len(grid.columns)),
        labels=grid.columns,
        rotation=90
    )
    
    plt.yticks(
        ticks=np.arange(len(grid.index)),
        labels=grid.index
    )
    
    plt.title(f"{asset} — EMA Fast/Slow Sharpe Heatmap")
    plt.xlabel("Slow EMA")
    plt.ylabel("Fast EMA")
    
    plt.tight_layout()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

for asset in close.columns:
    
    grid = compute_sharpe_grid(asset)
    
    bf = best_params[asset]["fast"]
    bs = best_params[asset]["slow"]
    
    plt.figure(figsize=(10, 6))
    
    # --- heatmap ---
    plt.imshow(grid.values, aspect="auto", origin="lower")
    plt.colorbar(label="Sharpe Ratio")
    
    # --- axis labels ---
    plt.xticks(
        ticks=np.arange(len(grid.columns)),
        labels=grid.columns,
        rotation=90
    )
    
    plt.yticks(
        ticks=np.arange(len(grid.index)),
        labels=grid.index
    )
    
    # --- map (fast, slow) → grid coordinates ---
    y = list(grid.index).index(bf)
    x = list(grid.columns).index(bs)
    
    # best solution marker
    plt.scatter(
        x, y,
        marker="*",
        s=300,
        color="white",
        edgecolors="black",
        linewidth=1.5
    )
    
    plt.title(f"{asset} — EMA Sharpe Surface")
    plt.xlabel("Slow EMA")
    plt.ylabel("Fast EMA")
    
    plt.tight_layout()
    plt.show()

for asset, pf in pf_dict.items():
    
    print("\n======================")
    print(asset)
    print("======================")
    print(pf.stats())

pf_dict["SPY"].plot().show()

pf_dict["GLD"].plot().show()

pf_dict["BTC"].plot().show()

for asset, pf in pf_dict.items():
    
    fig = pf.plot()
    fig.update_layout(title=f"{asset} — Equity + Drawdown")
    fig.show()

for asset, pf in pf_dict.items():
    
    equity = pf.value()
    dd = equity / equity.cummax() - 1
    
    plt.figure(figsize=(10, 4))
    dd.plot()
    
    plt.title(f"{asset} — Drawdown")
    plt.grid()
    plt.show()

import matplotlib.pyplot as plt

for asset in close.columns:
    
    pf = pf_dict[asset]
    
    # Strategy equity from vectorbt
    strat_equity = pf.value()
    strat_dd = strat_equity / strat_equity.cummax() - 1
    
    # Buy & hold equity
    bh_equity = close[asset] / close[asset].iloc[0]
    bh_dd = bh_equity / bh_equity.cummax() - 1

import matplotlib.pyplot as plt

for asset in close.columns:
    
    pf = pf_dict[asset]
    
    strat_equity = pf.value()
    strat_dd = strat_equity / strat_equity.cummax() - 1
    
    bh_equity = close[asset] / close[asset].iloc[0]
    bh_dd = bh_equity / bh_equity.cummax() - 1
    
    plt.figure(figsize=(10, 4))
    
    plt.plot(strat_dd.index, strat_dd, label="EMA Strategy", linewidth=2)
    plt.plot(bh_dd.index, bh_dd, label="Buy & Hold", linestyle="--")
    
    plt.title(f"{asset} — Drawdown Comparison")
    plt.ylabel("Drawdown")
    plt.xlabel("Date")
    plt.legend()
    plt.grid()
    
    plt.show()

import numpy as np
import matplotlib.pyplot as plt

for asset in close.columns:
    
    pf = pf_dict[asset]
    
    fast = best_params[asset]["fast"]
    slow = best_params[asset]["slow"]
    
    ema_fast = close[asset].ewm(span=fast, adjust=False).mean()
    ema_slow = close[asset].ewm(span=slow, adjust=False).mean()
    
    regime = ema_fast > ema_slow

strat_equity = pf.value()
strat_dd = strat_equity / strat_equity.cummax() - 1
    
bh_equity = close[asset] / close[asset].iloc[0]
bh_dd = bh_equity / bh_equity.cummax() - 1

plt.figure(figsize=(12, 4))
    
    # --- Strategy drawdown ---
plt.plot(strat_dd.index, strat_dd, color="black", label="EMA Drawdown", linewidth=2)
    
    # --- Shade regimes ---
plt.fill_between(
        strat_dd.index,
        strat_dd,
        0,
        where=regime,
        color="green",
        alpha=0.2,
        label="Bull Regime"
    )
    
plt.fill_between(
        strat_dd.index,
        strat_dd,
        0,
        where=~regime,
        color="red",
        alpha=0.15,
        label="Bear Regime"
    )
    
    # --- Buy & Hold for reference ---
plt.plot(bh_dd.index, bh_dd, "--", label="Buy & Hold DD", alpha=0.7)
    
plt.title(f"{asset} — Regime-Aware Drawdown")
plt.ylabel("Drawdown")
plt.xlabel("Date")
plt.legend()
plt.grid()
    
plt.show()

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for i, asset in enumerate(close.columns):
    
    ax = axes[i]
    
    pf = pf_dict[asset]
    
    # --- EMAs for regime ---
    fast = best_params[asset]["fast"]
    slow = best_params[asset]["slow"]
    
    ema_fast = close[asset].ewm(span=fast, adjust=False).mean()
    ema_slow = close[asset].ewm(span=slow, adjust=False).mean()
    
    regime = ema_fast > ema_slow
    
    # --- Strategy drawdown ---
    strat_equity = pf.value()
    strat_dd = strat_equity / strat_equity.cummax() - 1
    
    # --- Buy & Hold drawdown ---
    bh_equity = close[asset] / close[asset].iloc[0]
    bh_dd = bh_equity / bh_equity.cummax() - 1
    
    # --- Plot strategy DD ---
    ax.plot(strat_dd.index, strat_dd, color="black", linewidth=1.8, label="EMA Drawdown")
    
    # --- Regime shading ---
    ax.fill_between(
        strat_dd.index,
        strat_dd,
        0,
        where=regime,
        color="green",
        alpha=0.2
    )
    
    ax.fill_between(
        strat_dd.index,
        strat_dd,
        0,
        where=~regime,
        color="red",
        alpha=0.15
    )
    
    # --- Buy & Hold ---
    ax.plot(bh_dd.index, bh_dd, "--", color="blue", alpha=0.7, label="Buy & Hold DD")
    
    ax.set_title(f"{asset} — Regime-Aware Drawdown")
    ax.set_ylabel("Drawdown")
    ax.grid(True)
    ax.legend()

plt.xlabel("Date")
plt.tight_layout()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)

for i, asset in enumerate(close.columns):
    
    ax = axes[i]
    
    pf = pf_dict[asset]
    
    fast = best_params[asset]["fast"]
    slow = best_params[asset]["slow"]
    
    ema_fast = close[asset].ewm(span=fast, adjust=False).mean()
    ema_slow = close[asset].ewm(span=slow, adjust=False).mean()
    
    regime = ema_fast > ema_slow
    
    strat_equity = pf.value()
    strat_dd = strat_equity / strat_equity.cummax() - 1
    
    bh_equity = close[asset] / close[asset].iloc[0]
    bh_dd = bh_equity / bh_equity.cummax() - 1
    
    # --- plots ---
    ax.plot(strat_dd.index, strat_dd, color="black", label="EMA Drawdown")
    ax.plot(bh_dd.index, bh_dd, "--", color="blue", label="Buy & Hold DD")
    
    ax.fill_between(
        strat_dd.index,
        strat_dd.values,
        0,
        where=regime.values,
        color="green",
        alpha=0.2
    )
    
    ax.fill_between(
        strat_dd.index,
        strat_dd.values,
        0,
        where=(~regime).values,
        color="red",
        alpha=0.15
    )
    
    ax.set_title(asset)
    ax.grid(True)
    
    # legend only once per axis (safe)
    handles, labels = ax.get_legend_handles_labels()
    
    regime_legend = [
        Patch(facecolor="green", alpha=0.2, label="Bull Regime"),
        Patch(facecolor="red", alpha=0.15, label="Bear Regime")
    ]
    
    ax.legend(handles=handles + regime_legend)

# 
plt.tight_layout()
plt.show()

for asset, pf in pf_dict.items():
    
    fig = pf.trades.plot()
    fig.update_layout(
        title=f"{asset} — Trades",
        showlegend=False
    )
    
    fig.show()

import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(len(close.columns), 1, figsize=(14, 10), sharex=True)

for i, asset in enumerate(close.columns):
    
    ax = axes[i]
    
    fast = best_params[asset]["fast"]
    slow = best_params[asset]["slow"]
    
    ema_fast = close[asset].ewm(span=fast, adjust=False).mean()
    ema_slow = close[asset].ewm(span=slow, adjust=False).mean()
    
    position = (ema_fast > ema_slow).astype(int)

    ax.step(position.index, position.values, where="post", linewidth=2)
    
    ax.fill_between(
        position.index,
        0,
        position.values,
        alpha=0.2
        )
    
    ax.set_title(f"{asset} — EMA Position (0=Flat, 1=Long)")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Flat", "Long"])
    ax.grid(True)

plt.tight_layout()
plt.show()



