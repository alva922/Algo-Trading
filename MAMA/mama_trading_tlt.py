# https://medium.com/@alexzap922/from-spy-underperformance-to-macro-alpha-across-bonds-commodities-and-crypto-a-a735cbba2465?sk=941b9fd73ffe341654455c4e0d925e6e
# get 10% discount EODHD APIs access link https://eodhd.com/pricing-special-10?via=aiva

import requests
import pandas as pd

API_KEY = "YOUR API KEY"


START_DATE = "2015-01-02"
END_DATE = "2026-05-27"


# -----------------------------------
# EODHD LOADER
# -----------------------------------
def get_eodhd(symbol):

    url = f"https://eodhistoricaldata.com/api/eod/{symbol}"

    params = {
        "api_token": API_KEY,
        "period": "d",
        "from": START_DATE,
        "to": END_DATE,
        "fmt": "json"
    }

    r = requests.get(url, params=params)

    # important for debugging API issues
    r.raise_for_status()

    data = r.json()

    df = pd.DataFrame(data)

    if df.empty:
        raise ValueError(f"No data returned for {symbol}.{exchange}")

    # convert date
    df["date"] = pd.to_datetime(df["date"])

    # use date as index
    df = df.set_index("date").sort_index()

    return df


# -----------------------------------
# LOAD ASSETS
# -----------------------------------

# TLT
tlt = get_eodhd("TLT")
import mplfinance as mpf

# Rename columns to mplfinance format (lowercase required)
df = tlt[['open', 'high', 'low', 'close', 'volume']]

# Plot
mpf.plot(
    df,
    type='candle',
    volume=True,
    style='charles',
    title='TLT Candlestick Chart & Volume',
    ylabel='Price (USD)',
    xlabel='Date',
    ylabel_lower='Volume',figsize=(14, 6)
)

df_base = pd.DataFrame(index=tlt.index)
df_base["price"] = tlt["close"]

import numpy as np
import pandas as pd
import yfinance as yf
import talib
import optuna
import matplotlib.pyplot as plt


# =========================================================
# MAMA FUNCTION
# =========================================================
def mama(series, fast=0.5, slow=0.05):

    mama, fama = talib.MAMA(
        series.values,
        fastlimit=fast,
        slowlimit=slow
    )

    return pd.DataFrame({
        "mama": mama,
        "fama": fama
    }, index=series.index)


# =========================================================
# STRATEGY
# =========================================================
def build_strategy(
    df,
    fast,
    slow,
    slope_window,
    vol_filter
):

    df = df.copy()

    # -----------------------------------------------------
    # MAMA / FAMA
    # -----------------------------------------------------
    ma = mama(df["price"], fast=fast, slow=slow)

    df["mama"] = ma["mama"]
    df["fama"] = ma["fama"]

    # -----------------------------------------------------
    # SIGNALS
    # -----------------------------------------------------
    df["spread"] = (
        df["mama"] - df["fama"]
    ) / df["price"]

    df["slope"] = (
        df["mama"].diff(slope_window)
    ) / df["price"]

    # volatility filter
    df["vol"] = (
        df["price"]
        .pct_change()
        .rolling(20)
        .std()
    )

    # -----------------------------------------------------
    # TREND STRENGTH
    # -----------------------------------------------------
    strength = (
        5 * df["spread"] +
        20 * df["slope"]
    )

    # -----------------------------------------------------
    # EXPOSURE ENGINE
    # -----------------------------------------------------
    position = np.where(
        (strength > 0) &
        (df["vol"] < vol_filter),
        1.0,
        0.25
    )

    # reduce exposure in strong downtrends
    position = np.where(
        strength < 0,
        0.0,
        position
    )

    df["position"] = position

    return df


# =========================================================
# BACKTEST
# =========================================================
def backtest(df, initial=10000):

    df = df.copy()

    df["ret"] = (
        df["price"]
        .pct_change()
        .fillna(0)
    )

    df["strategy_ret"] = (
        df["position"]
        .shift(1)
        .fillna(0)
        * df["ret"]
    )

    # transaction costs
    turnover = (
        df["position"]
        .diff()
        .abs()
        .fillna(0)
    )

    cost = turnover * 0.0005

    df["strategy_ret"] -= cost

    # equity
    df["equity"] = (
        (1 + df["strategy_ret"])
        .cumprod()
        * initial
    )

    df["bh_equity"] = (
        (1 + df["ret"])
        .cumprod()
        * initial
    )

    # -----------------------------------------------------
    # METRICS
    # -----------------------------------------------------
    sharpe = 0

    if df["strategy_ret"].std() != 0:
        sharpe = (
            df["strategy_ret"].mean()
            / df["strategy_ret"].std()
        ) * np.sqrt(252)

    strategy_return = (
        df["equity"].iloc[-1] / initial - 1
    )

    bh_return = (
        df["bh_equity"].iloc[-1] / initial - 1
    )

    alpha = strategy_return - bh_return

    # drawdown
    peak = df["equity"].cummax()

    dd = (
        df["equity"] - peak
    ) / peak

    max_dd = dd.min()

    # CAGR
    years = len(df) / 252

    cagr = (
        (df["equity"].iloc[-1] / initial)
        ** (1 / years)
    ) - 1

    return {
        "sharpe": sharpe,
        "return": strategy_return,
        "bh_return": bh_return,
        "alpha": alpha,
        "drawdown": max_dd,
        "cagr": cagr
    }, df


# =========================================================
# BAYESIAN OPTIMIZATION
# =========================================================
def objective(trial):

    fast = trial.suggest_float(
        "fast",
        0.05,
        0.95
    )

    slow = trial.suggest_float(
        "slow",
        0.01,
        0.5
    )

    slope_window = trial.suggest_int(
        "slope_window",
        2,
        10
    )

    vol_filter = trial.suggest_float(
        "vol_filter",
        0.005,
        0.05
    )

    # constraint
    if slow >= fast:
        return -999

    df = build_strategy(
        df_base,
        fast,
        slow,
        slope_window,
        vol_filter
    )

    metrics, _ = backtest(df)

    sharpe = metrics["sharpe"]
    alpha = metrics["alpha"]
    dd = abs(metrics["drawdown"])

    # -----------------------------------------------------
    # OBJECTIVE
    # -----------------------------------------------------
    score = (
        sharpe * 4 +
        alpha * 2 -
        dd
    )

    # hard quality floor
    if sharpe < 1:
        score -= 100

    return score


# =========================================================
# RUN OPTIMIZATION
# =========================================================
study = optuna.create_study(
    direction="maximize"
)

study.optimize(
    objective,
    n_trials=300
)


# =========================================================
# BEST PARAMETERS
# =========================================================
print("\n=== BEST PARAMETERS ===")
print(study.best_params)


# =========================================================
# FINAL BACKTEST
# =========================================================
best = study.best_params

final_df = build_strategy(
    df_base,
    best["fast"],
    best["slow"],
    best["slope_window"],
    best["vol_filter"]
)

metrics, bt = backtest(final_df)

print("\n=== FINAL RESULTS ===")

for k, v in metrics.items():
    print(f"{k}: {v}")


# =========================================================
# EQUITY CURVES
# =========================================================
plt.figure(figsize=(14,6))

plt.plot(
    bt.index,
    bt["equity"],
    label="MAMA Strategy"
)

plt.plot(
    bt.index,
    bt["bh_equity"],
    linestyle="--",
    label="Buy & Hold"
)

plt.title(
    "TLT Bayesian Optimized MAMA Strategy vs Buy & Hold"
)

plt.legend()
plt.ylabel("Return USD")
plt.xlabel("Date")
#plt.grid()

plt.show()

import seaborn as sns

df = bt.copy()
df["cummax"] = df["equity"].cummax()
df["drawdown"] = df["equity"] / df["cummax"] - 1
df["turnover"] = df["position"].diff().abs().fillna(0)

# =========================================================
# 1. EQUITY + BUY HOLD + DRAWDOWN
# =========================================================
fig, ax = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

ax[0].plot(df.index, df["equity"], label="Strategy")
ax[0].plot(df.index, df["bh_equity"], linestyle="--", label="Buy & Hold")
ax[0].set_title("Equity Curve Comparison")
ax[0].legend()

ax[1].fill_between(df.index, df["drawdown"], 0, color="red")
ax[1].set_title("Drawdown")
ax[1].set_ylabel("Drawdown")

plt.tight_layout()
plt.show()


# =========================================================
# 2. POSITION EXPOSURE
# =========================================================
plt.figure(figsize=(14, 4))
plt.plot(df.index, df["position"])
plt.title("Position Exposure Over Time")
plt.ylim(-0.1, 1.1)
plt.show()


# =========================================================
# 3. RETURN DISTRIBUTION
# =========================================================
plt.figure(figsize=(10, 4))
plt.hist(df["strategy_ret"], bins=80)
plt.title("Strategy Return Distribution")
plt.show()


# =========================================================
# 4. ROLLING SHARPE (1 YEAR)
# =========================================================
roll_sharpe = (
    df["strategy_ret"]
    .rolling(252)
    .mean()
    / df["strategy_ret"].rolling(252).std()
) * np.sqrt(252)

plt.figure(figsize=(14, 4))
plt.plot(df.index, roll_sharpe)
plt.title("Rolling Sharpe (252D)")
plt.axhline(0, linestyle="--")
plt.show()


# =========================================================
# 5. TURNOVER (COST PRESSURE CHECK)
# =========================================================
plt.figure(figsize=(14, 4))
plt.plot(df.index, df["turnover"].rolling(20).mean())
plt.title("Rolling Turnover (20D)")
plt.show()


# =========================================================
# 6. MONTHLY RETURNS HEATMAP
# =========================================================
monthly = (
    df["strategy_ret"]
    .resample("M")
    .apply(lambda x: (1 + x).prod() - 1)
)

heatmap = monthly.to_frame()
heatmap["year"] = heatmap.index.year
heatmap["month"] = heatmap.index.month

pivot = heatmap.pivot(index="year", columns="month", values="strategy_ret")

plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="RdYlGn", center=0)
plt.title("Monthly Returns Heatmap")
plt.show()


# =========================================================
# 7. SUMMARY STATS TABLE
# =========================================================
summary = pd.DataFrame({
    "Metric": [
        "Sharpe",
        "CAGR",
        "Max Drawdown",
        "Total Return",
        "Buy & Hold Return",
        "Alpha"
    ],
    "Value": [
        metrics["sharpe"],
        metrics["cagr"],
        metrics["drawdown"],
        metrics["return"],
        metrics["bh_return"],
        metrics["alpha"]
    ]
})

print("\n=== SUMMARY ===")
print(summary)

import numpy as np

def compute_extra_metrics(df, initial=10000):

    rets = df["strategy_ret"].dropna()

    equity = df["equity"]
    bh = df["bh_equity"]

    # ================================
    # 1. Volatility (annualized)
    # ================================
    vol = rets.std() * np.sqrt(252)

    # ================================
    # 2. Sortino Ratio (downside risk)
    # ================================
    downside = rets[rets < 0]
    downside_std = downside.std() * np.sqrt(252)

    sortino = (rets.mean() * 252) / downside_std if downside_std != 0 else np.nan

    # ================================
    # 3. Calmar Ratio (CAGR / DD)
    # ================================
    peak = equity.cummax()
    dd = (equity / peak) - 1
    max_dd = dd.min()

    cagr = (equity.iloc[-1] / initial) ** (1 / (len(df) / 252)) - 1
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan

    # ================================
    # 4. Win rate
    # ================================
    win_rate = (rets > 0).mean()

    # ================================
    # 5. Profit factor
    # ================================
    gross_profit = rets[rets > 0].sum()
    gross_loss = abs(rets[rets < 0].sum())
    profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.nan

    # ================================
    # 6. Tail risk (VaR / CVaR)
    # ================================
    var_95 = np.percentile(rets, 5)
    cvar_95 = rets[rets <= var_95].mean()

    # ================================
    # 7. Skew & Kurtosis
    # ================================
    skew = rets.skew()
    kurt = rets.kurtosis()

    # ================================
    # 8. Exposure ratio
    # ================================
    exposure = df["position"].mean()

    # ================================
    # 9. CAGR / Vol (Sharpe alternative)
    # ================================
    cagr_vol = cagr / vol if vol != 0 else np.nan

    return {
        "volatility": vol,
        "sortino": sortino,
        "calmar": calmar,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "var_95": var_95,
        "cvar_95": cvar_95,
        "skew": skew,
        "kurtosis": kurt,
        "exposure": exposure,
        "cagr_vol": cagr_vol
    }

extra_metrics = compute_extra_metrics(df)

import pandas as pd
import numpy as np

def make_comparison_table(strategy_metrics, bh_metrics):
    """
    Builds a professional risk/return comparison table.
    
    strategy_metrics: dict from your model
    bh_metrics: dict for buy & hold benchmark
    """

    rows = []

    for key in strategy_metrics.keys():

        strat_val = strategy_metrics.get(key, np.nan)
        bh_val = bh_metrics.get(key, np.nan)

        rows.append({
            "Metric": key,
            "Strategy": float(strat_val),
            "Buy & Hold": float(bh_val),
            "Delta": float(strat_val - bh_val) if pd.notna(strat_val) and pd.notna(bh_val) else np.nan,
            "Relative (%)": (
                (strat_val / bh_val - 1) * 100
                if bh_val not in [0, np.nan] and pd.notna(bh_val)
                else np.nan
            )
        })

    df = pd.DataFrame(rows)

    # nicer formatting
    df = df.sort_values(by="Metric").reset_index(drop=True)

    return df

strategy_metrics, strategy_df = backtest(final_df)

def buy_hold_metrics(df_base, initial=10000):

    df = df_base.copy()

    df["ret"] = df["price"].pct_change().fillna(0)

    df["equity"] = (1 + df["ret"]).cumprod() * initial

    # metrics
    sharpe = (df["ret"].mean() / df["ret"].std()) * np.sqrt(252)

    max_dd = (df["equity"] / df["equity"].cummax() - 1).min()

    cagr = (df["equity"].iloc[-1] / initial) ** (1 / (len(df)/252)) - 1

    return {
        "sharpe": sharpe,
        "return": df["equity"].iloc[-1] / initial - 1,
        "drawdown": max_dd,
        "cagr": cagr,
        "volatility": df["ret"].std() * np.sqrt(252)
    }, df

# strategy
strategy_metrics, strategy_df = backtest(final_df)

# buy & hold (FIXED)
bh_metrics, bh_df = buy_hold_metrics(df_base)

# comparison table
table = make_comparison_table(strategy_metrics, bh_metrics)
print(table)

import pandas as pd
import numpy as np

def compare_strategy_vs_bh(df, extra_metrics, initial=10000):

    # =========================
    # Strategy metrics
    # =========================
    strategy_rets = df["strategy_ret"].dropna()
    strategy_equity = df["equity"]

    strategy_return = strategy_equity.iloc[-1] / initial

    strategy_cagr = (
        strategy_return ** (252 / len(df))
    ) - 1

    strategy_sharpe = (
        strategy_rets.mean() * 252
        / (strategy_rets.std() * np.sqrt(252))
    )

    strategy_dd = (
        strategy_equity / strategy_equity.cummax() - 1
    ).min()

    # =========================
    # Buy & Hold metrics
    # =========================
    bh_rets = df["price"].pct_change().dropna()
    bh_equity = df["bh_equity"]

    bh_return = bh_equity.iloc[-1] / initial

    bh_cagr = (
        bh_return ** (252 / len(df))
    ) - 1

    bh_vol = bh_rets.std() * np.sqrt(252)

    bh_sharpe = (
        bh_rets.mean() * 252
        / bh_vol
    )

    bh_dd = (
        bh_equity / bh_equity.cummax() - 1
    ).min()

    bh_downside = bh_rets[bh_rets < 0]
    bh_downside_std = bh_downside.std() * np.sqrt(252)

    bh_sortino = (
        (bh_rets.mean() * 252) / bh_downside_std
        if bh_downside_std != 0 else np.nan
    )

    bh_calmar = (
        bh_cagr / abs(bh_dd)
        if bh_dd != 0 else np.nan
    )

    bh_win_rate = (bh_rets > 0).mean()

    bh_gross_profit = bh_rets[bh_rets > 0].sum()
    bh_gross_loss = abs(bh_rets[bh_rets < 0].sum())

    bh_profit_factor = (
        bh_gross_profit / bh_gross_loss
        if bh_gross_loss != 0 else np.nan
    )

    bh_var95 = np.percentile(bh_rets, 5)
    bh_cvar95 = bh_rets[bh_rets <= bh_var95].mean()

    bh_skew = bh_rets.skew()
    bh_kurtosis = bh_rets.kurtosis()

    bh_cagr_vol = (
        bh_cagr / bh_vol
        if bh_vol != 0 else np.nan
    )

    # =========================
    # Summary table
    # =========================
    summary = pd.DataFrame({
        "Strategy": [
            strategy_return,
            strategy_cagr,
            strategy_sharpe,
            strategy_dd,
            extra_metrics["volatility"],
            extra_metrics["sortino"],
            extra_metrics["calmar"],
            extra_metrics["win_rate"],
            extra_metrics["profit_factor"],
            extra_metrics["var_95"],
            extra_metrics["cvar_95"],
            extra_metrics["skew"],
            extra_metrics["kurtosis"],
            extra_metrics["exposure"],
            extra_metrics["cagr_vol"]
        ],
        "Buy & Hold": [
            bh_return,
            bh_cagr,
            bh_sharpe,
            bh_dd,
            bh_vol,
            bh_sortino,
            bh_calmar,
            bh_win_rate,
            bh_profit_factor,
            bh_var95,
            bh_cvar95,
            bh_skew,
            bh_kurtosis,
            1.0,  # always invested
            bh_cagr_vol
        ]
    }, index=[
        "Total Return",
        "CAGR",
        "Sharpe Ratio",
        "Max Drawdown",
        "Volatility",
        "Sortino Ratio",
        "Calmar Ratio",
        "Win Rate",
        "Profit Factor",
        "VaR (95%)",
        "CVaR (95%)",
        "Skewness",
        "Kurtosis",
        "Exposure",
        "CAGR/Vol"
    ])

    return summary.round(4)


# Run
summary = compare_strategy_vs_bh(df, extra_metrics)

print("\n=== STRATEGY VS BUY & HOLD ===")
print(summary)

