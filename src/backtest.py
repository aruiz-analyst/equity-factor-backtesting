from __future__ import annotations

import numpy as np
import pandas as pd


def month_end_dates(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Return the last available trading day of each month."""
    s = pd.Series(index=index, data=1)
    return s.resample("M").last().index


def long_short_weights_from_signal(signal_row: pd.Series, q: float = 1/3) -> pd.Series:
    """
    Build dollar-neutral long/short weights from a cross-sectional signal at one date.
    Long top quantile, short bottom quantile. Weights sum to 0 and gross to 1.
    """
    s = signal_row.dropna()
    if s.empty:
        return pd.Series(index=signal_row.index, data=0.0)

    n = len(s)
    k = max(1, int(np.floor(n * q)))

    ranked = s.sort_values()
    short_names = ranked.index[:k]
    long_names = ranked.index[-k:]

    w = pd.Series(index=signal_row.index, data=0.0)
    w.loc[long_names] = 0.5 / k
    w.loc[short_names] = -0.5 / k
    return w


def backtest_monthly_long_short(
    returns: pd.DataFrame,
    signal: pd.DataFrame,
    cost_bps: float = 5.0,
    q: float = 1/3,
) -> pd.Series:
    """
    Monthly rebalance long/short strategy. Applies linear transaction costs on turnover.

    returns: daily returns (date index, tickers columns)
    signal:  daily signal values aligned on the same tickers (can have NaNs)
    cost_bps: cost per $ traded, in basis points
    """
    returns = returns.copy()
    signal = signal.reindex_like(returns)

    # Rebalance on month-end dates available in the returns index
    rebal_dates = month_end_dates(returns.index)
    rebal_dates = rebal_dates.intersection(returns.index)

    w_prev = pd.Series(index=returns.columns, data=0.0)
    daily_strat = []

    cost_rate = cost_bps / 10000.0

    for dt in returns.index:
        if dt in rebal_dates:
            w_new = long_short_weights_from_signal(signal.loc[dt], q=q)

            turnover = (w_new - w_prev).abs().sum()
            cost = cost_rate * turnover
            w_prev = w_new

        # portfolio return for the day using current weights
        r = (w_prev * returns.loc[dt]).sum()
        # apply cost on rebalance day (cost already computed) — cost treated as return drag
        if dt in rebal_dates:
            r -= cost

        daily_strat.append(r)

    strat = pd.Series(daily_strat, index=returns.index, name="strategy_return")
    return strat
