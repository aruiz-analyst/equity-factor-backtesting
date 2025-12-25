"""
Data loader for historical price data.

Initial target: pull daily adjusted close prices for a small universe of tickers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
import pandas as pd

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "Missing dependency: yfinance. Install with: pip install yfinance"
    ) from e


@dataclass(frozen=True)
class PriceRequest:
    tickers: List[str]
    start: str  # "YYYY-MM-DD"
    end: str    # "YYYY-MM-DD"


def load_adj_close(req: PriceRequest) -> pd.DataFrame:
    """
    Returns a DataFrame of adjusted close prices indexed by date with tickers as columns.
    """
    df = yf.download(
        tickers=req.tickers,
        start=req.start,
        end=req.end,
        auto_adjust=False,
        progress=False,
        group_by="column",
    )

    if df.empty:
        raise ValueError("No data returned. Check tickers and date range.")

    # yfinance returns a multi-index column set when multiple tickers are used.
    if isinstance(df.columns, pd.MultiIndex):
        adj = df["Adj Close"].copy()
    else:
        # single ticker case
        adj = df[["Adj Close"]].rename(columns={"Adj Close": req.tickers[0]})

    adj = adj.dropna(how="all")
    return adj
