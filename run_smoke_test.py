from src.data_loader import PriceRequest, load_adj_close
from src.factors import compute_returns, momentum_12m_1m
from src.metrics import annualized_return, annualized_vol, sharpe_ratio, max_drawdown

req = PriceRequest(["SPY", "QQQ", "IWM"], "2018-01-01", "2024-01-01")
prices = load_adj_close(req)

rets = compute_returns(prices)
mom = momentum_12m_1m(prices)

# Simple equal-weight benchmark sanity check
ew = rets.mean(axis=1)
equity = (1.0 + ew).cumprod()

print("Rows/Cols prices:", prices.shape)
print("Rows/Cols returns:", rets.shape)
print("Rows/Cols momentum:", mom.shape)
print("AnnReturn:", round(annualized_return(ew), 4))
print("AnnVol:", round(annualized_vol(ew), 4))
print("Sharpe:", round(sharpe_ratio(ew), 4))
print("MaxDD:", round(max_drawdown(equity), 4))
