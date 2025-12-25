import matplotlib.pyplot as plt

from src.data_loader import PriceRequest, load_adj_close
from src.factors import compute_returns, momentum_12m_1m
from src.backtest import backtest_monthly_long_short
from src.metrics import annualized_return, annualized_vol, sharpe_ratio, max_drawdown

req = PriceRequest(["SPY", "QQQ", "IWM"], "2018-01-01", "2024-01-01")
prices = load_adj_close(req)
rets = compute_returns(prices)
signal = momentum_12m_1m(prices)

strategy = backtest_monthly_long_short(rets, signal, cost_bps=5.0, q=1/3)
equity = (1.0 + strategy).cumprod()

print("AnnReturn:", round(annualized_return(strategy), 4))
print("AnnVol:", round(annualized_vol(strategy), 4))
print("Sharpe:", round(sharpe_ratio(strategy), 4))
print("MaxDD:", round(max_drawdown(equity), 4))

# Save plot for GitHub proof
plt.figure()
equity.plot()
plt.title("Momentum Long/Short (Monthly Rebalance) Equity Curve")
plt.ylabel("Cumulative Wealth")
plt.tight_layout()
plt.savefig("results/momentum_equity_curve.png")
print("Saved plot to results/momentum_equity_curve.png")
