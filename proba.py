import numpy as np
import pandas as pd
import data_handler as dh
from portfolio_simulation import *

# Set parameters
tickers = ["AAPL", "MSFT"]
start_date = "2020-01-01"
end_date = "2020-12-31"
a_initial_weights = np.array([0.6, 0.4])
initial_investment = 10000
rebalancing_frequency = 5

# Create portfolio
portfolio = create_portfolio_from_historical_prices(tickers, start_date, end_date, a_initial_weights, initial_investment)

# Choose a rebalancing strategy
strategy = FixedQuantitiesStrategy(frequency=rebalancing_frequency)

# Simulate the rebalance
simulate_rebalance(portfolio, strategy)

# Compute portfolio value and benchmark value
portfolio_value = np.sum(portfolio.a_investments, axis=0)
benchmark_value = np.sum(portfolio_value[0, :]) * np.array([0.5, 0.5])

# Plot results
df_portfolio_value = pd.DataFrame(portfolio_value, columns=["Portfolio"])
df_benchmark_value = pd.DataFrame(benchmark_value, columns=["Benchmark"])
df_value = pd.concat([df_portfolio_value, df_benchmark_value], axis=1)
df_value.plot()