import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, initial_investment, weights, delta_weight, trading_cost):
        self.initial_investment = initial_investment
        self.weights = weights
        self.current_values = None
        self.delta_weight = delta_weight
        self.trading_cost = trading_cost
        self.history = []

    def calculate_current_values(self, a_prices):
        self.current_values = np.array(a_prices) * self.weights * self.initial_investment
        self.history.append(np.sum(self.current_values))

    def rebalance(self, a_returns):
        epsilon = 1e-10
        current_weights = self.current_values / (np.sum(self.current_values) + epsilon)
        weight_difference = np.abs(current_weights - self.weights)

        if np.any(weight_difference >= self.delta_weight):
            self.weights = current_weights
            self.current_values *= (1 - self.trading_cost)

        self.history.append(np.sum(self.current_values))

    def get_value(self):
        return np.sum(self.current_values)

    def get_history(self):
        return np.array(self.history)


class InputData:
    def __init__(self, etf_name: str):
        self.etf_name = etf_name
        self.df_prices = None
        self.df_returns = None

    def read_csv_and_calculate_returns(self):
        self.df_prices = pd.read_csv(f"input/{self.etf_name}.csv", index_col="Date", parse_dates=True)["Adj Close"]
        self.df_returns = self.calculate_returns(self.df_prices)

    @staticmethod
    def calculate_returns(df_prices: pd.Series) -> pd.Series:
        return df_prices.pct_change().dropna()


class PortfolioReturn:
    @staticmethod
    def calculate(a_portfolio_values: np.ndarray) -> np.ndarray:
        return pd.Series(a_portfolio_values).pct_change().dropna().values


def simulate_portfolio(initial_investment, weights, a_returns, delta_weight, trading_cost):
    portfolio = Portfolio(initial_investment, weights, delta_weight, trading_cost)
    portfolio.calculate_current_values([1, 1])
    for i in range(1, len(a_returns)):
        portfolio.rebalance(a_returns[:i])
        portfolio.calculate_current_values(a_returns[i])
    return portfolio.get_history()

# Test case
VOO_data = InputData("VOO")
VOO_data.read_csv_and_calculate_returns()
IEI_data = InputData("IEI")
IEI_data.read_csv_and_calculate_returns()

df_returns = pd.concat([VOO_data.df_returns, IEI_data.df_returns], axis=1).dropna()
df_returns.columns = ["VOO", "IEI"]

initial_investment = 1000000  # 1 million dollars
weights = np.array([0.6, 0.4])
delta_weight = 0.02
trading_cost = 0.002  # 20 bps

a_portfolio_values = simulate_portfolio(initial_investment, weights, df_returns.values, delta_weight, trading_cost)

total_return = (a_portfolio_values[-1] / a_portfolio_values[0]) - 1
a_portfolio_returns = PortfolioReturn.calculate(a_portfolio_values)
annualized_return = np.power(np.prod(1 + a_portfolio_returns), 252 / len(a_portfolio_returns)) - 1
annualized_volatility = np.std(a_portfolio_returns) * np.sqrt(252)

print("Total return:", total_return)
print("Annualized return:", annualized_return)
print("Annualized volatility:", annualized_volatility)
``


