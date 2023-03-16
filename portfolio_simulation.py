import numpy as np
import pandas as pd
from scipy.stats import norm


class Asset:
    def __init__(self, name, a_prices):
        self.name = name
        self.a_prices = a_prices

    def calculate_returns(self):
        self.a_returns = np.diff(self.a_prices) / self.a_prices[:-1]
        return self.a_returns

    def get_prices(self):
        return self.a_prices


class Portfolio:
    def __init__(self, assets, a_initial_weights, a_initial_amounts):
        self.assets = assets
        self.a_initial_weights = a_initial_weights
        self.a_initial_amounts = a_initial_amounts
        self.df_weights = pd.DataFrame(columns=[asset.name for asset in assets])
        self.df_amounts = pd.DataFrame(columns=[asset.name for asset in assets])
        self.df_rebalancing = pd.DataFrame(columns=["Date", "New Weights"])
        self.df_trade_history = pd.DataFrame(columns=["Asset", "Purchase Price", "Quantity", "Time Index"])
        self.total_pnl = {"Trading Profit": 0, "Tax Payable": 0, "Trading Cost": 0}

    def calculate_portfolio_value(self, a_prices):
        return np.sum(a_prices * self.a_initial_amounts)

    def calculate_portfolio_returns(self, a_prices):
        a_returns = np.diff(a_prices, axis=0) / a_prices[:-1]
        a_weighted_returns = self.a_initial_weights * a_returns
        a_portfolio_returns = np.sum(a_weighted_returns, axis=1)
        return a_portfolio_returns

    def rebalance(self, date, a_new_weights):
        self.df_rebalancing = self.df_rebalancing.append({"Date": date, "New Weights": a_new_weights}, ignore_index=True)
        self.df_weights = self.df_weights.append(pd.Series(a_new_weights, index=self.df_weights.columns), ignore_index=True)

    def update_weights_and_amounts(self, a_prices):
        a_current_value = self.calculate_portfolio_value(a_prices)
        self.a_initial_amounts = a_current_value * self.a_initial_weights
        self.df_amounts = self.df_amounts.append(pd.Series(self.a_initial_amounts, index=self.df_amounts.columns), ignore_index=True)

    def evaluate_portfolio_performance(self):
        # Implement portfolio performance evaluation
        pass

    def sell_assets(self, strategy):
        # Implement the sell_assets method
        pass
    def calculate_trading_cost(self, trade):
        # Implement trading cost calculation
        pass
    def calculate_tax_payable(self, trade):
        # Implement tax payable calculation
        pass


class TaxMinimizationStrategy:
    def __init__(self, portfolio):
        self.portfolio = portfolio

    def identify_assets_to_sell(self):
        a_losses = []
        for asset in self.portfolio.assets:
            a_asset_returns = asset.calculate_returns()
            a_losses.append(np.sum(a_asset_returns < 0))

        a_losses = np.array(a_losses)
        assets_to_sell = np.argsort(a_losses)[::-1]

        return assets_to_sell


class Simulation:
    def __init__(self, days, num_simulations, a_mu, a_sigma, a_correlation_matrix, a_initial_prices, timestep=1 / 252):
        self.days = days
        self.num_simulations = num_simulations
        self.a_mu = a_mu
        self.a_sigma = a_sigma
        self.a_correlation_matrix = a_correlation_matrix
        self.a_initial_prices = a_initial_prices
        self.num_assets = len(a_initial_prices)
        self.timestep = timestep

    def generate_asset_prices(self, random_seed=None):
        if random_seed is not None:
            np.random.seed(random_seed)

        a_Z = np.random.multivariate_normal(np.zeros(self.num_assets), self.a_correlation_matrix,
                                            (self.num_simulations, self.days))
        a_Z = a_Z.swapaxes(0, 2)  # Swap dimensions to have the shape (num_assets, num_simulations, days)
        a_Z = a_Z.swapaxes(1, 2)
        a_drift = (self.a_mu * self.timestep - 0.5 * self.a_sigma ** 2 * self.timestep)[:, np.newaxis, np.newaxis]

        a_diffusion = np.empty((self.num_assets, self.num_simulations, self.days))
        for i in range(self.num_assets):
            a_diffusion[i] = self.a_sigma[i] * np.sqrt(self.timestep) * a_Z[i]

        a_asset_returns = a_drift + a_diffusion
        a_asset_prices = np.exp(np.cumsum(a_asset_returns, axis=2))
        a_asset_prices = a_asset_prices * np.array(self.a_initial_prices)[:, np.newaxis, np.newaxis]

        return a_asset_prices