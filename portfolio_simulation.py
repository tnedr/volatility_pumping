import numpy as np
import pandas as pd
from scipy.stats import norm
import data_handler as dh
from enum import Enum

class RebalancingStrategy(Enum):
    FIXED_QUANTITIES = 0
    FIXED_WEIGHTS = 1


class FixedQuantitiesStrategy:
    def __init__(self, frequency):
        self.frequency = frequency
        self.steps_since_rebalance = 0

    def should_rebalance(self):
        return False

    def rebalance(self, portfolio, step):
        portfolio.a_quantities[:, step, :] = portfolio.a_quantities[:, step-1, :]
        portfolio.a_investments[:, step, :] = portfolio.a_quantities[:, step, :] * portfolio.a_prices[:, step, :]
        portfolio.a_weights[:, step, :] = portfolio.a_investments[:, step, :] / np.sum(portfolio.a_investments[:, step, :], axis=0)

        # portfolio.a_quantities[:, step, :] = portfolio.a_quantities[:, step - 1, :]
        # portfolio.a_investments[:, step, :] = portfolio.a_quantities[:, step, :] * portfolio.a_prices[:, step, :]
        # portfolio.a_weights[:, step, :] = portfolio.a_investments[:, step, :] / np.sum(portfolio.a_investments[:, step, :], axis=0)

class FixedWeightsStrategy:
    def __init__(self, frequency):
        self.frequency = frequency
        self.steps_since_rebalance = 0

    def should_rebalance(self):
        return self.steps_since_rebalance == self.frequency


    def rebalance(self, portfolio, step):
        self.steps_since_rebalance += 1
        if self.should_rebalance():
            # print('y', step, self.steps_since_rebalance)
            # investment pre rebalance
            portfolio.a_investments[:, step, :] = portfolio.a_quantities[:, step-1, :] * portfolio.a_prices[:, step, :]
            portfolio.a_weights[:, step, :] = portfolio.a_initial_weights[:, np.newaxis]
            # post rebalance
            portfolio.a_investments[:, step, :] = portfolio.a_weights[:, step, :] * np.sum(portfolio.a_investments[:, step, :], axis=0)
            portfolio.a_quantities[:, step, :] = portfolio.a_investments[:, step, :] / portfolio.a_prices[:, step, :]
            self.steps_since_rebalance = 0
        else:
            # print('n', step, self.steps_since_rebalance)
            portfolio.a_quantities[:, step, :] = portfolio.a_quantities[:, step - 1, :]
            portfolio.a_investments[:, step, :] = portfolio.a_quantities[:, step, :] * portfolio.a_prices[:, step, :]
            portfolio.a_weights[:, step, :] = portfolio.a_investments[:, step, :] / np.sum(portfolio.a_investments[:, step, :], axis=0)


class Asset:

    def __init__(self, name, a_prices, time_dim=None):
        self.name = name
        self.a_prices = a_prices  # a_prices: (num_steps, num_paths)
        if time_dim is None:
            self.time_dim = range(self.a_prices.shape[0])
        else:
            self.time_dim = time_dim

    def calculate_returns(self):
        self.a_returns = np.diff(self.a_prices, axis=0) / self.a_prices[:-1, :]
        return self.a_returns

    def get_prices(self):
        return self.a_prices

    def get_time_dim(self):
        return self.time_dim


class Portfolio:

    def __init__(self, assets, a_initial_weights, initial_investment):
        self.assets = assets
        self.a_initial_weights = a_initial_weights  # a_initial_weights: (num_assets,)
        self.initial_investment = initial_investment

        self.num_assets = len(self.assets)
        self.step_dim = self.get_step_dim()
        self.num_steps = len(self.step_dim)
        self.num_paths = self.assets[0].get_prices().shape[1]

        self.step_from = self.step_dim[0]
        self.step_to = self.step_dim[-1]

        self.a_weights = np.zeros(
            (self.num_assets, self.num_steps, self.num_paths))  # a_weights: (num_assets, num_steps, num_paths)
        self.a_investments = np.zeros(
            (self.num_assets, self.num_steps, self.num_paths))  # a_investments: (num_assets, num_steps, num_paths)
        self.a_quantities = np.zeros(
            (self.num_assets, self.num_steps, self.num_paths))  # a_quantities: (num_assets, num_steps, num_paths)
        self.a_prices = self.get_all_asset_prices()  # a_prices: (num_assets, num_steps, num_paths)

        self.initialization()

        print('ok')
        # self.df_weights = pd.DataFrame(columns=[asset.name for asset in assets])
        #
        # a_initial_quantities = self.calculate_initial_quantities(initial_investment)  # a_initial_quantities: (num_assets, num_paths)
        # self.a_quantities = self.initialize_quantities(a_initial_quantities)  # a_quantities: (num_assets, num_steps, num_paths)
        #
        # self.compute_and_store_weights()

    def get_step_dim(self):
        return self.assets[0].get_time_dim()

    def get_all_asset_prices(self):
        a_prices = np.array(
            [asset.get_prices() for asset in self.assets])  # a_prices: (num_assets, num_steps, num_paths)
        return a_prices

    def initialization(self):
        self.a_weights[:, 0, :] = self.a_initial_weights[:, np.newaxis]
        self.a_investments[:, 0, :] = (self.initial_investment * self.a_initial_weights[:, np.newaxis])
        self.a_quantities[:, 0, :] = self.calculate_initial_quantities(self.initial_investment)


    def calculate_initial_quantities(self, initial_investment):
        a_initial_prices = np.array([asset.get_prices()[0] for asset in self.assets])
        # a_initial_quantities = (initial_investment * self.a_initial_weights) / a_initial_prices
        a_initial_quantities = (initial_investment * self.a_initial_weights[:, np.newaxis]) / a_initial_prices
        return a_initial_quantities

    def rebalance(self, strategy, step):
        strategy.rebalance(self, step)

class Simulation:
    def __init__(self, days, num_simulations, a_mu, a_sigma, a_correlation_matrix, a_initial_prices):
        self.days = days
        self.num_simulations = num_simulations
        self.a_mu = a_mu  # a_mu: (num_assets,)
        self.a_sigma = a_sigma  # a_sigma: (num_assets,)
        self.a_correlation_matrix = a_correlation_matrix  # a_correlation_matrix: (num_assets, num_assets)
        self.a_initial_prices = a_initial_prices  # a_initial_prices: (num_assets,)

    def generate_asset_prices(self):
        a_asset_prices = []
        for i in range(len(self.a_mu)):
            asset_prices = self.simulate_asset_prices(self.a_mu[i], self.a_sigma[i], self.a_correlation_matrix[i], self.a_initial_prices[i])
            a_asset_prices.append(asset_prices)

        a_asset_prices = np.array(a_asset_prices)
        return a_asset_prices

    def simulate_asset_prices(self, mu, sigma, a_correlation, initial_price):
        dt = 1/252
        dW = norm.rvs(size=(self.days, self.num_simulations)) * np.sqrt(dt)
        dW = np.vstack((np.zeros(self.num_simulations), dW))
        a_prices = np.zeros((self.days + 1, self.num_simulations))
        a_prices[0] = initial_price
        for i in range(1, self.days + 1):
            dS = mu * a_prices[i - 1] * dt + sigma * a_prices[i - 1] * dW[i]
            a_prices[i] = a_prices[i - 1] + dS

        return a_prices


def simulate_rebalance(portfolio, strategy):
    for step in range(1, portfolio.num_steps):
        portfolio.rebalance(strategy, step)


def create_portfolio_from_historical_prices(tickers, start_date, end_date, a_initial_weights, initial_investment):
    # Iterate over tickers and get adjusted close prices
    l_prices_df = [dh.get_adj_close(ticker, start_date, end_date) for ticker in tickers]

    merged_df = pd.concat(l_prices_df, axis=1, keys=tickers)

    # Drop the rows with missing values
    merged_df.dropna(inplace=True)

    # Convert the merged dataframe to a 3D numpy array
    # a_prices = merged_df.values.reshape(len(tickers), -1, 1)
    a_prices = merged_df.values.T[:, :, np.newaxis]

    # Create Asset objects
    assets = [Asset(ticker, a_prices[i, :, :], time_dim=merged_df.index) for i, ticker in enumerate(tickers)]

    # step_dim = merged_df.index

    # Create Portfolio object
    portfolio = Portfolio(assets, a_initial_weights, initial_investment)

    return portfolio


# tickers = ["AAPL", "MSFT"]
# start_date = "2020-01-01"
# end_date = "2020-12-31"
# a_initial_weights = np.array([0.6, 0.4])
# initial_investment = 10000
#
# portfolio = create_portfolio_from_historical_prices(tickers, start_date, end_date, a_initial_weights, initial_investment)
#
# # Choose a rebalancing strategy
# strategy = FixedQuantitiesStrategy(frequency=5)
#
# # Simulate the rebalance
# simulate_rebalance(portfolio, strategy)