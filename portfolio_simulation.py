import numpy as np
import pandas as pd
from scipy.stats import norm
import data_handler as dh


class Asset:

    def __init__(self, name, a_prices):
        self.name = name
        self.a_prices = a_prices

    def calculate_returns(self):
        self.a_returns = np.diff(self.a_prices, axis=0) / self.a_prices[:-1, :]
        return self.a_returns

    def get_prices(self):
        return self.a_prices


class Portfolio:

    def __init__(self, assets, a_initial_weights, initial_investment, global_start_date, global_end_date, time_info):
        self.assets = assets
        self.a_initial_weights = a_initial_weights
        self.initial_investment = initial_investment
        self.global_start_date = global_start_date
        self.global_end_date = global_end_date
        self.time_info = time_info
        self.df_weights = pd.DataFrame(columns=[asset.name for asset in assets])
        self.a_initial_quantities = self.calculate_initial_quantities(initial_investment)
        self.a_quantities = self.initialize_quantities(initial_investment)
        self.df_rebalancing = pd.DataFrame(columns=["Date", "New Weights"])
        self.df_trade_history = pd.DataFrame(columns=["Asset", "Purchase Price", "Quantity", "Time Index"])
        self.total_pnl = {"Trading Profit": 0, "Tax Payable": 0, "Trading Cost": 0}

        self.store_initial_quantities()


    def initialize_quantities(self, initial_investment):
        a_initial_prices = np.array([asset.get_prices()[0] for asset in self.assets])
        a_initial_quantities = (initial_investment * self.a_initial_weights) / a_initial_prices

        num_assets = len(self.assets)
        num_steps = len(self.time_info)
        num_paths = self.a_initial_quantities.shape[1]

        a_quantities = np.zeros((num_assets, num_steps, num_paths))
        a_quantities[:, 0, :] = a_initial_quantities

        return a_quantities

    def compute_and_store_weights(self):
        a_total_investments = np.sum(self.initial_investment)
        self.a_initial_weights = self.initial_investment / a_total_investments
        self.df_weights = pd.DataFrame(data=self.a_initial_weights.reshape(1, -1),
            columns=[asset.name for asset in self.assets])

    def calculate_initial_quantities(self, initial_investment):
        a_initial_prices = np.array([asset.get_prices()[0] for asset in self.assets])
        a_initial_quantities = (initial_investment * self.a_initial_weights) / a_initial_prices
        return a_initial_quantities

    def store_initial_quantities(self):
        # self.a_quantities = np.concatenate((self.a_quantities, self.a_initial_quantities[:, np.newaxis, :]), axis=1)
        self.a_quantities = np.concatenate((self.a_quantities, self.a_initial_quantities[:, :, np.newaxis]), axis=1)



    def get_all_asset_prices_old(self):
        return np.array([asset.get_prices() for asset in self.assets]).T

    def get_all_asset_prices(self):
        a_prices = np.array([asset.get_prices() for asset in self.assets])
        return a_prices

    def calculate_portfolio_value(self, a_prices, a_quantities):
        return np.sum(a_prices * a_quantities[:, np.newaxis], axis=0)

    # def compute_and_store_weights(self):
    #     a_total_investments = np.sum(self.a_initial_investments)
    #     self.a_initial_weights = self.a_initial_investments / a_total_investments
    #     self.df_weights = pd.DataFrame(self.a_initial_weights, columns=[asset.name for asset in self.assets])

    def compute_and_store_weights(self):
        a_total_investments = np.sum(self.a_initial_investments)
        self.a_initial_weights = self.a_initial_investments / a_total_investments
        self.df_weights = pd.DataFrame(data=self.a_initial_weights.reshape(1, -1),
                                       columns=[asset.name for asset in self.assets])

    def calculate_portfolio_returns(self, a_prices):
        a_returns = np.diff(a_prices, axis=1) / a_prices[:, :-1, :]
        a_weighted_returns = self.a_initial_weights[:, np.newaxis, np.newaxis] * a_returns
        a_portfolio_returns = np.sum(a_weighted_returns, axis=0)
        return a_portfolio_returns

    def calculate_asset_mu(self):
        asset_returns_list = [asset.calculate_returns() for asset in self.assets]
        a_asset_returns = np.array(asset_returns_list)
        a_mu = np.mean(a_asset_returns, axis=1) * 252
        return a_mu

    def calculate_asset_sigma(self):
        asset_returns_list = [asset.calculate_returns() for asset in self.assets]
        a_asset_returns = np.array(asset_returns_list)
        a_sigma = np.std(a_asset_returns, axis=1) * np.sqrt(252)
        return a_sigma

    def calculate_correlation_matrix(self):
        asset_returns_list = [asset.calculate_returns() for asset in self.assets]
        a_asset_returns = np.array(asset_returns_list)
        num_paths = a_asset_returns.shape[2]
        correlation_matrices = []
        for i in range(num_paths):
            corr_matrix = np.corrcoef(a_asset_returns[:, :, i])
            correlation_matrices.append(corr_matrix)

        a_correlation_matrix = np.array(correlation_matrices)
        a_correlation_matrix = a_correlation_matrix.swapaxes(0, 2)
        return a_correlation_matrix

    # Add any additional functions you need for the portfolio simulation


class Simulation:
    def __init__(self, days, num_simulations, a_mu, a_sigma, a_correlation_matrix, a_initial_prices):
        self.days = days
        self.num_simulations = num_simulations
        self.a_mu = a_mu
        self.a_sigma = a_sigma
        self.a_correlation_matrix = a_correlation_matrix
        self.a_initial_prices = a_initial_prices

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


def create_portfolio_from_historical_prices(tickers, start_date, end_date, a_initial_weights, initial_investment):
    # Iterate over tickers and get adjusted close prices
    l_prices_df = [dh.get_adj_close(ticker, start_date, end_date) for ticker in tickers]

    merged_df = pd.concat(l_prices_df, axis=1, keys=tickers)

    # Drop the rows with missing values
    merged_df.dropna(inplace=True)

    # Convert the merged dataframe to a 3D numpy array
    a_prices = merged_df.values.reshape(len(tickers), -1, 1)

    # Create Asset objects
    assets = [Asset(ticker, a_prices[i, :, :]) for i, ticker in enumerate(tickers)]

    time_info = merged_df.index

    # Create Portfolio object
    portfolio = Portfolio(assets, a_initial_weights, initial_investment, start_date, end_date, time_info)

    return portfolio

