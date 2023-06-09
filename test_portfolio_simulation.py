import unittest
import numpy as np
import pandas as pd
from portfolio_simulation import Asset, Portfolio, FixedQuantitiesStrategy, create_portfolio_from_historical_prices


class TestAsset(unittest.TestCase):

    def setUp(self):
        self.a_prices = np.array([[100, 110, 120],
                                  [100*1.1, 110*1.1, 120*1.1],
                                  [100*1.1*1.1, 110*1.1*1.1, 120*1.1*1.1]])
        self.asset = Asset("Test Asset", self.a_prices)

    def test_init(self):
        self.assertEqual(self.asset.name, "Test Asset")
        np.testing.assert_array_equal(self.asset.a_prices, self.a_prices)

    def test_calculate_returns(self):
        a_expected_returns = np.array([[0.1, 0.1, 0.1],
                                       [0.1, 0.1, 0.1]])
        a_returns = self.asset.calculate_returns()
        np.testing.assert_array_almost_equal(a_returns, a_expected_returns)

    def test_get_prices(self):
        a_prices = self.asset.get_prices()
        np.testing.assert_array_equal(a_prices, self.a_prices)


class TestPortfolio(unittest.TestCase):

    def setUp(self):
        a_prices1 = np.array([[100, 110, 120],
                              [110, 120, 130],
                              [120, 130, 140]])
        a_prices2 = np.array([[200, 210, 220],
                              [210, 220, 230],
                              [220, 230, 240]])
        asset1 = Asset("Asset 1", a_prices1)
        asset2 = Asset("Asset 2", a_prices2)
        self.assets = [asset1, asset2]
        self.a_initial_weights = np.array([0.6, 0.4])
        self.initial_investment = 10000
        self.portfolio = Portfolio(self.assets, self.a_initial_weights, self.initial_investment)

    def test_init(self):
        self.assertEqual(self.portfolio.assets, self.assets)
        np.testing.assert_array_equal(self.portfolio.a_initial_weights, self.a_initial_weights)
        self.assertEqual(self.portfolio.initial_investment, self.initial_investment)

    def test_get_step_dim(self):
        a_expected_step_dim = range(3)
        a_step_dim = self.portfolio.get_step_dim()
        np.testing.assert_array_equal(a_step_dim, a_expected_step_dim)

    def test_calculate_initial_quantities(self):
        a_expected_initial_quantities = np.array([[10000*0.6/100, 10000*0.6/110, 10000*0.6/120],
                                                  [10000*0.4/200, 10000*0.4/210, 10000*0.4/220]])
        a_initial_quantities = self.portfolio.calculate_initial_quantities(self.initial_investment)
        np.testing.assert_array_almost_equal(a_initial_quantities, a_expected_initial_quantities)

    # Add the rest of the tests that we have already discussed in our previous conversations
    def test_portfolio_initialization(self):
        tickers = ["AAPL", "MSFT"]
        start_date = "2020-01-01"
        end_date = "2020-12-31"
        a_initial_weights = np.array([0.6, 0.4])
        initial_investment = 10000

        portfolio = create_portfolio_from_historical_prices(tickers, start_date, end_date, a_initial_weights, initial_investment)

        self.assertEqual(len(portfolio.assets), 2)
        self.assertTrue(isinstance(portfolio.assets[0], Asset))
        self.assertTrue(isinstance(portfolio.assets[1], Asset))
        self.assertTrue(np.array_equal(portfolio.a_initial_weights, a_initial_weights))
        self.assertEqual(portfolio.initial_investment, initial_investment)


class TestFixedQuantitiesStrategy(unittest.TestCase):
    def test_fixed_quantities(self):
        # Create example data for testing
        tickers = ['AAPL', 'GOOGL']
        start_date = '2021-01-01'
        end_date = '2021-12-31'
        a_initial_weights = np.array([0.5, 0.5])
        initial_investment = 10000

        # Create a portfolio
        portfolio = create_portfolio_from_historical_prices(tickers, start_date, end_date, a_initial_weights, initial_investment)

        # Choose a rebalancing strategy
        strategy = FixedQuantitiesStrategy(frequency=5)

        # Iterate through each step and check if the quantities remain the same after rebalancing
        for step in range(1, portfolio.num_steps):
            prev_quantities = portfolio.a_quantities[:, step-1, :].copy()
            portfolio.rebalance(strategy, step)
            np.testing.assert_array_equal(prev_quantities, portfolio.a_quantities[:, step, :], 'Quantities changed after rebalancing')


class TestFixedWeightsStrategy(unittest.TestCase):
    def test_fixed_weights(self):
        # Create example data for testing
        tickers = ['AAPL', 'GOOGL']
        start_date = '2021-01-01'
        end_date = '2021-12-31'
        a_initial_weights = np.array([0.5, 0.5])
        initial_investment = 10000

        # Create a portfolio
        portfolio = create_portfolio_from_historical_prices(tickers, start_date, end_date, a_initial_weights, initial_investment)

        # Choose a rebalancing strategy
        strategy = FixedWeightsStrategy(frequency=5)

        # Iterate through each step and check if the weights are maintained after rebalancing
        for step in range(1, portfolio.num_steps):
            prev_weights = portfolio.a_weights[:, step - 1, :].copy()
            portfolio.rebalance(strategy, step)
            current_weights = portfolio.a_weights[:, step, :]

            if strategy.should_rebalance():
                np.testing.assert_array_almost_equal(prev_weights, a_initial_weights[:, np.newaxis], decimal=4, err_msg='Weights did not match after rebalancing')
            else:
                np.testing.assert_array_almost_equal(prev_weights, current_weights, decimal=4, err_msg='Weights changed without rebalancing')

if __name__ == "__main__":
    unittest.main()
