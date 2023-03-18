import numpy as np
from numpy.testing import assert_array_equal
import unittest
from portfolio_simulation import Asset, Portfolio, FixedQuantitiesStrategy, FixedWeightsStrategy, create_portfolio_from_historical_prices


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
                np.testing.assert_array_almost_equal(current_weights, a_initial_weights[:, np.newaxis], decimal=4, err_msg='Weights did not match after rebalancing')

if __name__ == "__main__":
    unittest.main()
