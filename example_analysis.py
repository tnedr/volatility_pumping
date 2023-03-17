import pandas as pd
import numpy as np
import portfolio_simulation as m
import data_handler as dm


def historical_example():

    tickers = ['IEI', 'VOO']
    start_date = '2001-01-01'
    end_date = '2021-09-01'

    initial_weights = np.array([0.6, 0.4])
    initial_amount = 100000

    # Create Portfolio
    portfolio = m.create_portfolio(tickers, start_date, end_date, initial_weights, initial_amount)

    # Get prices for the global date range
    a_prices = portfolio.get_all_asset_prices()

    # Calculate returns
    a_returns = np.diff(np.log(a_prices), axis=0)

    # Calculate statistics
    a_mu = np.mean(a_returns, axis=0) * 252
    a_sigma = np.std(a_returns, axis=0) * np.sqrt(252)
    a_correlation_matrix = np.corrcoef(a_returns.T)

    # Results
    print("Initial Weights:", initial_weights)
    print("Initial Amount:", initial_amount)
    print("Global Start Date:", portfolio.global_start_date)
    print("Global End Date:", portfolio.global_end_date)
    print("Mu:", a_mu)
    print("Sigma:", a_sigma)
    print("Correlation Matrix:\n", a_correlation_matrix)
    print("Portfolio Value:", portfolio.calculate_portfolio_value(a_prices))
    print("Portfolio Returns:", portfolio.calculate_portfolio_returns(a_prices))


def historical_example2():
    tickers = ['IEI', 'VOO']
    start_date = '2001-01-01'
    end_date = '2021-09-01'

    initial_weights = np.array([0.5, 0.5])
    initial_amount = np.array([1000, 1000])

    portfolio = m.create_portfolio(tickers, start_date, end_date, initial_weights, initial_amount)

    print("Global Start Date:", portfolio.global_start_date)
    print("Global End Date:", portfolio.global_end_date)

    a_avg_asset_prices = np.mean(portfolio.get_all_asset_prices(), axis=2)
    print("Portfolio Returns:", portfolio.calculate_portfolio_returns(a_avg_asset_prices))


def simulated_example():
    tickers = ['Asset1', 'Asset2']
    days = 252  # One year of daily data
    num_simulations = 1000

    a_initial_prices = np.array([100, 100])
    a_mu = np.array([0.08, 0.1])  # Expected annual returns
    a_sigma = np.array([0.2, 0.25])  # Annual volatilities
    a_correlation_matrix = np.array([[1, 0.5],
                                     [0.5, 1]])  # Correlation between assets

    initial_weights = np.array([0.6, 0.4])
    initial_amount = 100000

    # Create Simulation object
    sim = m.Simulation(days, num_simulations, a_mu, a_sigma, a_correlation_matrix, a_initial_prices)

    # Generate simulated asset prices
    a_asset_prices = sim.generate_asset_prices()

    # Calculate average asset prices across simulations
    # a_avg_asset_prices = np.mean(a_asset_prices, axis=1)

    # Create Asset objects
    assets = []
    for i, ticker in enumerate(tickers):
        assets.append(m.Asset(ticker, a_asset_prices[i, :, :]))

    # Dummy global start and end dates
    global_start_date = '2000-01-01'
    global_end_date = '2001-01-01'

    # Create Portfolio
    portfolio = m.Portfolio(assets, initial_weights, initial_amount, global_start_date, global_end_date)

    # Results
    print("Initial Weights:", initial_weights)
    print("Initial Amount:", initial_amount)
    print("Portfolio Value:", portfolio.calculate_portfolio_value(a_asset_prices))
    print("Portfolio Returns:", portfolio.calculate_portfolio_returns(a_asset_prices))


if __name__ == '__main__':
    # historical_example() #2765, 2
    simulated_example()