import pandas as pd
import numpy as np
import portfolio_simulation as psim
import data_handler as dm
from visualization import plot_subplot

def historical_example():

    tickers = ['IEI', 'GSG']
    start_date = '2001-01-01'
    end_date = '2021-09-01'

    initial_weights = np.array([0.6, 0.4])
    initial_investment = 100000

    a_initial_amounts = initial_investment * initial_weights

    portfolio = psim.create_portfolio_from_historical_prices(
        tickers, start_date, end_date, initial_weights,
        initial_investment)

    assets = portfolio.assets

    mu = portfolio.calculate_asset_mu()
    sigma = portfolio.calculate_asset_sigma()
    correlation_matrix = portfolio.calculate_correlation_matrix()
    initial_prices = np.array([asset.get_prices()[0] for asset in assets])

    days = 252
    num_simulations = 1000

    simulation = psim.Simulation(days, num_simulations, mu, sigma, correlation_matrix, initial_prices)
    asset_prices = simulation.generate_asset_prices()

    # Plot summary charts
    plot_subplot(portfolio, portfolio.time_info)
    # Results
    print("Initial Weights:", initial_weights)
    print("Initial Investment:", initial_investment)
    print("Global Start Date:", portfolio.step_from)
    print("Global End Date:", portfolio.step_to)
    print("Asset Mu:", portfolio.calculate_asset_mu())
    print("Asset Sigma:", portfolio.calculate_asset_sigma())
    print("Correlation Matrix:", portfolio.calculate_correlation_matrix())
    print("Portfolio Value:", portfolio.calculate_portfolio_value(a_prices, a_initial_amounts))
    print("Portfolio Returns:", portfolio.calculate_portfolio_returns(a_prices))

    # Plot summary charts

    plot_summary(assets, portfolio, simulation)


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
    historical_example() #2765, 2
    # simulated_example()