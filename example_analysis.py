import pandas as pd
import numpy as np
import portfolio_simulation as m
import data_handler as dm


def historical_example():
    tickers = ['IEI', 'VOO']
    start_date = '2018-01-01'
    end_date = '2021-09-01'

    dfs = [dm.get_data(ticker, start_date, end_date) for ticker in tickers]

    df_prices = pd.concat([dfs[i].rename(columns={tickers[i]: col}) for i, col in enumerate(dfs[0].columns)], axis=1)

    a_prices = df_prices.to_numpy()
    a_returns = np.diff(np.log(a_prices), axis=0)

    a_mu = np.mean(a_returns, axis=0) * 252
    a_sigma = np.std(a_returns, axis=0) * np.sqrt(252)

    a_correlation_matrix = np.corrcoef(a_returns.T)

    initial_weights = np.array([0.6, 0.4])
    initial_amount = 100000

    # Create Asset objects
    assets = []
    for i, ticker in enumerate(tickers):
        assets.append(m.Asset(ticker, a_prices[:, i]))

    # Create Portfolio
    portfolio = m.Portfolio(assets, initial_weights, initial_amount)

    # Results
    print("Initial Weights:", initial_weights)
    print("Initial Amount:", initial_amount)
    print("Portfolio Value:", portfolio.calculate_portfolio_value(a_prices))
    print("Portfolio Returns:", portfolio.calculate_portfolio_returns(a_prices))


if __name__ == '__main__':
    historical_example()
