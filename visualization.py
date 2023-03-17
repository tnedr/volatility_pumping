import matplotlib.pyplot as plt


def plot_asset_prices(portfolio, time_info):
    fig, ax = plt.subplots()
    for i, asset in enumerate(portfolio.assets):
        ax.plot(time_info, asset.get_prices(), label=asset.name)
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('Asset Prices')
    ax.legend()
    plt.show()


def plot_portfolio_value(portfolio, time_info):
    fig, ax = plt.subplots()
    a_portfolio_value = portfolio.calculate_portfolio_value(portfolio.get_all_asset_prices(), portfolio.a_quantities)
    ax.plot(time_info, a_portfolio_value)
    ax.set_xlabel('Time')
    ax.set_ylabel('Portfolio Value')
    ax.set_title('Portfolio Value')
    plt.show()


def plot_weights(portfolio, time_info):
    fig, ax = plt.subplots()
    for i, asset in enumerate(portfolio.assets):
        ax.plot(time_info, portfolio.df_weights[asset.name], label=asset.name)
    ax.set_xlabel('Time')
    ax.set_ylabel('Weight')
    ax.set_title('Asset Weights in Portfolio')
    ax.legend()
    plt.show()


def plot_subplot(portfolio, time_info):
    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # Asset Prices
    for i, asset in enumerate(portfolio.assets):
        axes[0].plot(time_info, asset.get_prices(), label=asset.name)
    axes[0].set_ylabel('Price')
    axes[0].set_title('Asset Prices')
    axes[0].legend()

    # Portfolio Value
    a_portfolio_value = portfolio.calculate_portfolio_value(portfolio.get_all_asset_prices(), portfolio.a_initial_quantities)
    axes[1].plot(time_info, a_portfolio_value)
    axes[1].set_ylabel('Portfolio Value')
    axes[1].set_title('Portfolio Value')

    # Asset Weights
    for i, asset in enumerate(portfolio.assets):
        axes[2].plot(time_info, portfolio.df_weights[asset.name], label=asset.name)
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Weight')
    axes[2].set_title('Asset Weights in Portfolio')
    axes[2].legend()

    plt.show()
