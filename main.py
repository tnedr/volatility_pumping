
class Portfolio:
    def __init__(self, initial_investment, weights, delta_weight, trading_cost):
        self.initial_investment = initial_investment
        self.weights = weights
        self.current_values = None
        self.delta_weight = delta_weight
        self.trading_cost = trading_cost

    def calculate_current_values(self, prices):
        self.current_values = np.array(prices) * self.weights * self.initial_investment

    def rebalance(self, returns):
        current_returns = returns[-1]
        vol_target = np.mean(returns) / np.std(returns)
        vol_current = np.std(current_returns)
        if vol_current < vol_target:
            self.weights[0] -= self.delta_weight
            self.weights[1] += self.delta_weight
        else:
            self.weights[0] += self.delta_weight
            self.weights[1] -= self.delta_weight
        self.weights /= np.sum(self.weights)
        self.current_values *= (1 - self.trading_cost)

    def get_value(self):
        return np.sum(self.current_values)


def simulate_portfolio(initial_investment, weights, returns, delta_weight, trading_cost):
    portfolio = Portfolio(initial_investment, weights, delta_weight, trading_cost)
    portfolio.calculate_current_values([1, 1])
    portfolio_values = [portfolio.get_value()]
    for i in range(1, len(returns)):
        portfolio.rebalance(returns[:i])
        portfolio.calculate_current_values(returns[i])
        portfolio_values.append(portfolio.get_value())
    return portfolio_values