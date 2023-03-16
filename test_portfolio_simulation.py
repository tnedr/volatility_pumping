import unittest
import numpy as np
import portfolio_simulation as m

class TestSimulation(unittest.TestCase):
    def test_init(self):
        sim = m.Simulation(252, 1000, np.array([0.07, 0.12]), np.array([0.2, 0.3]), np.array([[1, 0.1], [0.1, 1]]), np.array([100, 50]))
        self.assertEqual(sim.days, 252)
        self.assertEqual(sim.num_simulations, 1000)
        self.assertTrue(np.array_equal(sim.a_mu, np.array([0.07, 0.12])))
        self.assertTrue(np.array_equal(sim.a_sigma, np.array([0.2, 0.3])))
        self.assertTrue(np.array_equal(sim.a_correlation_matrix, np.array([[1, 0.1], [0.1, 1]])))
        self.assertTrue(np.array_equal(sim.a_initial_prices, np.array([100, 50])))

    def test_generate_asset_prices(self):
        sim = m.Simulation(252, 1000, np.array([0.07, 0.12]), np.array([0.2, 0.3]), np.array([[1, 0.1], [0.1, 1]]), np.array([100, 50]))
        a_asset_prices = sim.generate_asset_prices(random_seed=42)
        self.assertEqual(a_asset_prices.shape, (2, 1000, 252))

if __name__ == '__main__':
    unittest.main()