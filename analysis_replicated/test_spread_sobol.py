"""Checks spread indices on a function whose noise level depends on known inputs."""
import unittest

import numpy as np
from SALib.sample import saltelli

from replicated_sobol import PROBLEM
from spread_sobol import spread_indices


class SpreadIndicesTests(unittest.TestCase):
    def test_only_the_input_that_controls_noise_has_spread_sensitivity(self):
        x = saltelli.sample(PROBLEM, 2048, calc_second_order=False)
        rng = np.random.default_rng(0)
        # The mean depends on x2, the noise SD only on x1.
        sd = 0.1 * np.exp(1.5 * x[:, 0])
        outputs = 5 * x[:, 1][:, None] + sd[:, None] * rng.standard_normal((len(x), 32))
        result = spread_indices(outputs, x)
        self.assertGreater(result['S1']['theta_J'], 0.9)
        self.assertGreater(result['ST']['theta_J'], 0.9)
        for name in ('theta_M', 'theta_B', 'theta_T'):
            self.assertLess(abs(result['S1'][name]), 0.05)
            self.assertLess(abs(result['ST'][name]), 0.05)

    def test_binned_first_order_is_stable_when_spread_estimates_are_noisy(self):
        x = saltelli.sample(PROBLEM, 1024, calc_second_order=False)
        sd = 0.1 * np.exp(0.5 * x[:, 0])
        outputs = sd[:, None] * np.random.default_rng(2).standard_normal((len(x), 12))
        result = spread_indices(outputs, x)
        self.assertGreater(result['estimation_noise_share'], 0.5)
        self.assertAlmostEqual(result['S1']['theta_J'], 1.0, delta=0.25)
        for name in ('theta_M', 'theta_B', 'theta_T'):
            self.assertLess(abs(result['S1'][name]), 0.15)

    def test_constant_noise_has_no_spread_sensitivity(self):
        x = saltelli.sample(PROBLEM, 1024, calc_second_order=False)
        outputs = x[:, 1][:, None] + 0.2 * np.random.default_rng(1).standard_normal((len(x), 32))
        result = spread_indices(outputs)
        self.assertLess(result['signal_variance'], 0.02)


if __name__ == '__main__':
    unittest.main()
