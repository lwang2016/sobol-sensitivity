"""Checks the noise-corrected estimator on functions with known Sobol indices."""
import unittest

import numpy as np
from SALib.sample import saltelli

from replicated_sobol import PROBLEM, corrected_indices


def replicated(function, noise_sd, replicates, n=4096, seed=0):
    x = saltelli.sample(PROBLEM, n, calc_second_order=False)
    rng = np.random.default_rng(seed)
    runs = function(x)[:, None] + rng.normal(0, noise_sd, (len(x), replicates))
    return corrected_indices(runs.mean(axis=1), runs.var(axis=1, ddof=1), replicates, resamples=50)


class CorrectedIndicesTests(unittest.TestCase):
    def test_additive_function_recovered_despite_noise(self):
        # Var(x)=1/12, so x1 + 2*x2 gives S1 = ST = (0.2, 0.8, 0, 0).
        result = replicated(lambda x: x[:, 0] + 2 * x[:, 1], noise_sd=0.3, replicates=8)
        expected = [0.2, 0.8, 0.0, 0.0]
        for name, value in zip(PROBLEM['names'], expected):
            self.assertAlmostEqual(result['S1'][name], value, delta=0.04)
            self.assertAlmostEqual(result['ST'][name], value, delta=0.04)
        # Noise variance 0.09 against signal 5/12.
        self.assertAlmostEqual(result['noise_share'], 0.09 / (0.09 + 5 / 12), delta=0.01)

    def test_noise_is_not_reported_as_interaction(self):
        noisy = replicated(lambda x: x[:, 0] + 2 * x[:, 1], noise_sd=0.5, replicates=4)
        for name in ('theta_B', 'theta_T'):
            self.assertLess(abs(noisy['ST'][name]), 0.03)

    def test_true_interaction_is_kept(self):
        # x1*x2 has an interaction: ST exceeds S1 for both inputs.
        result = replicated(lambda x: (x[:, 0] - 0.5) * (x[:, 1] - 0.5), noise_sd=0.02, replicates=8)
        for name in ('theta_J', 'theta_M'):
            self.assertLess(result['S1'][name], 0.05)
            self.assertGreater(result['ST'][name], 0.9)


if __name__ == '__main__':
    unittest.main()
