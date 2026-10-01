"""The controller variant must match the original model exactly when the integral term is off."""
import unittest

import numpy as np

from controller_variants import forward_model, simulate_task_variant


class ControllerVariantTests(unittest.TestCase):
    def test_zero_integral_reproduces_original_exactly(self):
        rng = np.random.default_rng(0)
        for task in forward_model.TASKS:
            for _ in range(20):
                theta = rng.uniform(0, 1, 4)
                seed = int(rng.integers(1_000_000))
                original = forward_model.simulate_task(*theta, task, rng=np.random.default_rng(seed))
                variant = simulate_task_variant(*theta, task, rng=np.random.default_rng(seed), ki=0.0)
                self.assertEqual(original, variant, f'{task} {theta}')

    def test_integral_term_changes_drive_behaviour(self):
        theta = (0.5, 1.0, 0.5, 0.0)
        base = simulate_task_variant(*theta, 'transit', rng=np.random.default_rng(1), ki=0.0)
        with_integral = simulate_task_variant(*theta, 'transit', rng=np.random.default_rng(1), ki=2.0)
        self.assertNotEqual(base, with_integral)


if __name__ == '__main__':
    unittest.main()
