"""The thermal variant must reproduce the original model and the integral variant exactly at default settings."""
import unittest

import numpy as np

from controller_variants import simulate_task_variant
from thermal_variants import (CORRECTED_THERMAL, GYRO_TCO_RAD_S_PER_K, MOTOR_RISE_600S_K, PUBLISHED_THERMAL,
                              drift_growth, forward_model, simulate_task_thermal)


def cases(count=15, seed=0):
    rng = np.random.default_rng(seed)
    for task in forward_model.TASKS:
        for _ in range(count):
            yield task, rng.uniform(0, 1, 4), int(rng.integers(1_000_000))


class ThermalVariantTests(unittest.TestCase):
    def tearDown(self):
        for name, value in PUBLISHED_THERMAL.items():
            setattr(forward_model, name, value)

    def test_defaults_reproduce_original_exactly(self):
        for constants in (PUBLISHED_THERMAL, CORRECTED_THERMAL):
            for name, value in constants.items():
                setattr(forward_model, name, value)
            for task, theta, seed in cases():
                original = forward_model.simulate_task(*theta, task, rng=np.random.default_rng(seed))
                variant = simulate_task_thermal(*theta, task, rng=np.random.default_rng(seed))
                self.assertEqual(original, variant, f'{task} {theta}')

    def test_integral_matches_controller_variant_exactly(self):
        for task, theta, seed in cases(5, seed=1):
            expected = simulate_task_variant(*theta, task, rng=np.random.default_rng(seed), ki=5.0)
            actual = simulate_task_thermal(*theta, task, rng=np.random.default_rng(seed), ki=5.0)
            self.assertEqual(expected, actual, f'{task} {theta}')

    def test_no_thermal_severity_means_warm_start_has_no_effect(self):
        for task, theta, seed in cases(5, seed=2):
            theta = (*theta[:3], 0.0)
            cold = simulate_task_thermal(*theta, task, rng=np.random.default_rng(seed))
            warm = simulate_task_thermal(*theta, task, rng=np.random.default_rng(seed), warm_start_s=600.0)
            self.assertEqual(cold, warm, task)

    def test_carried_heading_offset_steers_transit_off_line(self):
        # Offset 1.5e-4 rad/s * 600 s = 0.09 rad; a 2 m drive held at that sensed heading ends ~180 mm off line.
        for name, value in CORRECTED_THERMAL.items():
            setattr(forward_model, name, value)
        theta = (0.5, 0.5, 0.5, 1.0)
        error = simulate_task_thermal(*theta, 'transit', rng=np.random.default_rng(3), warm_start_s=600.0,
                                      carry_heading_offset=True)
        self.assertGreater(error, 120.0)
        self.assertLess(error, 220.0)

    def test_signed_thermal_maps_severity_and_direction(self):
        for name, value in CORRECTED_THERMAL.items():
            setattr(forward_model, name, value)
        for task, theta, seed in cases(3, seed=4):
            def run(t, **options):
                return simulate_task_thermal(*theta[:3], t, task, rng=np.random.default_rng(seed), warm_start_s=300.0,
                                             carry_heading_offset=True, **options)
            self.assertEqual(run(0.5, signed_thermal=True), run(0.0), task)
            self.assertEqual(run(1.0, signed_thermal=True), run(1.0), task)
        # Matched motors: drift of either sign gives the same sideways error.
        positive, negative = (simulate_task_thermal(0.5, 0.5, 0.5, t, 'transit', rng=np.random.default_rng(5), warm_start_s=600.0,
                                                    carry_heading_offset=True, signed_thermal=True) for t in (1.0, 0.0))
        self.assertAlmostEqual(positive, negative, delta=0.05 * positive)

    def test_temperature_driven_drift(self):
        for name, value in CORRECTED_THERMAL.items():
            setattr(forward_model, name, value)
        growth = drift_growth(GYRO_TCO_RAD_S_PER_K['typical'], MOTOR_RISE_600S_K['FR'])

        def transit(warm, carry, rate):
            return simulate_task_thermal(0.5, 0.5, 0.5, 1.0, 'transit', rng=np.random.default_rng(6), warm_start_s=warm,
                                         carry_heading_offset=carry, drift_growth_rad_s2=rate)
        # Carried offset = growth * 600^2 / 2; the drive holds that sensed heading, so it ends ~2 m * sin(offset) off line.
        expected = 2000.0 * np.sin(growth * 600.0 ** 2 / 2)
        self.assertAlmostEqual(transit(600.0, True, growth), expected, delta=0.15 * expected)
        # Heading re-zeroed at task start: only the few seconds of in-task drift remain.
        self.assertLess(transit(600.0, False, growth), 20.0)
        # Cold start: the gyro was just calibrated, so drift is negligible.
        self.assertAlmostEqual(transit(0.0, False, growth), transit(0.0, False, 0.0), delta=1.0)

    def test_absolute_turn(self):
        for name, value in CORRECTED_THERMAL.items():
            setattr(forward_model, name, value)
        # Turn-in-place starts at heading 0, so both conventions give the same target.
        for _, theta, seed in cases(3, seed=7):
            relative = simulate_task_thermal(*theta, 'turn', rng=np.random.default_rng(seed))
            absolute = simulate_task_thermal(*theta, 'turn', rng=np.random.default_rng(seed), absolute_turn=True)
            self.assertEqual(relative, absolute)
        # Parking: the drive's steady mismatch offset carries into a relative turn but not an absolute one.
        forward_model.MOTOR_GAIN_HALF_RANGE = 0.2 / 2.2
        try:
            def parking(**options):
                return np.degrees(np.mean([simulate_task_thermal(0.5, 1.0, 0.5, 0.5, 'parking_hdg', rng=np.random.default_rng(i),
                                                                 signed_thermal=True, **options) for i in range(40)]))
            self.assertLess(abs(parking(absolute_turn=True) - 0.945), 0.1)
            self.assertGreater(abs(parking(sensed_phase_start=True) - 0.945), 0.2)
        finally:
            forward_model.MOTOR_GAIN_HALF_RANGE = 0.10

    def test_scaled_mismatch_turn(self):
        saved = {name: getattr(forward_model, name) for name in
                 ('BATTERY_V_MIN', 'BATTERY_V_MAX', 'BATTERY_SLOPE_STEEP', 'BATTERY_SLOPE_FLAT', 'BATTERY_NOISE_STD',
                  'MOTOR_NOISE_CV', 'THERMAL_SPEED_DECAY_MAX')}
        try:
            # At exactly 12 V, no heating and no motor noise the achieved speed equals P * v_max, so nothing changes.
            for name, value in (('BATTERY_V_MIN', 12.0), ('BATTERY_V_MAX', 12.0), ('BATTERY_SLOPE_STEEP', 0.0),
                                ('BATTERY_SLOPE_FLAT', 0.0), ('BATTERY_NOISE_STD', 0.0), ('MOTOR_NOISE_CV', 0.0),
                                ('THERMAL_SPEED_DECAY_MAX', 0.0)):
                setattr(forward_model, name, value)
            for task, theta, seed in cases(3, seed=8):
                original = simulate_task_thermal(*theta, task, rng=np.random.default_rng(seed))
                scaled = simulate_task_thermal(*theta, task, rng=np.random.default_rng(seed), scaled_mismatch_turn=True)
                self.assertAlmostEqual(original, scaled, delta=1e-6 * max(1.0, abs(original)), msg=task)
            # With the PD's steady error set by mismatch / correction, voltage now cancels: transit error is flat in theta_B.
            for name, value in saved.items():
                setattr(forward_model, name, value)
            forward_model.MOTOR_NOISE_CV = 0.0
            forward_model.BATTERY_NOISE_STD = 0.0

            def transit(theta_b, **options):
                return simulate_task_thermal(0.5, 1.0, theta_b, 0.0, 'transit', rng=np.random.default_rng(9), **options)
            self.assertLess(abs(transit(0.0, scaled_mismatch_turn=True) / transit(1.0, scaled_mismatch_turn=True) - 1), 0.03)
            self.assertGreater(transit(0.0) / transit(1.0), 1.05)
        finally:
            for name, value in saved.items():
                setattr(forward_model, name, value)


if __name__ == '__main__':
    unittest.main()
