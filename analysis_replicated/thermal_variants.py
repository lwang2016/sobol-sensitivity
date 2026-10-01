"""Warm-start variant of forward_model.simulate_task, plus a parallel evaluator that sets model constants for every job.

Built by patching forward_model.simulate_task's own source in memory (as in
controller_variants.py); forward_model.py on disk is not modified.

  warm_start_s          operating time before the task starts. Thermal speed loss
                        continues from that point: factor 1 - rate * (warm_start_s + t).
  carry_heading_offset  False: heading is referenced at task start, so the sensed-heading
                        offset from thermal bias builds only during the task (original model).
                        True: heading is referenced at the start of operation, so the offset
                        built over warm_start_s is carried in, and each phase starts from the
                        sensed heading, as a real controller would.
  ki                    optional integral term, identical to controller_variants.py.
  signed_thermal        False: theta_T is severity 0..1 and the heading drift always has one sign
                        (original model). True: severity is |2*theta_T - 1| and the drift sign is
                        sign(2*theta_T - 1), so drift direction is random with the same magnitude
                        distribution; a real gyro's bias can go either way.
  drift_growth_rad_s2   0: constant drift rate (above). > 0: temperature-driven drift. The gyro is
                        calibrated at power-on, and its bias grows with temperature rise, taken as
                        linear in time: rate(tau) = severity * drift_growth_rad_s2 * tau, where tau is
                        time since calibration. Sensed offset = integral of the rate from the heading
                        reference (power-on if carry_heading_offset, else task start, with the gyro not
                        recalibrated). Replaces THERMAL_HEADING_DRIFT_MAX; phases start from the sensed heading.
  sensed_phase_start    True: each phase starts from the sensed heading (a real robot knows no other).
                        The original starts from the true heading, which double-counts drift in a
                        relative turn. Implied by carry_heading_offset and drift_growth_rad_s2.
  absolute_turn         True: turn to the task's target heading in the sensed frame (IMU-referenced
                        turn), instead of turning by the target angle from the heading at phase start.
  scaled_mismatch_turn  True: in drive phases the mismatch turn rate uses the achieved forward speed
                        (voltage, heating and motor noise included), as the heading correction already
                        does, so both follow v_L,R = (1 -/+ delta) P V(t)/12 V (1 - rt) v_max and
                        omega = (v_R - v_L)/W. The original uses the nominal speed P * v_max for the
                        mismatch term only.

With the defaults the variant reproduces the original exactly (test_thermal_variants.py).
"""
import inspect
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import forward_model  # noqa: E402
from controller_variants import _PATCHES as _INTEGRAL_PATCHES  # noqa: E402
from replicated_sobol import TASK_INDEX  # noqa: E402

PUBLISHED_THERMAL = {'THERMAL_SPEED_DECAY_MAX': forward_model.THERMAL_SPEED_DECAY_MAX,
                     'THERMAL_HEADING_DRIFT_MAX': forward_model.THERMAL_HEADING_DRIFT_MAX}
# Speed: joint D_1+C1 fit gives 0.84e-4 (k=1.31) to 1.15e-4 /s (k=1.0); the code comment's own
# "6% over 600 s" is 1e-4. Heading: the code's stated estimate, without the 5x exploratory factor.
CORRECTED_THERMAL = {'THERMAL_SPEED_DECAY_MAX': 1.0e-4, 'THERMAL_HEADING_DRIFT_MAX': 1.5e-4}
REQUIRED_CONSTANTS = ('MOTOR_GAIN_HALF_RANGE', 'THERMAL_SPEED_DECAY_MAX', 'THERMAL_HEADING_DRIFT_MAX')

# Bosch BNO055 datasheet (BST-BNO055-DS000): gyro zero-rate offset change over temperature,
# typical 0.015 and maximum 0.03 deg/s per K.
GYRO_TCO_RAD_S_PER_K = {'typical': np.radians(0.015), 'maximum': np.radians(0.03)}
# Motor temperature rise over the 600 s D_1 run (7.8/1.9/2.2/4.2 F), a stand-in for IMU warming.
MOTOR_RISE_600S_K = {'FL': 7.8 / 1.8, 'FR': 1.9 / 1.8, 'BL': 2.2 / 1.8, 'BR': 4.2 / 1.8}


def drift_growth(tco, rise_k, over_s=600.0):
    """rad/s^2 such that the drift rate reaches tco * rise_k after over_s seconds."""
    return tco * rise_k / over_s

_PATCHES = [
    ('def simulate_task(theta_j, theta_m, theta_b, theta_t, task_name, rng=None):',
     'def simulate_task_thermal(theta_j, theta_m, theta_b, theta_t, task_name, rng=None,\n'
     '                          warm_start_s=0.0, carry_heading_offset=False, ki=0.0, signed_thermal=False,\n'
     '                          drift_growth_rad_s2=0.0, sensed_phase_start=False, absolute_turn=False,\n'
     '                          scaled_mismatch_turn=False):'),
    ('    speed_decay, heading_drift = map_thermal(theta_t)\n',
     '    if signed_thermal:\n'
     '        speed_decay, heading_drift = map_thermal(abs(2 * theta_t - 1))\n'
     '        heading_drift *= np.sign(2 * theta_t - 1)\n'
     '        drift_growth = drift_growth_rad_s2 * (2 * theta_t - 1)\n'
     '    else:\n'
     '        speed_decay, heading_drift = map_thermal(theta_t)\n'
     '        drift_growth = drift_growth_rad_s2 * theta_t\n'
     '\n'
     '    def _sensed_offset(elapsed):\n'
     '        if drift_growth_rad_s2:\n'
     '            reference = 0.0 if carry_heading_offset else warm_start_s\n'
     '            return drift_growth * ((warm_start_s + elapsed) ** 2 - reference ** 2) / 2\n'
     '        return heading_drift * ((warm_start_s if carry_heading_offset else 0.0) + elapsed)\n'),
    ('        heading_at_phase_start = heading  # capture actual heading at start of each phase\n',
     '        heading_at_phase_start = heading + (_sensed_offset(elapsed_total)\n'
     '                                            if carry_heading_offset or drift_growth_rad_s2 or sensed_phase_start else 0.0)\n'),
    ('                absolute_target = heading_at_phase_start + phase_target_heading\n',
     '                absolute_target = target_heading if absolute_turn else heading_at_phase_start + phase_target_heading\n'),
    ('            thermal_factor = max(1.0 - speed_decay * elapsed_total, 0.7)\n'
     '            heading_bias = heading_drift * elapsed_total\n',
     '            thermal_factor = max(1.0 - speed_decay * (warm_start_s + elapsed_total), 0.7)\n'
     '            heading_bias = _sensed_offset(elapsed_total)\n'),
    ('                asymmetry_omega = (gain_diff / avg_gain) * fwd_power * MAX_SPEED_MM_S / (TRACK_WIDTH_MM / 2.0)\n',
     '                asymmetry_omega = ((gain_diff / avg_gain) * fwd_speed / (TRACK_WIDTH_MM / 2.0) if scaled_mismatch_turn else\n'
     '                                   (gain_diff / avg_gain) * fwd_power * MAX_SPEED_MM_S / (TRACK_WIDTH_MM / 2.0))\n'),
] + _INTEGRAL_PATCHES[1:]


def _build():
    source = inspect.getsource(forward_model.simulate_task)
    for old, new in _PATCHES:
        if source.count(old) != 1:
            raise RuntimeError(f'Patch target not found exactly once: {old.strip()[:60]}')
        source = source.replace(old, new)
    exec(compile(source, '<simulate_task_thermal>', 'exec'), forward_model.__dict__)
    return forward_model.__dict__['simulate_task_thermal']


simulate_task_thermal = _build()


def _chunk(job):
    task, constants, options, rows, first_row, replicates, seed_base = job
    # Worker processes are reused, so every job sets every constant it depends on.
    for name, value in constants.items():
        setattr(forward_model, name, value)
    out = np.empty((len(rows), replicates))
    for i, theta in enumerate(rows):
        for r in range(replicates):
            rng = np.random.default_rng([seed_base, TASK_INDEX[task], first_row + i, r])
            out[i, r] = simulate_task_thermal(*theta, task, rng=rng, **options)
    return first_row, out


def evaluate(pool, samples, task, constants, replicates, seed_base, chunk=64, **options):
    missing = [name for name in REQUIRED_CONSTANTS if name not in constants]
    if missing:
        raise ValueError(f'constants must set {missing}')
    jobs = [(task, constants, options, samples[s:s + chunk], s, replicates, seed_base)
            for s in range(0, len(samples), chunk)]
    out = np.empty((len(samples), replicates))
    for first_row, block in pool.map(_chunk, jobs):
        out[first_row:first_row + len(block)] = block
    return out
