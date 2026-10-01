"""Warm-start simulation with temperature-driven gyro drift (thermal_variants, drift_growth_rad_s2 > 0).

The gyro is calibrated at power-on, so its bias starts at zero and grows with temperature
rise: rate = TCO * dT(t), dT linear to the measured 600 s motor rise. Each setting pairs a
BNO055 datasheet TCO (typical 0.015, maximum 0.03 deg/s per K) with one measured motor rise
(FR 1.1 K, BR 2.3 K, FL 4.3 K); the IMU's own temperature was not measured. The settings
span the referenced range, to find where the compromise lies, not to predict one robot.

Heading reference:
  task_start       heading re-zeroed before each task, gyro not recalibrated: the bias rate
                   reached so far acts during the task
  operation_start  heading referenced once at power-on: the whole accumulated offset carries in
(Recalibrating the gyro before a task, e.g. resetPosAndIMU(), returns it to the cold start.)
Drift direction is random (signed); in the mismatch grid drift is at full setting (severity 1),
with drift and mismatch directions balanced. Turn is omitted: its error is model randomness.
The drive-phase mismatch turn rate is scaled like the heading correction (scaled_mismatch_turn).
Outputs: results/warm_start_temperature.json and results/warm_start_temperature_log.txt.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from SALib.sample import saltelli

from mismatch_impact import offset_for, summarize
from replicated_sobol import PROBLEM, analyze, fmt, sha256
from thermal_variants import (CORRECTED_THERMAL, GYRO_TCO_RAD_S_PER_K, MOTOR_RISE_600S_K, drift_growth, evaluate,
                              forward_model)

HERE = Path(__file__).resolve().parent
TASKS = ('transit', 'tracking', 'parking_pos', 'parking_hdg')
SETTINGS = {'typical_FR': ('typical', 'FR'), 'typical_BR': ('typical', 'BR'),
            'typical_FL': ('typical', 'FL'), 'maximum_FL': ('maximum', 'FL')}
GRID = {'task_start': (False, (150.0, 600.0)), 'operation_start': (True, (30.0, 150.0, 300.0, 600.0))}
MISMATCH_PCT = (0.0, 5.0, 10.0)
SOBOL_SETTINGS = ('typical_FL', 'maximum_FL')
SOBOL_WARM_START_S = 600.0


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--quick', action='store_true', help='small sizes for a smoke test')
    args = parser.parse_args()
    sobol_n, sobol_r, draws, reps = (32, 2, 64, 2) if args.quick else (512, 8, 1024, 4)
    out_dir = HERE / ('results_quick' if args.quick else 'results')
    out_dir.mkdir(exist_ok=True)
    constants = {'MOTOR_GAIN_HALF_RANGE': forward_model.MOTOR_GAIN_HALF_RANGE, **CORRECTED_THERMAL}
    growth = {name: drift_growth(GYRO_TCO_RAD_S_PER_K[tco], MOTOR_RISE_600S_K[motor]) for name, (tco, motor) in SETTINGS.items()}
    report = {
        'method': __doc__.strip(), 'constants': constants,
        'settings': {name: {'tco': tco, 'motor_rise': motor, 'rise_K': MOTOR_RISE_600S_K[motor],
                            'drift_growth_rad_s2': growth[name], 'drift_rate_at_600s_rad_s': growth[name] * 600,
                            'carried_offset_at_600s_deg': float(np.degrees(growth[name] * 600 ** 2 / 2))}
                     for name, (tco, motor) in SETTINGS.items()},
        'provenance': {name: sha256(HERE / name) for name in
                       ('warm_start_temperature.py', 'thermal_variants.py', 'replicated_sobol.py', 'mismatch_impact.py')}
                      | {'forward_model.py': sha256(HERE.parent / 'forward_model.py')},
        'mismatch_impact': {'draws': draws, 'replicates': reps, 'seed_base': 20260926, 'results': {}},
        'sobol': {'N': sobol_n, 'replicates': sobol_r, 'seed_base': 20260925, 'warm_start_s': SOBOL_WARM_START_S,
                  'reference': 'operation_start', 'configs': {}},
    }
    log = []

    def say(line):
        print(line, flush=True)
        log.append(line)

    for name, entry in report['settings'].items():
        say(f"{name:11} rate at 600 s {entry['drift_rate_at_600s_rad_s']:.2e} rad/s, carried offset at 600 s {entry['carried_offset_at_600s_deg']:.1f} deg")
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        theta = np.random.default_rng(20260926).uniform(0, 1, (draws, 4))
        theta[:, 1] = np.arange(draws) % 2          # which motor is stronger
        theta[:, 3] = (np.arange(draws) // 2) % 2   # drift sign at full setting, balanced against motor direction
        say(f'\nMismatch impact, {draws} draws x {reps} runs')
        for name in SETTINGS:
            for reference, (carry, times) in GRID.items():
                for warm in times:
                    key = f'{name}|{reference}|warm_start_s={warm:g}'
                    results = report['mismatch_impact']['results'][key] = {}
                    for task in TASKS:
                        results[task] = {}
                        for mismatch in MISMATCH_PCT:
                            config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                            errors = evaluate(pool, theta, task, config, reps, 20260926, warm_start_s=warm, carry_heading_offset=carry,
                                              signed_thermal=True, drift_growth_rad_s2=growth[name], scaled_mismatch_turn=True)
                            summary = results[task][str(mismatch)] = summarize(errors, task)
                            say(f"  {key:40} {task:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")

        samples = saltelli.sample(PROBLEM, sobol_n, calc_second_order=False)
        say(f'\nSobol, operation_start, {SOBOL_WARM_START_S:g} s, N={sobol_n}, R={sobol_r}')
        for name in SOBOL_SETTINGS:
            tasks = report['sobol']['configs'][name] = {}
            for task in TASKS:
                outputs = evaluate(pool, samples, task, constants, sobol_r, 20260925, warm_start_s=SOBOL_WARM_START_S,
                                   carry_heading_offset=True, signed_thermal=True, drift_growth_rad_s2=growth[name],
                                   scaled_mismatch_turn=True)
                result = tasks[task] = analyze(samples, outputs, sobol_r, seed=4)
                say(f'  {name:11} {task:12} {fmt(result)}')

    report['runtime_sec'] = time.time() - started
    (out_dir / 'warm_start_temperature.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'warm_start_temperature_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    say(f'\nSaved to {out_dir} in {report["runtime_sec"] / 60:.1f} min')


if __name__ == '__main__':
    main()
