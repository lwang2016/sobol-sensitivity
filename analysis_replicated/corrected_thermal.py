"""Cold-start results rerun with the corrected thermal constants (thermal_variants.CORRECTED_THERMAL).

Published THERMAL_SPEED_DECAY_MAX = 1e-3/s is ten times its own comment ("6% over 600 s")
and the D_1+C1 fit (0.84e-4 to 1.15e-4 /s); corrected to 1e-4. Published
THERMAL_HEADING_DRIFT_MAX = 7.5e-4 rad/s is 5x the code's own estimate; corrected to
1.5e-4 and 7.5e-4 kept only as a labelled stress case in the range sweep.

Same designs, estimators and seeds as replicated_sobol.py, mismatch_impact.py and
controller_comparison.py, so every difference from those results comes from the constants.
Outputs: results/corrected_thermal.json and results/corrected_thermal_log.txt.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from SALib.sample import saltelli

from mismatch_impact import MISMATCH_PCT, offset_for, summarize
from replicated_sobol import PROBLEM, analyze, fmt, sha256
from thermal_variants import CORRECTED_THERMAL, PUBLISHED_THERMAL, evaluate, forward_model

HERE = Path(__file__).resolve().parent
TASKS = tuple(forward_model.TASKS)
CONTROLLER_TASKS = ('transit', 'tracking', 'parking_pos', 'parking_hdg')
CONTROLLER_MISMATCH = (0.0, 5.0, 10.0, 20.0)
CONTROLLERS = {'pd_original': 0.0, 'pid_ki1': 1.0, 'pid_ki5': 5.0}
FEEDFORWARD_RESIDUAL_PCT = (1.0, 2.0)
PUBLISHED_MOTOR = forward_model.MOTOR_GAIN_HALF_RANGE


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--quick', action='store_true', help='small sizes for a smoke test')
    args = parser.parse_args()
    base_n, base_r, sweep_n, sweep_r, draws, reps = (32, 2, 32, 2, 64, 2) if args.quick else (1024, 16, 512, 8, 1024, 4)
    out_dir = HERE / ('results_quick' if args.quick else 'results')
    out_dir.mkdir(exist_ok=True)
    constants = {'MOTOR_GAIN_HALF_RANGE': PUBLISHED_MOTOR, **CORRECTED_THERMAL}
    report = {
        'method': __doc__.strip(),
        'published_thermal': PUBLISHED_THERMAL, 'corrected_thermal': CORRECTED_THERMAL,
        'provenance': {name: sha256(HERE / name) for name in
                       ('corrected_thermal.py', 'thermal_variants.py', 'replicated_sobol.py', 'mismatch_impact.py', 'controller_variants.py')}
                      | {'forward_model.py': sha256(HERE.parent / 'forward_model.py')},
        'sobol_baseline': {'N': base_n, 'replicates': base_r, 'seed_base': 20260925, 'tasks': {}},
        'seed_stability': {},
        'range_sweep': {'N': sweep_n, 'replicates': sweep_r, 'seed_base': 20260925, 'configs': []},
        'mismatch_impact': {'draws': draws, 'replicates': reps, 'seed_base': 20260926, 'results': {}},
        'controller_comparison': {'draws': draws, 'replicates': reps, 'seed_base': 20260926, 'results': {}},
    }
    log = []

    def say(line):
        print(line, flush=True)
        log.append(line)

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        samples = saltelli.sample(PROBLEM, base_n, calc_second_order=False)
        say(f'Sobol baseline, corrected thermal, N={base_n}, R={base_r}')
        for task in TASKS:
            result = analyze(samples, evaluate(pool, samples, task, constants, base_r, 20260925), base_r, seed=1)
            report['sobol_baseline']['tasks'][task] = result
            say(f'  {task:12} {fmt(result)}')
        result = analyze(samples, evaluate(pool, samples, 'parking_hdg', constants, base_r, 777), base_r, seed=2)
        report['seed_stability']['parking_hdg'] = result
        say(f'  parking_hdg  independent seeds: {fmt(result)}')

        sweep = saltelli.sample(PROBLEM, sweep_n, calc_second_order=False)
        say(f'\nRange sweep, speed decay {CORRECTED_THERMAL["THERMAL_SPEED_DECAY_MAX"]:.0e}/s, N={sweep_n}, R={sweep_r}')
        for motor in (0.025, 0.05, 0.075, 0.10):
            for heading in (0.0, 1.5e-4, 7.5e-4):
                config = {**constants, 'MOTOR_GAIN_HALF_RANGE': motor, 'THERMAL_HEADING_DRIFT_MAX': heading}
                entry = {'MOTOR_GAIN_HALF_RANGE': motor, 'THERMAL_HEADING_DRIFT_MAX': heading, 'tasks': {}}
                for task in ('parking_hdg', 'transit'):
                    result = analyze(sweep, evaluate(pool, sweep, task, config, sweep_r, 20260925), sweep_r, seed=3)
                    result['top_source_S1'] = max(result['S1'], key=result['S1'].get)
                    entry['tasks'][task] = result
                    say(f'  motor +/-{motor:.3f} heading {heading:.1e} {task:12} {fmt(result)}')
                report['range_sweep']['configs'].append(entry)

        draws_theta = np.random.default_rng(20260926).uniform(0, 1, (draws, 4))
        draws_theta[:, 1] = np.arange(draws) % 2  # alternate which motor is stronger
        say(f'\nMismatch impact, {draws} draws x {reps} runs')
        for task in TASKS:
            report['mismatch_impact']['results'][task] = {}
            for mismatch in MISMATCH_PCT:
                config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                summary = summarize(evaluate(pool, draws_theta, task, config, reps, 20260926), task)
                report['mismatch_impact']['results'][task][str(mismatch)] = summary
                say(f"  {task:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")

        say('\nController comparison')
        for task in CONTROLLER_TASKS:
            results = report['controller_comparison']['results'][task] = {}
            for name, ki in CONTROLLERS.items():
                for mismatch in CONTROLLER_MISMATCH:
                    config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                    summary = summarize(evaluate(pool, draws_theta, task, config, reps, 20260926, ki=ki), task)
                    results[f'{name}|mismatch={mismatch}'] = summary
                    say(f"  {task:12} {name:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")
            for residual in FEEDFORWARD_RESIDUAL_PCT:
                config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(residual)}
                summary = summarize(evaluate(pool, draws_theta, task, config, reps, 20260926), task)
                results[f'feedforward|residual={residual}'] = summary
                say(f"  {task:12} feedforward  residual {residual:3.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")

    report['runtime_sec'] = time.time() - started
    (out_dir / 'corrected_thermal.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'corrected_thermal_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    say(f'\nSaved to {out_dir} in {report["runtime_sec"] / 60:.1f} min')


if __name__ == '__main__':
    main()
