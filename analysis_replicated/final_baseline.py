"""Final-configuration baseline for the tasks not covered by parking_absolute.py (transit, tracking, turn).

Final configuration used in the paper: corrected thermal constants, random drift direction,
absolute turn, each phase starting from the sensed heading, mismatch turn rate scaled like the
heading correction (scaled_mismatch_turn). Same designs and seeds as
corrected_thermal.py, so results are directly comparable. Together with parking_absolute.py
this gives every paper number from one configuration.
Outputs: results/final_baseline.json and results/final_baseline_log.txt.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from SALib.sample import saltelli

from corrected_thermal import CONTROLLER_MISMATCH, CONTROLLERS, FEEDFORWARD_RESIDUAL_PCT
from mismatch_impact import MISMATCH_PCT, offset_for, summarize
from replicated_sobol import PROBLEM, analyze, fmt, sha256
from thermal_variants import CORRECTED_THERMAL, evaluate, forward_model

HERE = Path(__file__).resolve().parent
TASKS = ('transit', 'tracking', 'turn')
CONTROLLER_TASKS = ('transit', 'tracking')
OPTIONS = {'signed_thermal': True, 'absolute_turn': True, 'sensed_phase_start': True, 'scaled_mismatch_turn': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    base_n, base_r, draws, reps = 1024, 16, 1024, 4
    constants = {'MOTOR_GAIN_HALF_RANGE': forward_model.MOTOR_GAIN_HALF_RANGE, **CORRECTED_THERMAL}
    report = {
        'method': __doc__.strip(), 'constants': constants, 'options': OPTIONS,
        'provenance': {name: sha256(HERE / name) for name in
                       ('final_baseline.py', 'thermal_variants.py', 'corrected_thermal.py', 'replicated_sobol.py', 'mismatch_impact.py')}
                      | {'forward_model.py': sha256(HERE.parent / 'forward_model.py')},
        'sobol_baseline': {'N': base_n, 'replicates': base_r, 'seed_base': 20260925, 'tasks': {}},
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
        for task in TASKS:
            result = report['sobol_baseline']['tasks'][task] = analyze(samples, evaluate(pool, samples, task, constants, base_r, 20260925, **OPTIONS), base_r, seed=1)
            say(f'  Sobol {task:10} {fmt(result)}')
        theta = np.random.default_rng(20260926).uniform(0, 1, (draws, 4))
        theta[:, 1] = np.arange(draws) % 2  # alternate which motor is stronger
        for task in TASKS:
            report['mismatch_impact']['results'][task] = {}
            for mismatch in MISMATCH_PCT:
                config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                summary = report['mismatch_impact']['results'][task][str(mismatch)] = summarize(evaluate(pool, theta, task, config, reps, 20260926, **OPTIONS), task)
                say(f"  {task:10} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")
        for task in CONTROLLER_TASKS:
            results = report['controller_comparison']['results'][task] = {}
            for name, ki in CONTROLLERS.items():
                for mismatch in CONTROLLER_MISMATCH:
                    config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                    summary = results[f'{name}|mismatch={mismatch}'] = summarize(evaluate(pool, theta, task, config, reps, 20260926, ki=ki, **OPTIONS), task)
                    say(f"  {task:10} {name:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}")
            for residual in FEEDFORWARD_RESIDUAL_PCT:
                config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(residual)}
                summary = results[f'feedforward|residual={residual}'] = summarize(evaluate(pool, theta, task, config, reps, 20260926, **OPTIONS), task)
                say(f"  {task:10} feedforward residual {residual:3.1f}%: mean {summary['mean']:8.3f} {summary['unit']}")

    report['runtime_sec'] = time.time() - started
    (HERE / 'results' / 'final_baseline.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (HERE / 'results' / 'final_baseline_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    say(f'\nSaved in {report["runtime_sec"] / 60:.1f} min')


if __name__ == '__main__':
    main()
