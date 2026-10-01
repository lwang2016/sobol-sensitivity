"""Sensitivity of the results to the assumed per-loop motor noise (MOTOR_NOISE_CV = 0.008, not measured).

Reruns the cold-start baseline at half and double the assumed value with the final settings:
corrected thermal constants, random drift direction, absolute turn, phases starting from the
sensed heading. Same draws, seeds and designs as corrected_thermal.py / parking_absolute.py.
Outputs: results/motor_noise_check.json and results/motor_noise_check_log.txt.
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
from thermal_variants import CORRECTED_THERMAL, evaluate, forward_model

HERE = Path(__file__).resolve().parent
TASKS = tuple(forward_model.TASKS)
NOISE_CV = (0.004, 0.008, 0.016)
MISMATCH_PCT = (0.0, 5.0, 10.0)
OPTIONS = {'signed_thermal': True, 'absolute_turn': True, 'sensed_phase_start': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()
    sobol_n, sobol_r, draws, reps = 512, 8, 1024, 4
    base = {'MOTOR_GAIN_HALF_RANGE': forward_model.MOTOR_GAIN_HALF_RANGE, **CORRECTED_THERMAL}
    report = {
        'method': __doc__.strip(), 'constants': base, 'options': OPTIONS, 'noise_cv': NOISE_CV,
        'provenance': {name: sha256(HERE / name) for name in ('motor_noise_check.py', 'thermal_variants.py', 'replicated_sobol.py', 'mismatch_impact.py')}
                      | {'forward_model.py': sha256(HERE.parent / 'forward_model.py')},
        'mismatch_impact': {}, 'sobol': {},
    }
    log = []

    def say(line):
        print(line, flush=True)
        log.append(line)

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        theta = np.random.default_rng(20260926).uniform(0, 1, (draws, 4))
        theta[:, 1] = np.arange(draws) % 2  # alternate which motor is stronger
        samples = saltelli.sample(PROBLEM, sobol_n, calc_second_order=False)
        for cv in NOISE_CV:
            key = f'cv={cv}'
            results = report['mismatch_impact'][key] = {}
            for task in TASKS:
                results[task] = {}
                for mismatch in MISMATCH_PCT:
                    config = {**base, 'MOTOR_NOISE_CV': cv, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                    summary = results[task][str(mismatch)] = summarize(evaluate(pool, theta, task, config, reps, 20260926, **OPTIONS), task)
                    say(f"  {key:10} {task:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  "
                        f"P95 {summary['p95']:8.3f}  run-to-run SD {summary['sd_run_to_run']:.4f}")
            tasks = report['sobol'][key] = {}
            for task in TASKS:
                config = {**base, 'MOTOR_NOISE_CV': cv}
                result = tasks[task] = analyze(samples, evaluate(pool, samples, task, config, sobol_r, 20260925, **OPTIONS), sobol_r, seed=5)
                say(f'  {key:10} Sobol {task:12} {fmt(result)}')

    report['runtime_sec'] = time.time() - started
    (HERE / 'results' / 'motor_noise_check.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (HERE / 'results' / 'motor_noise_check_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    say(f'\nSaved in {report["runtime_sec"] / 60:.1f} min')


if __name__ == '__main__':
    main()
