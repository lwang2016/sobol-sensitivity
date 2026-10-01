"""Warm-start rerun with the thermal heading drift in either direction (signed_thermal=True).

warm_start.py drew the drift in one direction only. That one-directional drift cancels or
adds to the turn's fixed stop-short and to one mismatch direction, so its parking-heading
results were an artifact. Only the affected runs are repeated:
  operation_start  the carried-in offset is large; full mismatch grid and Sobol, as in warm_start.py
  task_start       drift acts only within a task (<= 0.1 deg); t = 0 and 600 s grid only, to check
                   that the cold and task-start results are unaffected
Same draws, seeds and sizes as warm_start.py.
Outputs: results/warm_start_signed.json and results/warm_start_signed_log.txt.
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
from warm_start import MISMATCH_PCT, SOBOL_WARM_START_S, TASKS, WARM_START_S

HERE = Path(__file__).resolve().parent
GRID = {'operation_start': (True, WARM_START_S), 'task_start': (False, (0.0, 600.0))}
SOBOL = {'operation_start': (True, SOBOL_WARM_START_S)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--quick', action='store_true', help='small sizes for a smoke test')
    args = parser.parse_args()
    sobol_n, sobol_r, draws, reps = (32, 2, 64, 2) if args.quick else (512, 8, 1024, 4)
    out_dir = HERE / ('results_quick' if args.quick else 'results')
    out_dir.mkdir(exist_ok=True)
    constants = {'MOTOR_GAIN_HALF_RANGE': forward_model.MOTOR_GAIN_HALF_RANGE, **CORRECTED_THERMAL}
    report = {
        'method': __doc__.strip(), 'constants': constants, 'signed_thermal': True,
        'provenance': {name: sha256(HERE / name) for name in
                       ('warm_start_signed.py', 'warm_start.py', 'thermal_variants.py', 'replicated_sobol.py', 'mismatch_impact.py')}
                      | {'forward_model.py': sha256(HERE.parent / 'forward_model.py')},
        'mismatch_impact': {'draws': draws, 'replicates': reps, 'seed_base': 20260926, 'results': {}},
        'sobol': {'N': sobol_n, 'replicates': sobol_r, 'seed_base': 20260925, 'configs': {}},
    }
    log = []

    def say(line):
        print(line, flush=True)
        log.append(line)

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        theta = np.random.default_rng(20260926).uniform(0, 1, (draws, 4))
        theta[:, 1] = np.arange(draws) % 2  # alternate which motor is stronger
        say(f'Mismatch impact, signed drift, {draws} draws x {reps} runs')
        for reference, (carry, times) in GRID.items():
            for warm in times:
                key = f'{reference}|warm_start_s={warm:g}'
                results = report['mismatch_impact']['results'][key] = {}
                for task in TASKS:
                    results[task] = {}
                    for mismatch in MISMATCH_PCT:
                        config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                        errors = evaluate(pool, theta, task, config, reps, 20260926, warm_start_s=warm,
                                          carry_heading_offset=carry, signed_thermal=True)
                        summary = results[task][str(mismatch)] = summarize(errors, task)
                        say(f"  {key:32} {task:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")

        samples = saltelli.sample(PROBLEM, sobol_n, calc_second_order=False)
        say(f'\nSobol, signed drift, N={sobol_n}, R={sobol_r}')
        for reference, (carry, times) in SOBOL.items():
            for warm in times:
                key = f'{reference}|warm_start_s={warm:g}'
                tasks = report['sobol']['configs'][key] = {}
                for task in TASKS:
                    outputs = evaluate(pool, samples, task, constants, sobol_r, 20260925, warm_start_s=warm,
                                       carry_heading_offset=carry, signed_thermal=True)
                    result = tasks[task] = analyze(samples, outputs, sobol_r, seed=4)
                    result['top_source_S1'] = max(result['S1'], key=result['S1'].get)
                    say(f'  {key:32} {task:12} {fmt(result)}')

    report['runtime_sec'] = time.time() - started
    (out_dir / 'warm_start_signed.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'warm_start_signed_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    say(f'\nSaved to {out_dir} in {report["runtime_sec"] / 60:.1f} min')


if __name__ == '__main__':
    main()
