"""Which fix restores accuracy with mismatched motors: heading feedback alone, integral action, or feedforward compensation.

Controllers compared at each mismatch level (jitter, battery and thermal severity
drawn uniformly; same draws and seeds everywhere, i.e. common random numbers):
  pd_original        the published controller (PD heading correction in drive)
  pid_ki1, pid_ki5   the same plus an integral term (controller_variants.py)
  feedforward        per-motor gain compensation, as the shooter does with kV_Left/kV_Right.
                     In this model that is equivalent to leaving only the calibration
                     residual, so it is evaluated as the original controller at 1% and 2% mismatch.

Outputs: results/controller_comparison.json and results/controller_comparison_log.txt.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from controller_variants import forward_model, simulate_task_variant
from mismatch_impact import HEADING_TASKS, offset_for, summarize
from replicated_sobol import TASK_INDEX, sha256

HERE = Path(__file__).resolve().parent
TASKS = ('transit', 'tracking', 'parking_pos', 'parking_hdg')
MISMATCH_PCT = (0.0, 5.0, 10.0, 20.0)
CONTROLLERS = {'pd_original': 0.0, 'pid_ki1': 1.0, 'pid_ki5': 5.0}
FEEDFORWARD_RESIDUAL_PCT = (1.0, 2.0)


def _chunk(job):
    task, mismatch, ki, rows, first_row, replicates, seed_base = job
    forward_model.MOTOR_GAIN_HALF_RANGE = offset_for(mismatch)
    out = np.empty((len(rows), replicates))
    for i, theta in enumerate(rows):
        for r in range(replicates):
            rng = np.random.default_rng([seed_base, TASK_INDEX[task], first_row + i, r])
            out[i, r] = simulate_task_variant(*theta, task, rng=rng, ki=ki)
    return first_row, out


def run(pool, samples, task, mismatch, ki, replicates, seed_base=20260926, chunk=64):
    jobs = [(task, mismatch, ki, samples[s:s + chunk], s, replicates, seed_base) for s in range(0, len(samples), chunk)]
    out = np.empty((len(samples), replicates))
    for first_row, block in pool.map(_chunk, jobs):
        out[first_row:first_row + len(block)] = block
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--draws', type=int, default=1024)
    parser.add_argument('--replicates', type=int, default=4)
    args = parser.parse_args()
    samples = np.random.default_rng(20260926).uniform(0, 1, (args.draws, 4))
    samples[:, 1] = np.arange(args.draws) % 2  # alternate which motor is stronger
    report = {'method': __doc__.strip(), 'draws': args.draws, 'replicates': args.replicates,
              'provenance': {'this_script_sha256': sha256(__file__), 'controller_variants_sha256': sha256(HERE / 'controller_variants.py'),
                             'forward_model_sha256': sha256(HERE.parent / 'forward_model.py')},
              'results': {}}
    log = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in TASKS:
            report['results'][task] = {}
            log.append(task)
            print(task, flush=True)
            for name, ki in CONTROLLERS.items():
                for mismatch in MISMATCH_PCT:
                    summary = summarize(run(pool, samples, task, mismatch, ki, args.replicates), task)
                    report['results'][task][f'{name}|mismatch={mismatch}'] = summary
                    line = f"  {name:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}"
                    print(line, flush=True)
                    log.append(line)
            for residual in FEEDFORWARD_RESIDUAL_PCT:
                summary = summarize(run(pool, samples, task, residual, 0.0, args.replicates), task)
                report['results'][task][f'feedforward|residual={residual}'] = summary
                line = f"  feedforward  residual {residual:3.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}  (any original mismatch)"
                print(line, flush=True)
                log.append(line)
    report['runtime_sec'] = time.time() - started
    (HERE / 'results' / 'controller_comparison.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (HERE / 'results' / 'controller_comparison_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
