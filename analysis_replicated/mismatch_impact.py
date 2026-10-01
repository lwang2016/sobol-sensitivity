"""Task error at fixed motor mismatch levels, with the other noise sources varying.

A mismatch d means one motor's gain is (1 + d) times the other's; which motor is
stronger alternates across draws. In forward_model terms the gains are 1 -/+ offset,
so offset = d / (2 + d).
For each level, jitter, battery and thermal severities are drawn uniformly over
their ranges and each draw is run R times. The same draws and seeds are used at
every level (common random numbers), so differences come only from the mismatch.

Outputs: results/mismatch_impact.json and results/mismatch_impact_log.txt.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from replicated_sobol import evaluate, sha256

HERE = Path(__file__).resolve().parent
MISMATCH_PCT = (0.0, 2.5, 5.0, 7.5, 10.0, 15.0, 20.0)
TASKS = ('transit', 'tracking', 'parking_pos', 'parking_hdg', 'turn')
HEADING_TASKS = {'parking_hdg', 'turn'}


def offset_for(mismatch_pct):
    d = mismatch_pct / 100
    return d / (2 + d)


def summarize(errors, task):
    values = np.degrees(errors) if task in HEADING_TASKS else errors
    flat = values.ravel()
    per_draw_mean = values.mean(axis=1)
    return {
        'unit': 'deg' if task in HEADING_TASKS else 'mm',
        'mean': float(flat.mean()),
        'median': float(np.median(flat)),
        'p90': float(np.percentile(flat, 90)),
        'p95': float(np.percentile(flat, 95)),
        'sd_across_conditions': float(per_draw_mean.std(ddof=1)),
        'sd_run_to_run': float(np.sqrt(values.var(axis=1, ddof=1).mean())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--draws', type=int, default=1024)
    parser.add_argument('--replicates', type=int, default=4)
    args = parser.parse_args()
    rng = np.random.default_rng(20260926)
    samples = rng.uniform(0, 1, (args.draws, 4))
    # Alternate which motor is stronger; the thermal heading bias is one-directional, so direction matters.
    samples[:, 1] = np.arange(args.draws) % 2
    report = {'method': __doc__.strip(), 'draws': args.draws, 'replicates': args.replicates, 'seed_base': 20260926,
              'provenance': {'this_script_sha256': sha256(__file__), 'forward_model_sha256': sha256(HERE.parent / 'forward_model.py')},
              'heading_bias_settings_rad_per_s': {'published': 7.5e-4, 'code_estimate_from_data': 1.5e-4},
              'results': {}}
    log = []
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        scenarios = [(task, 'published', {}) for task in TASKS]
        scenarios += [(task, 'code_estimate_from_data', {'THERMAL_HEADING_DRIFT_MAX': 1.5e-4}) for task in sorted(HEADING_TASKS)]
        for task, setting, extra in scenarios:
            key = f'{task}|heading_bias={setting}'
            report['results'][key] = {}
            header = f'{key}'
            print(header, flush=True)
            log.append(header)
            for mismatch in MISMATCH_PCT:
                overrides = {'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch), **extra}
                errors = evaluate(pool, samples, task, overrides, args.replicates, 20260926)
                summary = summarize(errors, task)
                report['results'][key][str(mismatch)] = summary
                line = (f"  mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  median {summary['median']:8.3f}  "
                        f"P95 {summary['p95']:8.3f}  SD across conditions {summary['sd_across_conditions']:7.3f}  "
                        f"run-to-run SD {summary['sd_run_to_run']:7.3f}")
                print(line, flush=True)
                log.append(line)
    report['runtime_sec'] = time.time() - started
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'mismatch_impact.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (HERE / 'results' / 'mismatch_impact_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
