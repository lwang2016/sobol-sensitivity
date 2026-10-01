"""Warm-start simulation: tasks begin after the robot has already been operating for warm_start_s seconds.

Thermal state follows the corrected constants (thermal_variants.CORRECTED_THERMAL).
Warm-start times: 30 s (FTC autonomous period), 150 s (about one FTC match:
30 s autonomous + 2 min driver control), 300 s and 600 s (longest measured run;
no extrapolation beyond the D_1 data, which show no saturation within 600 s).

Heading reference:
  task_start       heading re-zeroed before each task (goBILDA Pinpoint guide recommends
                   resetPosAndIMU() at the start of the first OpMode); only the speed loss
                   is carried in, the heading offset builds during the task only.
  operation_start  heading referenced once at the start of operation; the offset built
                   over warm_start_s is carried in (worst case for heading).
Battery is not advanced: theta_B already samples the task-start voltage over the
observed 11.8-13.7 V range, which includes partly discharged states.

Outputs: results/warm_start.json and results/warm_start_log.txt.
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
WARM_START_S = (0.0, 30.0, 150.0, 300.0, 600.0)
SOBOL_WARM_START_S = (0.0, 150.0, 600.0)
REFERENCES = {'task_start': False, 'operation_start': True}
MISMATCH_PCT = (0.0, 5.0, 10.0, 20.0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--quick', action='store_true', help='small sizes for a smoke test')
    args = parser.parse_args()
    sobol_n, sobol_r, draws, reps = (32, 2, 64, 2) if args.quick else (512, 8, 1024, 4)
    out_dir = HERE / ('results_quick' if args.quick else 'results')
    out_dir.mkdir(exist_ok=True)
    constants = {'MOTOR_GAIN_HALF_RANGE': forward_model.MOTOR_GAIN_HALF_RANGE, **CORRECTED_THERMAL}
    rate, bias = CORRECTED_THERMAL['THERMAL_SPEED_DECAY_MAX'], CORRECTED_THERMAL['THERMAL_HEADING_DRIFT_MAX']
    report = {
        'method': __doc__.strip(), 'constants': constants,
        'thermal_state_at_max_severity': {str(t): {'speed_factor': 1 - rate * t, 'carried_heading_offset_deg': float(np.degrees(bias * t))}
                                          for t in WARM_START_S},
        'provenance': {name: sha256(HERE / name) for name in ('warm_start.py', 'thermal_variants.py', 'replicated_sobol.py', 'mismatch_impact.py')}
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
        say(f'Mismatch impact at warm start, {draws} draws x {reps} runs')
        for reference, carry in REFERENCES.items():
            for warm in WARM_START_S:
                key = f'{reference}|warm_start_s={warm:g}'
                results = report['mismatch_impact']['results'][key] = {}
                for task in TASKS:
                    results[task] = {}
                    for mismatch in MISMATCH_PCT:
                        config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                        errors = evaluate(pool, theta, task, config, reps, 20260926, warm_start_s=warm, carry_heading_offset=carry)
                        summary = results[task][str(mismatch)] = summarize(errors, task)
                        say(f"  {key:32} {task:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")

        samples = saltelli.sample(PROBLEM, sobol_n, calc_second_order=False)
        say(f'\nSobol at warm start, N={sobol_n}, R={sobol_r}')
        for reference, carry in REFERENCES.items():
            for warm in SOBOL_WARM_START_S:
                key = f'{reference}|warm_start_s={warm:g}'
                tasks = report['sobol']['configs'][key] = {}
                for task in TASKS:
                    outputs = evaluate(pool, samples, task, constants, sobol_r, 20260925, warm_start_s=warm, carry_heading_offset=carry)
                    result = tasks[task] = analyze(samples, outputs, sobol_r, seed=4)
                    result['top_source_S1'] = max(result['S1'], key=result['S1'].get)
                    say(f'  {key:32} {task:12} {fmt(result)}')

    report['runtime_sec'] = time.time() - started
    (out_dir / 'warm_start.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'warm_start_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    say(f'\nSaved to {out_dir} in {report["runtime_sec"] / 60:.1f} min')


if __name__ == '__main__':
    main()
