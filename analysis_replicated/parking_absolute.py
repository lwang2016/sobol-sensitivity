"""Parking tasks rerun with the absolute (IMU-referenced) turn and the review fixes (script review, AI_USE_LOG item 12).

Fixes: drift direction random (signed_thermal); each phase starts from the sensed heading;
drive-phase mismatch turn rate scaled like the heading correction (scaled_mismatch_turn).
Conventions: absolute_turn (primary: turn to -45 deg in the sensed frame) and relative
(turn 45 deg from the sensed heading at the end of the drive; secondary).
Only the parking tasks depend on the turn convention, so only they are rerun:
  A. cold start, corrected constants: Sobol baseline and seed check, range sweep,
     mismatch curves and controller comparison (designs and seeds as corrected_thermal.py)
  B. warm start, temperature-driven drift, absolute turn: grid and Sobol as warm_start_temperature.py
Outputs: results/parking_absolute.json and results/parking_absolute_log.txt.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from SALib.sample import saltelli

import warm_start_temperature as wst
from corrected_thermal import CONTROLLER_MISMATCH, CONTROLLERS, FEEDFORWARD_RESIDUAL_PCT
from mismatch_impact import MISMATCH_PCT, offset_for, summarize
from replicated_sobol import PROBLEM, analyze, fmt, sha256
from thermal_variants import CORRECTED_THERMAL, evaluate, forward_model

HERE = Path(__file__).resolve().parent
TASKS = ('parking_pos', 'parking_hdg')
CONVENTIONS = {'absolute': {'absolute_turn': True}, 'relative': {'sensed_phase_start': True}}
FIXES = {'signed_thermal': True, 'scaled_mismatch_turn': True}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--quick', action='store_true', help='small sizes for a smoke test')
    args = parser.parse_args()
    base_n, base_r, sweep_n, sweep_r, draws, reps = (32, 2, 32, 2, 64, 2) if args.quick else (1024, 16, 512, 8, 1024, 4)
    out_dir = HERE / ('results_quick' if args.quick else 'results')
    out_dir.mkdir(exist_ok=True)
    constants = {'MOTOR_GAIN_HALF_RANGE': forward_model.MOTOR_GAIN_HALF_RANGE, **CORRECTED_THERMAL}
    report = {
        'method': __doc__.strip(), 'constants': constants, 'fixes': FIXES, 'conventions': CONVENTIONS,
        'provenance': {name: sha256(HERE / name) for name in
                       ('parking_absolute.py', 'thermal_variants.py', 'corrected_thermal.py', 'warm_start_temperature.py',
                        'replicated_sobol.py', 'mismatch_impact.py')}
                      | {'forward_model.py': sha256(HERE.parent / 'forward_model.py')},
        'cold': {}, 'warm_temperature': {'settings': {}, 'mismatch_impact': {}, 'sobol': {}},
    }
    log = []

    def say(line):
        print(line, flush=True)
        log.append(line)

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        samples = saltelli.sample(PROBLEM, base_n, calc_second_order=False)
        sweep = saltelli.sample(PROBLEM, sweep_n, calc_second_order=False)
        theta = np.random.default_rng(20260926).uniform(0, 1, (draws, 4))
        theta[:, 1] = np.arange(draws) % 2  # alternate which motor is stronger
        for convention, turn in CONVENTIONS.items():
            options = {**FIXES, **turn}
            cold = report['cold'][convention] = {'sobol_baseline': {}, 'seed_stability': {}, 'range_sweep': [],
                                                 'mismatch_impact': {}, 'controller_comparison': {}}
            say(f'\n[{convention} turn] Sobol baseline, N={base_n}, R={base_r}')
            for task in TASKS:
                result = cold['sobol_baseline'][task] = analyze(samples, evaluate(pool, samples, task, constants, base_r, 20260925, **options), base_r, seed=1)
                say(f'  {task:12} {fmt(result)}')
            result = cold['seed_stability']['parking_hdg'] = analyze(samples, evaluate(pool, samples, 'parking_hdg', constants, base_r, 777, **options), base_r, seed=2)
            say(f'  parking_hdg  independent seeds: {fmt(result)}')

            say(f'[{convention} turn] range sweep, parking_hdg, N={sweep_n}, R={sweep_r}')
            for motor in (0.025, 0.05, 0.075, 0.10):
                for heading in (0.0, 1.5e-4, 7.5e-4):
                    config = {**constants, 'MOTOR_GAIN_HALF_RANGE': motor, 'THERMAL_HEADING_DRIFT_MAX': heading}
                    result = analyze(sweep, evaluate(pool, sweep, 'parking_hdg', config, sweep_r, 20260925, **options), sweep_r, seed=3)
                    cold['range_sweep'].append({'MOTOR_GAIN_HALF_RANGE': motor, 'THERMAL_HEADING_DRIFT_MAX': heading, 'parking_hdg': result})
                    say(f'  motor +/-{motor:.3f} heading {heading:.1e} {fmt(result)}')

            say(f'[{convention} turn] mismatch impact, {draws} draws x {reps} runs')
            for task in TASKS:
                cold['mismatch_impact'][task] = {}
                for mismatch in MISMATCH_PCT:
                    config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                    summary = cold['mismatch_impact'][task][str(mismatch)] = summarize(evaluate(pool, theta, task, config, reps, 20260926, **options), task)
                    say(f"  {task:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")

            say(f'[{convention} turn] controller comparison')
            for task in TASKS:
                results = cold['controller_comparison'][task] = {}
                for name, ki in CONTROLLERS.items():
                    for mismatch in CONTROLLER_MISMATCH:
                        config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                        summary = results[f'{name}|mismatch={mismatch}'] = summarize(evaluate(pool, theta, task, config, reps, 20260926, ki=ki, **options), task)
                        say(f"  {task:12} {name:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")
                for residual in FEEDFORWARD_RESIDUAL_PCT:
                    config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(residual)}
                    summary = results[f'feedforward|residual={residual}'] = summarize(evaluate(pool, theta, task, config, reps, 20260926, **options), task)
                    say(f"  {task:12} feedforward  residual {residual:3.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")

        warm = report['warm_temperature']
        options = {**FIXES, **CONVENTIONS['absolute']}
        growth = {name: wst.drift_growth(wst.GYRO_TCO_RAD_S_PER_K[tco], wst.MOTOR_RISE_600S_K[motor]) for name, (tco, motor) in wst.SETTINGS.items()}
        warm['settings'] = growth
        theta_warm = theta.copy()
        theta_warm[:, 3] = (np.arange(draws) // 2) % 2  # drift sign at full setting, balanced against motor direction
        say('\n[absolute turn] warm start, temperature-driven drift')
        for name in wst.SETTINGS:
            for reference, (carry, times) in wst.GRID.items():
                for warm_s in times:
                    key = f'{name}|{reference}|warm_start_s={warm_s:g}'
                    results = warm['mismatch_impact'][key] = {}
                    for task in TASKS:
                        results[task] = {}
                        for mismatch in wst.MISMATCH_PCT:
                            config = {**constants, 'MOTOR_GAIN_HALF_RANGE': offset_for(mismatch)}
                            errors = evaluate(pool, theta_warm, task, config, reps, 20260926, warm_start_s=warm_s,
                                              carry_heading_offset=carry, drift_growth_rad_s2=growth[name], **options)
                            summary = results[task][str(mismatch)] = summarize(errors, task)
                            say(f"  {key:40} {task:12} mismatch {mismatch:4.1f}%: mean {summary['mean']:8.3f} {summary['unit']}  P95 {summary['p95']:8.3f}")
        wsamples = saltelli.sample(PROBLEM, sweep_n, calc_second_order=False)
        for name in wst.SOBOL_SETTINGS:
            tasks = warm['sobol'][name] = {}
            for task in TASKS:
                outputs = evaluate(pool, wsamples, task, constants, sweep_r, 20260925, warm_start_s=wst.SOBOL_WARM_START_S,
                                   carry_heading_offset=True, drift_growth_rad_s2=growth[name], **options)
                result = tasks[task] = analyze(wsamples, outputs, sweep_r, seed=4)
                say(f'  Sobol {name:11} operation_start 600 s {task:12} {fmt(result)}')

    report['runtime_sec'] = time.time() - started
    (out_dir / 'parking_absolute.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'parking_absolute_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    say(f'\nSaved to {out_dir} in {report["runtime_sec"] / 60:.1f} min')


if __name__ == '__main__':
    main()
