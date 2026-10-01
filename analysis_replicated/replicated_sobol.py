"""Replicated Sobol analysis that separates the model's run-to-run randomness from input effects.

The original run_sobol.py evaluates the stochastic forward model once per Saltelli
sample, so run-to-run randomness is counted as input sensitivity. Here each sample
is evaluated R times with independent seeds. Indices are estimated on the replicate
mean and corrected for the residual noise that averaging leaves behind.

For one run  Y = m(theta) + noise:  Var(Y) = Var(m) + E[noise variance].
Reported per task:
  noise_share          E[noise variance] / Var(Y): randomness no input explains
  S1, ST               first/total-order indices of m(theta), noise removed
  S1_of_total_variance S1 * (1 - noise_share), comparable to one-run results

Outputs go to ./results; forward_model.py and the original results are only read.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from importlib import metadata
from pathlib import Path

import numpy as np
from SALib.analyze import sobol
from SALib.sample import saltelli

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import forward_model  # noqa: E402

NAMES = ['theta_J', 'theta_M', 'theta_B', 'theta_T']
PROBLEM = {'num_vars': 4, 'names': NAMES, 'bounds': [[0.0, 1.0]] * 4}
TASK_INDEX = {name: index for index, name in enumerate(forward_model.TASKS)}


def split_saltelli(y, k):
    """Split first-order-only Saltelli output into A, B and AB_i blocks."""
    y = np.asarray(y, dtype=float).reshape(-1, k + 2)
    return y[:, 0], y[:, k + 1], y[:, 1:k + 1]


def indices(a, b, ab, residual_noise=0.0):
    """Saltelli (2010) first-order and Jansen total-order estimators.

    residual_noise is the noise variance left in each output value; it biases
    Var(Y) and the Jansen numerator upward and is subtracted from both.
    """
    variance = np.var(np.r_[a, b], ddof=1) - residual_noise
    if variance <= 0:
        return np.full(ab.shape[1], np.nan), np.full(ab.shape[1], np.nan), float(variance)
    first = np.mean(b[:, None] * (ab - a[:, None]), axis=0) / variance
    total = (0.5 * np.mean((a[:, None] - ab) ** 2, axis=0) - residual_noise) / variance
    return first, total, float(variance)


def corrected_indices(mean_output, within_variance, replicates, k=4, resamples=1000, seed=0):
    a, b, ab = split_saltelli(mean_output, k)
    noise_variance = float(np.mean(within_variance))
    residual = noise_variance / replicates
    s1, st, signal_variance = indices(a, b, ab, residual)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(a), size=(resamples, len(a)))
    boot = np.array([np.r_[indices(a[d], b[d], ab[d], residual)[:2]] for d in draws])
    half_width = 1.96 * np.nanstd(boot, axis=0)
    noise_share = noise_variance / (signal_variance + noise_variance) if signal_variance > 0 else 1.0
    return {
        'S1': dict(zip(NAMES, s1.tolist())),
        'S1_conf': dict(zip(NAMES, half_width[:k].tolist())),
        'ST': dict(zip(NAMES, st.tolist())),
        'ST_conf': dict(zip(NAMES, half_width[k:].tolist())),
        'sum_S1': float(np.nansum(s1)),
        'noise_share': float(noise_share),
        'S1_of_total_variance': dict(zip(NAMES, (s1 * (1 - noise_share)).tolist())),
        'signal_variance': signal_variance,
        'noise_variance': noise_variance,
        'residual_noise_in_mean': residual,
    }


def _evaluate_chunk(job):
    task, overrides, rows, first_row, replicates, seed_base = job
    for name, value in overrides.items():
        setattr(forward_model, name, value)
    out = np.empty((len(rows), replicates))
    for i, theta in enumerate(rows):
        for r in range(replicates):
            rng = np.random.default_rng([seed_base, TASK_INDEX[task], first_row + i, r])
            out[i, r] = forward_model.simulate_task(*theta, task, rng=rng)
    return first_row, out


def evaluate(pool, samples, task, overrides, replicates, seed_base, chunk=64):
    jobs = [(task, overrides, samples[s:s + chunk], s, replicates, seed_base) for s in range(0, len(samples), chunk)]
    outputs = np.empty((len(samples), replicates))
    for first_row, block in pool.map(_evaluate_chunk, jobs):
        outputs[first_row:first_row + len(block)] = block
    return outputs


def analyze(samples, outputs, replicates, seed):
    mean_output = outputs.mean(axis=1)
    within = outputs.var(axis=1, ddof=1)
    single = sobol.analyze(PROBLEM, outputs[:, 0], calc_second_order=False, num_resamples=1000, seed=seed)
    result = corrected_indices(mean_output, within, replicates, seed=seed)
    result['single_run_SALib'] = {
        'S1': dict(zip(NAMES, single['S1'].tolist())),
        'ST': dict(zip(NAMES, single['ST'].tolist())),
        'sum_S1': float(np.sum(single['S1'])),
    }
    result['Y_mean'] = float(np.mean(outputs))
    result['Y_single_run_std'] = float(np.std(outputs[:, 0]))
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fmt(result):
    s1, st = result['S1'], result['ST']
    return (f"noise={result['noise_share']:.3f}  S1 J/M/B/T=" + '/'.join(f'{s1[n]:+.3f}' for n in NAMES)
            + '  ST=' + '/'.join(f'{st[n]:.3f}' for n in NAMES) + f"  sumS1={result['sum_S1']:.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--quick', action='store_true', help='small N and R for a smoke test')
    args = parser.parse_args()
    base_n, base_r, sweep_n, sweep_r = (64, 4, 64, 4) if args.quick else (1024, 16, 512, 8)
    out_dir = HERE / ('results_quick' if args.quick else 'results')
    out_dir.mkdir(exist_ok=True)
    published = {name: getattr(forward_model, name) for name in
                 ('MOTOR_GAIN_HALF_RANGE', 'THERMAL_HEADING_DRIFT_MAX', 'THERMAL_SPEED_DECAY_MAX')}
    report = {
        'method': __doc__.strip(),
        'provenance': {
            'forward_model_sha256': sha256(ROOT / 'forward_model.py'),
            'run_sobol_sha256': sha256(ROOT / 'run_sobol.py'),
            'original_results_sha256': sha256(ROOT / 'sobol_results.json'),
            'this_script_sha256': sha256(__file__),
            'python': platform.python_version(),
            'numpy': np.__version__,
            'SALib': metadata.version('SALib'),
        },
        'published_constants': published,
        'baseline': {'N': base_n, 'replicates': base_r, 'seed_base': 20260925, 'tasks': {}},
        'seed_stability': {},
        'range_sweep': {'N': sweep_n, 'replicates': sweep_r, 'seed_base': 20260925, 'configs': []},
    }
    log = []

    def say(line):
        print(line, flush=True)
        log.append(line)

    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        samples = saltelli.sample(PROBLEM, base_n, calc_second_order=False)
        say(f'Baseline: published constants, N={base_n}, R={base_r}, {len(samples)} samples x {base_r} runs per task')
        for task in forward_model.TASKS:
            t0 = time.time()
            outputs = evaluate(pool, samples, task, {}, base_r, 20260925)
            result = analyze(samples, outputs, base_r, seed=1)
            report['baseline']['tasks'][task] = result
            say(f'  {task:12} {fmt(result)}  ({time.time() - t0:.0f}s)')

        for task in ('parking_hdg',):
            outputs = evaluate(pool, samples, task, {}, base_r, 777)
            result = analyze(samples, outputs, base_r, seed=2)
            report['seed_stability'][task] = result
            say(f'  {task:12} independent seeds: {fmt(result)}')

        sweep_samples = saltelli.sample(PROBLEM, sweep_n, calc_second_order=False)
        say(f'\nRange sweep: N={sweep_n}, R={sweep_r}; common random numbers across configurations')
        for motor in (0.025, 0.05, 0.075, 0.10):
            for heading in (0.0, 1.5e-4, 7.5e-4):
                overrides = {'MOTOR_GAIN_HALF_RANGE': motor, 'THERMAL_HEADING_DRIFT_MAX': heading}
                entry = {'MOTOR_GAIN_HALF_RANGE': motor, 'THERMAL_HEADING_DRIFT_MAX': heading, 'tasks': {}}
                for task in ('parking_hdg', 'transit'):
                    outputs = evaluate(pool, sweep_samples, task, overrides, sweep_r, 20260925)
                    entry['tasks'][task] = analyze(sweep_samples, outputs, sweep_r, seed=3)
                    s1 = entry['tasks'][task]['S1']
                    entry['tasks'][task]['top_source_S1'] = max(s1, key=s1.get)
                    say(f'  motor +/-{motor:.3f} heading {heading:.1e} {task:12} {fmt(entry["tasks"][task])}')
                report['range_sweep']['configs'].append(entry)

    report['runtime_sec'] = time.time() - started
    (out_dir / 'replicated_sobol_results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'run_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')
    say(f'\nSaved to {out_dir} in {report["runtime_sec"] / 60:.1f} min')


if __name__ == '__main__':
    main()
