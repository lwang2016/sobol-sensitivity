"""Which severity settings change how repeatable each task is (Sobol indices of the run-to-run spread).

For each Saltelli sample the forward model runs R times. The replicates are split
into two independent halves; each half gives an estimate of log Var(Y | theta).
The two half-estimates act as two noisy replicates of the spread, so the same
noise-corrected estimator used for the mean (replicated_sobol.corrected_indices)
yields spread indices without assuming normality.

Outputs: results/spread_sobol_results.json and results/spread_run_log.txt.
"""
import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from SALib.sample import saltelli

from replicated_sobol import NAMES, PROBLEM, corrected_indices, evaluate, sha256

HERE = Path(__file__).resolve().parent


def binned_first_order(samples, halves, bins=20):
    """Given-data first-order indices from conditional means over bins of each input.

    Robust when outputs are noisy; the Saltelli first-order estimator is not.
    Estimation noise is removed from the total variance and from the bin means.
    """
    z = halves.mean(axis=1)
    residual = float(np.mean(halves.var(axis=1, ddof=1)) / 2)
    signal = np.var(z, ddof=1) - residual
    edges = np.linspace(0, 1, bins + 1)
    first = {}
    for i, name in enumerate(NAMES):
        index = np.clip(np.digitize(samples[:, i], edges[1:-1]), 0, bins - 1)
        counts = np.bincount(index, minlength=bins)
        means = np.bincount(index, weights=z, minlength=bins) / counts
        between = np.sum(counts * (means - z.mean()) ** 2) / len(z)
        within = np.sum([np.var(z[index == b], ddof=1) for b in range(bins)] * counts) / len(z)
        first[name] = float((between - within * bins / len(z)) / signal)
    return first, float(signal)


def spread_indices(outputs, samples=None, seed=0):
    """Sobol indices of log within-sample variance, from two independent replicate halves."""
    half = outputs.shape[1] // 2
    if half < 3:
        raise ValueError('Need at least 6 replicates per sample')
    variances = np.stack([outputs[:, :half].var(axis=1, ddof=1), outputs[:, half:2 * half].var(axis=1, ddof=1)], axis=1)
    if np.any(variances <= 0):
        raise ValueError('Zero within-sample variance: log spread undefined')
    log_var = np.log(variances)
    result = corrected_indices(log_var.mean(axis=1), log_var.var(axis=1, ddof=1), replicates=2, seed=seed)
    result['estimation_noise_share'] = result.pop('noise_share')
    result['S1_saltelli_unstable'] = result.pop('S1')
    result.pop('S1_conf')
    result.pop('S1_of_total_variance')
    if samples is not None:
        result['S1'], _ = binned_first_order(samples, log_var)
    result['log_var_halves'] = log_var
    return result


def binned_spread(samples, outputs, bins=5):
    """Mean within-sample SD in equal-width bins of each input, for an interpretable effect size."""
    sd = outputs.std(axis=1, ddof=1)
    edges = np.linspace(0, 1, bins + 1)
    return {name: [float(sd[(samples[:, i] >= lo) & (samples[:, i] <= hi if hi == 1 else samples[:, i] < hi)].mean())
                   for lo, hi in zip(edges[:-1], edges[1:])]
            for i, name in enumerate(NAMES)}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 2) - 2))
    parser.add_argument('--quick', action='store_true')
    args = parser.parse_args()
    n, r = (64, 8) if args.quick else (1024, 32)
    out_dir = HERE / ('results_quick' if args.quick else 'results')
    out_dir.mkdir(exist_ok=True)
    samples = saltelli.sample(PROBLEM, n, calc_second_order=False)
    report = {'method': __doc__.strip(), 'N': n, 'replicates': r, 'seed_base': 20260926,
              'provenance': {'this_script_sha256': sha256(__file__), 'forward_model_sha256': sha256(HERE.parent / 'forward_model.py')},
              'tasks': {}}
    log = []
    arrays = {'samples': samples}
    started = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for task in ('parking_hdg', 'turn', 'transit'):
            t0 = time.time()
            outputs = evaluate(pool, samples, task, {}, r, 20260926)
            result = spread_indices(outputs, samples, seed=4)
            arrays[f'{task}_log_var_halves'] = result.pop('log_var_halves')
            arrays[f'{task}_within_sd'] = outputs.std(axis=1, ddof=1)
            result['mean_within_sd_by_input_bin'] = binned_spread(samples, outputs)
            result['mean_within_sd'] = float(outputs.std(axis=1, ddof=1).mean())
            report['tasks'][task] = result
            s1, st = result['S1'], result['ST']
            line = (f'{task:12} spread S1(binned) J/M/B/T=' + '/'.join(f'{s1[k]:+.3f}' for k in NAMES)
                    + '  ST=' + '/'.join(f'{st[k]:.3f}' for k in NAMES)
                    + f"  estimation-noise share={result['estimation_noise_share']:.3f}  ({time.time() - t0:.0f}s)")
            print(line, flush=True)
            log.append(line)
            for name in NAMES:
                bins = result['mean_within_sd_by_input_bin'][name]
                line = f'    within-sample SD across {name} bins (low->high): ' + ', '.join(f'{v:.5g}' for v in bins)
                print(line, flush=True)
                log.append(line)
    report['runtime_sec'] = time.time() - started
    np.savez_compressed(out_dir / 'spread_per_sample.npz', **arrays)
    (out_dir / 'spread_sobol_results.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out_dir / 'spread_run_log.txt').write_text('\n'.join(log) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
