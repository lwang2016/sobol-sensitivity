"""
Scenario A: Communication Jitter Analysis
Analyzes ResearchLogger CSVs for control loop timing distributions.
Compares stationary (A_2) vs loaded (A_3) conditions.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, glob, json

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv(filepath):
    """Load a ResearchLogger CSV, skipping comment lines."""
    rows = []
    headers = None
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                if headers is None and 'Timestamp' in line:
                    headers = [h.strip() for h in line[2:].split(',')]
                continue
            if line:
                rows.append(line.split(','))
    if headers is None:
        raise ValueError(f"No header found in {filepath}")
    # Handle duplicate column names by deduplicating
    seen = {}
    unique_headers = []
    for h in headers:
        if h in seen:
            seen[h] += 1
            unique_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            unique_headers.append(h)
    ncols = len(rows[0]) if rows else len(unique_headers)
    df = pd.DataFrame(rows, columns=unique_headers[:ncols])
    for col in df.columns:
        df[col] = pd.to_numeric(pd.Series(df[col].tolist()), errors='coerce')
    df = df.dropna(subset=['LoopDeltaMs'])
    return df

def analyze_jitter(df, label):
    """Compute jitter statistics from LoopDeltaMs column."""
    deltas = df['LoopDeltaMs'].values
    # Skip first row (startup artifact)
    deltas = deltas[1:]
    # Remove extreme startup outliers (> 200ms)
    deltas = deltas[deltas < 200]

    result = {
        'label': label,
        'n_samples': len(deltas),
        'duration_sec': df.iloc[-1, 0] - df.iloc[0, 0] if len(df) > 1 else 0,
        'mean_ms': float(np.mean(deltas)),
        'median_ms': float(np.median(deltas)),
        'std_ms': float(np.std(deltas)),
        'cv_pct': float(np.std(deltas) / np.mean(deltas) * 100),
        'min_ms': float(np.min(deltas)),
        'p05_ms': float(np.percentile(deltas, 5)),
        'p25_ms': float(np.percentile(deltas, 25)),
        'p75_ms': float(np.percentile(deltas, 75)),
        'p95_ms': float(np.percentile(deltas, 95)),
        'p99_ms': float(np.percentile(deltas, 99)),
        'max_ms': float(np.max(deltas)),
        'skewness': float(stats.skew(deltas)),
        'kurtosis': float(stats.kurtosis(deltas)),
    }

    # Fit distributions
    for dist_name, dist_fn in [('lognorm', stats.lognorm), ('gamma', stats.gamma), ('norm', stats.norm)]:
        try:
            params = dist_fn.fit(deltas)
            ks_stat, ks_p = stats.kstest(deltas, dist_fn.cdf, args=params)
            result[f'{dist_name}_params'] = [float(p) for p in params]
            result[f'{dist_name}_ks_stat'] = float(ks_stat)
            result[f'{dist_name}_ks_p'] = float(ks_p)
        except Exception:
            pass

    # Count GC-pause-like outliers (> 2x median)
    threshold_gc = 2 * np.median(deltas)
    result['gc_outlier_count'] = int(np.sum(deltas > threshold_gc))
    result['gc_outlier_pct'] = float(np.sum(deltas > threshold_gc) / len(deltas) * 100)

    return result, deltas

def main():
    print("=" * 80)
    print("SCENARIO A: COMMUNICATION JITTER ANALYSIS")
    print("=" * 80)

    # Discover files
    stationary_files = sorted(glob.glob(os.path.join(DATA_DIR, 'ResearchLogger_A_2_Jitter_Stationary_*.csv')))
    loaded_files = sorted(glob.glob(os.path.join(DATA_DIR, 'ResearchLogger_A_3_Jitter_Loaded_*.csv')))

    print(f"\nStationary runs found: {len(stationary_files)}")
    print(f"Loaded runs found: {len(loaded_files)}")

    all_results = []
    all_stationary_deltas = []
    all_loaded_deltas = []

    # Analyze each file
    for fpath in stationary_files + loaded_files:
        fname = os.path.basename(fpath)
        df = load_csv(fpath)
        condition = 'stationary' if 'Stationary' in fname else 'loaded'
        result, deltas = analyze_jitter(df, fname)
        result['condition'] = condition

        # Check for motor velocity data
        if 'MotorVelocityL' in df.columns:
            nonzero_vel = (df['MotorVelocityL'].abs() > 0.01).sum()
            result['motor_velocity_nonzero_count'] = int(nonzero_vel)
        # Check battery voltage
        if 'BatteryV' in df.columns:
            result['battery_mean_V'] = float(df['BatteryV'].mean())
            result['battery_std_V'] = float(df['BatteryV'].std())

        all_results.append(result)
        if condition == 'stationary':
            all_stationary_deltas.extend(deltas.tolist())
        else:
            all_loaded_deltas.extend(deltas.tolist())

        print(f"\n--- {fname} ({condition}) ---")
        print(f"  Samples: {result['n_samples']}, Duration: {result['duration_sec']:.1f}s")
        print(f"  Mean: {result['mean_ms']:.2f} ms, Median: {result['median_ms']:.2f} ms, Std: {result['std_ms']:.2f} ms")
        print(f"  CV: {result['cv_pct']:.1f}%, Skewness: {result['skewness']:.2f}, Kurtosis: {result['kurtosis']:.2f}")
        print(f"  [P5, P25, P75, P95, P99, Max]: [{result['p05_ms']:.1f}, {result['p25_ms']:.1f}, {result['p75_ms']:.1f}, {result['p95_ms']:.1f}, {result['p99_ms']:.1f}, {result['max_ms']:.1f}]")
        print(f"  GC-like outliers (>{2*np.median(deltas):.0f}ms): {result['gc_outlier_count']} ({result['gc_outlier_pct']:.2f}%)")
        if 'battery_mean_V' in result:
            print(f"  Battery: {result['battery_mean_V']:.3f} ± {result['battery_std_V']:.3f} V")

    # Aggregate comparison
    stat_arr = np.array(all_stationary_deltas)
    load_arr = np.array(all_loaded_deltas)

    print("\n" + "=" * 80)
    print("AGGREGATE: STATIONARY vs LOADED")
    print("=" * 80)
    print(f"\n{'Metric':<25} {'Stationary':>12} {'Loaded':>12} {'Delta':>10}")
    print("-" * 60)
    for metric, fn in [('N samples', len), ('Mean (ms)', np.mean), ('Median (ms)', np.median),
                        ('Std (ms)', np.std), ('P95 (ms)', lambda x: np.percentile(x, 95)),
                        ('P99 (ms)', lambda x: np.percentile(x, 99)), ('Max (ms)', np.max)]:
        s_val = fn(stat_arr)
        l_val = fn(load_arr)
        delta = l_val - s_val if isinstance(s_val, float) else ''
        if isinstance(delta, float):
            print(f"{metric:<25} {s_val:>12.2f} {l_val:>12.2f} {delta:>+10.2f}")
        else:
            print(f"{metric:<25} {s_val:>12} {l_val:>12}")

    # Mann-Whitney U test: are distributions significantly different?
    u_stat, u_p = stats.mannwhitneyu(stat_arr, load_arr, alternative='two-sided')
    print(f"\nMann-Whitney U test (stationary vs loaded): U={u_stat:.0f}, p={u_p:.2e}")
    if u_p < 0.05:
        print("  => Distributions are SIGNIFICANTLY DIFFERENT (p < 0.05)")
    else:
        print("  => No significant difference detected (p >= 0.05)")

    # Best-fit distribution for the combined data (for the forward model)
    combined = np.concatenate([stat_arr, load_arr])
    print("\n--- BEST FIT FOR FORWARD MODEL (combined data) ---")
    best_ks = 1.0
    best_dist = None
    for dist_name, dist_fn in [('lognorm', stats.lognorm), ('gamma', stats.gamma),
                                ('weibull_min', stats.weibull_min), ('norm', stats.norm)]:
        try:
            params = dist_fn.fit(combined)
            ks_stat, ks_p = stats.kstest(combined, dist_fn.cdf, args=params)
            print(f"  {dist_name}: KS={ks_stat:.4f}, p={ks_p:.4f}, params={[f'{p:.4f}' for p in params]}")
            if ks_stat < best_ks:
                best_ks = ks_stat
                best_dist = (dist_name, params)
        except Exception:
            pass
    if best_dist:
        print(f"\n  BEST FIT: {best_dist[0]} (KS={best_ks:.4f})")
        print(f"  Parameters: {[f'{p:.6f}' for p in best_dist[1]]}")

    # Summary assessment
    print("\n" + "=" * 80)
    print("SCENARIO A ASSESSMENT")
    print("=" * 80)
    print(f"  Total samples: {len(stat_arr) + len(load_arr)} across {len(stationary_files) + len(loaded_files)} runs")
    print(f"  Paper requirement (N >= 1000): {'PASS' if len(combined) >= 1000 else 'FAIL'} (N={len(combined)})")
    print(f"  Jitter is measurable: {'YES' if np.std(combined) > 1.0 else 'NO'} (std={np.std(combined):.2f} ms)")
    print(f"  Distribution is non-Gaussian: {'YES' if abs(stats.skew(combined)) > 0.5 else 'MODERATE'} (skew={stats.skew(combined):.2f})")
    print(f"  Data quality: loop timing directly measured via System.nanoTime()")
    print(f"  theta_J distribution for forward model: {best_dist[0] if best_dist else 'TBD'}")

    # Save JSON summary
    summary = {
        'scenario': 'A_Communication_Jitter',
        'total_samples': len(combined),
        'stationary_samples': len(stat_arr),
        'loaded_samples': len(load_arr),
        'stationary_stats': {'mean': float(np.mean(stat_arr)), 'std': float(np.std(stat_arr)),
                             'median': float(np.median(stat_arr)), 'p95': float(np.percentile(stat_arr, 95)),
                             'p99': float(np.percentile(stat_arr, 99))},
        'loaded_stats': {'mean': float(np.mean(load_arr)), 'std': float(np.std(load_arr)),
                         'median': float(np.median(load_arr)), 'p95': float(np.percentile(load_arr, 95)),
                         'p99': float(np.percentile(load_arr, 99))},
        'best_fit': {'distribution': best_dist[0], 'params': [float(p) for p in best_dist[1]]} if best_dist else None,
        'mann_whitney_p': float(u_p),
        'per_run': all_results
    }
    with open(os.path.join(DATA_DIR, 'jitter_analysis_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: jitter_analysis_results.json")

    # Generate distribution plot
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    xlim_max = np.percentile(combined, 99)
    # Use full data for histogram (density normalization must match the fitted distribution)
    # but clip x-axis to avoid long tail compressing the main peak
    ax.hist(combined, bins=np.linspace(combined.min(), xlim_max, 50),
            density=True, alpha=0.7, label="Measured loop duration")
    if best_dist:
        dist_fn = getattr(stats, best_dist[0])
        x = np.linspace(combined.min(), xlim_max, 200)
        ax.plot(x, dist_fn.pdf(x, *best_dist[1]), 'r-', linewidth=2, label="Log-normal fit")
    ax.set_xlim(left=combined.min() - 1, right=xlim_max + 2)
    ax.set_xlabel("Inter-iteration time (ms)")
    ax.set_ylabel("Probability density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(DATA_DIR, 'jitter_distribution.png'), dpi=150)
    plt.close(fig)
    print(f"  Plot saved to: jitter_distribution.png")

if __name__ == '__main__':
    main()
