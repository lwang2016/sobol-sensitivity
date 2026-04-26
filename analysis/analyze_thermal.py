"""
Scenario D: Thermal Drift Analysis
Analyzes:
  D_1: Velocity/position drift during sustained motor operation (thermal effects on motors)
  D_2: Heading drift while stationary (IMU thermal bias)
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
    metadata = {}
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                if headers is None and 'Timestamp' in line:
                    headers = [h.strip() for h in line[2:].split(',')]
                for key in ['CommandedPower', 'CommandedPowerR', 'TestDurationSec', 'ExperimentLabel']:
                    if key + '=' in line:
                        metadata[key] = line.split(key + '=')[1].strip()
                continue
            if line:
                rows.append(line.split(','))
    df = pd.DataFrame(rows, columns=headers[:len(rows[0])])
    # Handle duplicate column names
    cols = df.columns.tolist()
    seen = {}
    new_cols = []
    for c in cols:
        if c in seen:
            seen[c] += 1
            new_cols.append(f"{c}_{seen[c]}")
        else:
            seen[c] = 0
            new_cols.append(c)
    df.columns = new_cols
    for col in df.columns:
        df[col] = pd.to_numeric(pd.Series(df[col].tolist()), errors='coerce')
    df = df.dropna(subset=[df.columns[0]])
    return df, metadata

def analyze_thermal_motor(df, label, meta):
    """Analyze D_1: motor performance drift during sustained operation."""
    df = df.iloc[5:].copy()  # skip startup
    
    timestamps = df.iloc[:, 0].values
    elapsed = timestamps - timestamps[0]
    
    voltage = df['BatteryV'].values
    heading = df['HeadingRad'].values
    px = df['PinpointX'].values
    py = df['PinpointY'].values
    
    # Compute instantaneous speed from position differentiation
    dx = np.diff(px)
    dy = np.diff(py)
    dist_mm = np.sqrt(dx**2 + dy**2)
    dt_s = np.diff(elapsed)
    dt_s = np.where(dt_s > 0.001, dt_s, np.nan)  # avoid div by zero
    speed_mm_s = dist_mm / dt_s
    speed_elapsed = (elapsed[:-1] + elapsed[1:]) / 2  # midpoints
    
    # Remove NaN and extreme outliers (> 5000 mm/s)
    valid = ~np.isnan(speed_mm_s) & (speed_mm_s < 5000)
    speed_mm_s = speed_mm_s[valid]
    speed_elapsed = speed_elapsed[valid]
    
    # Apply moving average to smooth (window = 50 samples ≈ 1 sec)
    window = min(50, len(speed_mm_s) // 10)
    if window > 2:
        speed_smooth = np.convolve(speed_mm_s, np.ones(window)/window, mode='valid')
        speed_smooth_t = speed_elapsed[window//2:window//2+len(speed_smooth)]
    else:
        speed_smooth = speed_mm_s
        speed_smooth_t = speed_elapsed
    
    # Segment into time windows (30-second buckets)
    bucket_sec = 30
    max_t = elapsed[-1]
    n_buckets = int(max_t / bucket_sec)
    time_buckets = []
    for i in range(n_buckets):
        t_start = i * bucket_sec
        t_end = (i + 1) * bucket_sec
        mask = (speed_elapsed >= t_start) & (speed_elapsed < t_end)
        if mask.sum() > 10:
            bucket_speeds = speed_mm_s[mask]
            time_buckets.append({
                'time_start': t_start,
                'time_end': t_end,
                'mean_speed_mm_s': float(np.mean(bucket_speeds)),
                'std_speed_mm_s': float(np.std(bucket_speeds)),
                'n_samples': int(mask.sum()),
            })
    
    # Compare first quarter speed vs last quarter speed
    q1_mask = speed_elapsed < max_t * 0.25
    q4_mask = speed_elapsed > max_t * 0.75
    q1_speed = speed_mm_s[q1_mask] if q1_mask.sum() > 10 else np.array([0])
    q4_speed = speed_mm_s[q4_mask] if q4_mask.sum() > 10 else np.array([0])
    
    # Linear trend fit on smoothed speed
    if len(speed_smooth) > 10:
        coeffs = np.polyfit(speed_smooth_t, speed_smooth, 1)
        speed_trend_slope = float(coeffs[0])  # mm/s per second of operation
    else:
        speed_trend_slope = 0.0
    
    # Heading analysis (does heading drift during sustained driving?)
    heading_unwrapped = np.unwrap(heading)
    heading_trend = np.polyfit(elapsed, heading_unwrapped, 1)
    heading_rate_deg_per_min = float(np.degrees(heading_trend[0]) * 60)
    
    result = {
        'label': label,
        'experiment': meta.get('ExperimentLabel', ''),
        'commanded_power_L': meta.get('CommandedPower', ''),
        'commanded_power_R': meta.get('CommandedPowerR', ''),
        'duration_sec': float(max_t),
        'n_samples': len(df),
        'voltage_start_V': float(np.mean(voltage[:20])),
        'voltage_end_V': float(np.mean(voltage[-20:])),
        'voltage_drop_V': float(np.mean(voltage[:20]) - np.mean(voltage[-20:])),
        'speed_q1_mean_mm_s': float(np.mean(q1_speed)),
        'speed_q4_mean_mm_s': float(np.mean(q4_speed)),
        'speed_drift_pct': float((np.mean(q1_speed) - np.mean(q4_speed)) / np.mean(q1_speed) * 100) if np.mean(q1_speed) > 0 else 0,
        'speed_trend_slope_mm_s2': speed_trend_slope,
        'heading_rate_deg_per_min': heading_rate_deg_per_min,
        'total_heading_change_deg': float(np.degrees(heading_unwrapped[-1] - heading_unwrapped[0])),
        'time_buckets': time_buckets,
    }
    
    return result

def analyze_heading_drift(df, label, meta):
    """Analyze D_2: IMU heading drift while stationary."""
    df = df.iloc[5:].copy()
    
    timestamps = df.iloc[:, 0].values
    elapsed = timestamps - timestamps[0]
    
    heading = df['HeadingRad'].values
    voltage = df['BatteryV'].values
    
    # Heading unwrap (handle wraparound)
    heading_unwrapped = np.unwrap(heading)
    
    # Overall drift
    total_drift_rad = heading_unwrapped[-1] - heading_unwrapped[0]
    total_drift_deg = np.degrees(total_drift_rad)
    duration = elapsed[-1]
    drift_rate_deg_per_min = total_drift_deg / (duration / 60)
    
    # Segment analysis (60-second windows)
    bucket_sec = 60
    n_buckets = int(duration / bucket_sec)
    time_buckets = []
    for i in range(n_buckets):
        t_start = i * bucket_sec
        t_end = (i + 1) * bucket_sec
        mask = (elapsed >= t_start) & (elapsed < t_end)
        if mask.sum() > 10:
            seg_heading = heading_unwrapped[mask]
            seg_drift = seg_heading[-1] - seg_heading[0]
            time_buckets.append({
                'time_start': t_start,
                'time_end': t_end,
                'drift_deg': float(np.degrees(seg_drift)),
                'drift_rate_deg_per_min': float(np.degrees(seg_drift) / (bucket_sec / 60)),
                'heading_std_rad': float(np.std(seg_heading)),
            })
    
    # Check if drift accelerates (compare first half vs second half rate)
    first_half = heading_unwrapped[elapsed < duration / 2]
    second_half = heading_unwrapped[elapsed >= duration / 2]
    if len(first_half) > 10 and len(second_half) > 10:
        first_half_rate = (first_half[-1] - first_half[0]) / (duration / 2)
        second_half_rate = (second_half[-1] - second_half[0]) / (duration / 2)
    else:
        first_half_rate = 0
        second_half_rate = 0
    
    # Noise around the trend (fit linear trend, measure residuals)
    coeffs = np.polyfit(elapsed, heading_unwrapped, 1)
    trend = np.polyval(coeffs, elapsed)
    residuals = heading_unwrapped - trend
    
    result = {
        'label': label,
        'experiment': meta.get('ExperimentLabel', ''),
        'duration_sec': float(duration),
        'n_samples': len(df),
        'heading_start_rad': float(heading_unwrapped[0]),
        'heading_start_deg': float(np.degrees(heading_unwrapped[0])),
        'heading_end_rad': float(heading_unwrapped[-1]),
        'heading_end_deg': float(np.degrees(heading_unwrapped[-1])),
        'total_drift_deg': total_drift_deg,
        'drift_rate_deg_per_min': drift_rate_deg_per_min,
        'drift_rate_first_half_deg_per_min': float(np.degrees(first_half_rate) * 60),
        'drift_rate_second_half_deg_per_min': float(np.degrees(second_half_rate) * 60),
        'heading_noise_std_deg': float(np.degrees(np.std(residuals))),
        'heading_noise_p95_deg': float(np.degrees(np.percentile(np.abs(residuals), 95))),
        'voltage_mean_V': float(np.mean(voltage)),
        'voltage_std_V': float(np.std(voltage)),
        'time_buckets': time_buckets,
    }
    
    return result

def main():
    print("=" * 80)
    print("SCENARIO D: THERMAL DRIFT ANALYSIS")
    print("=" * 80)

    # D_1: Motor thermal drift
    d1_files = sorted(glob.glob(os.path.join(DATA_DIR, 'ResearchLogger_D_1_*.csv')))
    d2_files = sorted(glob.glob(os.path.join(DATA_DIR, 'ResearchLogger_D_2_*.csv')))
    
    print(f"\nD_1 (motor thermal drift) files: {len(d1_files)}")
    print(f"D_2 (heading drift stationary) files: {len(d2_files)}")

    all_results = {'D1_motor': [], 'D2_heading': []}

    # ====== D_1: Motor Thermal Drift ======
    for fpath in d1_files:
        fname = os.path.basename(fpath)
        df, meta = load_csv(fpath)
        result = analyze_thermal_motor(df, fname, meta)
        all_results['D1_motor'].append(result)

        print(f"\n{'='*70}")
        print(f"D_1 MOTOR THERMAL DRIFT: {fname}")
        print(f"Power: L={result['commanded_power_L']}, R={result['commanded_power_R']}")
        print(f"Duration: {result['duration_sec']:.0f}s ({result['duration_sec']/60:.1f} min)")
        print(f"Samples: {result['n_samples']}")
        print(f"{'='*70}")

        print(f"\n  Voltage:")
        print(f"    Start: {result['voltage_start_V']:.3f}V → End: {result['voltage_end_V']:.3f}V")
        print(f"    Drop: {result['voltage_drop_V']:.3f}V")

        print(f"\n  Speed drift (motor thermal effect):")
        print(f"    Q1 (first 25%) mean speed: {result['speed_q1_mean_mm_s']:.1f} mm/s")
        print(f"    Q4 (last 25%) mean speed:  {result['speed_q4_mean_mm_s']:.1f} mm/s")
        print(f"    Speed reduction: {result['speed_drift_pct']:.1f}%")
        print(f"    Trend slope: {result['speed_trend_slope_mm_s2']:.4f} mm/s per second")

        print(f"\n  Heading drift during driving:")
        print(f"    Total: {result['total_heading_change_deg']:.1f}°")
        print(f"    Rate: {result['heading_rate_deg_per_min']:.2f} °/min")

        if result['time_buckets']:
            print(f"\n  Speed by 30-second windows:")
            for b in result['time_buckets'][:5]:  # first 5
                print(f"    {b['time_start']:.0f}-{b['time_end']:.0f}s: {b['mean_speed_mm_s']:.1f} ± {b['std_speed_mm_s']:.1f} mm/s")
            if len(result['time_buckets']) > 5:
                print(f"    ...")
                for b in result['time_buckets'][-3:]:  # last 3
                    print(f"    {b['time_start']:.0f}-{b['time_end']:.0f}s: {b['mean_speed_mm_s']:.1f} ± {b['std_speed_mm_s']:.1f} mm/s")

    # ====== D_2: Heading Drift Stationary ======
    for fpath in d2_files:
        fname = os.path.basename(fpath)
        df, meta = load_csv(fpath)
        result = analyze_heading_drift(df, fname, meta)
        all_results['D2_heading'].append(result)

        print(f"\n{'='*70}")
        print(f"D_2 HEADING DRIFT (STATIONARY): {fname}")
        print(f"Duration: {result['duration_sec']:.0f}s ({result['duration_sec']/60:.1f} min)")
        print(f"Samples: {result['n_samples']}")
        print(f"{'='*70}")

        print(f"\n  Starting heading: {result['heading_start_deg']:.3f}°")
        print(f"  Ending heading:   {result['heading_end_deg']:.3f}°")
        print(f"  Total drift:      {result['total_drift_deg']:.4f}°")
        print(f"  Drift rate:       {result['drift_rate_deg_per_min']:.4f} °/min")
        print(f"  First-half rate:  {result['drift_rate_first_half_deg_per_min']:.4f} °/min")
        print(f"  Second-half rate: {result['drift_rate_second_half_deg_per_min']:.4f} °/min")
        print(f"  Heading noise σ:  {result['heading_noise_std_deg']:.4f}°")
        print(f"  Heading noise P95: ±{result['heading_noise_p95_deg']:.4f}°")
        print(f"  Battery: {result['voltage_mean_V']:.3f} ± {result['voltage_std_V']:.3f} V")

        if result['time_buckets']:
            print(f"\n  Heading drift by 60-second windows:")
            for b in result['time_buckets']:
                print(f"    {b['time_start']:.0f}-{b['time_end']:.0f}s: drift={b['drift_deg']:.4f}° "
                      f"({b['drift_rate_deg_per_min']:.4f}°/min), noise σ={np.degrees(b['heading_std_rad']):.4f}°")

    # Summary assessment
    print("\n" + "=" * 80)
    print("SCENARIO D ASSESSMENT")
    print("=" * 80)

    # D_1 assessment
    if all_results['D1_motor']:
        r = all_results['D1_motor'][0]
        print(f"\n  D_1 Motor Thermal Drift:")
        print(f"    Duration: {r['duration_sec']:.0f}s ({r['duration_sec']/60:.1f} min)")
        print(f"    Speed change Q1→Q4: {r['speed_drift_pct']:.1f}%")
        speed_drift_detectable = abs(r['speed_drift_pct']) > 1.0
        print(f"    Thermal effect detectable: {'YES' if speed_drift_detectable else 'MARGINAL'}")
        print(f"    Paper requirement (5-10 min sustained): {'PASS' if r['duration_sec'] >= 300 else 'FAIL'}")

    # D_2 assessment
    if all_results['D2_heading']:
        r = all_results['D2_heading'][0]
        print(f"\n  D_2 Heading Drift (Stationary):")
        print(f"    Duration: {r['duration_sec']:.0f}s ({r['duration_sec']/60:.1f} min)")
        print(f"    Total drift: {r['total_drift_deg']:.4f}°")
        print(f"    Drift rate: {r['drift_rate_deg_per_min']:.4f} °/min")
        drift_detectable = abs(r['total_drift_deg']) > 0.1
        print(f"    IMU drift detectable: {'YES' if drift_detectable else 'MARGINAL/NO'}")
        print(f"    Paper requirement (5-10 min): {'PASS' if r['duration_sec'] >= 300 else 'FAIL'}")

    total_samples = sum(r['n_samples'] for rlist in all_results.values() for r in rlist)
    print(f"\n  Total samples: {total_samples}")
    print(f"  θ_T characterization:")
    if all_results['D1_motor']:
        r1 = all_results['D1_motor'][0]
        print(f"    Motor thermal: speed trend = {r1['speed_trend_slope_mm_s2']:.4f} mm/s²")
    if all_results['D2_heading']:
        r2 = all_results['D2_heading'][0]
        print(f"    IMU heading drift rate = {r2['drift_rate_deg_per_min']:.4f} °/min")
        print(f"    IMU heading noise σ = {r2['heading_noise_std_deg']:.4f}°")

    # Save JSON
    summary = {
        'scenario': 'D_Thermal_Drift',
        'D1_motor_files': len(all_results['D1_motor']),
        'D2_heading_files': len(all_results['D2_heading']),
        'total_samples': total_samples,
        'D1_results': all_results['D1_motor'],
        'D2_results': all_results['D2_heading'],
    }
    with open(os.path.join(DATA_DIR, 'thermal_analysis_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: thermal_analysis_results.json")

if __name__ == '__main__':
    main()
