"""
Scenario C: Battery Sag Analysis
Analyzes voltage trajectory over time under sustained motor load.
Characterizes θ_B distribution for the forward model.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, glob, json

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load_csv(filepath):
    """Load a ResearchLogger CSV, skipping comment lines. Returns df and metadata."""
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
    df = df.dropna(subset=['BatteryV'])
    return df, metadata

def analyze_battery(df, label, meta):
    """Analyze battery voltage trajectory."""
    # Skip first 2 rows (startup)
    df = df.iloc[2:].copy()
    
    timestamps = df.iloc[:, 0].values  # first Timestamp column (elapsed seconds from OpMode start)
    # Use second column (relative timestamp) if it seems more reasonable
    ts_col = df.columns[1]  # Second Timestamp column
    elapsed = df[ts_col].values if ts_col == 'Timestamp' else timestamps
    # Actually use the first numerical timestamp, compute relative
    elapsed = timestamps - timestamps[0]
    
    voltage = df['BatteryV'].values
    
    # Basic stats
    result = {
        'label': label,
        'experiment': meta.get('ExperimentLabel', ''),
        'commanded_power_L': meta.get('CommandedPower', ''),
        'commanded_power_R': meta.get('CommandedPowerR', ''),
        'duration_sec': float(elapsed[-1]),
        'n_samples': len(voltage),
        'voltage_start_V': float(np.mean(voltage[:20])),  # avg first 20 samples
        'voltage_end_V': float(np.mean(voltage[-20:])),    # avg last 20 samples
        'voltage_mean_V': float(np.mean(voltage)),
        'voltage_std_V': float(np.std(voltage)),
        'voltage_min_V': float(np.min(voltage)),
        'voltage_max_V': float(np.max(voltage)),
    }
    result['voltage_drop_V'] = result['voltage_start_V'] - result['voltage_end_V']
    result['voltage_drop_pct'] = result['voltage_drop_V'] / result['voltage_start_V'] * 100
    result['sag_rate_mV_per_min'] = result['voltage_drop_V'] / (result['duration_sec'] / 60) * 1000

    # Segment analysis: divide into quarters
    n = len(voltage)
    quarter = n // 4
    quarters = []
    for i, qname in enumerate(['Q1 (0-25%)', 'Q2 (25-50%)', 'Q3 (50-75%)', 'Q4 (75-100%)']):
        start = i * quarter
        end = (i + 1) * quarter if i < 3 else n
        seg_v = voltage[start:end]
        seg_t = elapsed[start:end]
        quarters.append({
            'quarter': qname,
            'time_range_sec': f"{seg_t[0]:.0f}-{seg_t[-1]:.0f}",
            'mean_V': float(np.mean(seg_v)),
            'std_V': float(np.std(seg_v)),
            'min_V': float(np.min(seg_v)),
            'max_V': float(np.max(seg_v)),
        })
    result['quarters'] = quarters

    # Voltage noise (high-frequency fluctuation) - detrend first
    # Fit a linear trend and look at residuals
    coeffs = np.polyfit(elapsed, voltage, 1)
    trend = np.polyval(coeffs, elapsed)
    residuals = voltage - trend
    result['trend_slope_V_per_sec'] = float(coeffs[0])
    result['residual_std_V'] = float(np.std(residuals))
    result['residual_p95_V'] = float(np.percentile(np.abs(residuals), 95))

    # Check for position/heading data (to correlate with motion)
    if 'PinpointX' in df.columns:
        px = df['PinpointX'].values
        py = df['PinpointY'].values
        total_distance_mm = float(np.sum(np.sqrt(np.diff(px)**2 + np.diff(py)**2)))
        result['total_distance_mm'] = total_distance_mm
    if 'HeadingRad' in df.columns:
        heading = df['HeadingRad'].values
        result['heading_start_rad'] = float(heading[0])
        result['heading_end_rad'] = float(heading[-1])
        result['heading_total_change_deg'] = float(np.degrees(heading[-1] - heading[0]))

    return result, elapsed, voltage, trend

def main():
    print("=" * 80)
    print("SCENARIO C: BATTERY SAG ANALYSIS")
    print("=" * 80)

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'ResearchLogger_C*.csv')))
    print(f"\nBattery sag files found: {len(files)}")

    all_results = []

    for fpath in files:
        fname = os.path.basename(fpath)
        df, meta = load_csv(fpath)
        result, elapsed, voltage, trend = analyze_battery(df, fname, meta)
        all_results.append(result)

        print(f"\n{'='*60}")
        print(f"FILE: {fname}")
        print(f"Experiment: {result['experiment']}")
        print(f"Power: L={result['commanded_power_L']}, R={result['commanded_power_R']}")
        print(f"Duration: {result['duration_sec']:.1f}s, Samples: {result['n_samples']}")
        print(f"{'='*60}")

        print(f"\n  Voltage trajectory:")
        print(f"    Start:  {result['voltage_start_V']:.3f} V")
        print(f"    End:    {result['voltage_end_V']:.3f} V")
        print(f"    Drop:   {result['voltage_drop_V']:.3f} V ({result['voltage_drop_pct']:.1f}%)")
        print(f"    Rate:   {result['sag_rate_mV_per_min']:.1f} mV/min")
        print(f"    Range:  [{result['voltage_min_V']:.3f}, {result['voltage_max_V']:.3f}] V")

        print(f"\n  Voltage by quarter:")
        for q in result['quarters']:
            print(f"    {q['quarter']}: Mean={q['mean_V']:.3f}V, Std={q['std_V']:.3f}V, "
                  f"Range=[{q['min_V']:.3f}, {q['max_V']:.3f}]")

        print(f"\n  Trend analysis:")
        print(f"    Linear slope: {result['trend_slope_V_per_sec']*1000:.3f} mV/sec")
        print(f"    Residual noise (σ): {result['residual_std_V']*1000:.1f} mV")
        print(f"    Residual P95: ±{result['residual_p95_V']*1000:.1f} mV")

        if 'total_distance_mm' in result:
            print(f"\n  Motion data:")
            print(f"    Total distance: {result['total_distance_mm']:.0f} mm ({result['total_distance_mm']/1000:.2f} m)")
        if 'heading_total_change_deg' in result:
            print(f"    Heading change: {result['heading_total_change_deg']:.1f}°")

    # Cross-file comparison
    print("\n" + "=" * 80)
    print("CROSS-FILE COMPARISON")
    print("=" * 80)
    print(f"\n{'Experiment':<30} {'Start V':>8} {'End V':>8} {'Drop V':>8} {'Drop%':>6} {'mV/min':>8} {'Duration':>9}")
    print("-" * 80)
    for r in all_results:
        print(f"{r['experiment']:<30} {r['voltage_start_V']:>8.3f} {r['voltage_end_V']:>8.3f} "
              f"{r['voltage_drop_V']:>8.3f} {r['voltage_drop_pct']:>5.1f}% {r['sag_rate_mV_per_min']:>8.1f} "
              f"{r['duration_sec']:>8.0f}s")

    # Characterize the voltage-vs-time relationship for θ_B
    print("\n" + "=" * 80)
    print("θ_B CHARACTERIZATION FOR FORWARD MODEL")
    print("=" * 80)
    all_slopes = [r['trend_slope_V_per_sec'] for r in all_results]
    all_residual_std = [r['residual_std_V'] for r in all_results]
    print(f"\n  Voltage sag slopes (V/sec): {[f'{s*1000:.2f} mV/s' for s in all_slopes]}")
    print(f"  Mean slope: {np.mean(all_slopes)*1000:.2f} mV/sec")
    print(f"  Residual noise (σ): {[f'{s*1000:.1f} mV' for s in all_residual_std]}")
    print(f"  Mean residual σ: {np.mean(all_residual_std)*1000:.1f} mV")

    # Summary assessment
    print("\n" + "=" * 80)
    print("SCENARIO C ASSESSMENT")
    print("=" * 80)
    total_samples = sum(r['n_samples'] for r in all_results)
    total_duration = sum(r['duration_sec'] for r in all_results)
    max_drop = max(r['voltage_drop_V'] for r in all_results)
    print(f"  Total samples: {total_samples} across {len(all_results)} runs")
    print(f"  Total duration: {total_duration:.0f} sec ({total_duration/60:.1f} min)")
    print(f"  Max voltage drop observed: {max_drop:.3f} V")
    print(f"  Sag is measurable: {'YES' if max_drop > 0.05 else 'NO'} (max drop = {max_drop:.3f} V)")
    print(f"  Multiple SOC levels tested: {'YES' if len(all_results) >= 2 else 'NO'} ({len(all_results)} runs)")
    print(f"  Data quality: voltage sampled at control-loop rate, ~{total_samples/total_duration:.0f} Hz")
    has_motion = any('total_distance_mm' in r and r['total_distance_mm'] > 100 for r in all_results)
    print(f"  Robot motion during tests: {'YES' if has_motion else 'UNCERTAIN'}")

    # Save JSON
    summary = {
        'scenario': 'C_Battery_Sag',
        'total_files': len(all_results),
        'total_samples': total_samples,
        'total_duration_sec': total_duration,
        'sag_slopes_V_per_sec': all_slopes,
        'residual_noise_V': all_residual_std,
        'per_file': all_results,
    }
    with open(os.path.join(DATA_DIR, 'battery_analysis_results.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to: battery_analysis_results.json")

if __name__ == '__main__':
    main()
