"""
Scenario B: Motor Variability Analysis
Analyzes ResearchLogger CSVs for motor response at different power levels.
Extracts what data exists and identifies gaps.
"""
import pandas as pd
import numpy as np
from scipy import stats
import os, glob, json, re

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
                # Parse metadata from comments
                for key in ['CommandedPower', 'MotorNames', 'ExperimentLabel', 'TestDurationSec']:
                    if key + '=' in line:
                        val = line.split(key + '=')[1].strip()
                        metadata[key] = val
                continue
            if line:
                rows.append(line.split(','))
    if not headers:
        raise ValueError(f"No header found in {filepath}")
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
    return df, metadata

def parse_filename(fname):
    """Extract motor name and power level from filename."""
    # ResearchLogger_B_2_Motor_FL_Power0p2.csv
    m = re.search(r'Motor_(\w+)_Power(\d+)p(\d+)', fname)
    if m:
        motor = m.group(1)
        power = float(f"{m.group(2)}.{m.group(3)}")
        return motor, power
    return None, None

def main():
    print("=" * 80)
    print("SCENARIO B: MOTOR VARIABILITY ANALYSIS")
    print("=" * 80)

    files = sorted(glob.glob(os.path.join(DATA_DIR, 'ResearchLogger_B_2_Motor_*.csv')))
    print(f"\nMotor test files found: {len(files)}")

    # Parse filenames and organize
    motors = set()
    powers = set()
    results = []

    for fpath in files:
        fname = os.path.basename(fpath)
        motor, power = parse_filename(fname)
        if motor is None:
            print(f"  WARNING: Could not parse {fname}")
            continue
        motors.add(motor)
        powers.add(power)

        df, meta = load_csv(fpath)
        # Skip first 5 rows (startup transient)
        df = df.iloc[5:]

        has_velocity_cols = 'MotorVelocityL' in df.columns
        pinpoint_x_changed = df['PinpointX'].std() > 0.1 if 'PinpointX' in df.columns else False
        heading_changed = df['HeadingRad'].std() > 0.001 if 'HeadingRad' in df.columns else False

        # Battery voltage under this specific motor load
        bat_v = df['BatteryV'].values
        loop_dt = df['LoopDeltaMs'].values
        loop_dt = loop_dt[loop_dt < 200]  # filter startup outliers

        # Compute velocity from Pinpoint position if possible
        has_usable_velocity = False
        velocity_derived = None
        if 'PinpointX' in df.columns and 'PinpointY' in df.columns:
            px = df['PinpointX'].values
            py = df['PinpointY'].values
            dx = np.diff(px)
            dy = np.diff(py)
            dist_mm = np.sqrt(dx**2 + dy**2)
            dt_s = np.diff(df.iloc[:, 0].values)  # timestamp diff
            dt_s = np.where(dt_s > 0, dt_s, np.nan)
            speed_mm_s = dist_mm / dt_s
            speed_mm_s = speed_mm_s[~np.isnan(speed_mm_s)]
            if np.mean(speed_mm_s) > 1.0:  # meaningful motion
                has_usable_velocity = True
                velocity_derived = speed_mm_s

        result = {
            'file': fname,
            'motor': motor,
            'power': power,
            'n_samples': len(df),
            'duration_sec': float(df.iloc[-1, 0] - df.iloc[0, 0]) if len(df) > 1 else 0,
            'has_motor_velocity_cols': has_velocity_cols,
            'pinpoint_position_changed': bool(pinpoint_x_changed),
            'heading_changed': bool(heading_changed),
            'has_usable_derived_velocity': has_usable_velocity,
            'battery_mean_V': float(np.mean(bat_v)),
            'battery_std_V': float(np.std(bat_v)),
            'battery_min_V': float(np.min(bat_v)),
            'battery_max_V': float(np.max(bat_v)),
            'loop_mean_ms': float(np.mean(loop_dt)),
            'loop_std_ms': float(np.std(loop_dt)),
        }

        if has_usable_velocity:
            result['velocity_mean_mm_s'] = float(np.mean(velocity_derived))
            result['velocity_std_mm_s'] = float(np.std(velocity_derived))
            result['velocity_cv_pct'] = float(np.std(velocity_derived) / np.mean(velocity_derived) * 100)

        results.append(result)

    motors = sorted(motors)
    powers = sorted(powers)
    print(f"Motors: {motors}")
    print(f"Power levels: {powers}")

    # Print per-file results
    print(f"\n{'File':<45} {'Motor':>5} {'Pwr':>4} {'N':>5} {'BatV':>7} {'BatSD':>6} {'LoopMs':>7} {'HasVel':>7} {'PinMove':>8}")
    print("-" * 110)
    for r in results:
        print(f"{r['file']:<45} {r['motor']:>5} {r['power']:>4.1f} {r['n_samples']:>5} "
              f"{r['battery_mean_V']:>7.3f} {r['battery_std_V']:>6.3f} {r['loop_mean_ms']:>7.2f} "
              f"{'YES' if r['has_usable_derived_velocity'] else 'NO':>7} "
              f"{'YES' if r['pinpoint_position_changed'] else 'NO':>8}")

    # Battery voltage vs power level analysis
    print("\n" + "=" * 80)
    print("BATTERY VOLTAGE vs POWER LEVEL (Motor Load Effect)")
    print("=" * 80)
    print(f"\n{'Motor':<6} {'Power':>6} {'Mean V':>8} {'Std V':>7} {'Min V':>7} {'Max V':>7} {'V Drop from 0.2':>16}")
    print("-" * 60)
    baseline_v = {}
    for motor in motors:
        motor_results = [r for r in results if r['motor'] == motor]
        motor_results.sort(key=lambda x: x['power'])
        if motor_results:
            baseline_v[motor] = motor_results[0]['battery_mean_V']
        for r in motor_results:
            v_drop = baseline_v.get(motor, r['battery_mean_V']) - r['battery_mean_V']
            print(f"{r['motor']:<6} {r['power']:>6.1f} {r['battery_mean_V']:>8.3f} {r['battery_std_V']:>7.3f} "
                  f"{r['battery_min_V']:>7.3f} {r['battery_max_V']:>7.3f} {v_drop:>+16.3f}")

    # Loop timing comparison across power levels
    print("\n" + "=" * 80)
    print("LOOP TIMING vs POWER LEVEL (Computational Load Effect)")
    print("=" * 80)
    print(f"\n{'Motor':<6} {'Power':>6} {'Mean dt':>8} {'Std dt':>7}")
    print("-" * 30)
    for motor in motors:
        motor_results = sorted([r for r in results if r['motor'] == motor], key=lambda x: x['power'])
        for r in motor_results:
            print(f"{r['motor']:<6} {r['power']:>6.1f} {r['loop_mean_ms']:>8.2f} {r['loop_std_ms']:>7.2f}")

    # Critical assessment of motor velocity data
    print("\n" + "=" * 80)
    print("MOTOR VELOCITY DATA ASSESSMENT")
    print("=" * 80)
    has_vel = sum(1 for r in results if r['has_motor_velocity_cols'])
    has_derived = sum(1 for r in results if r['has_usable_derived_velocity'])
    has_motion = sum(1 for r in results if r['pinpoint_position_changed'])

    print(f"\n  Files with MotorVelocity columns: {has_vel}/{len(results)}")
    print(f"  Files with Pinpoint position change: {has_motion}/{len(results)}")
    print(f"  Files with usable derived velocity: {has_derived}/{len(results)}")

    if has_derived == 0 and has_vel == 0:
        print("\n  *** CRITICAL GAP: No motor velocity data available ***")
        print("  The ResearchLogger does not log motor encoder velocity (getVelocity()).")
        print("  The robot appears to be on blocks (Pinpoint position unchanged).")
        print("  Therefore, there is NO way to determine actual motor output velocity")
        print("  from these CSV files.")
        print("")
        print("  What IS available from motor variability experiments:")
        print("    - Battery voltage under different motor loads (power levels)")
        print("    - Loop timing during motor operation")
        print("    - Confirmation that 4 motors at 5 power levels were tested")
        print("")
        print("  What is MISSING for the paper:")
        print("    - Actual velocity (ticks/sec) at each commanded power level")
        print("    - Motor-to-motor velocity differences at the same power")
        print("    - Within-motor velocity variance at constant power")
        print("    - The commanded-vs-actual gain ratio (θ_M) for the forward model")

    # Summary
    print("\n" + "=" * 80)
    print("SCENARIO B ASSESSMENT")
    print("=" * 80)
    print(f"  Data collected: {len(results)} files across {len(motors)} motors × {len(powers)} power levels")
    print(f"  Total rows: {sum(r['n_samples'] for r in results)}")
    if has_derived > 0:
        print(f"  Motor velocity data: AVAILABLE (derived from Pinpoint)")
        print(f"  θ_M can be characterized: YES")
    else:
        print(f"  Motor velocity data: MISSING")
        print(f"  θ_M characterization: BLOCKED - need to re-collect with encoder velocity logging")
        print(f"  Usable secondary data: battery voltage vs load (can support θ_B characterization)")

    # Save JSON
    summary = {
        'scenario': 'B_Motor_Variability',
        'total_files': len(results),
        'motors': motors,
        'powers': powers,
        'velocity_data_available': has_derived > 0,
        'velocity_columns_present': has_vel > 0,
        'critical_gap': has_derived == 0 and has_vel == 0,
        'per_file': results,
    }
    with open(os.path.join(DATA_DIR, 'motor_analysis_results.json'), 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n  Results saved to: motor_analysis_results.json")

if __name__ == '__main__':
    main()
