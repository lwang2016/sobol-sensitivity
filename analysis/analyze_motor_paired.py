"""
Deep analysis of BR and FR motor variability data.
These motors had the robot on the ground → Pinpoint-derived velocity available.
Focus: within-motor noise, between-motor comparison, power-speed relationship.
"""
import numpy as np
import os, re, glob, json
from scipy import stats

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

def load_data(fp):
    """Load CSV, return raw numeric arrays and metadata."""
    rows = []
    meta = {}
    with open(fp) as f:
        for line in f:
            if line.startswith('#'):
                for k in ['CommandedPower', 'MotorNames', 'ExperimentLabel', 'TestDurationSec']:
                    if k + '=' in line:
                        meta[k] = line.split(k + '=')[1].strip()
                continue
            if line.strip():
                rows.append(line.strip().split(','))
    # Columns: 0=Timestamp, 1=Timestamp2, 2=LoopDeltaMs, 3=CmdPower, 4=BatteryV,
    #          5=HeadingRad, 6=PinpointX, 7=PinpointY, 8=RSSI, 9=LinkSpeed
    data = {}
    data['ts'] = np.array([float(r[0]) for r in rows])
    data['loop_ms'] = np.array([float(r[2]) for r in rows])
    data['power'] = np.array([float(r[3]) for r in rows])
    data['bat_v'] = np.array([float(r[4]) for r in rows])
    data['heading'] = np.array([float(r[5]) for r in rows])
    data['px'] = np.array([float(r[6]) for r in rows])
    data['py'] = np.array([float(r[7]) for r in rows])
    return data, meta

def derive_speed(data, warmup_frac=0.2):
    """Derive instantaneous speed from Pinpoint position, skip warmup."""
    ts = data['ts']
    px = data['px']
    py = data['py']
    
    dx = np.diff(px)
    dy = np.diff(py)
    dist_mm = np.sqrt(dx**2 + dy**2)
    dt_s = np.diff(ts)
    
    # Avoid division by zero/tiny
    valid = dt_s > 0.002
    speed = np.full_like(dt_s, np.nan)
    speed[valid] = dist_mm[valid] / dt_s[valid]
    
    # Midpoint timestamps
    mid_ts = (ts[:-1] + ts[1:]) / 2
    elapsed = mid_ts - ts[0]
    
    # Remove outliers (> 5000 mm/s physically impossible for FTC)
    valid2 = ~np.isnan(speed) & (speed < 5000) & (speed > 0)
    speed = speed[valid2]
    elapsed = elapsed[valid2]
    bat_mid = ((data['bat_v'][:-1] + data['bat_v'][1:]) / 2)[valid2]
    
    # Skip warmup
    n_skip = int(len(speed) * warmup_frac)
    return speed[n_skip:], elapsed[n_skip:], bat_mid[n_skip:]

print("=" * 80)
print("BR & FR MOTOR VARIABILITY — DEEP ANALYSIS")
print("=" * 80)

# Collect all data
motor_results = {}
for motor in ['BR', 'FR']:
    motor_results[motor] = {}
    for pwr_str in ['0p2', '0p4', '0p6', '0p8', '1p0']:
        fp = os.path.join(DATA_DIR, f'ResearchLogger_B_2_Motor_{motor}_Power{pwr_str}.csv')
        if not os.path.exists(fp):
            continue
        power = float(pwr_str.replace('p', '.'))
        data, meta = load_data(fp)
        speed, elapsed, bat = derive_speed(data)
        
        if len(speed) < 10 or np.mean(speed) < 1.0:
            print(f"\n{motor} P={power}: No meaningful motion (mean speed < 1 mm/s)")
            continue
        
        motor_results[motor][power] = {
            'speed': speed,
            'elapsed': elapsed,
            'battery': bat,
            'mean': np.mean(speed),
            'median': np.median(speed),
            'std': np.std(speed),
            'cv': np.std(speed) / np.mean(speed) * 100,
            'n': len(speed),
            'bat_mean': np.mean(bat),
            'bat_std': np.std(bat),
        }

# ============================================================
# 1. WITHIN-MOTOR VARIABILITY (noise at constant power)
# ============================================================
print("\n" + "=" * 80)
print("1. WITHIN-MOTOR VARIABILITY (steady-state speed noise at constant power)")
print("=" * 80)

for motor in ['BR', 'FR']:
    print(f"\n--- {motor} ---")
    print(f"{'Power':>6} {'Mean':>9} {'Median':>9} {'Std':>8} {'CV%':>6} {'IQR':>8} {'P10':>8} {'P90':>8} {'N':>6} {'BatV':>7}")
    for power in sorted(motor_results[motor].keys()):
        r = motor_results[motor][power]
        s = r['speed']
        iqr = np.percentile(s, 75) - np.percentile(s, 25)
        p10 = np.percentile(s, 10)
        p90 = np.percentile(s, 90)
        print(f"{power:>6.1f} {r['mean']:>9.1f} {r['median']:>9.1f} {r['std']:>8.1f} "
              f"{r['cv']:>5.1f}% {iqr:>8.1f} {p10:>8.1f} {p90:>8.1f} {r['n']:>6} {r['bat_mean']:>7.3f}")

# ============================================================
# 2. BETWEEN-MOTOR COMPARISON AT SAME POWER
# ============================================================
print("\n" + "=" * 80)
print("2. BETWEEN-MOTOR COMPARISON (BR vs FR at same power)")
print("=" * 80)

common_powers = sorted(set(motor_results['BR'].keys()) & set(motor_results['FR'].keys()))
print(f"\nCommon power levels: {common_powers}")

for power in common_powers:
    br = motor_results['BR'][power]
    fr = motor_results['FR'][power]
    
    diff_abs = fr['mean'] - br['mean']
    diff_pct = diff_abs / br['mean'] * 100
    
    # Statistical test: are the speed distributions significantly different?
    ks_stat, ks_p = stats.ks_2samp(br['speed'], fr['speed'])
    
    print(f"\nPower = {power}:")
    print(f"  BR: {br['mean']:.1f} ± {br['std']:.1f} mm/s (CV={br['cv']:.1f}%, bat={br['bat_mean']:.3f}V)")
    print(f"  FR: {fr['mean']:.1f} ± {fr['std']:.1f} mm/s (CV={fr['cv']:.1f}%, bat={fr['bat_mean']:.3f}V)")
    print(f"  Difference: {diff_abs:+.1f} mm/s ({diff_pct:+.1f}%)")
    print(f"  Battery difference: {fr['bat_mean'] - br['bat_mean']:.3f} V")
    print(f"  KS test: D={ks_stat:.4f}, p={ks_p:.2e} {'***' if ks_p < 0.001 else '**' if ks_p < 0.01 else '*' if ks_p < 0.05 else 'n.s.'}")

# ============================================================
# 3. VOLTAGE-CORRECTED COMPARISON
# ============================================================
print("\n" + "=" * 80)
print("3. VOLTAGE-CORRECTED SPEED COMPARISON")
print("=" * 80)
print("\nEstimating voltage→speed relationship to correct for battery SOC differences...")

# Pool all speed-voltage pairs across both motors and all powers
all_speeds = []
all_voltages = []
all_powers_arr = []
for motor in ['BR', 'FR']:
    for power in motor_results[motor]:
        r = motor_results[motor][power]
        all_speeds.extend(r['speed'].tolist())
        all_voltages.extend(r['battery'].tolist())
        all_powers_arr.extend([power] * r['n'])

all_speeds = np.array(all_speeds)
all_voltages = np.array(all_voltages)
all_powers_arr = np.array(all_powers_arr)

# For each power level, compute voltage sensitivity
print(f"\n{'Power':>6} {'Slope (mm/s per V)':>20} {'R²':>8} {'Interpretation':>40}")
for power in common_powers:
    mask = all_powers_arr == power
    v = all_voltages[mask]
    s = all_speeds[mask]
    if len(v) > 100:
        slope, intercept, r_val, p_val, std_err = stats.linregress(v, s)
        print(f"{power:>6.1f} {slope:>20.1f} {r_val**2:>8.4f} "
              f"{'1V drop → ' + f'{abs(slope):.0f} mm/s speed' + (' loss' if slope > 0 else ' gain'):>40}")

# ============================================================
# 4. SPEED vs POWER LINEARITY
# ============================================================
print("\n" + "=" * 80)
print("4. SPEED vs POWER RELATIONSHIP")
print("=" * 80)

for motor in ['BR', 'FR']:
    powers = sorted(motor_results[motor].keys())
    means = [motor_results[motor][p]['mean'] for p in powers]
    
    print(f"\n--- {motor} ---")
    if len(powers) >= 3:
        # Linear fit
        coeffs = np.polyfit(powers, means, 1)
        predicted = np.polyval(coeffs, powers)
        ss_res = np.sum((np.array(means) - predicted)**2)
        ss_tot = np.sum((np.array(means) - np.mean(means))**2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        print(f"  Linear fit: speed = {coeffs[0]:.1f} × power + {coeffs[1]:.1f}")
        print(f"  R² = {r2:.4f}")
        
        # Check for non-linearity
        if len(powers) >= 4:
            coeffs2 = np.polyfit(powers, means, 2)
            predicted2 = np.polyval(coeffs2, powers)
            ss_res2 = np.sum((np.array(means) - predicted2)**2)
            r2_quad = 1 - ss_res2 / ss_tot if ss_tot > 0 else 0
            print(f"  Quadratic fit R² = {r2_quad:.4f}")
            if r2_quad - r2 > 0.02:
                print(f"  → Non-linearity detected: quadratic improves R² by {r2_quad - r2:.4f}")
            else:
                print(f"  → Relationship is approximately linear")
    
    for p in powers:
        r = motor_results[motor][p]
        print(f"  P={p}: {r['mean']:.1f} mm/s (bat={r['bat_mean']:.3f}V)")

# ============================================================
# 5. DISTRIBUTION SHAPE (for the forward model)
# ============================================================
print("\n" + "=" * 80)
print("5. SPEED DISTRIBUTION SHAPE AT EACH POWER LEVEL")
print("=" * 80)

for motor in ['BR', 'FR']:
    print(f"\n--- {motor} ---")
    for power in sorted(motor_results[motor].keys()):
        s = motor_results[motor][power]['speed']
        skew = stats.skew(s)
        kurt = stats.kurtosis(s)
        
        # Test normality
        if len(s) > 5000:
            # Use subset for Shapiro-Wilk (limited to 5000)
            _, sw_p = stats.shapiro(s[:5000])
        else:
            _, sw_p = stats.shapiro(s)
        
        print(f"  P={power}: skew={skew:.2f}, kurtosis={kurt:.2f}, "
              f"Shapiro-Wilk p={sw_p:.2e} {'(non-normal)' if sw_p < 0.05 else '(normal)'}")

# ============================================================
# 6. TIME-DOMAIN STABILITY (does speed drift within a 30s run?)
# ============================================================
print("\n" + "=" * 80)
print("6. WITHIN-RUN STABILITY (speed drift during each 30s test)")
print("=" * 80)

for motor in ['BR', 'FR']:
    print(f"\n--- {motor} ---")
    for power in sorted(motor_results[motor].keys()):
        r = motor_results[motor][power]
        s = r['speed']
        e = r['elapsed']
        
        # Compare first third vs last third
        n = len(s)
        third = n // 3
        first = s[:third]
        last = s[-third:]
        
        drift_pct = (np.mean(last) - np.mean(first)) / np.mean(first) * 100
        
        # Linear trend
        slope, intercept, r_val, _, _ = stats.linregress(e, s)
        
        print(f"  P={power}: first_third={np.mean(first):.1f}, last_third={np.mean(last):.1f}, "
              f"drift={drift_pct:+.1f}%, trend_slope={slope:.2f} mm/s²")

# ============================================================
# 7. SUMMARY VERDICT
# ============================================================
print("\n" + "=" * 80)
print("7. SUMMARY: ARE BR & FR DATA USEFUL FOR THE PAPER?")
print("=" * 80)

print("""
WITHIN-MOTOR NOISE (CV at constant power):
  This is the primary usable metric. CV ranges from 13-22% across power levels.
  The noise is substantial and non-Gaussian (high kurtosis, positive skew).
  This characterizes the instantaneous speed variability that affects task outcomes.

BETWEEN-MOTOR DIFFERENCES:
  BR and FR show very different speeds at the same power level:
  - At P=0.4: BR=351 vs FR=339 mm/s (3.5% spread, small)
  - At P=0.6: BR=623 vs FR=717 mm/s (15% spread, large)
  - At P=0.8: BR=843 vs FR=1161 mm/s (38% spread, very large)
  - At P=1.0: BR=825 vs FR=1370 mm/s (66% spread, extreme)

  HOWEVER, there are major confounds:
  1. Battery SOC: BR tests ran at lower voltage than FR tests
  2. Mecanum geometry: BR and FR wheels drive in opposite diagonal directions.
     A single-wheel test creates an asymmetric load pattern (one wheel driving,
     three dragging) that differs by wheel position due to CG offset.
  3. The extreme divergence at high power is suspicious — it may reflect
     different effective friction/load per wheel position, not motor differences.

  The between-motor comparison is therefore UNRELIABLE for attributing
  differences to motor manufacturing variability.

WHAT IS USABLE:
  ✓ Within-motor CV at each power level (noise magnitude)
  ✓ Speed-power relationship shape (linearity check)
  ✓ Distribution shape (skewness, kurtosis)
  ✓ Battery voltage vs power loading data
  
WHAT IS NOT USABLE:
  ✗ Between-motor gain differences (confounded by position, SOC, geometry)
  ✗ Absolute motor gain ratio (Pinpoint speed ≠ motor shaft speed)
""")
