"""Checks behind three non-obvious findings; raw logs and forward_model.py are only read.

1. Shooter logs: the logged right/left power ratio depends on whether the target speed
   is a multiple of the 20 TPS velocity resolution.
2. Forward model: sideways error from motor mismatch grows with drive power, while
   error from a time-growing heading bias shrinks with speed (closed-form vs simulation).
3. Forward model: turn heading error is set by the stop rule (tolerance minus a random
   fraction of one loop's rotation step).
"""
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
import forward_model as fm  # noqa: E402
from mismatch_impact import offset_for  # noqa: E402
from replicated_sobol import sha256  # noqa: E402

LOGS = ROOT / 'Data' / 'prev_logs'
TPS_STEP = 20.0


def load_shooter(path):
    with open(path) as fh:
        head = [h.strip() for h in fh.readline().lstrip('#').strip().split(',')]
        kp = float(re.search(r'kP=([\d.Ee-]+)', fh.readline()).group(1))
    d = pd.read_csv(path, comment='#', names=head, on_bad_lines='skip')
    return d.apply(pd.to_numeric, errors='coerce').dropna(), kp


def shooter_windows():
    windows, integral, zero = [], [], {True: np.zeros(3), False: np.zeros(3)}
    files = [p for p in sorted(LOGS.glob('*ShooterLog*.csv')) if 'raw' not in p.name]  # "- raw" duplicates another log
    used = []
    for path in files:
        d, kp = load_shooter(path)
        if len(d) < 200:
            continue
        used.append(path.name)
        d['seg'] = d.TargetTPS.ne(d.TargetTPS.shift()).cumsum()
        ok = (d.RightError.abs() <= 40) & (d.LeftError.abs() <= 40)
        ok &= d.RightPower.between(0.02, 0.98) & d.LeftPower.between(0.02, 0.98)
        if 'IsShooting' in d:
            ok &= d.IsShooting == 0
        s = d[ok]
        for side in ('Right', 'Left'):
            integral.append((s[side + 'PowerPID'] - kp * s[side + 'Error']).to_numpy())
        for grid, g in s.groupby(s.TargetTPS % TPS_STEP == 0):
            zero[grid] += [(g.RightError == 0).sum(), (g.LeftError == 0).sum(), len(g)]
        for _, g in s.groupby('seg'):
            gt = g.iloc[:, 0]
            if gt.iloc[-1] - gt.iloc[0] < 2.0:
                continue
            g = g[gt > gt.iloc[0] + 1.0]  # skip the first second after a target change
            if len(g) < 300:
                continue
            speed_r = g.TargetTPS.iloc[0] - g.RightError.mean()
            speed_l = g.TargetTPS.iloc[0] - g.LeftError.mean()
            windows.append({
                'file': path.name, 'target': float(g.TargetTPS.iloc[0]), 'rows': len(g),
                'on_grid': bool(g.TargetTPS.iloc[0] % TPS_STEP == 0),
                'ratio': float((g.RightPower.mean() / speed_r) / (g.LeftPower.mean() / speed_l)),
                'error_sd_right': float(g.RightError.std()), 'error_sd_left': float(g.LeftError.std()),
                'error_corr': float(np.corrcoef(g.RightError, g.LeftError)[0, 1]),
            })
    w = pd.DataFrame(windows)
    groups = {}
    for grid, g in w.groupby('on_grid'):
        groups['target_multiple_of_20' if grid else 'target_between_steps'] = {
            'windows': len(g), 'rows': int(g.rows.sum()), 'targets': sorted(g.target.unique().tolist()),
            'ratio_weighted_mean': float(np.average(g.ratio, weights=g.rows)),
            'ratio_median': float(g.ratio.median()),
            'ratio_iqr': [float(g.ratio.quantile(0.25)), float(g.ratio.quantile(0.75))],
            'error_sd_tps': float(np.average((g.error_sd_right + g.error_sd_left) / 2, weights=g.rows)),
            'error_corr_right_left': float(np.average(g.error_corr, weights=g.rows)),
            'share_rows_error_exactly_zero': float(zero[grid][:2].sum() / (2 * zero[grid][2])),
        }
    i = np.concatenate(integral)
    return {
        'files_used': used,
        'configured_kV_ratio': 3.8e-4 / 3.2e-4,
        'quantization_only_error_sd_tps': TPS_STEP / np.sqrt(12),
        'groups': groups,
        'implied_integral_term': {'min': float(i.min()), 'max': float(i.max()),
                                  'percentiles_5_50_95': np.percentile(i, [5, 50, 95]).tolist()},
        'windows': windows,
    }


def mean_error(task, power, dist, mismatch, theta_t, reps=24):
    saved = fm.MOTOR_GAIN_HALF_RANGE
    fm.TASKS['probe'] = dict(fm.TASKS[task], power=power, target_distance_mm=dist)
    fm.MOTOR_GAIN_HALF_RANGE = max(offset_for(mismatch), 1e-12)
    try:
        return float(np.mean([fm.simulate_task(0.5, 1.0, 0.5, theta_t, 'probe', rng=np.random.default_rng(i))
                              for i in range(reps)]))
    finally:
        fm.MOTOR_GAIN_HALF_RANGE = saved
        del fm.TASKS['probe']


def scaling_laws():
    vs = fm.map_battery(0.5)[0] / fm.NOMINAL_VOLTAGE
    rows = []
    for dist in (1000.0, 2000.0, 3000.0):
        for power in (0.2, 0.4, 0.6, 0.8):
            speed = power * fm.MAX_SPEED_MM_S * vs
            rows.append({
                'distance_mm': dist, 'power': power,
                'mismatch10_sim_mm': mean_error('tracking', power, dist, 10.0, 0.0),
                'mismatch10_pred_mm': offset_for(10.0) * power * dist / (fm.KP_HEADING * vs),
                'thermal_max_sim_mm': mean_error('tracking', power, dist, 0.0, 1.0),
                'thermal_max_pred_mm': fm.THERMAL_HEADING_DRIFT_MAX * dist ** 2 / (2 * speed),
            })
    return {'task': 'tracking (max lateral), theta_J=0.5, theta_B=0.5', 'rows': rows}


def turn_stop_rule(draws=2000, tolerance=0.02, min_power=0.10):
    rows = []
    for tj in (0.0, 0.5, 1.0):
        for tb in (0.0, 1.0):
            e = np.array([fm.simulate_task(tj, 0.5, tb, 0.0, 'turn', rng=np.random.default_rng(i)) for i in range(draws)])
            sigma, vs = fm.map_jitter(tj), fm.map_battery(tb)[0] / fm.NOMINAL_VOLTAGE
            dt = (fm.JITTER_LOC + fm.JITTER_SCALE * np.exp(sigma ** 2 / 2)) / 1000.0
            step = 2 * min_power * fm.MAX_SPEED_MM_S * vs / fm.TRACK_WIDTH_MM * dt
            rows.append({'theta_J': tj, 'theta_B': tb,
                         'sim_mean_deg': float(np.degrees(e.mean())), 'sim_sd_deg': float(np.degrees(e.std())),
                         'pred_mean_deg': float(np.degrees(tolerance - step / 2)),
                         'pred_sd_deg': float(np.degrees(step / np.sqrt(12))), 'step_deg': float(np.degrees(step))})
    return {'draws': draws, 'rows': rows}


def main():
    result = {
        'shooter': shooter_windows(),
        'scaling_laws': scaling_laws(),
        'turn_stop_rule': turn_stop_rule(),
        'provenance': {'forward_model.py': sha256(ROOT / 'forward_model.py'),
                       'counterintuitive_checks.py': sha256(Path(__file__))},
    }
    out = HERE / 'results' / 'counterintuitive_checks.json'
    out.write_text(json.dumps(result, indent=2))
    for name, g in result['shooter']['groups'].items():
        print(f"{name}: windows {g['windows']}, ratio weighted {g['ratio_weighted_mean']:.3f} median {g['ratio_median']:.3f} "
              f"IQR {g['ratio_iqr'][0]:.3f}-{g['ratio_iqr'][1]:.3f}, error exactly 0 in {g['share_rows_error_exactly_zero']:.1%} of rows, "
              f"error SD {g['error_sd_tps']:.2f} TPS, R-L corr {g['error_corr_right_left']:.2f}")
    print('implied integral term range:', result['shooter']['implied_integral_term'])
    for r in result['scaling_laws']['rows']:
        print(f"L={r['distance_mm']:.0f} P={r['power']}: mismatch {r['mismatch10_sim_mm']:.1f} (pred {r['mismatch10_pred_mm']:.1f}), "
              f"thermal {r['thermal_max_sim_mm']:.1f} (pred {r['thermal_max_pred_mm']:.1f})")
    for r in result['turn_stop_rule']['rows']:
        print(f"turn theta_J={r['theta_J']} theta_B={r['theta_B']}: sim {r['sim_mean_deg']:.3f}+/-{r['sim_sd_deg']:.3f}, "
              f"pred {r['pred_mean_deg']:.3f}+/-{r['pred_sd_deg']:.3f}")
    print('wrote', out)


if __name__ == '__main__':
    main()
