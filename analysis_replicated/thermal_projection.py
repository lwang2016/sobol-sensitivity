"""What the collected thermal data support about speed loss and turning drift at a given operating time.

Data: D_1 (600 s) and C1 (300 s) used the same command. Per ResearchLogger.java,
motor 0 (FL) received CommandedPower=0.5 and FR, BL, BR received 0.3. Speed and
turning are measured in 30 s windows (path length / elapsed time, logging clock).

Speed: log(speed) = run intercept + k*log(V) + g(t), with g(t) linear or
saturating, g(t) = -A*(1 - exp(-t/tau)), k fitted jointly (voltage and time are
nearly collinear within a run, so both runs are needed). Turning: yaw per
distance (ratio of window sums), log fitted the same way. Uncertainty: moving-
block bootstrap of window residuals within each run.

Outputs: results/thermal_projection.json.
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from Data.Motor_Variability.analyze_motor import load_csv  # noqa: E402
from Data.Thermal_Drift.analyze_thermal import analyze_thermal_motor  # noqa: E402

RUNS = {
    'D_1': ROOT / 'Data' / 'Thermal_Drift' / 'ResearchLogger_D_1_ThermalDrift_FullCharge.csv',
    'C1': ROOT / 'Data' / 'Battery_Sag' / 'ResearchLogger_C1_Full_Charge.csv',
}
TAUS = np.r_[np.arange(30, 600, 15), np.arange(600, 3001, 100)]
HORIZONS = (60, 120, 300, 600)


def windows(path, width=30.0):
    frame, meta = load_csv(path)
    summary = analyze_thermal_motor(frame, path.name, meta)
    frame = frame.iloc[5:]
    t = frame['Timestamp'].to_numpy() - frame['Timestamp'].iloc[0]
    step = np.hypot(np.diff(frame['PinpointX']), np.diff(frame['PinpointY']))
    turn = np.abs(np.diff(np.unwrap(frame['HeadingRad'].to_numpy())))
    rows = []
    for bucket in summary['time_buckets']:
        inside = (t[:-1] >= bucket['time_start']) & (t[1:] < bucket['time_end']) & (step / np.maximum(np.diff(t), 1e-9) < 5000)
        rows.append(((bucket['time_start'] + bucket['time_end']) / 2, bucket['mean_speed_mm_s'],
                     bucket['mean_voltage_V'], turn[inside].sum() / step[inside].sum()))
    return np.array(rows)


def design(run_ids, t, v, time_term, use_voltage=True):
    columns = [(run_ids == r).astype(float) for r in np.unique(run_ids)]
    if use_voltage:
        columns.append(np.log(v))
    columns.append(time_term)
    return np.column_stack(columns)


def fit(run_ids, t, v, y, model, use_voltage=True):
    """Least squares for linear or saturating time effect; returns g(t) function, SSE, parameters."""
    candidates = [None] if model == 'linear' else TAUS
    best = None
    for tau in candidates:
        term = t if tau is None else -(1 - np.exp(-t / tau))
        x = design(run_ids, t, v, term, use_voltage)
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        sse = float(np.sum((y - x @ beta) ** 2))
        if best is None or sse < best[1]:
            best = (tau, sse, beta)
    tau, sse, beta = best
    coefficient = beta[-1]
    if tau is None:
        g = lambda h: coefficient * h
    else:
        g = lambda h: -coefficient * (1 - np.exp(-h / tau))
    return g, sse, {'tau_s': None if tau is None else float(tau), 'time_coefficient': float(coefficient),
                    'voltage_exponent': float(beta[-2]) if use_voltage else None, 'n_params': x.shape[1]}


def block_bootstrap(run_ids, t, v, y, model, use_voltage, resamples=2000, block=3, seed=0):
    g, _, _ = fit(run_ids, t, v, y, model, use_voltage)
    tau = fit(run_ids, t, v, y, model, use_voltage)[2]['tau_s']
    term = t if tau is None else -(1 - np.exp(-t / tau))
    x = design(run_ids, t, v, term, use_voltage)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    fitted, residual = x @ beta, y - x @ beta
    rng = np.random.default_rng(seed)
    projections = []
    for _ in range(resamples):
        resampled = np.empty_like(residual)
        for r in np.unique(run_ids):
            index = np.where(run_ids == r)[0]
            blocks = [index[s:s + block] for s in range(len(index) - block + 1)]
            chosen = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(index) // block + 1)])[:len(index)]
            resampled[index] = residual[chosen]
        g_b, _, _ = fit(run_ids, t, v, fitted + resampled, model, use_voltage)
        projections.append([g_b(h) for h in HORIZONS])
    return np.percentile(np.array(projections), [5, 95], axis=0)


def analyze(outcome, run_ids, t, v, y, use_voltage):
    result = {}
    n = len(y)
    for model in ('linear', 'saturating'):
        g, sse, params = fit(run_ids, t, v, y, model, use_voltage)
        low, high = block_bootstrap(run_ids, t, v, y, model, use_voltage)
        k = params['n_params'] + (model == 'saturating')
        result[model] = {
            **params,
            'aic': float(n * np.log(sse / n) + 2 * k),
            'change_pct': {str(h): float(100 * (np.exp(g(h)) - 1)) for h in HORIZONS},
            'change_pct_90pct_block_bootstrap': {str(h): [float(100 * (np.exp(a) - 1)), float(100 * (np.exp(b) - 1))]
                                                 for h, a, b in zip(HORIZONS, low, high)},
        }
        print(f"{outcome:9} {model:10} tau={params['tau_s']} k={params['voltage_exponent']} AIC={result[model]['aic']:.1f}  "
              + '  '.join(f"{h}s: {result[model]['change_pct'][str(h)]:+.2f}% "
                          f"[{result[model]['change_pct_90pct_block_bootstrap'][str(h)][0]:+.2f}, "
                          f"{result[model]['change_pct_90pct_block_bootstrap'][str(h)][1]:+.2f}]" for h in HORIZONS), flush=True)
    return result


def main():
    data = {name: windows(path) for name, path in RUNS.items()}
    run_ids = np.concatenate([[name] * len(rows) for name, rows in data.items()])
    rows = np.vstack(list(data.values()))
    t, speed, v, turning = rows.T
    report = {
        'method': __doc__.strip(),
        'motor_commands': 'FL 0.5, FR/BL/BR 0.3 (only motor index 0 receives CommandedPower)',
        'temperature_rise_10min_F': {'FL': 85.2 - 77.4, 'FR': 69.3 - 67.4, 'BL': 71.0 - 68.8, 'BR': 73.3 - 69.1},
        'windows': {name: rows.tolist() for name, rows in data.items()},
        'within_run_time_voltage_corr': {name: float(np.corrcoef(rows[:, 0], rows[:, 2])[0, 1]) for name, rows in data.items()},
        'speed': analyze('speed', run_ids, t, v, np.log(speed), use_voltage=True),
        # A DC motor at fixed duty and load has speed elasticity to voltage of roughly 1-2.
        'speed_fixed_voltage_exponent': {str(k): analyze(f'speed k={k}', run_ids, t, v, np.log(speed) - k * np.log(v), use_voltage=False)
                                         for k in (1.0, 1.31, 1.5, 2.0)},
        'turning_per_distance': analyze('turning', run_ids, t, v, np.log(turning), use_voltage=True),
        'turning_per_distance_no_voltage_term': analyze('turning*', run_ids, t, v, np.log(turning), use_voltage=False),
    }
    (HERE / 'results').mkdir(exist_ok=True)
    (HERE / 'results' / 'thermal_projection.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
