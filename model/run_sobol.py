"""
Sobol Sensitivity Analysis
===========================
Uses SALib to compute first-order (S_i) and total-order (S_Ti) Sobol indices
for each noise source across all task types.

Produces the noise-task criticality matrix — the paper's central result.
"""

import numpy as np
import json
import time
from SALib.sample import saltelli
from SALib.analyze import sobol

from forward_model import evaluate_batch, TASKS

# ============================================================
# PROBLEM DEFINITION
# ============================================================

problem = {
    'num_vars': 4,
    'names': ['theta_J', 'theta_M', 'theta_B', 'theta_T'],
    'bounds': [[0.0, 1.0]] * 4,
}

N = 1024  # Base sample size (paper specifies N=1024)
# With calc_second_order=False: N * (k + 2) = 1024 * 6 = 6144 evaluations per task

print("=" * 80)
print("SOBOL SENSITIVITY ANALYSIS")
print("=" * 80)
print(f"Base samples (N): {N}")
print(f"Parameters (k): {problem['num_vars']}")
print(f"Evaluations per task: {N * (problem['num_vars'] + 2)} (first-order only, no second-order)")
print(f"Tasks: {list(TASKS.keys())}")

# ============================================================
# GENERATE SALTELLI SAMPLES
# ============================================================

print("\nGenerating Saltelli quasi-random samples...")
param_values = saltelli.sample(problem, N, calc_second_order=False)
print(f"Sample matrix shape: {param_values.shape}")

# ============================================================
# EVALUATE FORWARD MODEL FOR EACH TASK
# ============================================================

results = {}
total_start = time.time()

for task_name in TASKS:
    print(f"\nEvaluating task: {task_name} ({param_values.shape[0]} runs)...", flush=True)
    t0 = time.time()
    
    Y = evaluate_batch(param_values, task_name, seed=42)
    
    dt = time.time() - t0
    print(f"  Completed in {dt:.1f}s ({dt/len(Y)*1000:.1f} ms/eval)")
    print(f"  Output: mean={np.mean(Y):.4f}, std={np.std(Y):.4f}, "
          f"min={np.min(Y):.4f}, max={np.max(Y):.4f}")
    
    # Check for degenerate output
    if np.std(Y) < 1e-10:
        print(f"  WARNING: Output variance is essentially zero — Sobol indices will be meaningless")
        results[task_name] = {
            'warning': 'degenerate_output',
            'Y_mean': float(np.mean(Y)),
            'Y_std': float(np.std(Y)),
        }
        continue
    
    # Run Sobol analysis
    print(f"  Computing Sobol indices...", flush=True)
    Si = sobol.analyze(problem, Y, calc_second_order=False, 
                       num_resamples=1000, conf_level=0.95)
    
    results[task_name] = {
        'S1': {name: float(val) for name, val in zip(problem['names'], Si['S1'])},
        'S1_conf': {name: float(val) for name, val in zip(problem['names'], Si['S1_conf'])},
        'ST': {name: float(val) for name, val in zip(problem['names'], Si['ST'])},
        'ST_conf': {name: float(val) for name, val in zip(problem['names'], Si['ST_conf'])},
        'sum_S1': float(np.sum(Si['S1'])),
        'Y_mean': float(np.mean(Y)),
        'Y_std': float(np.std(Y)),
    }
    
    # Print results
    unit = 'mm' if TASKS[task_name]['error_type'] in ('position', 'lateral') else 'rad'
    print(f"\n  {'Parameter':<10} {'S1':>8} {'S1_conf':>8} {'ST':>8} {'ST_conf':>8} {'Interaction':>12}")
    print(f"  {'-'*56}")
    for name in problem['names']:
        s1 = results[task_name]['S1'][name]
        s1c = results[task_name]['S1_conf'][name]
        st = results[task_name]['ST'][name]
        stc = results[task_name]['ST_conf'][name]
        interaction = st - s1
        print(f"  {name:<10} {s1:>8.4f} {s1c:>8.4f} {st:>8.4f} {stc:>8.4f} {interaction:>12.4f}")
    print(f"  Sum S1 = {results[task_name]['sum_S1']:.4f} "
          f"({'additive' if results[task_name]['sum_S1'] > 0.9 else 'significant interactions'})")

total_time = time.time() - total_start
print(f"\nTotal computation time: {total_time:.0f}s ({total_time/60:.1f} min)")

# ============================================================
# CRITICALITY MATRIX
# ============================================================

print("\n" + "=" * 80)
print("NOISE-TASK CRITICALITY MATRIX (First-Order Indices S_i)")
print("=" * 80)

header = f"{'Task':<20}" + "".join(f"{'S1('+n+')':>12}" for n in problem['names']) + f"{'Sum S1':>10}" + f"{'Interactions':>14}"
print(header)
print("-" * len(header))

for task_name in TASKS:
    if 'warning' in results.get(task_name, {}):
        print(f"{task_name:<20} {'DEGENERATE OUTPUT':>50}")
        continue
    r = results[task_name]
    row = f"{task_name:<20}"
    for name in problem['names']:
        val = r['S1'][name]
        row += f"{val:>12.4f}"
    row += f"{r['sum_S1']:>10.4f}"
    interactions = sum(r['ST'][n] - r['S1'][n] for n in problem['names'])
    row += f"{interactions:>14.4f}"
    print(row)

print("\n" + "=" * 80)
print("TOTAL-ORDER INDICES (S_Ti)")
print("=" * 80)

header = f"{'Task':<20}" + "".join(f"{'ST('+n+')':>12}" for n in problem['names'])
print(header)
print("-" * len(header))

for task_name in TASKS:
    if 'warning' in results.get(task_name, {}):
        continue
    r = results[task_name]
    row = f"{task_name:<20}"
    for name in problem['names']:
        row += f"{r['ST'][name]:>12.4f}"
    print(row)

# ============================================================
# KEY FINDINGS
# ============================================================

print("\n" + "=" * 80)
print("KEY FINDINGS")
print("=" * 80)

# Check if rankings shift across tasks
dominant_by_task = {}
for task_name in TASKS:
    if 'warning' in results.get(task_name, {}):
        continue
    r = results[task_name]
    dominant = max(problem['names'], key=lambda n: r['S1'][n])
    dominant_val = r['S1'][dominant]
    dominant_by_task[task_name] = (dominant, dominant_val)
    print(f"  {task_name:<20}: dominant = {dominant} (S1={dominant_val:.4f})")

dominant_sources = set(v[0] for v in dominant_by_task.values())
if len(dominant_sources) > 1:
    print(f"\n  FINDING: Sensitivity rankings SHIFT across tasks!")
    print(f"  Dominant sources: {dominant_sources}")
    print(f"  This confirms the paper's hypothesis that noise-source importance is task-dependent.")
else:
    print(f"\n  FINDING: Sensitivity rankings are UNIFORM across all tasks.")
    print(f"  Dominant source everywhere: {list(dominant_sources)[0]}")
    print(f"  This means a single mitigation strategy suffices for all tasks.")

# ============================================================
# SAVE RESULTS
# ============================================================

output = {
    'config': {
        'N': N,
        'k': problem['num_vars'],
        'evaluations_per_task': N * (problem['num_vars'] + 2),
        'total_evaluations': N * (problem['num_vars'] + 2) * len(TASKS),
        'total_time_sec': total_time,
    },
    'problem': problem,
    'tasks': results,
    'dominant_by_task': {k: {'source': v[0], 'S1': v[1]} for k, v in dominant_by_task.items()},
}

output_path = 'sobol_results.json'
with open(output_path, 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nResults saved to: {output_path}")
