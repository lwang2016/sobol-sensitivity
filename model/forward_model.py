"""
Forward Model & Sobol Sensitivity Analysis
===========================================
Implements the stochastic forward model f(θ_J, θ_M, θ_B, θ_T; τ) described in
draft.tex Section IV-B, and runs Sobol variance decomposition via SALib.

The model simulates a two-channel synced actuation system executing autonomous
tasks under four noise sources, with a PID controller in the loop. The two
channels represent any paired motor system that must coordinate to produce a
desired outcome (e.g., left/right drivetrain, paired flywheels, dual-motor lift).
Channel mismatch (θ_M) produces heading deviation and path curvature.

Sobol inputs (each ∈ [0,1], mapped to physical ranges):
  θ_J: jitter severity — scales the loop timing distribution
  θ_M: motor gain asymmetry — left/right channel gain mismatch
  θ_B: battery initial SOC — determines voltage trajectory
  θ_T: thermal drift severity — motor speed decay + heading bias rate
"""

import numpy as np
from scipy import stats
import os
import json
import time

# ============================================================
# PLATFORM CONSTANTS (from FTC Control Hub + GoBILDA drivetrain)
# ============================================================
NOMINAL_DT_S = 0.0135          # 13.5 ms nominal loop period
TRACK_WIDTH_MM = 350.0         # effective channel separation (distance between actuation channels)
NOMINAL_VOLTAGE = 12.0         # reference voltage for torque scaling
MAX_SPEED_MM_S = 800.0         # approximate max robot speed at full power, 12V

# Jitter distribution parameters (from jitter analysis: lognorm fit)
JITTER_SHAPE = 0.674
JITTER_LOC = 9.978
JITTER_SCALE = 2.801

# Motor variability parameters (from shooter + drivetrain analysis)
MOTOR_GAIN_HALF_RANGE = 0.10   # ±10% gain range (covers observed 18.7% L-R diff)
# Per-iteration velocity noise CV. The measured 13-22% CV from Pinpoint-derived
# speed includes position differentiation noise and Pinpoint quantization.
# The actual per-command-cycle motor output variation is much lower.
# The shooter under PID shows 6.7% CV which includes PID oscillation and the PID
# actively corrects; real single-cycle open-loop variation is estimated at ~0.8%.
# The measured cumulative CV is the compound effect over many cycles + measurement noise.
MOTOR_NOISE_CV = 0.008

# Battery parameters (from battery sag analysis)
BATTERY_V_MIN = 11.8           # lowest observed starting voltage (partial charge)
BATTERY_V_MAX = 13.7           # highest observed starting voltage (full charge)
BATTERY_SLOPE_STEEP = -3.5e-3  # V/s at full charge
BATTERY_SLOPE_FLAT = -0.3e-3   # V/s at plateau
BATTERY_NOISE_STD = 0.05       # V, typical voltage fluctuation

# Thermal parameters (from D_1 analysis)
# Motor speed decays ~11.2% over 600s, but ~5% is battery sag, leaving ~6% thermal
# After deconvolution estimate: thermal causes ~1% speed decay per minute
THERMAL_SPEED_DECAY_MAX = 0.001  # per second, max (= 6% over 600s / 600)
THERMAL_HEADING_DRIFT_MAX = 0.00075  # rad/s, max heading bias rate
# +4.3% heading/distance ratio drift over 600s at ~240 mm/s avg
# = 0.043 * mean_heading_rate / 600 ~ 0.043 * 2.06 rad/s / 600 ≈ 1.5e-4 rad/s
# But that's the ratio drift. Convert: at ~240mm/s, heading rate ~2.06 rad/s,
# 4.3% of that over 600s = 0.043 * 2.06 / 600 ≈ 1.5e-4 rad/s drift rate
# Use 7.5e-4 as max to allow Sobol to explore a wider range

# PID Controller gains (typical FTC heading PID)
KP_HEADING = 1.0      # proportional (rad -> differential power)
KD_HEADING = 0.01     # derivative

# ============================================================
# NOISE PARAMETER MAPPING: [0,1] → physical range
# ============================================================

def map_jitter(theta_j):
    """Map θ_J ∈ [0,1] to jitter severity multiplier.
    0 = minimal jitter (tight distribution), 1 = maximum jitter (wide distribution).
    Scales the shape parameter of the lognormal."""
    # Scale shape from 0.3 (tight) to 1.0 (very spread)
    shape = 0.3 + theta_j * 0.7
    return shape

def map_motor(theta_m):
    """Map θ_M ∈ [0,1] to left/right motor gain offset.
    0 = left motor weaker, 1 = right motor weaker.
    Returns (gain_left, gain_right) relative to nominal."""
    offset = (theta_m - 0.5) * 2 * MOTOR_GAIN_HALF_RANGE  # [-0.10, +0.10]
    gain_left = 1.0 - offset
    gain_right = 1.0 + offset
    return gain_left, gain_right

def map_battery(theta_b):
    """Map θ_B ∈ [0,1] to initial voltage and sag rate.
    0 = worst case (partial charge), 1 = best case (full charge)."""
    v0 = BATTERY_V_MIN + theta_b * (BATTERY_V_MAX - BATTERY_V_MIN)
    # Sag rate: steeper at higher voltage (non-linear)
    slope = BATTERY_SLOPE_FLAT + theta_b * (BATTERY_SLOPE_STEEP - BATTERY_SLOPE_FLAT)
    return v0, slope

def map_thermal(theta_t):
    """Map θ_T ∈ [0,1] to thermal drift rates.
    0 = no thermal drift, 1 = maximum observed drift."""
    speed_decay_rate = theta_t * THERMAL_SPEED_DECAY_MAX    # per second
    heading_drift_rate = theta_t * THERMAL_HEADING_DRIFT_MAX  # rad/s
    return speed_decay_rate, heading_drift_rate


# ============================================================
# TASK DEFINITIONS
# ============================================================

TASKS = {
    'transit': {
        'description': 'Point-to-point transit: 2.0m straight at 0.8 power',
        'target_distance_mm': 2000.0,
        'target_heading_rad': 0.0,
        'power': 0.8,
        'error_type': 'position',  # ||p(T) - p_target||
    },
    'turn': {
        'description': 'Turn-in-place: 90° CW at 0.5 power',
        'target_distance_mm': 0.0,
        'target_heading_rad': -np.pi / 2,  # 90° CW = -π/2
        'power': 0.5,
        'error_type': 'heading',  # |ψ(T) - ψ_target|
    },
    'tracking': {
        'description': 'Linear tracking: 3.0m at 0.4 power, constant vel',
        'target_distance_mm': 3000.0,
        'target_heading_rad': 0.0,
        'power': 0.4,
        'error_type': 'lateral',  # max |y(t)|
    },
    'parking_pos': {
        'description': 'Precision parking (position): 1.0m + 45° at 0.3 power',
        'target_distance_mm': 1000.0,
        'target_heading_rad': -np.pi / 4,  # 45° CW
        'power': 0.3,
        'error_type': 'position',
    },
    'parking_hdg': {
        'description': 'Precision parking (heading): 1.0m + 45° at 0.3 power',
        'target_distance_mm': 1000.0,
        'target_heading_rad': -np.pi / 4,
        'power': 0.3,
        'error_type': 'heading',
    },
}


# ============================================================
# FORWARD MODEL
# ============================================================

def simulate_task(theta_j, theta_m, theta_b, theta_t, task_name, rng=None):
    """
    Run one simulation of a task under the given noise parameters.
    
    Parameters (all ∈ [0,1]):
        theta_j: jitter severity
        theta_m: motor gain asymmetry
        theta_b: battery SOC
        theta_t: thermal drift severity
        task_name: key into TASKS dict
    
    Returns:
        error: scalar task error (mm for position, rad for heading)
    """
    if rng is None:
        rng = np.random.default_rng()
    
    task = TASKS[task_name]
    target_dist = task['target_distance_mm']
    target_heading = task['target_heading_rad']
    command_power = task['power']
    error_type = task['error_type']
    
    # Map Sobol inputs to physical parameters
    jitter_shape = map_jitter(theta_j)
    gain_left, gain_right = map_motor(theta_m)
    v0, v_slope = map_battery(theta_b)
    speed_decay, heading_drift = map_thermal(theta_t)
    
    # Robot state: [x, y, heading] in mm and radians
    x, y, heading = 0.0, 0.0, 0.0
    
    # For tracking task, record max lateral deviation
    max_lateral = 0.0
    
    # Build phase list
    # Transit/tracking: drive only. Turn: turn only. Parking: drive then turn.
    if target_dist > 0 and abs(target_heading) > 0.01:
        phases = [('drive', target_dist, 0.0), ('turn', 0.0, target_heading)]
    elif target_dist > 0:
        phases = [('drive', target_dist, 0.0)]
    else:
        phases = [('turn', 0.0, target_heading)]
    
    elapsed_total = 0.0
    
    for phase_type, phase_dist, phase_target_heading in phases:
        # Track how far we've gone in this phase
        phase_distance = 0.0
        heading_at_phase_start = heading  # capture actual heading at start of each phase
        prev_heading_error = 0.0
        
        # Max iterations (generous bound)
        max_iterations = 3000
        
        for iteration in range(max_iterations):
            # 1. JITTER: sample loop period
            dt_ms = rng.lognormal(
                mean=np.log(JITTER_SCALE),
                sigma=jitter_shape
            ) + JITTER_LOC
            dt_s = np.clip(dt_ms / 1000.0, 0.005, 0.100)
            
            elapsed_total += dt_s
            
            # 2. BATTERY: voltage at current time
            voltage = v0 + v_slope * elapsed_total
            voltage += rng.normal(0, BATTERY_NOISE_STD)
            voltage = np.clip(voltage, 8.0, 15.0)
            voltage_scale = voltage / NOMINAL_VOLTAGE
            
            # 3. THERMAL: time-dependent degradation
            thermal_factor = max(1.0 - speed_decay * elapsed_total, 0.7)
            heading_bias = heading_drift * elapsed_total
            
            # 4. SENSE: noisy heading (includes thermal bias)
            sensed_heading = heading + heading_bias
            
            # 5. PID CONTROLLER
            if phase_type == 'drive':
                remaining = phase_dist - phase_distance
                if remaining <= 5.0:  # close enough (5mm)
                    break
                
                # Forward power: constant (open-loop for distance)
                # Ramp down only in last 100mm for stop control
                fwd_power = command_power
                if remaining < 100.0:
                    fwd_power = max(command_power * (remaining / 100.0), 0.05)
                
                # Heading PID: correct drift to keep straight
                # But sensor is noisy and delayed — PID sees biased heading
                h_error = 0.0 - sensed_heading  # want heading = 0
                h_correction = KP_HEADING * h_error + KD_HEADING * (h_error - prev_heading_error) / dt_s
                # Clamp correction: heading fix should never exceed 30% of forward power
                max_corr = 0.3 * fwd_power
                h_correction = np.clip(h_correction, -max_corr, max_corr)
                prev_heading_error = h_error
                
                power_left = fwd_power + h_correction
                power_right = fwd_power - h_correction
                
            else:  # turn
                # Turn uses heading PID to reach absolute target heading
                # (for parking: target accumulated from drive phase + turn spec)
                absolute_target = heading_at_phase_start + phase_target_heading
                remaining_turn = absolute_target - sensed_heading
                
                if abs(remaining_turn) < 0.02:  # ~1.1° tolerance
                    break
                
                # P-controller with saturation
                turn_power = KP_HEADING * remaining_turn
                turn_power = np.clip(turn_power, -command_power, command_power)
                
                # Minimum power to overcome static friction
                if abs(turn_power) < 0.10 and abs(remaining_turn) > 0.02:
                    turn_power = 0.10 * np.sign(remaining_turn)
                
                # Differential drive: positive turn_power → CW (negative heading change)
                # Left wheel forward, right wheel backward → robot turns CW
                power_left = -turn_power
                power_right = turn_power
            
            power_left = np.clip(power_left, -1.0, 1.0)
            power_right = np.clip(power_right, -1.0, 1.0)
            
            # 6. ACTUATE: compute achieved forward speed and angular rate
            # Forward speed: commanded power → speed, scaled by gains, voltage, thermal
            avg_gain = (gain_left + gain_right) / 2.0
            gain_diff = (gain_right - gain_left) / 2.0  # persistent L-R asymmetry
            
            # Common-mode noise affects forward speed
            speed_noise = 1.0 + rng.normal(0, MOTOR_NOISE_CV)
            
            if phase_type == 'drive':
                # Forward velocity
                fwd_speed = fwd_power * avg_gain * voltage_scale * thermal_factor * speed_noise * MAX_SPEED_MM_S
                
                # Angular rate from heading correction + motor asymmetry
                # Motor asymmetry creates a persistent omega offset proportional to forward power
                asymmetry_omega = (gain_diff / avg_gain) * fwd_power * MAX_SPEED_MM_S / (TRACK_WIDTH_MM / 2.0)
                correction_omega = (h_correction / fwd_power if fwd_power > 0.01 else 0.0) * fwd_speed / (TRACK_WIDTH_MM / 2.0)
                # Differential motor noise adds random angular rate
                diff_noise = rng.normal(0, MOTOR_NOISE_CV * 0.3)
                omega_noise = diff_noise * fwd_power * MAX_SPEED_MM_S / (TRACK_WIDTH_MM / 2.0)
                
                omega = asymmetry_omega + correction_omega + omega_noise
                v_linear = fwd_speed
            else:
                # Turn: direct differential drive
                eff_left = power_left * gain_left * voltage_scale * thermal_factor * speed_noise
                eff_right = power_right * gain_right * voltage_scale * thermal_factor * speed_noise
                v_left = eff_left * MAX_SPEED_MM_S
                v_right = eff_right * MAX_SPEED_MM_S
                v_linear = (v_left + v_right) / 2.0
                omega = (v_right - v_left) / TRACK_WIDTH_MM
            
            # 7. PROPAGATE: two-channel synced actuation kinematics
            # Channel mismatch produces angular rate; coordinated output produces forward speed
            if abs(omega) < 1e-6:
                x += v_linear * np.cos(heading) * dt_s
                y += v_linear * np.sin(heading) * dt_s
            else:
                r = v_linear / omega
                x += r * (np.sin(heading + omega * dt_s) - np.sin(heading))
                y += r * (np.cos(heading) - np.cos(heading + omega * dt_s))
            heading += omega * dt_s
            
            # Track distance for drive phases
            if phase_type == 'drive':
                phase_distance += abs(v_linear) * dt_s
            
            # Track lateral deviation
            if error_type == 'lateral':
                max_lateral = max(max_lateral, abs(y))
    
    # Compute error metric
    if error_type == 'position':
        error = np.sqrt((x - target_dist)**2 + y**2)
    elif error_type == 'heading':
        error = abs(heading - target_heading)
        error = error % (2 * np.pi)
        if error > np.pi:
            error = 2 * np.pi - error
    elif error_type == 'lateral':
        error = max_lateral
    
    return error


def evaluate_batch(param_array, task_name, seed=42):
    """
    Evaluate the forward model for a batch of Sobol samples.
    
    Parameters:
        param_array: (N, 4) array of [θ_J, θ_M, θ_B, θ_T] ∈ [0,1]
        task_name: key into TASKS dict
        seed: random seed for reproducibility
    
    Returns:
        errors: (N,) array of scalar task errors
    """
    n = param_array.shape[0]
    errors = np.zeros(n)
    rng = np.random.default_rng(seed)
    
    for i in range(n):
        errors[i] = simulate_task(
            param_array[i, 0],
            param_array[i, 1],
            param_array[i, 2],
            param_array[i, 3],
            task_name,
            rng=rng
        )
    
    return errors


# ============================================================
# SANITY CHECKS
# ============================================================

def run_sanity_checks():
    """Run basic forward model validation."""
    print("=" * 70)
    print("FORWARD MODEL SANITY CHECKS")
    print("=" * 70)
    
    rng = np.random.default_rng(42)
    
    # 1. Nominal case (mid-range everything)
    print("\n1. Nominal case (all θ = 0.5):")
    for task_name, task in TASKS.items():
        errors = [simulate_task(0.5, 0.5, 0.5, 0.5, task_name, rng=np.random.default_rng(i)) 
                  for i in range(20)]
        unit = 'mm' if task['error_type'] in ('position', 'lateral') else 'rad'
        print(f"   {task_name:>15}: mean={np.mean(errors):.2f} {unit}, "
              f"std={np.std(errors):.2f}, range=[{np.min(errors):.2f}, {np.max(errors):.2f}]")
    
    # 2. Extreme cases
    print("\n2. Extreme jitter (θ_J=1.0 vs 0.0), transit task:")
    for tj in [0.0, 0.5, 1.0]:
        errors = [simulate_task(tj, 0.5, 0.5, 0.5, 'transit', rng=np.random.default_rng(i)) 
                  for i in range(20)]
        print(f"   θ_J={tj}: mean={np.mean(errors):.2f} mm, std={np.std(errors):.2f}")
    
    print("\n3. Extreme motor asymmetry (θ_M=0.0 vs 1.0), turn task:")
    for tm in [0.0, 0.5, 1.0]:
        errors = [simulate_task(0.5, tm, 0.5, 0.5, 'turn', rng=np.random.default_rng(i)) 
                  for i in range(20)]
        print(f"   θ_M={tm}: mean={np.mean(errors):.4f} rad ({np.degrees(np.mean(errors)):.2f}°), std={np.std(errors):.4f}")
    
    print("\n4. Extreme battery (θ_B=0.0 vs 1.0), tracking task:")
    for tb in [0.0, 0.5, 1.0]:
        errors = [simulate_task(0.5, 0.5, tb, 0.5, 'tracking', rng=np.random.default_rng(i)) 
                  for i in range(20)]
        print(f"   θ_B={tb}: mean={np.mean(errors):.2f} mm, std={np.std(errors):.2f}")
    
    print("\n5. Extreme thermal (θ_T=0.0 vs 1.0), tracking task:")
    for tt in [0.0, 0.5, 1.0]:
        errors = [simulate_task(0.5, 0.5, 0.5, tt, 'tracking', rng=np.random.default_rng(i)) 
                  for i in range(20)]
        print(f"   θ_T={tt}: mean={np.mean(errors):.2f} mm, std={np.std(errors):.2f}")
    
    # 3. Timing check
    print("\n6. Performance check:")
    t0 = time.time()
    n_eval = 200
    params = np.random.default_rng(42).uniform(0, 1, (n_eval, 4))
    _ = evaluate_batch(params, 'transit', seed=42)
    dt = time.time() - t0
    print(f"   {n_eval} evaluations in {dt:.2f}s ({dt/n_eval*1000:.1f} ms/eval)")
    est_sobol = dt / n_eval * 10240
    print(f"   Estimated Sobol time per task: {est_sobol:.0f}s ({est_sobol/60:.1f} min)")
    print(f"   Estimated total (5 tasks): {est_sobol*5/60:.1f} min")


if __name__ == '__main__':
    run_sanity_checks()
