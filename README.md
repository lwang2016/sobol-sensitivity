# Noise-Source Sensitivity Framework

A framework for decomposing robot task-error variance by noise source using Sobol sensitivity analysis. Given a robot affected by several concurrent noise sources (actuator mismatch, control-loop timing jitter, battery voltage sag, heating and gyro drift), it takes a systematic, task-level approach: the sources are measured on the robot, simulated together with the robot's own controller, and ranked by their effect on the accuracy of each task. Remedies are then compared in the same model.

## Framework

1. **Measure** each noise source on the robot (loop timing, battery voltage, motor speed and power, temperature) and fit the model inputs to the logs.
2. **Simulate** each task with a closed-loop forward model (`forward_model.py`), with every source scaled to an input between 0 and 1 over its measured or bounded range.
3. **Decompose** the task-error variance with Sobol first-order and total-order indices (Saltelli sampling, N = 1024). The simulator has built-in run-to-run randomness, so every input setting is simulated 16 times and the random variance is removed from the indices (`analysis_replicated/replicated_sobol.py`).
4. **Rank and compare remedies**: rank the sources per task, then simulate remedies and longer operation at fixed input levels.

## Case Study

A four-wheel mecanum robot built from widely used commercial parts: REV Robotics Control Hub (Android), four goBILDA planetary DC gear motors of the same model, and a goBILDA Pinpoint V2 odometry computer for heading. Four sources (loop-timing jitter, left-right motor mismatch, battery state, heating) are studied in four tasks: transit, turn in place, line tracking and parking.

Current results (final configuration, `analysis_replicated/results/`):

- **Position error** (transit, line tracking, parking position): motor mismatch has total-order indices of 0.996–0.999. Jitter, battery and heating each explain less than 1%, because they affect both motors equally and the controller and the distance-based stop compensate for them. First-order and total-order indices differ by less than 0.01, so interactions are negligible.
- **Heading error** (turn in place, parking heading): randomness makes up 98% and 95% of the variance, so the sources are not ranked. The mean error of 0.95° comes from the stopping rule (the turn ends inside a 1.15° tolerance).
- **Remedies** (simulation): an integral term in the heading controller (K_i = 5) reduces straight-driving error 7 to 30 times at 5% and 10% mismatch (transit 36.2 → 5.3 mm and 70.3 → 6.7 mm), below feedforward calibration to a 1% residual. For the shorter parking drive, calibration performs as well at 5% and better at 10%.
- **Continuous operation**: with gyro drift growing to the manufacturer's 1°/min fault threshold at 10 min, transit error with matched motors reaches 10.9, 39.2 and 155 mm after 2.5, 5 and 10 min if heading is set only at power-on. Resetting heading before each task returns errors to their cold-start values.
- **Method**: running each input setting once, as is standard for deterministic models, reports the simulator's randomness as interactions of every input (e.g. heating 0.30 instead of 0.01 for a relative-turn parking heading). The replicated, noise-corrected estimator removes this and recovers known indices on test functions with added noise.

The initial single-run analysis (`run_sobol.py`, `sobol_results.json`) is kept for reference. Its thermal and interaction findings came from that artifact and from heating constants that were later corrected; they are superseded by the results above.

`forward_model.py` keeps its original constants and comments. The analyses in `analysis_replicated/` replace the heating constants (speed loss 1.0×10⁻⁴ /s instead of 1.0×10⁻³ /s; heading drift 1.5×10⁻⁴ rad/s instead of 7.5×10⁻⁴ rad/s) and apply the other model options in memory. The "18.7% L-R diff" comment in that file refers to the programmed feedforward constants of the flywheel, not a measured mismatch; the measured mismatch for that pair is about 5%.

## Repository Structure

```
forward_model.py          Original closed-loop task model (unchanged)
run_sobol.py              Initial single-run Sobol analysis (superseded, kept for reference)
sobol_results.json        Output of run_sobol.py
requirements.txt

Data/                     Raw logs and characterization scripts (original file names)
  ResearchLogger.java     FTC OpMode used to collect the isolation-experiment data
  Communication_Jitter/   6 runs (3 stationary, 3 loaded); analyze_jitter.py, fitted loop-period distribution
  Motor_Variability/      20 runs (4 motors x 5 power levels); analyze_motor.py
  Battery_Sag/            3 runs (full charge, partial charge, transient load); analyze_battery.py
  Thermal_Drift/          sustained-drive and stationary heading-drift runs; analyze_thermal.py
  prev_logs/              flywheel (shooter) logs used for the measured motor mismatch

analysis_replicated/      Corrected analysis; model options are patched in memory, forward_model.py is not modified
  replicated_sobol.py     Replicated, noise-corrected Sobol estimator
  thermal_variants.py     Model options: corrected heating constants, signed drift, absolute turn,
                          sensed phase start, warm start, temperature-driven drift, scaled mismatch turn
  controller_variants.py  Optional integral term in the heading controller
  final_baseline.py       Final configuration: Sobol indices, mismatch curves, remedies (transit, tracking, turn)
  parking_absolute.py     Final configuration for the parking tasks (cold and warm start)
  warm_start_temperature.py  Continuous operation with temperature-driven gyro drift
  thermal_projection.py   Heat-related speed loss fitted to the logs
  counterintuitive_checks.py  Measured flywheel-pair mismatch and model checks
  test_*.py               Unit tests (16)
  (other scripts)         Earlier and supporting studies: corrected_thermal, warm_start, warm_start_signed,
                          mismatch_impact, controller_comparison, spread_sobol, motor_noise_check
  results/                JSON outputs and run logs

paper_figures/
  make_figures.py         Builds all figures from the saved results (no simulation)
  figures/                PNG (600 dpi), PDF and figure_data.json
```

Figure files map to the manuscript figures as follows: `fig1_workflow` (Fig. 1), `fig2_loop_timing` (Fig. 2), `fig3_motor_mismatch` (Fig. 3), `fig_fits` (Fig. 4), `fig_input_ranges` (Fig. 5), `fig_forward_model` (Fig. 6), `fig4_mismatch_remedies` (Fig. 7), `fig5_heading_reference` (Fig. 8), `fig6_single_vs_replicated` (Fig. 9).

## Data

Data collection was done using DataLogger: https://github.com/SounderBots/FTC-Datalogger

File names keep the experiment codes used by the analysis scripts:

| Code | Experiment | Files |
|---|---|---|
| A_2, A_3 | Loop timing, stationary and loaded | `Data/Communication_Jitter/ResearchLogger_A_*_Jitter_*.csv` |
| B_2 | Motor speed at 5 power levels per motor | `Data/Motor_Variability/ResearchLogger_B_2_Motor_*.csv` |
| C1, C_2, C_3 | Battery sag: full charge, partial charge, transient load | `Data/Battery_Sag/ResearchLogger_C*.csv` |
| D_1, D_2 | Sustained drive (heating), stationary heading drift | `Data/Thermal_Drift/ResearchLogger_D_*.csv` |
| Shooter logs | Two same-model motors on one flywheel | `Data/prev_logs/*ShooterLog*.csv` |

## Requirements

Python 3.10+ with NumPy, SciPy, pandas, matplotlib and SALib. The results were produced with Python 3.12.10, NumPy 2.4.3, SciPy 1.17.1, pandas 3.0.2, matplotlib 3.10.8 and SALib 1.5.2.

```bash
pip install -r requirements.txt
```

## Reproducing Results

```bash
# 1. Characterize the noise sources (writes *_results.json next to each script)
python Data/Communication_Jitter/analyze_jitter.py
python Data/Motor_Variability/analyze_motor.py
python Data/Battery_Sag/analyze_battery.py
python Data/Thermal_Drift/analyze_thermal.py

# 2. Fits and checks from the logs
python analysis_replicated/thermal_projection.py
python analysis_replicated/counterintuitive_checks.py

# 3. Simulations in the final configuration (multi-process; about 10, 43 and 33 min on 14 workers)
python analysis_replicated/final_baseline.py
python analysis_replicated/parking_absolute.py
python analysis_replicated/warm_start_temperature.py

# 4. Figures
python paper_figures/make_figures.py

# Tests
cd analysis_replicated && python -m unittest discover -p "test_*.py"
```

Seeds are fixed, so rerunning reproduces the saved numbers in `analysis_replicated/results/` (only run times differ). Each result file except `thermal_projection.json` and the `Data/` outputs records the SHA-256 of the scripts that produced it. The final-configuration results match the scripts in this repository. The supporting studies `corrected_thermal`, `motor_noise_check`, `warm_start` and `warm_start_signed` were run with an earlier `thermal_variants.py`, before the scaled mismatch-turn option was added; rerunning them with the current file can change their numbers.

## License

MIT
