# Noise-Source Sensitivity Framework

A reusable framework for decomposing robot task-error variance by noise source using Sobol sensitivity analysis. Given a robot affected by multiple concurrent noise sources (actuator variability, communication jitter, power supply fluctuations, sensor drift, etc.), the framework identifies which source matters most for each task type and whether the ranking changes across tasks.

## Paper

**"A Sobol Sensitivity Framework for Identifying Dominant Noise Sources in Robot Tasks"**

Aaron Wang and Robert Chun

IEEE MIT Undergraduate Research Technology Conference (URTC), 2026

## Framework

The framework follows four steps:

1. **Characterize** each noise source independently through isolation experiments on physical hardware
2. **Model** task execution as a stochastic forward function using the measured noise distributions
3. **Decompose** task-error variance using Sobol first-order and total-order indices via Monte Carlo simulation
4. **Output** a noise-task criticality matrix identifying the dominant source for each task

## Case Study

The framework is demonstrated on the FIRST Tech Challenge (FTC) Control Hub, a heterogeneous-processor platform pairing a non-real-time application processor (Android) with a real-time I/O co-processor. Four noise sources are characterized across three task types.

Key findings:
- Motor gain asymmetry dominates position-sensitive tasks (S_i > 0.98)
- Heading error shows a qualitatively different structure with thermal drift (S_i = 0.21) and significant cross-source interactions (S_Ti = 0.47)
- PID control renders all four sources negligible for short rotational tasks
- Correcting the top-ranked source (motor matching) eliminates the corresponding performance deficiency

## Repository Structure

```
data/                  Raw experimental data (CSV) from isolation experiments
  jitter/              6 runs: 3 stationary, 3 loaded (26,648 samples total)
  motor/               20 runs: 4 motors x 5 power levels
  battery/             3 runs: full charge, partial charge, transient load
  thermal/             2 runs: thermal drift, heading drift stationary

analysis/              Python scripts to reproduce all characterization results
  analyze_jitter.py    Jitter distribution fitting, Mann-Whitney test
  analyze_motor.py     Motor CV, Shapiro-Wilk test, gain variability
  analyze_motor_paired.py   Between-motor gain and residual correlation
  analyze_battery.py   SOC-dependent sag characterization
  analyze_thermal.py   Thermal speed decay and heading drift
  requirements.txt     Python dependencies

model/                 Stochastic forward model and Sobol decomposition
  forward_model.py     Closed-loop task simulation
  run_sobol.py         Saltelli sampling + SALib Sobol analysis
  sobol_results.json   Computed Sobol indices (reproducible output)

collection/            Data collection infrastructure
  ResearchLogger.java  FTC OpMode used to collect all experimental data

results/               Generated outputs
  jitter_distribution.png      Figure 1 in paper
  motor_variability.png        Figure 2 in paper
  analysis_results/            JSON output from each analysis script

paper/                 Paper source
  draft2_5page.tex     LaTeX source
```

## Data File Naming

Original filenames were shortened for clarity. Mapping:

| Repo path | Original filename |
|---|---|
| `data/jitter/stationary_run1.csv` | `ResearchLogger_A_2_Jitter_Stationary_Run1.csv` |
| `data/jitter/loaded_run1.csv` | `ResearchLogger_A_3_Jitter_Loaded_Run1.csv` |
| `data/motor/BL_Power0p4.csv` | `ResearchLogger_B_2_Motor_BL_Power0p4.csv` |
| `data/battery/full_charge.csv` | `ResearchLogger_C1_Full_Charge.csv` |
| `data/thermal/thermal_drift.csv` | `ResearchLogger_D_1_ThermalDrift_FullCharge.csv` |

## Requirements

- Python 3.8+
- NumPy, SciPy, pandas, matplotlib, SALib

```bash
pip install -r analysis/requirements.txt
```

## Reproducing Results

```bash
# Characterize noise sources
python analysis/analyze_jitter.py
python analysis/analyze_motor.py
python analysis/analyze_motor_paired.py
python analysis/analyze_battery.py
python analysis/analyze_thermal.py

# Run Sobol decomposition
python model/run_sobol.py
```

## License

MIT
