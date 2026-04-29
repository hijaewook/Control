# Motor PID Gain Scheduling Project

## 1. Project Overview

This project aims to construct a PID gain scheduling framework for motor control using Simulink-based simulation data.

The overall objective is to find appropriate PID gains according to the target value and to use them as initial gains for adaptive PID control. The current framework consists of:

1. Simulink-based PID gain sweep
2. Performance dataset construction
3. Surrogate model training
4. Surrogate-based gain optimization
5. Simulink validation of recommended gains
6. Target-based PID gain database construction
7. Adaptive PID control using gain DB and linear interpolation

---

## 2. Project Structure

```text
Motor/
├── config.py
├── main.py
├── dashboard.py
├── data/
│   ├── raw/
│   ├── processed/
│   │   ├── pid_performance_prediction_dataset.csv
│   │   └── pid_gain_recommendation_dataset.csv
│   └── simulation/
├── results/
│   ├── logs/
│   ├── figures/
│   ├── models/
│   ├── simulink_gain_db/
│   ├── model_gain_optimization/
│   └── surrogate_validation/
└── src/
    ├── pid_controller.py
    ├── gain_scheduler.py
    ├── safety_guard.py
    ├── motor_env.py
    ├── simulink_runner.py
    ├── simulink_gain_sweep.py
    ├── build_training_dataset.py
    ├── train_model.py
    ├── optimize_gain_with_model.py
    ├── validate_surrogate_gains.py
    ├── compare_logs.py
    └── compare_adaptive_targets.py