import sys
from pathlib import Path
from datetime import datetime
import itertools

import numpy as np
import pandas as pd


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from simulink_runner import SimulinkRunner
from config import (
    RESULTS_DIR,
    SIMULINK_MODEL_NAME,
    SIMULINK_MAT_FILE,
    SIMULINK_SWEEP_STOP_TIME,
    TARGET_LIST,
    SWEEP_KP_LIST,
    SWEEP_KI_LIST,
    SWEEP_KD_LIST,
    PWM_MAX,
    PWM_MIN,
    OVERSHOOT_WEIGHT,
    PWM_WEIGHT,
    SATURATION_WEIGHT,
    PWM_SATURATION_TOL,
)


# ============================================================
# Result path
# ============================================================

SAVE_DIR = RESULTS_DIR / "simulink_gain_db"


# ============================================================
# Metric functions
# ============================================================

def compute_dt(time: np.ndarray) -> np.ndarray:
    """
    time 배열에서 dt 배열 계산.
    """
    if len(time) <= 1:
        return np.zeros_like(time)

    dt = np.diff(time, prepend=time[0])
    dt[0] = 0.0

    return dt


def calculate_control_metrics(df: pd.DataFrame) -> dict:
    """
    Simulink log dataframe에서 제어 성능 지표 계산.
    """

    required_cols = ["time", "target", "current", "error", "pwm"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    time = df["time"].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    current = df["current"].to_numpy(dtype=float)
    error = df["error"].to_numpy(dtype=float)
    pwm = df["pwm"].to_numpy(dtype=float)

    dt = compute_dt(time)
    abs_error = np.abs(error)

    target_value = float(target[-1])

    # --------------------------------------------------------
    # Error metrics
    # --------------------------------------------------------

    iae = float(np.sum(abs_error * dt))
    ise = float(np.sum((error ** 2) * dt))
    mean_abs_error = float(np.mean(abs_error))
    final_error = float(abs_error[-1])

    # --------------------------------------------------------
    # Overshoot
    # --------------------------------------------------------

    max_current = float(np.max(current))
    overshoot = max(0.0, max_current - target_value)
    overshoot_percent = overshoot / max(abs(target_value), 1e-6) * 100.0

    # --------------------------------------------------------
    # PWM effort
    # --------------------------------------------------------

    abs_pwm = np.abs(pwm)

    mean_pwm = float(np.mean(abs_pwm))
    total_pwm = float(np.sum(abs_pwm * dt))
    max_pwm = float(np.max(pwm))
    min_pwm = float(np.min(pwm))

    # --------------------------------------------------------
    # Saturation metrics
    # --------------------------------------------------------

    high_saturation = pwm >= PWM_MAX - PWM_SATURATION_TOL
    low_saturation = pwm <= PWM_MIN + PWM_SATURATION_TOL
    saturation_mask = high_saturation | low_saturation

    saturation_count = int(np.sum(saturation_mask))
    saturation_ratio_percent = float(
        saturation_count / max(len(pwm), 1) * 100.0
    )
    saturation_duration = float(np.sum(dt[saturation_mask]))

    high_saturation_count = int(np.sum(high_saturation))
    high_saturation_ratio_percent = float(
        high_saturation_count / max(len(pwm), 1) * 100.0
    )
    high_saturation_duration = float(np.sum(dt[high_saturation]))

    # --------------------------------------------------------
    # Rise time: 10% -> 90%
    # --------------------------------------------------------

    y_10 = 0.1 * target_value
    y_90 = 0.9 * target_value

    try:
        t_10 = time[np.where(current >= y_10)[0][0]]
        t_90 = time[np.where(current >= y_90)[0][0]]
        rise_time = float(t_90 - t_10)
    except IndexError:
        rise_time = np.nan

    # --------------------------------------------------------
    # Settling time: ±2%
    # --------------------------------------------------------

    tolerance = 0.02 * abs(target_value)
    lower = target_value - tolerance
    upper = target_value + tolerance

    within_band = (current >= lower) & (current <= upper)

    settling_time = np.nan

    for i in range(len(time)):
        if np.all(within_band[i:]):
            settling_time = float(time[i])
            break

    # --------------------------------------------------------
    # Saturation-aware score
    # --------------------------------------------------------

    score = (
        iae
        + OVERSHOOT_WEIGHT * overshoot_percent
        + PWM_WEIGHT * total_pwm
        + SATURATION_WEIGHT * saturation_ratio_percent
    )

    return {
        "IAE": iae,
        "ISE": ise,
        "mean_abs_error": mean_abs_error,
        "final_error": final_error,

        "rise_time": rise_time,
        "settling_time": settling_time,
        "overshoot_percent": overshoot_percent,

        "mean_pwm": mean_pwm,
        "total_pwm": total_pwm,
        "max_pwm": max_pwm,
        "min_pwm": min_pwm,

        "saturation_count": saturation_count,
        "saturation_ratio_percent": saturation_ratio_percent,
        "saturation_duration": saturation_duration,
        "high_saturation_count": high_saturation_count,
        "high_saturation_ratio_percent": high_saturation_ratio_percent,
        "high_saturation_duration": high_saturation_duration,

        "score": score,
    }


# ============================================================
# Sweep
# ============================================================

def run_gain_sweep():
    """
    Simulink PID gain sweep 실행.
    """

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    runner = SimulinkRunner(
        model_name=SIMULINK_MODEL_NAME,
        mat_file_name=SIMULINK_MAT_FILE,
    )

    rows = []

    total_cases = (
        len(TARGET_LIST)
        * len(SWEEP_KP_LIST)
        * len(SWEEP_KI_LIST)
        * len(SWEEP_KD_LIST)
    )

    case_idx = 0

    try:
        for target, kp, ki, kd in itertools.product(
            TARGET_LIST,
            SWEEP_KP_LIST,
            SWEEP_KI_LIST,
            SWEEP_KD_LIST,
        ):
            case_idx += 1

            print(
                f"[{case_idx}/{total_cases}] "
                f"Run Simulink: target={target}, "
                f"Kp={kp}, Ki={ki}, Kd={kd}"
            )

            try:
                df = runner.run_simulation(
                    target=target,
                    kp=kp,
                    ki=ki,
                    kd=kd,
                    stop_time=SIMULINK_SWEEP_STOP_TIME,
                    save_log=False,
                )

                metrics = calculate_control_metrics(df)

                row = {
                    "target": target,
                    "kp": kp,
                    "ki": ki,
                    "kd": kd,
                }
                row.update(metrics)

                rows.append(row)

            except Exception as e:
                print(
                    f"Failed: target={target}, "
                    f"Kp={kp}, Ki={ki}, Kd={kd}"
                )
                print(f"Reason: {e}")

    finally:
        runner.stop()

    result_df = pd.DataFrame(rows)

    save_path = SAVE_DIR / f"simulink_gain_sweep_{timestamp}.csv"
    result_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("Simulink Gain Sweep Finished")
    print(f"Saved: {save_path}")
    print("=" * 80)

    if len(result_df) > 0:
        print("\nTop 5 by Saturation-aware Score:")
        display_cols = [
            "target",
            "kp",
            "ki",
            "kd",
            "score",
            "IAE",
            "settling_time",
            "rise_time",
            "overshoot_percent",
            "mean_pwm",
            "total_pwm",
            "max_pwm",
            "saturation_ratio_percent",
        ]
        display_cols = [col for col in display_cols if col in result_df.columns]

        print(
            result_df
            .sort_values("score")
            [display_cols]
            .head(5)
        )

        print("\nBest gain by target using Saturation-aware Score:")
        best_by_target = (
            result_df
            .sort_values("score")
            .groupby("target", as_index=False)
            .first()
        )

        print(best_by_target[display_cols])

    return result_df, save_path


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    run_gain_sweep()