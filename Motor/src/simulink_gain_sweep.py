import sys
from pathlib import Path
from datetime import datetime
import itertools

import pandas as pd
import numpy as np


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent

sys.path.append(str(MOTOR_DIR))
sys.path.append(str(CURRENT_DIR))


# ============================================================
# Import project modules
# ============================================================

from simulink_runner import SimulinkRunner

from config import (
    TARGET_LIST,
    SWEEP_KP_LIST,
    SWEEP_KI_LIST,
    SWEEP_KD_LIST,
    SIMULINK_SWEEP_STOP_TIME,
)


# ============================================================
# Output directory
# ============================================================

SAVE_DIR = MOTOR_DIR / "results" / "simulink_gain_db"


# ============================================================
# Score weights
# ============================================================
# score = IAE + OVERSHOOT_WEIGHT * overshoot_percent + PWM_WEIGHT * total_pwm
# target 150, 200에서 overshoot가 커졌으므로 overshoot penalty를 강하게 둠

OVERSHOOT_WEIGHT = 30.0
PWM_WEIGHT = 0.01


# ============================================================
# Metric calculation
# ============================================================

def calculate_metrics(df: pd.DataFrame, target: float) -> dict:
    """
    Simulink simulation result dataframe에서 성능 지표 계산
    """

    time = df["time"].to_numpy()
    current = df["current"].to_numpy()
    error = df["error"].to_numpy()
    pwm = df["pwm"].to_numpy()

    abs_error = np.abs(error)

    # Time step
    dt = np.diff(time, prepend=time[0])
    dt[0] = 0.0

    # Error metrics
    final_error = float(abs_error[-1])
    mean_abs_error = float(np.mean(abs_error))

    iae = float(np.sum(abs_error * dt))
    ise = float(np.sum((error ** 2) * dt))

    # PWM metrics
    mean_pwm = float(np.mean(np.abs(pwm)))
    total_pwm = float(np.sum(np.abs(pwm) * dt))
    max_pwm = float(np.max(np.abs(pwm)))

    # Overshoot
    max_current = float(np.max(current))
    overshoot = max(0.0, max_current - target)
    overshoot_percent = overshoot / max(abs(target), 1e-6) * 100.0

    # Rise time: 10% -> 90%
    y_10 = 0.1 * target
    y_90 = 0.9 * target

    try:
        t_10 = time[np.where(current >= y_10)[0][0]]
        t_90 = time[np.where(current >= y_90)[0][0]]
        rise_time = float(t_90 - t_10)
    except IndexError:
        rise_time = np.nan

    # Settling time: ±2% band
    tolerance = 0.02 * abs(target)
    lower = target - tolerance
    upper = target + tolerance

    settling_time = np.nan
    within_band = (current >= lower) & (current <= upper)

    for i in range(len(time)):
        if np.all(within_band[i:]):
            settling_time = float(time[i])
            break

    # Score
    score = (
        iae
        + OVERSHOOT_WEIGHT * overshoot_percent
        + PWM_WEIGHT * total_pwm
    )

    return {
        "final_error": final_error,
        "mean_abs_error": mean_abs_error,
        "IAE": iae,
        "ISE": ise,
        "overshoot": overshoot,
        "overshoot_percent": overshoot_percent,
        "max_pwm": max_pwm,
        "mean_pwm": mean_pwm,
        "total_pwm": total_pwm,
        "rise_time": rise_time,
        "settling_time": settling_time,
        "score": float(score),
    }


# ============================================================
# Gain sweep
# ============================================================

def run_gain_sweep():
    """
    Target list와 PID gain 조합에 대해 Simulink PID gain sweep 실행
    """

    SAVE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SAVE_DIR / f"simulink_gain_sweep_{timestamp}.csv"

    runner = SimulinkRunner()

    results = []

    sweep_combinations = list(
        itertools.product(
            TARGET_LIST,
            SWEEP_KP_LIST,
            SWEEP_KI_LIST,
            SWEEP_KD_LIST,
        )
    )

    total_cases = len(sweep_combinations)

    print("=" * 80)
    print("Simulink Gain Sweep Start")
    print(f"Target list: {TARGET_LIST}")
    print(f"Kp list: {SWEEP_KP_LIST}")
    print(f"Ki list: {SWEEP_KI_LIST}")
    print(f"Kd list: {SWEEP_KD_LIST}")
    print(f"Stop time: {SIMULINK_SWEEP_STOP_TIME}")
    print(f"Overshoot weight: {OVERSHOOT_WEIGHT}")
    print(f"PWM weight: {PWM_WEIGHT}")
    print(f"Total cases: {total_cases}")
    print(f"Save path: {save_path}")
    print("=" * 80)

    try:
        for idx, (target, kp, ki, kd) in enumerate(sweep_combinations, start=1):
            print(
                f"[{idx}/{total_cases}] "
                f"target={target}, Kp={kp}, Ki={ki}, Kd={kd}"
            )

            try:
                df = runner.run_simulation(
                    kp=kp,
                    ki=ki,
                    kd=kd,
                    target=target,
                    stop_time=SIMULINK_SWEEP_STOP_TIME,
                    save_log=False,
                )

                metrics = calculate_metrics(df, target=target)

                row = {
                    "case_id": idx,
                    "target": target,
                    "stop_time": SIMULINK_SWEEP_STOP_TIME,
                    "kp": kp,
                    "ki": ki,
                    "kd": kd,
                    **metrics,
                    "success": True,
                    "error_message": "",
                }

            except Exception as e:
                row = {
                    "case_id": idx,
                    "target": target,
                    "stop_time": SIMULINK_SWEEP_STOP_TIME,
                    "kp": kp,
                    "ki": ki,
                    "kd": kd,
                    "final_error": np.nan,
                    "mean_abs_error": np.nan,
                    "IAE": np.nan,
                    "ISE": np.nan,
                    "overshoot": np.nan,
                    "overshoot_percent": np.nan,
                    "max_pwm": np.nan,
                    "mean_pwm": np.nan,
                    "total_pwm": np.nan,
                    "rise_time": np.nan,
                    "settling_time": np.nan,
                    "score": np.nan,
                    "success": False,
                    "error_message": str(e),
                }

                print(f"  Failed: {e}")

            results.append(row)

            # 중간 저장
            pd.DataFrame(results).to_csv(
                save_path,
                index=False,
                encoding="utf-8-sig",
            )

    finally:
        runner.stop()

    result_df = pd.DataFrame(results)

    print("=" * 80)
    print("Simulink Gain Sweep Finished")
    print(f"Saved: {save_path}")
    print("=" * 80)

    success_df = result_df[result_df["success"] == True].copy()

    if success_df.empty:
        print("\nNo successful simulation results.")
        return result_df

    # ========================================================
    # 전체 기준 출력
    # ========================================================

    print("\nTop 5 by IAE:")
    print(
        success_df.sort_values("IAE")
        .head(5)
        [
            [
                "target",
                "kp",
                "ki",
                "kd",
                "IAE",
                "score",
                "settling_time",
                "rise_time",
                "overshoot_percent",
                "mean_pwm",
            ]
        ]
    )

    print("\nTop 5 by Score:")
    print(
        success_df.sort_values("score")
        .head(5)
        [
            [
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
            ]
        ]
    )

    # ========================================================
    # Target별 Best 출력
    # ========================================================

    print("\nBest gain by target using IAE:")
    print(
        success_df.sort_values("IAE")
        .groupby("target", as_index=False)
        .head(1)
        [
            [
                "target",
                "kp",
                "ki",
                "kd",
                "IAE",
                "score",
                "settling_time",
                "rise_time",
                "overshoot_percent",
                "mean_pwm",
            ]
        ]
        .sort_values("target")
    )

    print("\nBest gain by target using Score:")
    print(
        success_df.sort_values("score")
        .groupby("target", as_index=False)
        .head(1)
        [
            [
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
            ]
        ]
        .sort_values("target")
    )

    return result_df


if __name__ == "__main__":
    run_gain_sweep()