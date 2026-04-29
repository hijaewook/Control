import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from pid_controller import PIDController
from gain_scheduler import GainScheduler
from motor_env import SimpleMotorEnv
from config import (
    DT,
    PWM_MIN,
    PWM_MAX,
    RESULTS_DIR,
    FIGURE_DIR,
)


# ============================================================
# Delay-aware experiment settings
# ============================================================

RESULT_DIR = RESULTS_DIR / "delay_aware"

SIM_TIME = 10.0
N_STEPS = int(SIM_TIME / DT)

TARGET_BEFORE = 100.0
TARGET_AFTER = 200.0
TARGET_CHANGE_TIME = 3.0

# 여러 inference delay 조건
INFERENCE_DELAY_LIST = [0.0, 0.2, 0.5, 1.0, 1.5, 2.0]

CONTROL_MODES = [
    "fixed_pid",
    "immediate_gain_db",
    "delayed_gain_db",
]


# ============================================================
# Utility functions
# ============================================================

def get_target_at_time(t: float) -> float:
    if t < TARGET_CHANGE_TIME:
        return TARGET_BEFORE
    return TARGET_AFTER


def get_gain_from_db(target: float):
    scheduler = GainScheduler(target=target)
    kp, ki, kd = scheduler.get_gains()
    return float(kp), float(ki), float(kd)


def calculate_metrics(df: pd.DataFrame, inference_delay: float) -> dict:
    time = df["time"].to_numpy()
    target = df["target"].to_numpy()
    current = df["current"].to_numpy()
    error = df["error"].to_numpy()
    pwm = df["pwm"].to_numpy()

    abs_error = np.abs(error)

    if len(time) > 1:
        dt_array = np.diff(time, prepend=time[0])
        dt_array[0] = 0.0
    else:
        dt_array = np.array([0.0])

    iae = float(np.sum(abs_error * dt_array))
    ise = float(np.sum((error ** 2) * dt_array))
    mean_abs_error = float(np.mean(abs_error))
    final_error = float(abs_error[-1])

    mean_pwm = float(np.mean(np.abs(pwm)))
    total_pwm = float(np.sum(np.abs(pwm) * dt_array))
    max_pwm = float(np.max(np.abs(pwm)))

    pwm_saturation_ratio = float(np.mean(np.abs(pwm) >= PWM_MAX - 1e-9) * 100.0)

    after_mask = time >= TARGET_CHANGE_TIME

    if after_mask.any():
        after_iae = float(np.sum(abs_error[after_mask] * dt_array[after_mask]))
        after_mean_abs_error = float(np.mean(abs_error[after_mask]))
        after_max_error = float(np.max(abs_error[after_mask]))
    else:
        after_iae = np.nan
        after_mean_abs_error = np.nan
        after_max_error = np.nan

    after_current = current[after_mask]

    if len(after_current) > 0:
        max_after_current = float(np.max(after_current))
        overshoot = max(0.0, max_after_current - TARGET_AFTER)
        overshoot_percent = overshoot / max(abs(TARGET_AFTER), 1e-6) * 100.0
    else:
        overshoot_percent = np.nan

    tolerance = 0.02 * abs(TARGET_AFTER)
    settling_time_after_change = np.nan

    after_indices = np.where(after_mask)[0]

    for idx in after_indices:
        if np.all(np.abs(target[idx:] - current[idx:]) <= tolerance):
            settling_time_after_change = float(time[idx] - TARGET_CHANGE_TIME)
            break

    return {
        "inference_delay": inference_delay,
        "IAE": iae,
        "ISE": ise,
        "mean_abs_error": mean_abs_error,
        "final_error": final_error,
        "after_change_IAE": after_iae,
        "after_change_mean_abs_error": after_mean_abs_error,
        "after_change_max_error": after_max_error,
        "overshoot_percent_after_change": overshoot_percent,
        "settling_time_after_change": settling_time_after_change,
        "mean_pwm": mean_pwm,
        "total_pwm": total_pwm,
        "max_pwm": max_pwm,
        "pwm_saturation_ratio_percent": pwm_saturation_ratio,
    }


# ============================================================
# Simulation
# ============================================================

def run_single_mode(mode: str, inference_delay: float) -> pd.DataFrame:
    if mode not in CONTROL_MODES:
        raise ValueError(f"Unknown mode: {mode}")

    env = SimpleMotorEnv(
        initial_value=0.0,
        dt=DT,
        pwm_min=PWM_MIN,
        pwm_max=PWM_MAX,
        use_disturbance=False,
        disturbance_mode="none",
    )

    initial_target = TARGET_BEFORE
    initial_kp, initial_ki, initial_kd = get_gain_from_db(initial_target)

    pid = PIDController()

    if mode == "fixed_pid":
        # fixed는 PIDController 기본 gain 사용
        pass
    else:
        pid.set_gains(initial_kp, initial_ki, initial_kd)

    current = env.get_state()
    prev_error = initial_target - current

    target_change_detected = False
    gain_request_time = np.nan
    gain_arrival_time = np.nan
    pending_gain = None
    delayed_gain_applied = False

    rows = []

    for step in range(N_STEPS):
        t = step * DT
        target = get_target_at_time(t)

        target_changed_now = (
            (not target_change_detected)
            and t >= TARGET_CHANGE_TIME
        )

        if target_changed_now:
            target_change_detected = True

            new_kp, new_ki, new_kd = get_gain_from_db(target)

            if mode == "immediate_gain_db":
                pid.set_gains(new_kp, new_ki, new_kd)

            elif mode == "delayed_gain_db":
                gain_request_time = t
                gain_arrival_time = t + inference_delay
                pending_gain = (new_kp, new_ki, new_kd)

        gain_update_event = False

        if mode == "delayed_gain_db":
            if (
                target_change_detected
                and not delayed_gain_applied
                and pending_gain is not None
                and t >= gain_arrival_time
            ):
                kp_new, ki_new, kd_new = pending_gain
                pid.set_gains(kp_new, ki_new, kd_new)
                delayed_gain_applied = True
                gain_update_event = True

        error = target - current
        error_derivative = (error - prev_error) / DT

        pwm = pid.compute(target, current)

        pwm_range = PWM_MAX - PWM_MIN
        high_saturation = pwm > PWM_MAX - 0.1 * pwm_range
        low_saturation = pwm < PWM_MIN + 0.02 * pwm_range
        pwm_saturated = high_saturation

        pid_state = pid.get_state()
        kp = float(pid_state["kp"])
        ki = float(pid_state["ki"])
        kd = float(pid_state["kd"])
        integral = float(pid_state["integral"])

        current = env.step(pwm)

        rows.append(
            {
                "mode": mode,
                "step": step,
                "time": t,
                "target": target,
                "current": current,
                "error": error,
                "error_derivative": error_derivative,
                "pwm": pwm,
                "pwm_saturated": pwm_saturated,
                "high_saturation": high_saturation,
                "low_saturation": low_saturation,
                "kp": kp,
                "ki": ki,
                "kd": kd,
                "integral": integral,
                "target_change_time": TARGET_CHANGE_TIME,
                "inference_delay": inference_delay,
                "gain_request_time": gain_request_time,
                "gain_arrival_time": gain_arrival_time,
                "gain_update_event": gain_update_event,
            }
        )

        prev_error = error

    return pd.DataFrame(rows)


def run_all_delay_conditions():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_logs = {}
    metric_rows = []

    for inference_delay in INFERENCE_DELAY_LIST:
        print("=" * 80)
        print(f"Run inference delay: {inference_delay:.2f} s")
        print("=" * 80)

        all_logs[inference_delay] = {}

        for mode in CONTROL_MODES:
            print("-" * 80)
            print(f"Run mode: {mode}, delay={inference_delay:.2f}")

            df = run_single_mode(
                mode=mode,
                inference_delay=inference_delay,
            )

            all_logs[inference_delay][mode] = df

            save_path = (
                RESULT_DIR
                / f"delay_aware_{mode}_delay_{inference_delay:.2f}_{timestamp}.csv"
            )
            df.to_csv(save_path, index=False, encoding="utf-8-sig")
            print(f"Saved: {save_path}")

            metrics = calculate_metrics(
                df=df,
                inference_delay=inference_delay,
            )
            metrics["mode"] = mode
            metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)

    metrics_path = RESULT_DIR / f"delay_aware_metrics_sweep_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved metrics: {metrics_path}")

    return all_logs, metrics_df, timestamp


# ============================================================
# Plot functions
# ============================================================

def shade_delay_region(delay: float):
    plt.axvline(
        TARGET_CHANGE_TIME,
        linestyle="--",
        linewidth=1.5,
        label="target change",
    )

    if delay > 0:
        plt.axvspan(
            TARGET_CHANGE_TIME,
            TARGET_CHANGE_TIME + delay,
            alpha=0.15,
            label=f"delay window ({delay:.2f}s)",
        )


def plot_response_for_each_delay(all_logs: dict, timestamp: str):
    for delay, mode_logs in all_logs.items():
        plt.figure(figsize=(10, 5))

        for mode, df in mode_logs.items():
            plt.plot(df["time"], df["current"], label=mode)

        time_arr = next(iter(mode_logs.values()))["time"].to_numpy()
        target_profile = np.array([get_target_at_time(t) for t in time_arr])
        plt.plot(time_arr, target_profile, linestyle="--", label="target")

        shade_delay_region(delay)

        plt.xlabel("Time [s]")
        plt.ylabel("Current")
        plt.title(f"Delay-aware Response Comparison, delay={delay:.2f}s")
        plt.legend()
        plt.grid(True)

        save_path = FIGURE_DIR / f"delay_sweep_response_delay_{delay:.2f}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        plt.show()


def plot_delayed_response_across_delays(all_logs: dict, timestamp: str):
    plt.figure(figsize=(10, 5))

    for delay, mode_logs in sorted(all_logs.items()):
        df = mode_logs["delayed_gain_db"]
        plt.plot(
            df["time"],
            df["current"],
            label=f"delay={delay:.2f}s",
        )

    time_arr = next(iter(next(iter(all_logs.values())).values()))["time"].to_numpy()
    target_profile = np.array([get_target_at_time(t) for t in time_arr])
    plt.plot(time_arr, target_profile, linestyle="--", label="target")

    plt.axvline(TARGET_CHANGE_TIME, linestyle="--", linewidth=1.5)

    plt.xlabel("Time [s]")
    plt.ylabel("Current")
    plt.title("Delayed Gain DB Response Across Inference Delays")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"delay_sweep_delayed_response_all_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_metric_by_delay(metrics_df: pd.DataFrame, timestamp: str):
    metric_cols = [
        "after_change_IAE",
        "after_change_max_error",
        "settling_time_after_change",
        "overshoot_percent_after_change",
        "total_pwm",
        "pwm_saturation_ratio_percent",
    ]

    for metric in metric_cols:
        if metric not in metrics_df.columns:
            continue

        plt.figure(figsize=(9, 5))

        for mode in CONTROL_MODES:
            sub_df = metrics_df[metrics_df["mode"] == mode].sort_values("inference_delay")

            plt.plot(
                sub_df["inference_delay"],
                sub_df[metric],
                marker="o",
                label=mode,
            )

        plt.xlabel("Inference delay [s]")
        plt.ylabel(metric)
        plt.title(f"{metric} vs Inference Delay")
        plt.legend()
        plt.grid(True)

        save_path = FIGURE_DIR / f"delay_sweep_metric_{metric}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        plt.show()


def save_summary_table(metrics_df: pd.DataFrame, timestamp: str):
    """
    delayed_gain_db만 따로 요약 저장
    """

    delayed_df = (
        metrics_df[metrics_df["mode"] == "delayed_gain_db"]
        .sort_values("inference_delay")
        .reset_index(drop=True)
    )

    save_path = RESULT_DIR / f"delay_sweep_delayed_summary_{timestamp}.csv"
    delayed_df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Saved delayed summary: {save_path}")

    print("\n" + "=" * 80)
    print("Delayed gain DB summary")
    print("=" * 80)
    print(
        delayed_df[
            [
                "inference_delay",
                "after_change_IAE",
                "after_change_max_error",
                "settling_time_after_change",
                "overshoot_percent_after_change",
                "total_pwm",
                "pwm_saturation_ratio_percent",
            ]
        ]
    )


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Delay-aware gain update sweep simulation")
    print("=" * 80)
    print(f"Target: {TARGET_BEFORE} -> {TARGET_AFTER}")
    print(f"Target change time: {TARGET_CHANGE_TIME}")
    print(f"Inference delay list: {INFERENCE_DELAY_LIST}")
    print("=" * 80)

    all_logs, metrics_df, timestamp = run_all_delay_conditions()

    print("\n" + "=" * 80)
    print("Delay-aware metrics sweep")
    print("=" * 80)
    print(metrics_df)

    plot_response_for_each_delay(all_logs, timestamp)
    plot_delayed_response_across_delays(all_logs, timestamp)
    plot_metric_by_delay(metrics_df, timestamp)
    save_summary_table(metrics_df, timestamp)


if __name__ == "__main__":
    main()