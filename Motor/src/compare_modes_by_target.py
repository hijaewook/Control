import sys
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import LOG_DIR, FIGURE_DIR


# ============================================================
# Settings
# ============================================================

TARGET_LIST = [50, 100, 150, 200]

MODE_PATTERNS = {
    "fixed_pid": "fixed_pid_simple_motor_*.csv",
    "adaptive_pid": "adaptive_pid_simple_motor_*.csv",
    "simulink_pid": "simulink_motor_*.csv",
}


# ============================================================
# Log loading
# ============================================================

def load_latest_log_for_mode_target(mode: str, target: float) -> pd.DataFrame:
    """
    특정 mode와 target에 해당하는 최신 로그를 찾는다.
    """

    pattern = MODE_PATTERNS[mode]
    log_files = sorted(LOG_DIR.glob(pattern))

    candidates = []

    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)

            if "target" not in df.columns or len(df) == 0:
                continue

            log_target = float(df["target"].iloc[0])

            if abs(log_target - target) < 1e-6:
                candidates.append(
                    {
                        "file": log_file,
                        "mtime": log_file.stat().st_mtime,
                        "df": df,
                    }
                )

        except Exception as e:
            print(f"Skip file: {log_file}, reason: {e}")

    if not candidates:
        raise FileNotFoundError(
            f"No log found for mode={mode}, target={target}"
        )

    latest = max(candidates, key=lambda x: x["mtime"])

    print(f"Load {mode}, target={target}: {latest['file']}")

    return latest["df"]


def load_all_logs():
    """
    target별 fixed/adaptive/simulink 최신 로그 로드
    """

    logs = {}

    for target in TARGET_LIST:
        logs[target] = {}

        for mode in MODE_PATTERNS.keys():
            logs[target][mode] = load_latest_log_for_mode_target(
                mode=mode,
                target=target,
            )

    return logs


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    제어 성능 지표 계산
    """

    target = float(df["target"].iloc[0])
    time = df["time"].to_numpy()
    current = df["current"].to_numpy()
    error = df["error"].to_numpy()
    pwm = df["pwm"].to_numpy()

    abs_error = np.abs(error)

    if len(time) > 1:
        dt = np.diff(time, prepend=time[0])
        dt[0] = 0.0
    else:
        dt = np.array([0.0])

    final_error = float(abs_error[-1])
    mean_abs_error = float(np.mean(abs_error))

    iae = float(np.sum(abs_error * dt))
    ise = float(np.sum((error ** 2) * dt))

    mean_pwm = float(np.mean(np.abs(pwm)))
    total_pwm = float(np.sum(np.abs(pwm) * dt))
    max_pwm = float(np.max(np.abs(pwm)))

    max_current = float(np.max(current))
    overshoot = max(0.0, max_current - target)
    overshoot_percent = overshoot / max(abs(target), 1e-6) * 100.0

    # rise time: 10% -> 90%
    y_10 = 0.1 * target
    y_90 = 0.9 * target

    try:
        t_10 = time[np.where(current >= y_10)[0][0]]
        t_90 = time[np.where(current >= y_90)[0][0]]
        rise_time = float(t_90 - t_10)
    except IndexError:
        rise_time = np.nan

    # settling time: ±2%
    tolerance = 0.02 * abs(target)
    lower = target - tolerance
    upper = target + tolerance

    settling_time = np.nan
    within_band = (current >= lower) & (current <= upper)

    for i in range(len(time)):
        if np.all(within_band[i:]):
            settling_time = float(time[i])
            break

    return {
        "target": target,
        "final_error": final_error,
        "mean_abs_error": mean_abs_error,
        "IAE": iae,
        "ISE": ise,
        "overshoot_percent": overshoot_percent,
        "max_pwm": max_pwm,
        "mean_pwm": mean_pwm,
        "total_pwm": total_pwm,
        "rise_time": rise_time,
        "settling_time": settling_time,
    }


def build_metrics_summary(logs: dict) -> pd.DataFrame:
    rows = []

    for target, mode_dict in logs.items():
        for mode, df in mode_dict.items():
            metrics = calculate_metrics(df)
            metrics["mode"] = mode
            rows.append(metrics)

    metrics_df = pd.DataFrame(rows)

    metrics_df = metrics_df[
        [
            "target",
            "mode",
            "final_error",
            "mean_abs_error",
            "IAE",
            "ISE",
            "overshoot_percent",
            "rise_time",
            "settling_time",
            "mean_pwm",
            "total_pwm",
            "max_pwm",
        ]
    ].sort_values(["target", "mode"])

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURE_DIR / "final_mode_comparison_metrics.csv"
    metrics_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Final mode comparison metrics")
    print("=" * 80)
    print(metrics_df)
    print(f"\nSaved: {save_path}")

    return metrics_df


# ============================================================
# Plot functions
# ============================================================

def plot_response_by_target(logs: dict):
    """
    target별 fixed/adaptive/simulink response 비교
    """

    for target, mode_dict in sorted(logs.items()):
        plt.figure(figsize=(10, 5))

        for mode, df in mode_dict.items():
            plt.plot(
                df["time"],
                df["current"],
                label=mode,
            )

        plt.axhline(target, linestyle="--", label="target")

        plt.xlabel("Time [s]")
        plt.ylabel("Current")
        plt.title(f"Response Comparison at Target={target}")
        plt.legend()
        plt.grid(True)

        save_path = FIGURE_DIR / f"final_response_comparison_target_{int(target)}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        plt.show()


def plot_normalized_response_all_targets(logs: dict):
    """
    모든 target/mode의 normalized response 비교
    """

    plt.figure(figsize=(10, 6))

    for target, mode_dict in sorted(logs.items()):
        for mode, df in mode_dict.items():
            normalized_current = df["current"] / df["target"]

            plt.plot(
                df["time"],
                normalized_current,
                label=f"{mode}, T={target:.0f}",
            )

    plt.axhline(1.0, linestyle="--", label="target = 1.0")

    plt.xlabel("Time [s]")
    plt.ylabel("Current / Target")
    plt.title("Normalized Response Comparison Across Targets")
    plt.legend(ncol=2, fontsize=8)
    plt.grid(True)

    save_path = FIGURE_DIR / "final_normalized_response_all_targets.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_metric_by_target(metrics_df: pd.DataFrame):
    """
    target별 주요 metric 비교
    """

    metric_cols = [
        "IAE",
        "overshoot_percent",
        "settling_time",
        "rise_time",
        "mean_pwm",
        "total_pwm",
    ]

    for metric in metric_cols:
        plt.figure(figsize=(9, 5))

        for mode in metrics_df["mode"].unique():
            sub_df = metrics_df[metrics_df["mode"] == mode].sort_values("target")

            plt.plot(
                sub_df["target"],
                sub_df[metric],
                marker="o",
                label=mode,
            )

        plt.xlabel("Target")
        plt.ylabel(metric)
        plt.title(f"{metric} by Target and Mode")
        plt.legend()
        plt.grid(True)

        save_path = FIGURE_DIR / f"final_metric_{metric}_by_target.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        plt.show()


def plot_adaptive_gain_by_target(logs: dict):
    """
    adaptive PID의 초기/final gain target별 확인
    """

    rows = []

    for target, mode_dict in logs.items():
        df = mode_dict["adaptive_pid"]

        rows.append(
            {
                "target": target,
                "initial_kp": df["kp"].iloc[0],
                "initial_ki": df["ki"].iloc[0],
                "initial_kd": df["kd"].iloc[0],
                "final_kp": df["kp"].iloc[-1],
                "final_ki": df["ki"].iloc[-1],
                "final_kd": df["kd"].iloc[-1],
            }
        )

    gain_df = pd.DataFrame(rows).sort_values("target")

    plt.figure(figsize=(9, 5))
    plt.plot(gain_df["target"], gain_df["initial_kp"], marker="o", label="Initial Kp")
    plt.plot(gain_df["target"], gain_df["initial_ki"], marker="o", label="Initial Ki")
    plt.plot(gain_df["target"], gain_df["initial_kd"], marker="o", label="Initial Kd")

    plt.xlabel("Target")
    plt.ylabel("Initial gain")
    plt.title("Adaptive PID Initial Gains by Target")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / "final_adaptive_initial_gains_by_target.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()

    save_csv_path = FIGURE_DIR / "final_adaptive_gain_by_target.csv"
    gain_df.to_csv(save_csv_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {save_csv_path}")


# ============================================================
# Main
# ============================================================

def main():
    logs = load_all_logs()

    metrics_df = build_metrics_summary(logs)

    plot_response_by_target(logs)
    plot_normalized_response_all_targets(logs)
    plot_metric_by_target(metrics_df)
    plot_adaptive_gain_by_target(logs)


if __name__ == "__main__":
    main()