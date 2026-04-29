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

LOG_PATTERN = "adaptive_pid_simple_motor_*.csv"

# None이면 adaptive 로그에 존재하는 모든 target을 사용
# 특정 target만 보고 싶으면 예: TARGET_FILTER = [50, 100, 120, 150, 180, 200]
TARGET_FILTER = None


# ============================================================
# Log loading
# ============================================================

def load_adaptive_logs():
    """
    adaptive_pid_simple_motor_*.csv 로그를 모두 읽고,
    target별 최신 로그만 선택한다.
    """

    log_files = sorted(LOG_DIR.glob(LOG_PATTERN))

    if not log_files:
        raise FileNotFoundError(f"No adaptive PID logs found in: {LOG_DIR}")

    records = []

    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)

            if "target" not in df.columns or len(df) == 0:
                continue

            target = float(df["target"].iloc[0])

            if TARGET_FILTER is not None and target not in TARGET_FILTER:
                continue

            records.append(
                {
                    "target": target,
                    "log_file": log_file,
                    "modified_time": log_file.stat().st_mtime,
                    "df": df,
                }
            )

        except Exception as e:
            print(f"Skip log file: {log_file}")
            print(f"Reason: {e}")

    if not records:
        raise FileNotFoundError("No valid adaptive PID logs found.")

    # target별 최신 파일만 선택
    record_df = pd.DataFrame(
        [
            {
                "target": r["target"],
                "log_file": r["log_file"],
                "modified_time": r["modified_time"],
            }
            for r in records
        ]
    )

    latest_records = []

    for target in sorted(record_df["target"].unique()):
        target_records = [r for r in records if r["target"] == target]
        latest_record = max(target_records, key=lambda x: x["modified_time"])
        latest_records.append(latest_record)

    print("=" * 80)
    print("Loaded adaptive PID logs")
    print("=" * 80)

    logs = {}

    for record in latest_records:
        target = record["target"]
        logs[target] = record["df"]

        print(f"target={target:.1f} | {record['log_file']}")

    return logs


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    adaptive PID 로그에서 제어 성능 지표 계산
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

    # gain 정보
    initial_kp = float(df["kp"].iloc[0]) if "kp" in df.columns else np.nan
    initial_ki = float(df["ki"].iloc[0]) if "ki" in df.columns else np.nan
    initial_kd = float(df["kd"].iloc[0]) if "kd" in df.columns else np.nan

    final_kp = float(df["kp"].iloc[-1]) if "kp" in df.columns else np.nan
    final_ki = float(df["ki"].iloc[-1]) if "ki" in df.columns else np.nan
    final_kd = float(df["kd"].iloc[-1]) if "kd" in df.columns else np.nan

    gain_update_count = (
        int(df["gain_update_flag"].astype(bool).sum())
        if "gain_update_flag" in df.columns
        else 0
    )

    return {
        "target": target,
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
        "initial_kp": initial_kp,
        "initial_ki": initial_ki,
        "initial_kd": initial_kd,
        "final_kp": final_kp,
        "final_ki": final_ki,
        "final_kd": final_kd,
        "gain_update_count": gain_update_count,
    }


def save_metrics_summary(logs: dict) -> pd.DataFrame:
    """
    target별 metrics summary 저장
    """

    rows = []

    for target, df in logs.items():
        rows.append(calculate_metrics(df))

    metrics_df = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURE_DIR / "adaptive_target_metrics_summary.csv"

    metrics_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Adaptive target metrics summary")
    print("=" * 80)
    print(metrics_df)
    print(f"\nSaved: {save_path}")

    return metrics_df


# ============================================================
# Plot functions
# ============================================================

def plot_response(logs: dict, save: bool = True):
    """
    target별 response 비교
    """

    plt.figure(figsize=(10, 5))

    for target, df in sorted(logs.items()):
        plt.plot(
            df["time"],
            df["current"],
            label=f"target={target:.0f}",
        )

    # target line은 각 target이 다르므로 current curve만 비교
    plt.xlabel("Time [s]")
    plt.ylabel("Current")
    plt.title("Adaptive PID Response by Target")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "adaptive_targets_response.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_normalized_response(logs: dict, save: bool = True):
    """
    target별 normalized response 비교
    current / target 기준
    """

    plt.figure(figsize=(10, 5))

    for target, df in sorted(logs.items()):
        normalized_current = df["current"] / df["target"]

        plt.plot(
            df["time"],
            normalized_current,
            label=f"target={target:.0f}",
        )

    plt.axhline(1.0, linestyle="--", label="target = 1.0")

    plt.xlabel("Time [s]")
    plt.ylabel("Current / Target")
    plt.title("Adaptive PID Normalized Response by Target")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "adaptive_targets_normalized_response.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_error(logs: dict, save: bool = True):
    """
    target별 error 비교
    """

    plt.figure(figsize=(10, 5))

    for target, df in sorted(logs.items()):
        plt.plot(
            df["time"],
            df["error"],
            label=f"target={target:.0f}",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Error")
    plt.title("Adaptive PID Error by Target")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "adaptive_targets_error.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_normalized_error(logs: dict, save: bool = True):
    """
    target별 normalized error 비교
    error / target 기준
    """

    plt.figure(figsize=(10, 5))

    for target, df in sorted(logs.items()):
        normalized_error = df["error"] / df["target"]

        plt.plot(
            df["time"],
            normalized_error,
            label=f"target={target:.0f}",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Error / Target")
    plt.title("Adaptive PID Normalized Error by Target")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "adaptive_targets_normalized_error.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_pwm(logs: dict, save: bool = True):
    """
    target별 PWM 비교
    """

    plt.figure(figsize=(10, 5))

    for target, df in sorted(logs.items()):
        plt.plot(
            df["time"],
            df["pwm"],
            label=f"target={target:.0f}",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("Adaptive PID PWM by Target")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "adaptive_targets_pwm.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_gains(logs: dict, save: bool = True):
    """
    target별 Kp/Ki/Kd 변화 비교
    """

    for gain_col in ["kp", "ki", "kd"]:
        plt.figure(figsize=(10, 5))

        for target, df in sorted(logs.items()):
            if gain_col not in df.columns:
                continue

            plt.plot(
                df["time"],
                df[gain_col],
                label=f"target={target:.0f}",
            )

        plt.xlabel("Time [s]")
        plt.ylabel(gain_col)
        plt.title(f"Adaptive PID {gain_col.upper()} by Target")
        plt.legend()
        plt.grid(True)

        if save:
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            save_path = FIGURE_DIR / f"adaptive_targets_{gain_col}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()


def plot_gain_update_flag(logs: dict, save: bool = True):
    """
    target별 gain update flag 비교
    """

    if not all("gain_update_flag" in df.columns for df in logs.values()):
        print("Skip gain_update_flag plot: column not found.")
        return

    plt.figure(figsize=(10, 5))

    for target, df in sorted(logs.items()):
        plt.plot(
            df["time"],
            df["gain_update_flag"].astype(int),
            label=f"target={target:.0f}",
        )

    plt.xlabel("Time [s]")
    plt.ylabel("Gain update flag")
    plt.title("Adaptive PID Gain Update Flag by Target")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "adaptive_targets_gain_update_flag.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_metrics_summary(metrics_df: pd.DataFrame, save: bool = True):
    """
    target별 주요 metrics 요약 그래프
    """

    metric_cols = [
        "IAE",
        "overshoot_percent",
        "mean_pwm",
        "settling_time",
        "rise_time",
    ]

    for metric in metric_cols:
        if metric not in metrics_df.columns:
            continue

        plt.figure(figsize=(8, 5))
        plt.plot(
            metrics_df["target"],
            metrics_df[metric],
            marker="o",
        )

        plt.xlabel("Target")
        plt.ylabel(metric)
        plt.title(f"Adaptive PID {metric} by Target")
        plt.grid(True)

        if save:
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            save_path = FIGURE_DIR / f"adaptive_targets_metric_{metric}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()


# ============================================================
# Main
# ============================================================

def main():
    logs = load_adaptive_logs()

    metrics_df = save_metrics_summary(logs)

    plot_response(logs)
    plot_normalized_response(logs)
    plot_error(logs)
    plot_normalized_error(logs)
    plot_pwm(logs)
    plot_gains(logs)
    plot_gain_update_flag(logs)
    plot_metrics_summary(metrics_df)


if __name__ == "__main__":
    main()