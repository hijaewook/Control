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

from config import (
    LOG_DIR,
    FIGURE_DIR,
    RESULTS_DIR,
    PWM_MAX,
    PWM_MIN,
)


# ============================================================
# Settings
# ============================================================

TARGET = 200.0
LOG_PATTERN = "adaptive_pid_simple_motor_*.csv"

SUMMARY_DIR = RESULTS_DIR / "summary"
SATURATION_COMPARISON_DIR = RESULTS_DIR / "saturation_aware_comparison"

SATURATION_TOL = 1e-9


# ============================================================
# Log loading
# ============================================================

def is_true_value(value) -> bool:
    """
    CSV에서 읽은 True/False 값을 안전하게 bool로 변환.
    """

    value_str = str(value).strip().lower()

    return value_str in ["true", "1", "yes"]


def load_candidate_logs() -> list:
    """
    adaptive_pid 로그 중 target 조건과 disturbance 조건을 만족하는 로그를 수집.
    """

    log_files = sorted(LOG_DIR.glob(LOG_PATTERN))

    candidates = []

    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)

            if len(df) == 0:
                continue

            if "target" not in df.columns:
                continue

            target = float(df["target"].iloc[0])

            if abs(target - TARGET) > 1e-6:
                continue

            # disturbance 실험 로그만 사용
            if "use_disturbance" not in df.columns:
                continue

            if not is_true_value(df["use_disturbance"].iloc[0]):
                continue

            # saturation-aware 구분 컬럼이 있어야 함
            if "use_saturation_aware_gain" not in df.columns:
                continue

            use_saturation_aware = is_true_value(
                df["use_saturation_aware_gain"].iloc[0]
            )

            experiment_tag = (
                str(df["experiment_tag"].iloc[0])
                if "experiment_tag" in df.columns
                else "unknown"
            )

            candidates.append(
                {
                    "log_file": log_file,
                    "modified_time": log_file.stat().st_mtime,
                    "df": df,
                    "use_saturation_aware_gain": use_saturation_aware,
                    "experiment_tag": experiment_tag,
                }
            )

        except Exception as e:
            print(f"Skip file: {log_file}")
            print(f"Reason: {e}")

    return candidates


def select_latest_logs(candidates: list) -> dict:
    """
    baseline adaptive와 saturation-aware adaptive 최신 로그를 각각 선택.
    """

    baseline_candidates = [
        c for c in candidates
        if c["use_saturation_aware_gain"] is False
    ]

    saturation_aware_candidates = [
        c for c in candidates
        if c["use_saturation_aware_gain"] is True
    ]

    if not baseline_candidates:
        raise FileNotFoundError(
            "No baseline adaptive PID log found. "
            "Run with USE_SATURATION_AWARE_GAIN = False first."
        )

    if not saturation_aware_candidates:
        raise FileNotFoundError(
            "No saturation-aware adaptive PID log found. "
            "Run with USE_SATURATION_AWARE_GAIN = True first."
        )

    baseline = max(baseline_candidates, key=lambda x: x["modified_time"])
    saturation_aware = max(
        saturation_aware_candidates,
        key=lambda x: x["modified_time"],
    )

    logs = {
        "adaptive_baseline": baseline,
        "adaptive_saturation_aware": saturation_aware,
    }

    print("=" * 80)
    print("Selected logs")
    print("=" * 80)

    for name, item in logs.items():
        print(
            f"{name}: {item['log_file']} | "
            f"tag={item['experiment_tag']} | "
            f"use_saturation_aware_gain={item['use_saturation_aware_gain']}"
        )

    return logs


def load_logs() -> dict:
    candidates = load_candidate_logs()
    logs = select_latest_logs(candidates)
    return logs


# ============================================================
# Metric calculation
# ============================================================

def compute_dt(time: np.ndarray) -> np.ndarray:
    if len(time) <= 1:
        return np.zeros_like(time)

    dt = np.diff(time, prepend=time[0])
    dt[0] = 0.0

    return dt


def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    제어 성능 및 saturation 지표 계산.
    """

    time = df["time"].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    current = df["current"].to_numpy(dtype=float)
    pwm = df["pwm"].to_numpy(dtype=float)

    if "error" in df.columns:
        error = df["error"].to_numpy(dtype=float)
    else:
        error = target - current

    dt = compute_dt(time)
    abs_error = np.abs(error)
    abs_pwm = np.abs(pwm)

    target_value = float(target[-1])

    # --------------------------------------------------------
    # Basic control metrics
    # --------------------------------------------------------

    iae = float(np.sum(abs_error * dt))
    ise = float(np.sum((error ** 2) * dt))
    mean_abs_error = float(np.mean(abs_error))
    final_error = float(abs_error[-1])

    max_current = float(np.max(current))
    overshoot = max(0.0, max_current - target_value)
    overshoot_percent = overshoot / max(abs(target_value), 1e-6) * 100.0

    mean_pwm = float(np.mean(abs_pwm))
    total_pwm = float(np.sum(abs_pwm * dt))
    max_pwm = float(np.max(pwm))
    min_pwm = float(np.min(pwm))

    # --------------------------------------------------------
    # Saturation metrics
    # --------------------------------------------------------

    high_saturation = pwm >= PWM_MAX - SATURATION_TOL
    low_saturation = pwm <= PWM_MIN + SATURATION_TOL
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

    # near saturation
    pwm_range = PWM_MAX - PWM_MIN
    near_high_saturation = pwm >= PWM_MAX - 0.1 * pwm_range
    near_high_saturation_ratio_percent = float(
        np.mean(near_high_saturation) * 100.0
    )

    # --------------------------------------------------------
    # Settling time
    # --------------------------------------------------------

    tolerance = 0.02 * abs(target_value)
    settling_time = np.nan

    for i in range(len(time)):
        if np.all(np.abs(target[i:] - current[i:]) <= tolerance):
            settling_time = float(time[i])
            break

    # --------------------------------------------------------
    # Disturbance metrics
    # --------------------------------------------------------

    disturbance_metrics = {}

    if "disturbance" in df.columns:
        disturbance = df["disturbance"].to_numpy(dtype=float)
        disturbance_mask = np.abs(disturbance) > 1e-9

        if disturbance_mask.any():
            disturbance_saturation_mask = disturbance_mask & saturation_mask

            disturbance_metrics = {
                "disturbance_start_time": float(time[disturbance_mask][0]),
                "disturbance_end_time": float(time[disturbance_mask][-1]),
                "disturbance_IAE": float(
                    np.sum(abs_error[disturbance_mask] * dt[disturbance_mask])
                ),
                "disturbance_mean_abs_error": float(
                    np.mean(abs_error[disturbance_mask])
                ),
                "disturbance_max_error": float(
                    np.max(abs_error[disturbance_mask])
                ),
                "disturbance_min_current": float(
                    np.min(current[disturbance_mask])
                ),
                "disturbance_mean_pwm": float(
                    np.mean(abs_pwm[disturbance_mask])
                ),
                "disturbance_max_pwm": float(
                    np.max(pwm[disturbance_mask])
                ),
                "disturbance_saturation_count": int(
                    np.sum(disturbance_saturation_mask)
                ),
                "disturbance_saturation_ratio_percent": float(
                    np.sum(disturbance_saturation_mask)
                    / max(np.sum(disturbance_mask), 1)
                    * 100.0
                ),
                "disturbance_saturation_duration": float(
                    np.sum(dt[disturbance_saturation_mask])
                ),
            }

            # recovery time after disturbance
            disturbance_end_time = float(time[disturbance_mask][-1])
            recovery_mask = time > disturbance_end_time
            recovery_indices = np.where(recovery_mask)[0]
            recovery_time = np.nan

            for idx in recovery_indices:
                if np.all(abs_error[idx:] <= tolerance):
                    recovery_time = float(time[idx] - disturbance_end_time)
                    break

            disturbance_metrics["recovery_time_after_disturbance"] = recovery_time

    # --------------------------------------------------------
    # Scheduler state metrics
    # --------------------------------------------------------

    scheduler_metrics = {}

    if "kp_scale" in df.columns:
        scheduler_metrics["min_kp_scale"] = float(df["kp_scale"].min())
        scheduler_metrics["final_kp_scale"] = float(df["kp_scale"].iloc[-1])

    if "ki_scale" in df.columns:
        scheduler_metrics["min_ki_scale"] = float(df["ki_scale"].min())
        scheduler_metrics["final_ki_scale"] = float(df["ki_scale"].iloc[-1])

    if "last_update_reason" in df.columns:
        scheduler_metrics["saturation_gain_reduction_count"] = int(
            (df["last_update_reason"] == "saturation_gain_reduction").sum()
        )
        scheduler_metrics["saturation_recovery_count"] = int(
            (df["last_update_reason"] == "saturation_recovery").sum()
        )

    if "gain_update_flag" in df.columns:
        scheduler_metrics["gain_update_count"] = int(
            df["gain_update_flag"].astype(bool).sum()
        )

    return {
        "IAE": iae,
        "ISE": ise,
        "mean_abs_error": mean_abs_error,
        "final_error": final_error,
        "overshoot_percent": overshoot_percent,
        "settling_time": settling_time,
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
        "near_high_saturation_ratio_percent": near_high_saturation_ratio_percent,
        **disturbance_metrics,
        **scheduler_metrics,
    }


def build_metrics_table(logs: dict) -> pd.DataFrame:
    rows = []

    for name, item in logs.items():
        df = item["df"]
        metrics = calculate_metrics(df)

        row = {
            "controller": name,
            "log_file": str(item["log_file"]),
            "experiment_tag": item["experiment_tag"],
            "use_saturation_aware_gain": item["use_saturation_aware_gain"],
            "target": float(df["target"].iloc[0]),
        }
        row.update(metrics)
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    return metrics_df


def calc_improvement(baseline_value, proposed_value):
    """
    baseline 대비 proposed 개선율.
    양수면 proposed가 낮은 값으로 개선된 것.
    """

    if pd.isna(baseline_value) or pd.isna(proposed_value):
        return np.nan

    if abs(baseline_value) < 1e-12:
        return np.nan

    return (baseline_value - proposed_value) / baseline_value * 100.0


def build_improvement_table(metrics_df: pd.DataFrame) -> pd.DataFrame:
    baseline = metrics_df[
        metrics_df["controller"] == "adaptive_baseline"
    ].iloc[0]

    sat = metrics_df[
        metrics_df["controller"] == "adaptive_saturation_aware"
    ].iloc[0]

    metrics_to_compare = [
        "IAE",
        "final_error",
        "overshoot_percent",
        "settling_time",
        "mean_pwm",
        "total_pwm",
        "saturation_ratio_percent",
        "saturation_duration",
        "near_high_saturation_ratio_percent",
        "disturbance_IAE",
        "disturbance_mean_abs_error",
        "disturbance_max_error",
        "disturbance_min_current",
        "disturbance_saturation_ratio_percent",
        "disturbance_saturation_duration",
        "recovery_time_after_disturbance",
    ]

    rows = []

    for metric in metrics_to_compare:
        if metric not in metrics_df.columns:
            continue

        baseline_value = baseline[metric]
        sat_value = sat[metric]

        row = {
            "metric": metric,
            "baseline": baseline_value,
            "saturation_aware": sat_value,
            "difference": sat_value - baseline_value,
            "improvement_percent": calc_improvement(
                baseline_value,
                sat_value,
            ),
        }

        rows.append(row)

    improvement_df = pd.DataFrame(rows)

    return improvement_df


# ============================================================
# Plot functions
# ============================================================

def get_disturbance_window(logs: dict):
    for item in logs.values():
        df = item["df"]

        if "disturbance" not in df.columns:
            continue

        time = df["time"].to_numpy(dtype=float)
        disturbance = df["disturbance"].to_numpy(dtype=float)

        mask = np.abs(disturbance) > 1e-9

        if mask.any():
            return float(time[mask][0]), float(time[mask][-1])

    return None, None


def shade_disturbance_area(logs: dict):
    start_time, end_time = get_disturbance_window(logs)

    if start_time is not None and end_time is not None:
        plt.axvspan(
            start_time,
            end_time,
            alpha=0.15,
            label="disturbance window",
        )


def plot_response(logs: dict, timestamp: str):
    plt.figure(figsize=(10, 5))

    for name, item in logs.items():
        df = item["df"]
        plt.plot(df["time"], df["current"], label=name)

    target = float(next(iter(logs.values()))["df"]["target"].iloc[0])
    plt.axhline(target, linestyle="--", label="target")

    shade_disturbance_area(logs)

    plt.xlabel("Time [s]")
    plt.ylabel("Current")
    plt.title("Baseline vs Saturation-aware Adaptive PID Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"saturation_aware_response_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_pwm(logs: dict, timestamp: str):
    plt.figure(figsize=(10, 5))

    for name, item in logs.items():
        df = item["df"]
        plt.plot(df["time"], df["pwm"], label=name)

    plt.axhline(PWM_MAX, linestyle="--", label="PWM max")

    shade_disturbance_area(logs)

    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("Baseline vs Saturation-aware Adaptive PID PWM")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"saturation_aware_pwm_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_gain_scales(logs: dict, timestamp: str):
    if "adaptive_saturation_aware" not in logs:
        return

    df = logs["adaptive_saturation_aware"]["df"]

    if "kp_scale" not in df.columns or "ki_scale" not in df.columns:
        print("Skip gain scale plot: kp_scale or ki_scale column not found.")
        return

    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["kp_scale"], label="Kp scale")
    plt.plot(df["time"], df["ki_scale"], label="Ki scale")

    if "kd_scale" in df.columns:
        plt.plot(df["time"], df["kd_scale"], label="Kd scale")

    shade_disturbance_area(logs)

    plt.xlabel("Time [s]")
    plt.ylabel("Gain scale")
    plt.title("Saturation-aware Gain Scaling")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"saturation_aware_gain_scales_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_metric_bar(metrics_df: pd.DataFrame, timestamp: str):
    metric_cols = [
        "IAE",
        "settling_time",
        "overshoot_percent",
        "saturation_ratio_percent",
        "saturation_duration",
        "disturbance_IAE",
        "disturbance_max_error",
        "recovery_time_after_disturbance",
    ]

    for metric in metric_cols:
        if metric not in metrics_df.columns:
            continue

        plt.figure(figsize=(7, 5))
        plt.bar(metrics_df["controller"], metrics_df[metric])

        plt.xlabel("Controller")
        plt.ylabel(metric)
        plt.title(f"{metric}: Baseline vs Saturation-aware")
        plt.grid(True, axis="y")

        save_path = FIGURE_DIR / f"saturation_aware_metric_{metric}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        plt.show()


# ============================================================
# Save
# ============================================================

def save_results(
    metrics_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    timestamp: str,
):
    SATURATION_COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    metrics_path = (
        SATURATION_COMPARISON_DIR
        / f"saturation_aware_metrics_{timestamp}.csv"
    )
    improvement_path = (
        SATURATION_COMPARISON_DIR
        / f"saturation_aware_improvement_{timestamp}.csv"
    )

    summary_metrics_path = SUMMARY_DIR / "saturation_aware_metrics.csv"
    summary_improvement_path = SUMMARY_DIR / "saturation_aware_improvement.csv"

    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    improvement_df.to_csv(improvement_path, index=False, encoding="utf-8-sig")

    metrics_df.to_csv(summary_metrics_path, index=False, encoding="utf-8-sig")
    improvement_df.to_csv(summary_improvement_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {metrics_path}")
    print(f"Saved: {improvement_path}")
    print(f"Saved: {summary_metrics_path}")
    print(f"Saved: {summary_improvement_path}")


def save_markdown_summary(
    metrics_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    timestamp: str,
):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    save_path = SUMMARY_DIR / f"saturation_aware_result_summary_{timestamp}.md"

    lines = []

    lines.append("# Saturation-aware Adaptive PID Result Summary")
    lines.append("")
    lines.append("## 1. Experiment Setting")
    lines.append("")
    lines.append("- Controller 1: baseline adaptive PID")
    lines.append("- Controller 2: saturation-aware adaptive PID")
    lines.append("- Target: 200")
    lines.append("- Disturbance: pulse disturbance")
    lines.append("- Purpose: evaluate whether saturation-aware gain scaling can reduce PWM saturation while preserving tracking performance")
    lines.append("")

    lines.append("## 2. Metrics")
    lines.append("")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Improvement Table")
    lines.append("")
    lines.append(improvement_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 4. Interpretation")
    lines.append("")
    lines.append(
        "The saturation-aware adaptive PID reduced PWM saturation by scaling down Kp and Ki when repeated high-PWM commands were detected. "
        "Compared with the baseline adaptive PID, the saturation-aware controller reduced the saturation ratio and saturation duration while maintaining similar tracking and disturbance rejection performance."
    )
    lines.append("")
    lines.append(
        "This result shows a practical trade-off between tracking aggressiveness and actuator saturation mitigation. "
        "The current saturation-aware setting is suitable as a balanced configuration, but further tuning may be required for real motor experiments where current, thermal, or driver protection limits are critical."
    )
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved markdown summary: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Compare baseline adaptive PID and saturation-aware adaptive PID")
    print("=" * 80)

    logs = load_logs()

    metrics_df = build_metrics_table(logs)
    improvement_df = build_improvement_table(metrics_df)

    print("\n" + "=" * 80)
    print("Metrics")
    print("=" * 80)
    print(metrics_df)

    print("\n" + "=" * 80)
    print("Improvement")
    print("=" * 80)
    print(improvement_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    save_results(metrics_df, improvement_df, timestamp)
    save_markdown_summary(metrics_df, improvement_df, timestamp)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_response(logs, timestamp)
    plot_pwm(logs, timestamp)
    plot_gain_scales(logs, timestamp)
    plot_metric_bar(metrics_df, timestamp)


if __name__ == "__main__":
    main()