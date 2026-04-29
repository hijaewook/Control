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

TARGET = 200.0

MODE_PATTERNS = {
    "fixed_pid": "fixed_pid_simple_motor_*.csv",
    "adaptive_pid": "adaptive_pid_simple_motor_*.csv",
}


# ============================================================
# Log loading
# ============================================================

def load_latest_log(mode: str, target: float) -> pd.DataFrame:
    """
    특정 mode와 target에 해당하는 가장 최근 로그 파일을 불러온다.
    disturbance 컬럼이 있는 로그를 우선 사용한다.
    """

    pattern = MODE_PATTERNS[mode]
    log_files = sorted(LOG_DIR.glob(pattern))

    candidates = []

    for log_file in log_files:
        try:
            df = pd.read_csv(log_file)

            if len(df) == 0:
                continue

            if "target" not in df.columns:
                continue

            log_target = float(df["target"].iloc[0])

            if abs(log_target - target) > 1e-6:
                continue

            # disturbance 실험 로그만 선택
            if "disturbance" not in df.columns:
                continue

            if "use_disturbance" in df.columns:
                use_disturbance = str(df["use_disturbance"].iloc[0]).lower()
                if use_disturbance not in ["true", "1"]:
                    continue

            candidates.append(
                {
                    "file": log_file,
                    "mtime": log_file.stat().st_mtime,
                    "df": df,
                }
            )

        except Exception as e:
            print(f"Skip file: {log_file}")
            print(f"Reason: {e}")

    if not candidates:
        raise FileNotFoundError(
            f"No disturbance log found for mode={mode}, target={target}"
        )

    latest = max(candidates, key=lambda x: x["mtime"])

    print(f"Load {mode}: {latest['file']}")

    return latest["df"]


def load_logs(target: float = TARGET):
    """
    fixed_pid와 adaptive_pid disturbance 로그 로드
    """

    logs = {}

    for mode in MODE_PATTERNS.keys():
        logs[mode] = load_latest_log(mode=mode, target=target)

    return logs


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    전체 구간 및 disturbance 구간 성능 지표 계산
    """

    target = float(df["target"].iloc[0])
    time = df["time"].to_numpy()
    current = df["current"].to_numpy()
    error = df["error"].to_numpy()
    pwm = df["pwm"].to_numpy()

    disturbance = (
        df["disturbance"].to_numpy()
        if "disturbance" in df.columns
        else np.zeros_like(time)
    )

    abs_error = np.abs(error)

    if len(time) > 1:
        dt = np.diff(time, prepend=time[0])
        dt[0] = 0.0
    else:
        dt = np.array([0.0])

    # 전체 구간
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

    # disturbance 구간
    disturbance_mask = np.abs(disturbance) > 1e-9

    if disturbance_mask.any():
        disturbance_iae = float(np.sum(abs_error[disturbance_mask] * dt[disturbance_mask]))
        disturbance_mean_abs_error = float(np.mean(abs_error[disturbance_mask]))
        disturbance_mean_pwm = float(np.mean(np.abs(pwm[disturbance_mask])))
        min_current_during_disturbance = float(np.min(current[disturbance_mask]))
        max_error_during_disturbance = float(np.max(abs_error[disturbance_mask]))
    else:
        disturbance_iae = np.nan
        disturbance_mean_abs_error = np.nan
        disturbance_mean_pwm = np.nan
        min_current_during_disturbance = np.nan
        max_error_during_disturbance = np.nan

    # disturbance 제거 후 회복성
    recovery_mask = np.zeros_like(time, dtype=bool)

    if disturbance_mask.any():
        disturbance_end_time = float(time[disturbance_mask][-1])
        recovery_mask = time > disturbance_end_time

    if recovery_mask.any():
        recovery_iae = float(np.sum(abs_error[recovery_mask] * dt[recovery_mask]))
        recovery_mean_abs_error = float(np.mean(abs_error[recovery_mask]))

        tolerance = 0.02 * abs(target)
        recovery_time = np.nan

        recovery_indices = np.where(recovery_mask)[0]

        for idx in recovery_indices:
            if np.all(abs_error[idx:] <= tolerance):
                recovery_time = float(time[idx] - time[recovery_indices[0]])
                break
    else:
        recovery_iae = np.nan
        recovery_mean_abs_error = np.nan
        recovery_time = np.nan

    return {
        "target": target,
        "final_error": final_error,
        "mean_abs_error": mean_abs_error,
        "IAE": iae,
        "ISE": ise,
        "overshoot_percent": overshoot_percent,
        "mean_pwm": mean_pwm,
        "total_pwm": total_pwm,
        "max_pwm": max_pwm,
        "disturbance_IAE": disturbance_iae,
        "disturbance_mean_abs_error": disturbance_mean_abs_error,
        "disturbance_mean_pwm": disturbance_mean_pwm,
        "min_current_during_disturbance": min_current_during_disturbance,
        "max_error_during_disturbance": max_error_during_disturbance,
        "recovery_IAE": recovery_iae,
        "recovery_mean_abs_error": recovery_mean_abs_error,
        "recovery_time_after_disturbance": recovery_time,
    }


def save_metrics_summary(logs: dict) -> pd.DataFrame:
    """
    fixed/adaptive disturbance metrics 저장
    """

    rows = []

    for mode, df in logs.items():
        row = calculate_metrics(df)
        row["mode"] = mode
        rows.append(row)

    metrics_df = pd.DataFrame(rows)

    metrics_df = metrics_df[
        [
            "mode",
            "target",
            "final_error",
            "mean_abs_error",
            "IAE",
            "ISE",
            "overshoot_percent",
            "mean_pwm",
            "total_pwm",
            "max_pwm",
            "disturbance_IAE",
            "disturbance_mean_abs_error",
            "disturbance_mean_pwm",
            "min_current_during_disturbance",
            "max_error_during_disturbance",
            "recovery_IAE",
            "recovery_mean_abs_error",
            "recovery_time_after_disturbance",
        ]
    ]

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURE_DIR / f"disturbance_metrics_target_{int(TARGET)}.csv"

    metrics_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Disturbance metrics")
    print("=" * 80)
    print(metrics_df)
    print(f"\nSaved: {save_path}")

    return metrics_df


# ============================================================
# Plot functions
# ============================================================

def get_disturbance_window(logs: dict):
    """
    disturbance가 존재하는 시간 구간 반환
    """

    for df in logs.values():
        if "disturbance" not in df.columns:
            continue

        disturbance = df["disturbance"].to_numpy()
        time = df["time"].to_numpy()

        mask = np.abs(disturbance) > 1e-9

        if mask.any():
            return float(time[mask][0]), float(time[mask][-1])

    return None, None


def shade_disturbance_area(logs: dict):
    """
    모든 plot에 disturbance 구간 음영 표시
    """

    start_time, end_time = get_disturbance_window(logs)

    if start_time is not None and end_time is not None:
        plt.axvspan(
            start_time,
            end_time,
            alpha=0.15,
            label="disturbance window",
        )


def plot_response(logs: dict, target: float, save: bool = True):
    plt.figure(figsize=(10, 5))

    for mode, df in logs.items():
        plt.plot(df["time"], df["current"], label=mode)

    plt.axhline(target, linestyle="--", label="target")
    shade_disturbance_area(logs)

    plt.xlabel("Time [s]")
    plt.ylabel("Current")
    plt.title(f"Disturbance Response Comparison at Target={target:.0f}")
    plt.legend()
    plt.grid(True)

    if save:
        save_path = FIGURE_DIR / f"disturbance_response_target_{int(target)}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_normalized_response(logs: dict, save: bool = True):
    plt.figure(figsize=(10, 5))

    for mode, df in logs.items():
        normalized_current = df["current"] / df["target"]
        plt.plot(df["time"], normalized_current, label=mode)

    plt.axhline(1.0, linestyle="--", label="target = 1.0")
    shade_disturbance_area(logs)

    plt.xlabel("Time [s]")
    plt.ylabel("Current / Target")
    plt.title("Normalized Disturbance Response Comparison")
    plt.legend()
    plt.grid(True)

    if save:
        save_path = FIGURE_DIR / f"disturbance_normalized_response_target_{int(TARGET)}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_error(logs: dict, save: bool = True):
    plt.figure(figsize=(10, 5))

    for mode, df in logs.items():
        plt.plot(df["time"], df["error"], label=mode)

    shade_disturbance_area(logs)

    plt.xlabel("Time [s]")
    plt.ylabel("Error")
    plt.title("Disturbance Error Comparison")
    plt.legend()
    plt.grid(True)

    if save:
        save_path = FIGURE_DIR / f"disturbance_error_target_{int(TARGET)}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_pwm(logs: dict, save: bool = True):
    plt.figure(figsize=(10, 5))

    for mode, df in logs.items():
        plt.plot(df["time"], df["pwm"], label=mode)

    shade_disturbance_area(logs)

    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("Disturbance PWM Comparison")
    plt.legend()
    plt.grid(True)

    if save:
        save_path = FIGURE_DIR / f"disturbance_pwm_target_{int(TARGET)}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_disturbance(logs: dict, save: bool = True):
    plt.figure(figsize=(10, 4))

    plotted = False

    for mode, df in logs.items():
        if "disturbance" not in df.columns:
            continue

        plt.plot(df["time"], df["disturbance"], label=f"disturbance ({mode})")
        plotted = True
        break

    if not plotted:
        print("Skip disturbance plot: disturbance column not found.")
        return

    plt.xlabel("Time [s]")
    plt.ylabel("Disturbance")
    plt.title("Applied Disturbance")
    plt.legend()
    plt.grid(True)

    if save:
        save_path = FIGURE_DIR / f"disturbance_input_target_{int(TARGET)}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_gains(logs: dict, save: bool = True):
    """
    adaptive PID gain 변화 확인
    """

    if "adaptive_pid" not in logs:
        return

    df = logs["adaptive_pid"]

    for gain_col in ["kp", "ki", "kd"]:
        if gain_col not in df.columns:
            continue

        plt.figure(figsize=(10, 4))
        plt.plot(df["time"], df[gain_col], label=gain_col)

        shade_disturbance_area(logs)

        plt.xlabel("Time [s]")
        plt.ylabel(gain_col)
        plt.title(f"Adaptive PID {gain_col.upper()} under Disturbance")
        plt.legend()
        plt.grid(True)

        if save:
            save_path = FIGURE_DIR / f"disturbance_adaptive_{gain_col}_target_{int(TARGET)}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()


# ============================================================
# Main
# ============================================================

def main():
    logs = load_logs(target=TARGET)

    metrics_df = save_metrics_summary(logs)

    target = float(metrics_df["target"].iloc[0])

    plot_response(logs, target=target)
    plot_normalized_response(logs)
    plot_error(logs)
    plot_pwm(logs)
    plot_disturbance(logs)
    plot_gains(logs)


if __name__ == "__main__":
    main()