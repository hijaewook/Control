import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import LOG_DIR, FIGURE_DIR


# ============================================================
# Log loading
# ============================================================

def get_latest_log(pattern: str) -> Path:
    """
    특정 패턴에 해당하는 가장 최근 로그 파일을 찾는다.
    """
    log_files = sorted(LOG_DIR.glob(pattern))

    if not log_files:
        raise FileNotFoundError(f"No log files found for pattern: {pattern}")

    return log_files[-1]

def save_metrics_summary(fixed_df, adaptive_df, simulink_df):
    """
    세 모드의 성능 지표를 CSV로 저장한다.
    """

    fixed_metrics = calculate_metrics(fixed_df)
    adaptive_metrics = calculate_metrics(adaptive_df)
    simulink_metrics = calculate_metrics(simulink_df)

    metrics_df = pd.DataFrame(
        [
            {"mode": "fixed_pid_simple_motor", **fixed_metrics},
            {"mode": "adaptive_pid_simple_motor", **adaptive_metrics},
            {"mode": "simulink_pid_motor", **simulink_metrics},
        ]
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    save_path = FIGURE_DIR / "comparison_metrics_summary.csv"

    metrics_df.to_csv(save_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {save_path}")

    return metrics_df

def plot_cumulative_metrics(fixed_df, adaptive_df, simulink_df, save: bool = True):
    """
    누적 절대 오차와 누적 PWM 사용량 비교
    """

    plt.figure(figsize=(10, 5))

    for df, label in [
        (fixed_df, "Fixed PID - SimpleMotorEnv"),
        (adaptive_df, "Adaptive PID - SimpleMotorEnv"),
        (simulink_df, "Simulink PID - Motor.slx"),
    ]:
        dt = df["time"].diff().fillna(0)
        cumulative_iae = (df["error"].abs() * dt).cumsum()

        plt.plot(df["time"], cumulative_iae, label=label)

    plt.xlabel("Time [s]")
    plt.ylabel("Cumulative absolute error")
    plt.title("Cumulative IAE Comparison")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "comparison_cumulative_iae_3mode.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()

    plt.figure(figsize=(10, 5))

    for df, label in [
        (fixed_df, "Fixed PID - SimpleMotorEnv"),
        (adaptive_df, "Adaptive PID - SimpleMotorEnv"),
        (simulink_df, "Simulink PID - Motor.slx"),
    ]:
        dt = df["time"].diff().fillna(0)
        cumulative_pwm = (df["pwm"].abs() * dt).cumsum()

        plt.plot(df["time"], cumulative_pwm, label=label)

    plt.xlabel("Time [s]")
    plt.ylabel("Cumulative PWM")
    plt.title("Control Effort Comparison")
    plt.legend()
    plt.grid(True)

    if save:
        save_path = FIGURE_DIR / "comparison_control_effort_3mode.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()

def plot_adaptive_diagnostics(adaptive_df, save: bool = True):
    """
    Adaptive PID 로그에 포함된 진단 변수 시각화
    """

    diagnostic_cols = [
        "integral",
        "pwm_saturated",
        "high_saturation",
        "low_saturation",
        "gain_update_flag",
    ]

    available_cols = [col for col in diagnostic_cols if col in adaptive_df.columns]

    if not available_cols:
        print("Skip adaptive diagnostics: diagnostic columns not found.")
        return

    for col in available_cols:
        plt.figure(figsize=(10, 5))
        plt.plot(adaptive_df["time"], adaptive_df[col], label=col)

        plt.xlabel("Time [s]")
        plt.ylabel(col)
        plt.title(f"Adaptive PID Diagnostic - {col}")
        plt.legend()
        plt.grid(True)

        if save:
            FIGURE_DIR.mkdir(parents=True, exist_ok=True)
            save_path = FIGURE_DIR / f"adaptive_diagnostic_{col}.png"
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved: {save_path}")

        plt.show()

def load_logs():
    """
    최신 fixed_pid, adaptive_pid, simulink_pid 로그를 불러온다.
    """

    fixed_log = get_latest_log("fixed_pid_simple_motor_*.csv")
    adaptive_log = get_latest_log("adaptive_pid_simple_motor_*.csv")
    simulink_log = get_latest_log("simulink_motor_*.csv")

    fixed_df = pd.read_csv(fixed_log)
    adaptive_df = pd.read_csv(adaptive_log)
    simulink_df = pd.read_csv(simulink_log)

    print(f"Fixed PID log:     {fixed_log}")
    print(f"Adaptive PID log:  {adaptive_log}")
    print(f"Simulink PID log:  {simulink_log}")

    return fixed_df, adaptive_df, simulink_df


# ============================================================
# Plot functions
# ============================================================

def plot_response_comparison(fixed_df, adaptive_df, simulink_df, save: bool = True):
    """
    Target-current 응답 비교
    """
    plt.figure(figsize=(10, 5))

    plt.plot(
        fixed_df["time"],
        fixed_df["target"],
        linestyle="--",
        label="Target",
    )

    plt.plot(
        fixed_df["time"],
        fixed_df["current"],
        label="Fixed PID - SimpleMotorEnv",
    )

    plt.plot(
        adaptive_df["time"],
        adaptive_df["current"],
        label="Adaptive PID - SimpleMotorEnv",
    )

    plt.plot(
        simulink_df["time"],
        simulink_df["current"],
        label="Simulink PID - Motor.slx",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Current value")
    plt.title("Response Comparison")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "comparison_response_3mode.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_normalized_response_comparison(fixed_df, adaptive_df, simulink_df, save: bool = True):
    """
    target 기준 정규화 응답 비교.
    target이 달라도 current / target 기준으로 비교 가능.
    """
    plt.figure(figsize=(10, 5))

    fixed_norm = fixed_df["current"] / fixed_df["target"]
    adaptive_norm = adaptive_df["current"] / adaptive_df["target"]
    simulink_norm = simulink_df["current"] / simulink_df["target"]

    plt.plot(
        fixed_df["time"],
        fixed_norm,
        label="Fixed PID - SimpleMotorEnv",
    )

    plt.plot(
        adaptive_df["time"],
        adaptive_norm,
        label="Adaptive PID - SimpleMotorEnv",
    )

    plt.plot(
        simulink_df["time"],
        simulink_norm,
        label="Simulink PID - Motor.slx",
    )

    plt.axhline(1.0, linestyle="--", label="Target = 1.0")

    plt.xlabel("Time [s]")
    plt.ylabel("Normalized response")
    plt.title("Normalized Response Comparison")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "comparison_normalized_response_3mode.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_error_comparison(fixed_df, adaptive_df, simulink_df, save: bool = True):
    """
    Error 비교
    """
    plt.figure(figsize=(10, 5))

    plt.plot(
        fixed_df["time"],
        fixed_df["error"],
        label="Fixed PID Error",
    )

    plt.plot(
        adaptive_df["time"],
        adaptive_df["error"],
        label="Adaptive PID Error",
    )

    plt.plot(
        simulink_df["time"],
        simulink_df["error"],
        label="Simulink PID Error",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("Error")
    plt.title("Error Comparison")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "comparison_error_3mode.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_pwm_comparison(fixed_df, adaptive_df, simulink_df, save: bool = True):
    """
    PWM 비교
    """
    plt.figure(figsize=(10, 5))

    plt.plot(
        fixed_df["time"],
        fixed_df["pwm"],
        label="Fixed PID PWM",
    )

    plt.plot(
        adaptive_df["time"],
        adaptive_df["pwm"],
        label="Adaptive PID PWM",
    )

    plt.plot(
        simulink_df["time"],
        simulink_df["pwm"],
        label="Simulink PID PWM",
    )

    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("PWM Comparison")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "comparison_pwm_3mode.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_adaptive_gains(adaptive_df, save: bool = True):
    """
    Adaptive PID의 gain 변화 확인
    """
    if not all(col in adaptive_df.columns for col in ["kp", "ki", "kd"]):
        print("Skip adaptive gain plot: kp, ki, kd columns not found.")
        return

    plt.figure(figsize=(10, 5))

    plt.plot(adaptive_df["time"], adaptive_df["kp"], label="Kp")
    plt.plot(adaptive_df["time"], adaptive_df["ki"], label="Ki")
    plt.plot(adaptive_df["time"], adaptive_df["kd"], label="Kd")

    plt.xlabel("Time [s]")
    plt.ylabel("Gain")
    plt.title("Adaptive PID Gains")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "comparison_adaptive_gains.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(df: pd.DataFrame) -> dict:
    """
    제어 성능 지표 계산
    """

    target = df["target"].iloc[0]
    time = df["time"]
    current = df["current"]
    error = df["error"]
    pwm = df["pwm"]

    dt = time.iloc[1] - time.iloc[0] if len(time) > 1 else 0.0

    final_error = abs(error.iloc[-1])
    mean_abs_error = error.abs().mean()

    # Integral error metrics
    iae = (error.abs() * dt).sum()
    ise = ((error ** 2) * dt).sum()

    # Control effort
    mean_pwm = pwm.mean()
    max_pwm = pwm.max()
    total_pwm = (pwm.abs() * dt).sum()

    # Overshoot
    overshoot = max(0.0, current.max() - target)
    overshoot_percent = overshoot / target * 100 if target != 0 else 0.0

    # Rise time: target의 90% 도달 시간
    rise_threshold = 0.9 * target
    rise_candidates = df[current >= rise_threshold]

    if len(rise_candidates) > 0:
        rise_time = rise_candidates["time"].iloc[0]
    else:
        rise_time = None

    # Settling time: target ±5% 범위에 들어간 뒤 끝까지 유지되는 첫 시간
    tolerance = 0.05 * abs(target)
    lower_bound = target - tolerance
    upper_bound = target + tolerance

    settling_time = None
    within_band = (current >= lower_bound) & (current <= upper_bound)

    for idx in range(len(df)):
        if within_band.iloc[idx] and within_band.iloc[idx:].all():
            settling_time = time.iloc[idx]
            break

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
    }


def print_metric_dict(metrics: dict):
    for key, value in metrics.items():
        if value is None:
            print(f"{key}: N/A")
        else:
            print(f"{key}: {value:.4f}")


def print_metrics(fixed_df, adaptive_df, simulink_df):
    fixed_metrics = calculate_metrics(fixed_df)
    adaptive_metrics = calculate_metrics(adaptive_df)
    simulink_metrics = calculate_metrics(simulink_df)

    print("\n" + "=" * 80)
    print("Performance Metrics")
    print("=" * 80)

    print("\n[Fixed PID - SimpleMotorEnv]")
    print_metric_dict(fixed_metrics)

    print("\n[Adaptive PID - SimpleMotorEnv]")
    print_metric_dict(adaptive_metrics)

    print("\n[Simulink PID - Motor.slx]")
    print_metric_dict(simulink_metrics)

    print("\n" + "=" * 80)
    print("Improvement relative to Fixed PID")
    print("=" * 80)

    for metric_name in ["mean_abs_error", "IAE", "ISE", "final_error"]:
        fixed_value = fixed_metrics[metric_name]

        adaptive_value = adaptive_metrics[metric_name]
        simulink_value = simulink_metrics[metric_name]

        if fixed_value is not None and fixed_value > 0:
            adaptive_improvement = (fixed_value - adaptive_value) / fixed_value * 100
            simulink_improvement = (fixed_value - simulink_value) / fixed_value * 100

            print(f"{metric_name} improvement - Adaptive: {adaptive_improvement:.2f}%")
            print(f"{metric_name} improvement - Simulink: {simulink_improvement:.2f}%")

    fixed_pwm = fixed_metrics["mean_pwm"]

    if fixed_pwm is not None and fixed_pwm > 0:
        adaptive_pwm_change = (adaptive_metrics["mean_pwm"] - fixed_pwm) / fixed_pwm * 100
        simulink_pwm_change = (simulink_metrics["mean_pwm"] - fixed_pwm) / fixed_pwm * 100

        print(f"mean_pwm change - Adaptive: {adaptive_pwm_change:.2f}%")
        print(f"mean_pwm change - Simulink: {simulink_pwm_change:.2f}%")


# ============================================================
# Main
# ============================================================

def main():
    fixed_df, adaptive_df, simulink_df = load_logs()

    print_metrics(fixed_df, adaptive_df, simulink_df)

    save_metrics_summary(fixed_df, adaptive_df, simulink_df)

    plot_response_comparison(fixed_df, adaptive_df, simulink_df)
    plot_normalized_response_comparison(fixed_df, adaptive_df, simulink_df)
    plot_error_comparison(fixed_df, adaptive_df, simulink_df)
    plot_pwm_comparison(fixed_df, adaptive_df, simulink_df)
    plot_adaptive_gains(adaptive_df)

    plot_cumulative_metrics(fixed_df, adaptive_df, simulink_df)
    plot_adaptive_diagnostics(adaptive_df)


if __name__ == "__main__":
    main()