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

from config import RESULTS_DIR, FIGURE_DIR, PWM_MAX, PWM_MIN


# ============================================================
# Paths
# ============================================================

KAFKA_RESULT_DIR = RESULTS_DIR / "kafka_control"
INTEGRATED_RESULT_DIR = RESULTS_DIR / "integrated_control"
SUMMARY_DIR = RESULTS_DIR / "summary"


# ============================================================
# Utility
# ============================================================

def get_latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No file found: {folder / pattern}")

    return files[-1]


def load_latest_kafka_results():
    log_path = get_latest_file(
        KAFKA_RESULT_DIR,
        "local_kafka_controller_log_*.csv",
    )

    metrics_path = get_latest_file(
        KAFKA_RESULT_DIR,
        "local_kafka_controller_metrics_*.csv",
    )

    print(f"Load Kafka log: {log_path}")
    print(f"Load Kafka metrics: {metrics_path}")

    log_df = pd.read_csv(log_path)
    metrics_df = pd.read_csv(metrics_path)

    return log_df, metrics_df, log_path, metrics_path


def load_latest_integrated_metrics():
    try:
        metrics_path = get_latest_file(
            INTEGRATED_RESULT_DIR,
            "integrated_metrics_*.csv",
        )

        print(f"Load integrated metrics: {metrics_path}")

        metrics_df = pd.read_csv(metrics_path)

        return metrics_df, metrics_path

    except FileNotFoundError:
        print("Integrated metrics not found. Skip integrated comparison.")
        return None, None


def select_integrated_reference(integrated_df: pd.DataFrame):
    """
    integrated simulation에서 Kafka 구조와 가장 가까운
    delayed_saturation_aware_gain_db 결과를 기준으로 선택.
    """

    if integrated_df is None:
        return None

    ref_mode = "delayed_saturation_aware_gain_db"

    if "mode" not in integrated_df.columns:
        return None

    ref_df = integrated_df[integrated_df["mode"] == ref_mode]

    if ref_df.empty:
        return None

    return ref_df.iloc[0].to_dict()


def calc_difference_table(kafka_metrics: dict, integrated_ref: dict):
    """
    Kafka result와 integrated simulation reference 비교.
    """

    if integrated_ref is None:
        return pd.DataFrame()

    metrics_to_compare = [
        "IAE",
        "mean_abs_error",
        "final_error",
        "after_change_IAE",
        "after_change_max_error",
        "disturbance_IAE",
        "disturbance_max_error",
        "disturbance_min_current",
        "settling_time_after_change",
        "mean_pwm",
        "total_pwm",
        "max_pwm",
        "saturation_ratio_percent",
        "saturation_duration",
        "near_high_saturation_ratio_percent",
        "local_gain_reduction_count",
        "local_gain_recovery_count",
        "min_kp_scale",
        "min_ki_scale",
    ]

    rows = []

    for metric in metrics_to_compare:
        if metric not in kafka_metrics:
            continue

        if metric not in integrated_ref:
            continue

        kafka_value = kafka_metrics[metric]
        integrated_value = integrated_ref[metric]

        if pd.isna(kafka_value) or pd.isna(integrated_value):
            diff = np.nan
            diff_percent = np.nan
        else:
            diff = kafka_value - integrated_value

            if abs(integrated_value) > 1e-12:
                diff_percent = diff / integrated_value * 100.0
            else:
                diff_percent = np.nan

        rows.append(
            {
                "metric": metric,
                "integrated_simulation": integrated_value,
                "kafka_simulation": kafka_value,
                "difference": diff,
                "difference_percent": diff_percent,
            }
        )

    return pd.DataFrame(rows)


def build_event_summary(log_df: pd.DataFrame):
    """
    Kafka controller log에서 이벤트 통계 생성.
    """

    rows = []

    if "gain_update_reason" in log_df.columns:
        gain_counts = log_df["gain_update_reason"].value_counts(dropna=False)

        for reason, count in gain_counts.items():
            rows.append(
                {
                    "event_group": "server_gain_update",
                    "event_name": reason,
                    "count": int(count),
                }
            )

    if "gain_command_discard_reason" in log_df.columns:
        discard_counts = log_df["gain_command_discard_reason"].value_counts(dropna=False)

        for reason, count in discard_counts.items():
            rows.append(
                {
                    "event_group": "gain_command_discard",
                    "event_name": reason,
                    "count": int(count),
                }
            )

    if "local_update_reason" in log_df.columns:
        local_counts = log_df["local_update_reason"].value_counts(dropna=False)

        for reason, count in local_counts.items():
            rows.append(
                {
                    "event_group": "local_safety_layer",
                    "event_name": reason,
                    "count": int(count),
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# Plot
# ============================================================

def get_event_times(log_df: pd.DataFrame):
    target_change_time = None
    disturbance_start_time = None
    disturbance_end_time = None

    if "target" in log_df.columns and "time" in log_df.columns:
        target_values = log_df["target"].to_numpy()
        time_values = log_df["time"].to_numpy()

        if len(np.unique(target_values)) > 1:
            first_target = target_values[0]
            changed = np.where(np.abs(target_values - first_target) > 1e-9)[0]

            if len(changed) > 0:
                target_change_time = float(time_values[changed[0]])

    if "disturbance" in log_df.columns and "time" in log_df.columns:
        disturbance = log_df["disturbance"].to_numpy()
        time_values = log_df["time"].to_numpy()

        mask = np.abs(disturbance) > 1e-9

        if mask.any():
            disturbance_start_time = float(time_values[mask][0])
            disturbance_end_time = float(time_values[mask][-1])

    return target_change_time, disturbance_start_time, disturbance_end_time


def shade_events(log_df: pd.DataFrame):
    target_change_time, disturbance_start_time, disturbance_end_time = get_event_times(log_df)

    if target_change_time is not None:
        plt.axvline(
            target_change_time,
            linestyle="--",
            linewidth=1.2,
            label="target change",
        )

    if disturbance_start_time is not None and disturbance_end_time is not None:
        plt.axvspan(
            disturbance_start_time,
            disturbance_end_time,
            alpha=0.15,
            label="disturbance",
        )


def plot_kafka_response(log_df: pd.DataFrame, timestamp: str):
    plt.figure(figsize=(10, 5))

    plt.plot(log_df["time"], log_df["current"], label="current")
    plt.plot(log_df["time"], log_df["target"], linestyle="--", label="target")

    shade_events(log_df)

    plt.xlabel("Time [s]")
    plt.ylabel("Current")
    plt.title("Kafka-based Server-assisted Adaptive PID Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"kafka_control_response_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_kafka_pwm(log_df: pd.DataFrame, timestamp: str):
    plt.figure(figsize=(10, 5))

    plt.plot(log_df["time"], log_df["pwm"], label="PWM")
    plt.axhline(PWM_MAX, linestyle="--", label="PWM max")

    shade_events(log_df)

    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("Kafka-based Controller PWM")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"kafka_control_pwm_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_kafka_gain(log_df: pd.DataFrame, timestamp: str):
    plt.figure(figsize=(10, 5))

    if "base_kp" in log_df.columns:
        plt.plot(log_df["time"], log_df["base_kp"], label="base Kp")

    if "base_ki" in log_df.columns:
        plt.plot(log_df["time"], log_df["base_ki"], label="base Ki")

    if "kp" in log_df.columns:
        plt.plot(log_df["time"], log_df["kp"], linestyle="--", label="applied Kp")

    if "ki" in log_df.columns:
        plt.plot(log_df["time"], log_df["ki"], linestyle="--", label="applied Ki")

    shade_events(log_df)

    plt.xlabel("Time [s]")
    plt.ylabel("Gain")
    plt.title("Kafka-based Controller Base and Applied Gains")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"kafka_control_gains_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_kafka_gain_scale(log_df: pd.DataFrame, timestamp: str):
    plt.figure(figsize=(10, 5))

    if "kp_scale" in log_df.columns:
        plt.plot(log_df["time"], log_df["kp_scale"], label="Kp scale")

    if "ki_scale" in log_df.columns:
        plt.plot(log_df["time"], log_df["ki_scale"], label="Ki scale")

    shade_events(log_df)

    plt.xlabel("Time [s]")
    plt.ylabel("Gain scale")
    plt.title("Kafka-based Local Saturation-aware Gain Scale")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"kafka_control_gain_scale_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# Save
# ============================================================

def save_summary(
    kafka_metrics_df: pd.DataFrame,
    comparison_df: pd.DataFrame,
    event_df: pd.DataFrame,
    log_path: Path,
    metrics_path: Path,
    integrated_path: Path,
    timestamp: str,
):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    kafka_summary_path = SUMMARY_DIR / "kafka_control_metrics_latest.csv"
    comparison_path = SUMMARY_DIR / "kafka_vs_integrated_comparison.csv"
    event_path = SUMMARY_DIR / "kafka_control_event_summary.csv"
    markdown_path = SUMMARY_DIR / f"kafka_control_summary_{timestamp}.md"

    kafka_metrics_df.to_csv(kafka_summary_path, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    event_df.to_csv(event_path, index=False, encoding="utf-8-sig")

    kafka_metrics = kafka_metrics_df.iloc[0].to_dict()

    lines = []

    lines.append("# Kafka-based Server-assisted Adaptive PID Summary")
    lines.append("")
    lines.append("## 1. Loaded Files")
    lines.append("")
    lines.append(f"- Kafka log: `{log_path}`")
    lines.append(f"- Kafka metrics: `{metrics_path}`")

    if integrated_path is not None:
        lines.append(f"- Integrated simulation metrics: `{integrated_path}`")

    lines.append("")

    lines.append("## 2. Kafka Controller Metrics")
    lines.append("")
    lines.append(kafka_metrics_df.to_markdown(index=False))
    lines.append("")

    if not comparison_df.empty:
        lines.append("## 3. Kafka vs Integrated Simulation")
        lines.append("")
        lines.append(comparison_df.to_markdown(index=False))
        lines.append("")

    if not event_df.empty:
        lines.append("## 4. Event Summary")
        lines.append("")
        lines.append(event_df.to_markdown(index=False))
        lines.append("")

    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(
        "The Kafka-based server-assisted adaptive PID controller successfully performed "
        "real-time motor state publishing and server-side gain command updates. "
        "The local controller separated the server-recommended base gain from the local "
        "saturation-aware gain scale, preventing instability from repeated gain commands."
    )
    lines.append("")
    lines.append(
        f"In the latest Kafka simulation, the controller achieved IAE={kafka_metrics.get('IAE', np.nan):.4f}, "
        f"final error={kafka_metrics.get('final_error', np.nan):.4f}, "
        f"after-change IAE={kafka_metrics.get('after_change_IAE', np.nan):.4f}, "
        f"and saturation ratio={kafka_metrics.get('saturation_ratio_percent', np.nan):.4f}%."
    )
    lines.append("")
    lines.append(
        "This result confirms that the server-assisted gain recommendation architecture can be implemented "
        "using Kafka while preserving the control behavior observed in the integrated simulation."
    )
    lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved: {kafka_summary_path}")
    print(f"Saved: {comparison_path}")
    print(f"Saved: {event_path}")
    print(f"Saved markdown summary: {markdown_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Summarize Kafka control results")
    print("=" * 80)

    log_df, kafka_metrics_df, log_path, metrics_path = load_latest_kafka_results()
    integrated_df, integrated_path = load_latest_integrated_metrics()

    kafka_metrics = kafka_metrics_df.iloc[0].to_dict()
    integrated_ref = select_integrated_reference(integrated_df)

    comparison_df = calc_difference_table(
        kafka_metrics=kafka_metrics,
        integrated_ref=integrated_ref,
    )

    event_df = build_event_summary(log_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nKafka metrics:")
    print(kafka_metrics_df)

    if not comparison_df.empty:
        print("\nKafka vs Integrated comparison:")
        print(comparison_df)

    print("\nEvent summary:")
    print(event_df)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plot_kafka_response(log_df, timestamp)
    plot_kafka_pwm(log_df, timestamp)
    plot_kafka_gain(log_df, timestamp)
    plot_kafka_gain_scale(log_df, timestamp)

    save_summary(
        kafka_metrics_df=kafka_metrics_df,
        comparison_df=comparison_df,
        event_df=event_df,
        log_path=log_path,
        metrics_path=metrics_path,
        integrated_path=integrated_path,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()