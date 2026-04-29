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

from config import RESULTS_DIR, FIGURE_DIR


# ============================================================
# Paths
# ============================================================

KAFKA_RESULT_DIR = RESULTS_DIR / "kafka_control"
SUMMARY_DIR = RESULTS_DIR / "summary"


# ============================================================
# Utility
# ============================================================

def get_latest_file(folder: Path, pattern: str) -> Path:
    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No file found: {folder / pattern}")

    return files[-1]


def load_latest_esp32_kafka_results():
    log_path = get_latest_file(
        KAFKA_RESULT_DIR,
        "local_kafka_controller_log_esp32_*.csv",
    )

    metrics_path = get_latest_file(
        KAFKA_RESULT_DIR,
        "local_kafka_controller_metrics_esp32_*.csv",
    )

    print(f"Load ESP32 Kafka log: {log_path}")
    print(f"Load ESP32 Kafka metrics: {metrics_path}")

    log_df = pd.read_csv(log_path)
    metrics_df = pd.read_csv(metrics_path)

    return log_df, metrics_df, log_path, metrics_path


def get_target_change_time(log_df: pd.DataFrame):
    if "target" not in log_df.columns or "time" not in log_df.columns:
        return None

    target_values = log_df["target"].to_numpy(dtype=float)
    time_values = log_df["time"].to_numpy(dtype=float)

    if len(np.unique(target_values)) <= 1:
        return None

    first_target = target_values[0]
    changed_idx = np.where(np.abs(target_values - first_target) > 1e-9)[0]

    if len(changed_idx) == 0:
        return None

    return float(time_values[changed_idx[0]])


def shade_target_change(log_df: pd.DataFrame):
    target_change_time = get_target_change_time(log_df)

    if target_change_time is not None:
        plt.axvline(
            target_change_time,
            linestyle="--",
            linewidth=1.2,
            label="target change",
        )


def build_event_summary(log_df: pd.DataFrame):
    rows = []

    if "gain_update_reason" in log_df.columns:
        for reason, count in log_df["gain_update_reason"].value_counts(dropna=False).items():
            rows.append(
                {
                    "event_group": "server_gain_update",
                    "event_name": reason,
                    "count": int(count),
                }
            )

    if "gain_command_discard_reason" in log_df.columns:
        for reason, count in log_df["gain_command_discard_reason"].value_counts(dropna=False).items():
            rows.append(
                {
                    "event_group": "gain_command_discard",
                    "event_name": reason,
                    "count": int(count),
                }
            )

    if "local_update_reason" in log_df.columns:
        for reason, count in log_df["local_update_reason"].value_counts(dropna=False).items():
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

def plot_rpm_response(log_df: pd.DataFrame, timestamp: str):
    plt.figure(figsize=(10, 5))

    plt.plot(log_df["time"], log_df["current"], label="measured RPM")
    plt.plot(log_df["time"], log_df["target"], linestyle="--", label="target RPM")

    shade_target_change(log_df)

    plt.xlabel("Time [s]")
    plt.ylabel("RPM")
    plt.title("ESP32 Kafka-based Adaptive PID RPM Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_kafka_rpm_response_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_pwm_command(log_df: pd.DataFrame, timestamp: str):
    plt.figure(figsize=(10, 5))

    if "raw_pwm" in log_df.columns:
        plt.plot(log_df["time"], log_df["raw_pwm"], linestyle="--", label="raw PWM")

    plt.plot(log_df["time"], log_df["pwm"], label="applied PWM")

    shade_target_change(log_df)

    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("ESP32 Kafka-based Adaptive PID PWM Command")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_kafka_pwm_command_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_gain_history(log_df: pd.DataFrame, timestamp: str):
    plt.figure(figsize=(10, 5))

    if "base_kp" in log_df.columns:
        plt.plot(log_df["time"], log_df["base_kp"], label="base Kp")

    if "base_ki" in log_df.columns:
        plt.plot(log_df["time"], log_df["base_ki"], label="base Ki")

    if "kp" in log_df.columns:
        plt.plot(log_df["time"], log_df["kp"], linestyle="--", label="applied Kp")

    if "ki" in log_df.columns:
        plt.plot(log_df["time"], log_df["ki"], linestyle="--", label="applied Ki")

    shade_target_change(log_df)

    plt.xlabel("Time [s]")
    plt.ylabel("Gain")
    plt.title("ESP32 Kafka-based Adaptive PID Gain History")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_kafka_gain_history_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_gain_scale(log_df: pd.DataFrame, timestamp: str):
    plt.figure(figsize=(10, 5))

    if "kp_scale" in log_df.columns:
        plt.plot(log_df["time"], log_df["kp_scale"], label="Kp scale")

    if "ki_scale" in log_df.columns:
        plt.plot(log_df["time"], log_df["ki_scale"], label="Ki scale")

    shade_target_change(log_df)

    plt.xlabel("Time [s]")
    plt.ylabel("Gain scale")
    plt.title("ESP32 Kafka-based Local Safety Gain Scale")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_kafka_gain_scale_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_error_response(log_df: pd.DataFrame, timestamp: str):
    plt.figure(figsize=(10, 5))

    if "measured_error" in log_df.columns:
        plt.plot(log_df["time"], log_df["measured_error"], label="measured error")
    else:
        plt.plot(log_df["time"], log_df["target"] - log_df["current"], label="measured error")

    if "control_error" in log_df.columns:
        plt.plot(log_df["time"], log_df["control_error"], linestyle="--", label="control error")

    plt.axhline(0.0, linewidth=1.0)

    shade_target_change(log_df)

    plt.xlabel("Time [s]")
    plt.ylabel("Error [RPM]")
    plt.title("ESP32 Kafka-based Adaptive PID Error Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_kafka_error_response_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


# ============================================================
# Save summary
# ============================================================

def save_summary(
    log_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
    event_df: pd.DataFrame,
    log_path: Path,
    metrics_path: Path,
    timestamp: str,
):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    latest_metrics_path = SUMMARY_DIR / "esp32_kafka_metrics_latest.csv"
    event_summary_path = SUMMARY_DIR / "esp32_kafka_event_summary.csv"
    markdown_path = SUMMARY_DIR / f"esp32_kafka_summary_{timestamp}.md"

    metrics_df.to_csv(latest_metrics_path, index=False, encoding="utf-8-sig")
    event_df.to_csv(event_summary_path, index=False, encoding="utf-8-sig")

    metrics = metrics_df.iloc[0].to_dict()

    final_error = metrics.get("final_error", np.nan)
    iae = metrics.get("IAE", np.nan)
    after_change_iae = metrics.get("after_change_IAE", np.nan)
    max_pwm = metrics.get("max_pwm", np.nan)
    saturation_ratio = metrics.get("saturation_ratio_percent", np.nan)
    server_gain_applied = metrics.get("server_gain_applied_count", np.nan)
    duplicate_discard = metrics.get("duplicate_gain_discard_count", np.nan)
    unsafe_discard = metrics.get("unsafe_gain_discard_count", np.nan)

    lines = []

    lines.append("# ESP32 Kafka-based Server-assisted Adaptive PID Summary")
    lines.append("")
    lines.append("## 1. Loaded Files")
    lines.append("")
    lines.append(f"- ESP32 Kafka log: `{log_path}`")
    lines.append(f"- ESP32 Kafka metrics: `{metrics_path}`")
    lines.append("")

    lines.append("## 2. Metrics")
    lines.append("")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")

    if not event_df.empty:
        lines.append("## 3. Event Summary")
        lines.append("")
        lines.append(event_df.to_markdown(index=False))
        lines.append("")

    lines.append("## 4. Interpretation")
    lines.append("")
    lines.append(
        "The ESP32-based Kafka controller successfully executed a server-assisted adaptive PID control loop "
        "for a real DC encoder motor. The local controller published measured motor states to Kafka, "
        "received gain commands from the server-side recommender, and applied valid server-recommended gains "
        "while discarding duplicate or unsafe commands."
    )
    lines.append("")
    lines.append(
        f"In the latest experiment, the controller achieved IAE={iae:.4f}, "
        f"after-change IAE={after_change_iae:.4f}, final error={final_error:.4f} RPM, "
        f"and maximum PWM={max_pwm:.4f}. The saturation ratio was {saturation_ratio:.4f}%."
    )
    lines.append("")
    lines.append(
        f"The server gain command was applied {server_gain_applied} time(s), "
        f"duplicate commands were discarded {duplicate_discard} time(s), "
        f"and unsafe gain commands were discarded {unsafe_discard} time(s)."
    )
    lines.append("")
    lines.append(
        "These results indicate that the motor interface abstraction, Kafka communication, "
        "server-side gain recommendation, duplicate-command filtering, and local safety guard operated "
        "together in the real ESP32 motor setup."
    )
    lines.append("")

    markdown_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved latest metrics: {latest_metrics_path}")
    print(f"Saved event summary: {event_summary_path}")
    print(f"Saved markdown summary: {markdown_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Summarize ESP32 Kafka control results")
    print("=" * 80)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    log_df, metrics_df, log_path, metrics_path = load_latest_esp32_kafka_results()
    event_df = build_event_summary(log_df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\nMetrics:")
    print(metrics_df)

    print("\nEvent summary:")
    print(event_df)

    plot_rpm_response(log_df, timestamp)
    plot_pwm_command(log_df, timestamp)
    plot_error_response(log_df, timestamp)
    plot_gain_history(log_df, timestamp)
    plot_gain_scale(log_df, timestamp)

    save_summary(
        log_df=log_df,
        metrics_df=metrics_df,
        event_df=event_df,
        log_path=log_path,
        metrics_path=metrics_path,
        timestamp=timestamp,
    )


if __name__ == "__main__":
    main()