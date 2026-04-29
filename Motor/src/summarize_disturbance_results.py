import sys
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import FIGURE_DIR, RESULTS_DIR


# ============================================================
# Paths
# ============================================================

SUMMARY_DIR = RESULTS_DIR / "summary"

DISTURBANCE_METRIC_FILES = [
    FIGURE_DIR / "disturbance_metrics_target_100.csv",
    FIGURE_DIR / "disturbance_metrics_target_200.csv",
]


def load_disturbance_metrics() -> pd.DataFrame:
    """
    disturbance metrics CSV 파일들을 병합한다.
    """

    dfs = []

    for file_path in DISTURBANCE_METRIC_FILES:
        if not file_path.exists():
            print(f"Skip missing file: {file_path}")
            continue

        print(f"Load: {file_path}")
        df = pd.read_csv(file_path)
        dfs.append(df)

    if not dfs:
        raise FileNotFoundError("No disturbance metric files found.")

    merged_df = pd.concat(dfs, axis=0).reset_index(drop=True)

    return merged_df


def compute_improvement_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    target별 fixed_pid 대비 adaptive_pid 개선율 계산.

    개선율 정의:
        improvement [%] = (fixed - adaptive) / fixed * 100

    값이 클수록 adaptive가 좋음.
    """

    rows = []

    target_list = sorted(metrics_df["target"].unique())

    for target in target_list:
        target_df = metrics_df[metrics_df["target"] == target]

        fixed_df = target_df[target_df["mode"] == "fixed_pid"]
        adaptive_df = target_df[target_df["mode"] == "adaptive_pid"]

        if fixed_df.empty or adaptive_df.empty:
            print(f"Skip target={target}: missing fixed or adaptive result")
            continue

        fixed = fixed_df.iloc[0]
        adaptive = adaptive_df.iloc[0]

        row = {
            "target": target,

            "fixed_IAE": fixed["IAE"],
            "adaptive_IAE": adaptive["IAE"],
            "IAE_improvement_percent": calc_improvement(
                fixed["IAE"],
                adaptive["IAE"],
            ),

            "fixed_disturbance_IAE": fixed["disturbance_IAE"],
            "adaptive_disturbance_IAE": adaptive["disturbance_IAE"],
            "disturbance_IAE_improvement_percent": calc_improvement(
                fixed["disturbance_IAE"],
                adaptive["disturbance_IAE"],
            ),

            "fixed_max_error_during_disturbance": fixed["max_error_during_disturbance"],
            "adaptive_max_error_during_disturbance": adaptive["max_error_during_disturbance"],
            "max_error_improvement_percent": calc_improvement(
                fixed["max_error_during_disturbance"],
                adaptive["max_error_during_disturbance"],
            ),

            "fixed_min_current_during_disturbance": fixed["min_current_during_disturbance"],
            "adaptive_min_current_during_disturbance": adaptive["min_current_during_disturbance"],

            "fixed_final_error": fixed["final_error"],
            "adaptive_final_error": adaptive["final_error"],
            "final_error_improvement_percent": calc_improvement(
                fixed["final_error"],
                adaptive["final_error"],
            ),

            "fixed_recovery_time_after_disturbance": fixed["recovery_time_after_disturbance"],
            "adaptive_recovery_time_after_disturbance": adaptive["recovery_time_after_disturbance"],

            "fixed_overshoot_percent": fixed["overshoot_percent"],
            "adaptive_overshoot_percent": adaptive["overshoot_percent"],

            "fixed_max_pwm": fixed["max_pwm"],
            "adaptive_max_pwm": adaptive["max_pwm"],
            "fixed_mean_pwm": fixed["mean_pwm"],
            "adaptive_mean_pwm": adaptive["mean_pwm"],
        }

        rows.append(row)

    summary_df = pd.DataFrame(rows)

    return summary_df


def calc_improvement(fixed_value, adaptive_value):
    """
    fixed 대비 adaptive 개선율 계산.
    fixed_value가 0 또는 NaN이면 NaN 반환.
    """

    if pd.isna(fixed_value) or pd.isna(adaptive_value):
        return np.nan

    if abs(fixed_value) < 1e-12:
        return np.nan

    return (fixed_value - adaptive_value) / fixed_value * 100.0


def save_csv(metrics_df: pd.DataFrame, improvement_df: pd.DataFrame):
    """
    CSV 결과 저장.
    """

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = SUMMARY_DIR / "disturbance_metrics_merged.csv"
    improvement_path = SUMMARY_DIR / "disturbance_improvement_summary.csv"

    metrics_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    improvement_df.to_csv(improvement_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {raw_path}")
    print(f"Saved: {improvement_path}")

    return raw_path, improvement_path


def save_markdown_summary(metrics_df: pd.DataFrame, improvement_df: pd.DataFrame):
    """
    disturbance 결과를 Markdown 요약으로 저장.
    """

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SUMMARY_DIR / f"disturbance_result_summary_{timestamp}.md"

    lines = []

    lines.append("# Disturbance Result Summary")
    lines.append("")
    lines.append("## 1. Experiment Setting")
    lines.append("")
    lines.append("- Disturbance type: pulse disturbance")
    lines.append("- Disturbance window: 3.0 s to 6.0 s")
    lines.append("- Disturbance magnitude: 20.0")
    lines.append("- Compared controllers: fixed PID and gain-scheduled adaptive PID")
    lines.append("")

    lines.append("## 2. Raw Disturbance Metrics")
    lines.append("")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Improvement Summary")
    lines.append("")
    lines.append(improvement_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 4. Interpretation")
    lines.append("")
    lines.append(
        "The fixed PID controller exhibited large steady-state errors and poor disturbance rejection "
        "under both target conditions. In contrast, the gain-scheduled adaptive PID controller maintained "
        "the output close to the target during the disturbance interval and rapidly recovered after the "
        "disturbance was removed."
    )
    lines.append("")
    lines.append(
        "The adaptive PID achieved substantial reductions in total IAE, disturbance-window IAE, and maximum "
        "disturbance error compared with the fixed PID. However, the adaptive controller used higher PWM effort, "
        "including saturation in the disturbance condition, which should be considered in future real-system validation."
    )
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved markdown summary: {save_path}")

    return save_path


def print_key_summary(improvement_df: pd.DataFrame):
    """
    콘솔에 핵심 결과 출력.
    """

    print("\n" + "=" * 80)
    print("Disturbance improvement summary")
    print("=" * 80)

    display_cols = [
        "target",
        "IAE_improvement_percent",
        "disturbance_IAE_improvement_percent",
        "max_error_improvement_percent",
        "fixed_min_current_during_disturbance",
        "adaptive_min_current_during_disturbance",
        "adaptive_recovery_time_after_disturbance",
        "adaptive_overshoot_percent",
        "adaptive_max_pwm",
    ]

    display_cols = [col for col in display_cols if col in improvement_df.columns]

    print(improvement_df[display_cols])


def main():
    print("=" * 80)
    print("Summarize disturbance results")
    print("=" * 80)

    metrics_df = load_disturbance_metrics()
    improvement_df = compute_improvement_summary(metrics_df)

    print("\nMerged disturbance metrics:")
    print(metrics_df)

    print_key_summary(improvement_df)

    save_csv(metrics_df, improvement_df)
    save_markdown_summary(metrics_df, improvement_df)


if __name__ == "__main__":
    main()