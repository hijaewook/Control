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

from config import RESULTS_DIR


# ============================================================
# Paths
# ============================================================

DELAY_RESULT_DIR = RESULTS_DIR / "delay_aware"
SUMMARY_DIR = RESULTS_DIR / "summary"


# ============================================================
# Utility
# ============================================================

def get_latest_file(folder: Path, pattern: str):
    """
    특정 폴더에서 가장 최근 파일을 찾는다.
    """

    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No files found: {folder / pattern}")

    return files[-1]


def calc_improvement(baseline_value, proposed_value):
    """
    baseline 대비 proposed 개선율 계산.

    improvement [%] = (baseline - proposed) / baseline * 100
    """

    if pd.isna(baseline_value) or pd.isna(proposed_value):
        return np.nan

    if abs(baseline_value) < 1e-12:
        return np.nan

    return (baseline_value - proposed_value) / baseline_value * 100.0


# ============================================================
# Load
# ============================================================

def load_delay_metrics() -> pd.DataFrame:
    """
    가장 최근 delay-aware metrics sweep 결과를 로드한다.
    """

    metrics_path = get_latest_file(
        DELAY_RESULT_DIR,
        "delay_aware_metrics_sweep_*.csv",
    )

    print(f"Load delay metrics: {metrics_path}")

    df = pd.read_csv(metrics_path)

    return df


# ============================================================
# Summary tables
# ============================================================

def build_delayed_gain_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    delayed_gain_db 결과만 추출하여 delay별 요약 테이블 생성.
    """

    delayed_df = (
        metrics_df[metrics_df["mode"] == "delayed_gain_db"]
        .sort_values("inference_delay")
        .reset_index(drop=True)
    )

    selected_cols = [
        "inference_delay",
        "IAE",
        "after_change_IAE",
        "after_change_max_error",
        "final_error",
        "settling_time_after_change",
        "overshoot_percent_after_change",
        "mean_pwm",
        "total_pwm",
        "max_pwm",
        "pwm_saturation_ratio_percent",
    ]

    selected_cols = [col for col in selected_cols if col in delayed_df.columns]

    return delayed_df[selected_cols].copy()


def build_improvement_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    fixed_pid 대비 delayed_gain_db의 delay별 개선율 계산.
    """

    rows = []

    fixed_df = metrics_df[metrics_df["mode"] == "fixed_pid"].copy()

    if fixed_df.empty:
        raise ValueError("fixed_pid result not found in metrics_df.")

    # fixed_pid는 delay와 무관하게 동일하게 반복 저장되어 있으므로 첫 행 사용
    fixed = fixed_df.iloc[0]

    delayed_df = (
        metrics_df[metrics_df["mode"] == "delayed_gain_db"]
        .sort_values("inference_delay")
        .reset_index(drop=True)
    )

    for _, delayed in delayed_df.iterrows():
        row = {
            "inference_delay": delayed["inference_delay"],

            "fixed_after_change_IAE": fixed["after_change_IAE"],
            "delayed_after_change_IAE": delayed["after_change_IAE"],
            "after_change_IAE_improvement_percent": calc_improvement(
                fixed["after_change_IAE"],
                delayed["after_change_IAE"],
            ),

            "fixed_after_change_max_error": fixed["after_change_max_error"],
            "delayed_after_change_max_error": delayed["after_change_max_error"],
            "after_change_max_error_improvement_percent": calc_improvement(
                fixed["after_change_max_error"],
                delayed["after_change_max_error"],
            ),

            "fixed_final_error": fixed["final_error"],
            "delayed_final_error": delayed["final_error"],
            "final_error_improvement_percent": calc_improvement(
                fixed["final_error"],
                delayed["final_error"],
            ),

            "fixed_settling_time_after_change": fixed["settling_time_after_change"],
            "delayed_settling_time_after_change": delayed["settling_time_after_change"],

            "delayed_overshoot_percent_after_change": delayed["overshoot_percent_after_change"],
            "delayed_pwm_saturation_ratio_percent": delayed["pwm_saturation_ratio_percent"],
            "delayed_total_pwm": delayed["total_pwm"],
        }

        rows.append(row)

    improvement_df = pd.DataFrame(rows)

    return improvement_df


def build_immediate_vs_delayed_summary(metrics_df: pd.DataFrame) -> pd.DataFrame:
    """
    immediate_gain_db 대비 delayed_gain_db 성능 차이 계산.
    """

    rows = []

    immediate_df = (
        metrics_df[metrics_df["mode"] == "immediate_gain_db"]
        .sort_values("inference_delay")
        .reset_index(drop=True)
    )

    delayed_df = (
        metrics_df[metrics_df["mode"] == "delayed_gain_db"]
        .sort_values("inference_delay")
        .reset_index(drop=True)
    )

    if immediate_df.empty or delayed_df.empty:
        return pd.DataFrame()

    for _, delayed in delayed_df.iterrows():
        delay = delayed["inference_delay"]

        # 같은 delay row가 있으면 사용
        matched = immediate_df[immediate_df["inference_delay"] == delay]

        if matched.empty:
            immediate = immediate_df.iloc[0]
        else:
            immediate = matched.iloc[0]

        row = {
            "inference_delay": delay,

            "immediate_after_change_IAE": immediate["after_change_IAE"],
            "delayed_after_change_IAE": delayed["after_change_IAE"],
            "after_change_IAE_difference": delayed["after_change_IAE"] - immediate["after_change_IAE"],
            "after_change_IAE_difference_percent": (
                (delayed["after_change_IAE"] - immediate["after_change_IAE"])
                / immediate["after_change_IAE"]
                * 100.0
                if abs(immediate["after_change_IAE"]) > 1e-12
                else np.nan
            ),

            "immediate_settling_time": immediate["settling_time_after_change"],
            "delayed_settling_time": delayed["settling_time_after_change"],
            "settling_time_delay_penalty": (
                delayed["settling_time_after_change"]
                - immediate["settling_time_after_change"]
                if not pd.isna(delayed["settling_time_after_change"])
                and not pd.isna(immediate["settling_time_after_change"])
                else np.nan
            ),

            "immediate_final_error": immediate["final_error"],
            "delayed_final_error": delayed["final_error"],
            "final_error_difference": delayed["final_error"] - immediate["final_error"],

            "immediate_total_pwm": immediate["total_pwm"],
            "delayed_total_pwm": delayed["total_pwm"],
            "total_pwm_difference": delayed["total_pwm"] - immediate["total_pwm"],
        }

        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================
# Save
# ============================================================

def save_csv_tables(
    metrics_df: pd.DataFrame,
    delayed_summary_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    immediate_vs_delayed_df: pd.DataFrame,
):
    """
    CSV 결과 저장.
    """

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    raw_path = SUMMARY_DIR / "delay_aware_metrics_sweep_latest.csv"
    delayed_path = SUMMARY_DIR / "delay_aware_delayed_gain_summary.csv"
    improvement_path = SUMMARY_DIR / "delay_aware_improvement_vs_fixed.csv"
    immediate_vs_delayed_path = SUMMARY_DIR / "delay_aware_immediate_vs_delayed.csv"

    metrics_df.to_csv(raw_path, index=False, encoding="utf-8-sig")
    delayed_summary_df.to_csv(delayed_path, index=False, encoding="utf-8-sig")
    improvement_df.to_csv(improvement_path, index=False, encoding="utf-8-sig")
    immediate_vs_delayed_df.to_csv(immediate_vs_delayed_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {raw_path}")
    print(f"Saved: {delayed_path}")
    print(f"Saved: {improvement_path}")
    print(f"Saved: {immediate_vs_delayed_path}")

    return raw_path, delayed_path, improvement_path, immediate_vs_delayed_path


def save_markdown_summary(
    delayed_summary_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    immediate_vs_delayed_df: pd.DataFrame,
):
    """
    Delay-aware 결과 Markdown summary 저장.
    """

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SUMMARY_DIR / f"delay_aware_result_summary_{timestamp}.md"

    lines = []

    lines.append("# Delay-aware Gain Update Result Summary")
    lines.append("")
    lines.append("## 1. Experiment Setting")
    lines.append("")
    lines.append("- Target step: 100 to 200")
    lines.append("- Target change time: 3.0 s")
    lines.append("- Compared controllers: fixed PID, immediate gain DB, delayed gain DB")
    lines.append("- Inference delay list: 0.0, 0.2, 0.5, 1.0, 1.5, 2.0 s")
    lines.append("")
    lines.append("The delayed gain DB controller keeps the previous gain during the inference delay window and applies the new target-based gain after the delay.")
    lines.append("")

    lines.append("## 2. Delayed Gain DB Summary")
    lines.append("")
    lines.append(delayed_summary_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Improvement over Fixed PID")
    lines.append("")
    lines.append(improvement_df.to_markdown(index=False))
    lines.append("")

    if not immediate_vs_delayed_df.empty:
        lines.append("## 4. Immediate vs Delayed Gain Update")
        lines.append("")
        lines.append(immediate_vs_delayed_df.to_markdown(index=False))
        lines.append("")

    lines.append("## 5. Interpretation")
    lines.append("")
    lines.append(
        "The delayed gain DB controller maintained stable target tracking even when the gain update was delayed. "
        "Compared with the fixed PID controller, the delayed gain DB controller substantially reduced the after-change IAE, "
        "maximum error, and final error for all tested delay conditions."
    )
    lines.append("")
    lines.append(
        "As the inference delay increased, the settling time after the target change gradually increased. "
        "However, the controller still converged to the target even with a 2.0 s delay, supporting the feasibility of delay-aware gain scheduling."
    )
    lines.append("")
    lines.append(
        "The results also show that PWM saturation occurs for a short period after the target step. "
        "Future work should include saturation-aware constraints or control effort penalties in the gain recommendation process."
    )
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved markdown summary: {save_path}")

    return save_path


# ============================================================
# Print
# ============================================================

def print_key_summary(
    delayed_summary_df: pd.DataFrame,
    improvement_df: pd.DataFrame,
    immediate_vs_delayed_df: pd.DataFrame,
):
    print("\n" + "=" * 80)
    print("Delayed Gain DB Summary")
    print("=" * 80)

    cols = [
        "inference_delay",
        "after_change_IAE",
        "after_change_max_error",
        "settling_time_after_change",
        "overshoot_percent_after_change",
        "final_error",
        "pwm_saturation_ratio_percent",
    ]
    cols = [col for col in cols if col in delayed_summary_df.columns]
    print(delayed_summary_df[cols])

    print("\n" + "=" * 80)
    print("Improvement over Fixed PID")
    print("=" * 80)

    cols = [
        "inference_delay",
        "after_change_IAE_improvement_percent",
        "after_change_max_error_improvement_percent",
        "final_error_improvement_percent",
        "delayed_settling_time_after_change",
        "delayed_pwm_saturation_ratio_percent",
    ]
    cols = [col for col in cols if col in improvement_df.columns]
    print(improvement_df[cols])

    if not immediate_vs_delayed_df.empty:
        print("\n" + "=" * 80)
        print("Immediate vs Delayed")
        print("=" * 80)

        cols = [
            "inference_delay",
            "after_change_IAE_difference",
            "after_change_IAE_difference_percent",
            "settling_time_delay_penalty",
            "final_error_difference",
            "total_pwm_difference",
        ]
        cols = [col for col in cols if col in immediate_vs_delayed_df.columns]
        print(immediate_vs_delayed_df[cols])


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Summarize delay-aware results")
    print("=" * 80)

    metrics_df = load_delay_metrics()

    delayed_summary_df = build_delayed_gain_summary(metrics_df)
    improvement_df = build_improvement_summary(metrics_df)
    immediate_vs_delayed_df = build_immediate_vs_delayed_summary(metrics_df)

    print_key_summary(
        delayed_summary_df,
        improvement_df,
        immediate_vs_delayed_df,
    )

    save_csv_tables(
        metrics_df,
        delayed_summary_df,
        improvement_df,
        immediate_vs_delayed_df,
    )

    save_markdown_summary(
        delayed_summary_df,
        improvement_df,
        immediate_vs_delayed_df,
    )


if __name__ == "__main__":
    main()