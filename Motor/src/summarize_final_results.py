import sys
from pathlib import Path
from datetime import datetime

import pandas as pd


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import RESULTS_DIR, FIGURE_DIR, PID_GAIN_DB


# ============================================================
# Paths
# ============================================================

SUMMARY_DIR = RESULTS_DIR / "summary"
SURROGATE_VALIDATION_DIR = RESULTS_DIR / "surrogate_validation"

MODEL_METRICS_PATH = FIGURE_DIR / "surrogate_model_metrics.csv"
ADAPTIVE_METRICS_PATH = FIGURE_DIR / "adaptive_target_metrics_summary.csv"


def get_latest_file(folder: Path, pattern: str):
    files = sorted(folder.glob(pattern))

    if not files:
        return None

    return files[-1]


def build_pid_gain_db_table():
    rows = []

    for target, gains in PID_GAIN_DB.items():
        rows.append(
            {
                "target": float(target),
                "kp": gains["kp"],
                "ki": gains["ki"],
                "kd": gains["kd"],
            }
        )

    df = pd.DataFrame(rows).sort_values("target").reset_index(drop=True)

    return df


def load_csv_if_exists(path: Path, name: str):
    if path is None or not path.exists():
        print(f"Skip {name}: file not found")
        return None

    print(f"Load {name}: {path}")
    return pd.read_csv(path)


def save_markdown_summary(
    gain_db_df,
    model_metrics_df=None,
    validation_df=None,
    adaptive_metrics_df=None,
):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SUMMARY_DIR / f"final_result_summary_{timestamp}.md"

    lines = []

    lines.append("# Final Result Summary")
    lines.append("")
    lines.append("## 1. Final PID Gain Database")
    lines.append("")
    lines.append(gain_db_df.to_markdown(index=False))
    lines.append("")

    if model_metrics_df is not None:
        lines.append("## 2. Surrogate Model Metrics")
        lines.append("")
        lines.append(model_metrics_df.to_markdown(index=False))
        lines.append("")

    if validation_df is not None:
        lines.append("## 3. Surrogate vs Sweep Validation")
        lines.append("")

        selected_cols = [
            "target",
            "surrogate_kp",
            "surrogate_ki",
            "surrogate_score",
            "sweep_kp",
            "sweep_ki",
            "sweep_score",
            "winner",
            "score_improvement_percent",
        ]

        selected_cols = [col for col in selected_cols if col in validation_df.columns]
        lines.append(validation_df[selected_cols].to_markdown(index=False))
        lines.append("")

    if adaptive_metrics_df is not None:
        lines.append("## 4. Adaptive PID Target Metrics")
        lines.append("")

        selected_cols = [
            "target",
            "final_error",
            "mean_abs_error",
            "IAE",
            "overshoot_percent",
            "rise_time",
            "settling_time",
            "initial_kp",
            "initial_ki",
            "final_kp",
            "final_ki",
            "gain_update_count",
        ]

        selected_cols = [col for col in selected_cols if col in adaptive_metrics_df.columns]
        lines.append(adaptive_metrics_df[selected_cols].to_markdown(index=False))
        lines.append("")

    lines.append("## 5. Current Interpretation")
    lines.append("")
    lines.append(
        "The current controller can be interpreted as a "
        "**Simulink-informed gain-scheduled adaptive PID framework**. "
        "The initial gains are selected from the validated PID gain database, "
        "and linear interpolation is used for unseen target values."
    )
    lines.append("")
    lines.append(
        "The rule-based scheduler remains as an optional fine-tuning layer, "
        "but recent results show that the validated gain database already provides "
        "stable responses with near-zero final error and zero overshoot for tested targets."
    )
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved markdown summary: {save_path}")

    return save_path


def save_csv_tables(
    gain_db_df,
    model_metrics_df=None,
    validation_df=None,
    adaptive_metrics_df=None,
):
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    gain_db_path = SUMMARY_DIR / "final_pid_gain_db.csv"
    gain_db_df.to_csv(gain_db_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {gain_db_path}")

    if model_metrics_df is not None:
        path = SUMMARY_DIR / "final_surrogate_model_metrics.csv"
        model_metrics_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved: {path}")

    if validation_df is not None:
        path = SUMMARY_DIR / "final_surrogate_validation_comparison.csv"
        validation_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved: {path}")

    if adaptive_metrics_df is not None:
        path = SUMMARY_DIR / "final_adaptive_target_metrics.csv"
        adaptive_metrics_df.to_csv(path, index=False, encoding="utf-8-sig")
        print(f"Saved: {path}")


def main():
    print("=" * 80)
    print("Final result summary")
    print("=" * 80)

    gain_db_df = build_pid_gain_db_table()

    validation_path = get_latest_file(
        SURROGATE_VALIDATION_DIR,
        "surrogate_vs_sweep_comparison_*.csv",
    )

    model_metrics_df = load_csv_if_exists(
        MODEL_METRICS_PATH,
        "surrogate model metrics",
    )

    validation_df = load_csv_if_exists(
        validation_path,
        "surrogate validation comparison",
    )

    adaptive_metrics_df = load_csv_if_exists(
        ADAPTIVE_METRICS_PATH,
        "adaptive target metrics",
    )

    print("\nFinal PID_GAIN_DB:")
    print(gain_db_df)

    if model_metrics_df is not None:
        print("\nSurrogate model metrics:")
        print(model_metrics_df)

    if validation_df is not None:
        print("\nSurrogate validation comparison:")
        print(validation_df)

    if adaptive_metrics_df is not None:
        print("\nAdaptive target metrics:")
        print(adaptive_metrics_df)

    save_csv_tables(
        gain_db_df=gain_db_df,
        model_metrics_df=model_metrics_df,
        validation_df=validation_df,
        adaptive_metrics_df=adaptive_metrics_df,
    )

    save_markdown_summary(
        gain_db_df=gain_db_df,
        model_metrics_df=model_metrics_df,
        validation_df=validation_df,
        adaptive_metrics_df=adaptive_metrics_df,
    )


if __name__ == "__main__":
    main()