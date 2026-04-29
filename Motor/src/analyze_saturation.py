import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd


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
# Paths
# ============================================================

SUMMARY_DIR = RESULTS_DIR / "summary"
SATURATION_DIR = RESULTS_DIR / "saturation_analysis"


# ============================================================
# Settings
# ============================================================

SATURATION_TOL = 1e-9

LOG_PATTERNS = {
    "fixed_pid": "fixed_pid_simple_motor_*.csv",
    "adaptive_pid": "adaptive_pid_simple_motor_*.csv",
    "simulink_pid": "simulink_motor_*.csv",
}


# ============================================================
# Utility
# ============================================================

def infer_experiment_type(df: pd.DataFrame) -> str:
    """
    로그 컬럼을 기반으로 실험 종류를 추정한다.
    """

    if "use_disturbance" in df.columns:
        use_disturbance = str(df["use_disturbance"].iloc[0]).lower()
        if use_disturbance in ["true", "1"]:
            return "disturbance"

    if "disturbance" in df.columns:
        if np.abs(df["disturbance"].to_numpy()).max() > 1e-9:
            return "disturbance"

    if "target" in df.columns:
        target_unique = df["target"].dropna().unique()
        if len(target_unique) > 1:
            return "target_step"

    return "target_tracking"


def compute_dt(time: np.ndarray) -> np.ndarray:
    """
    time 배열로부터 dt 배열 계산.
    """

    if len(time) <= 1:
        return np.zeros_like(time)

    dt = np.diff(time, prepend=time[0])
    dt[0] = 0.0

    return dt


def calculate_basic_control_metrics(df: pd.DataFrame) -> dict:
    """
    IAE, overshoot, settling time 등 기본 제어 지표 계산.
    """

    time = df["time"].to_numpy()
    target = df["target"].to_numpy()
    current = df["current"].to_numpy()

    if "error" in df.columns:
        error = df["error"].to_numpy()
    else:
        error = target - current

    abs_error = np.abs(error)
    dt = compute_dt(time)

    iae = float(np.sum(abs_error * dt))
    ise = float(np.sum((error ** 2) * dt))
    mean_abs_error = float(np.mean(abs_error))
    final_error = float(abs_error[-1])

    target_final = float(target[-1])
    max_current = float(np.max(current))

    overshoot = max(0.0, max_current - target_final)
    overshoot_percent = overshoot / max(abs(target_final), 1e-6) * 100.0

    tolerance = 0.02 * abs(target_final)
    settling_time = np.nan

    for i in range(len(time)):
        if np.all(np.abs(target[i:] - current[i:]) <= tolerance):
            settling_time = float(time[i])
            break

    return {
        "IAE": iae,
        "ISE": ise,
        "mean_abs_error": mean_abs_error,
        "final_error": final_error,
        "overshoot_percent": overshoot_percent,
        "settling_time": settling_time,
    }


def calculate_saturation_metrics(df: pd.DataFrame) -> dict:
    """
    PWM saturation 관련 지표 계산.
    """

    time = df["time"].to_numpy()
    pwm = df["pwm"].to_numpy()
    dt = compute_dt(time)

    abs_pwm = np.abs(pwm)

    high_saturation = pwm >= PWM_MAX - SATURATION_TOL
    low_saturation = pwm <= PWM_MIN + SATURATION_TOL
    saturation_mask = high_saturation | low_saturation

    saturation_count = int(np.sum(saturation_mask))
    high_saturation_count = int(np.sum(high_saturation))
    low_saturation_count = int(np.sum(low_saturation))

    n = len(pwm)

    saturation_ratio_percent = float(saturation_count / max(n, 1) * 100.0)
    high_saturation_ratio_percent = float(high_saturation_count / max(n, 1) * 100.0)
    low_saturation_ratio_percent = float(low_saturation_count / max(n, 1) * 100.0)

    saturation_duration = float(np.sum(dt[saturation_mask]))
    high_saturation_duration = float(np.sum(dt[high_saturation]))
    low_saturation_duration = float(np.sum(dt[low_saturation]))

    if saturation_mask.any():
        first_saturation_time = float(time[saturation_mask][0])
        last_saturation_time = float(time[saturation_mask][-1])
    else:
        first_saturation_time = np.nan
        last_saturation_time = np.nan

    pwm_range = PWM_MAX - PWM_MIN

    near_high_saturation = pwm >= PWM_MAX - 0.1 * pwm_range
    near_low_saturation = pwm <= PWM_MIN + 0.02 * pwm_range

    near_high_saturation_ratio_percent = float(
        np.mean(near_high_saturation) * 100.0
    )
    near_low_saturation_ratio_percent = float(
        np.mean(near_low_saturation) * 100.0
    )

    return {
        "max_pwm": float(np.max(pwm)),
        "min_pwm": float(np.min(pwm)),
        "mean_pwm": float(np.mean(abs_pwm)),
        "total_pwm": float(np.sum(abs_pwm * dt)),

        "saturation_count": saturation_count,
        "saturation_ratio_percent": saturation_ratio_percent,
        "saturation_duration": saturation_duration,

        "high_saturation_count": high_saturation_count,
        "high_saturation_ratio_percent": high_saturation_ratio_percent,
        "high_saturation_duration": high_saturation_duration,

        "low_saturation_count": low_saturation_count,
        "low_saturation_ratio_percent": low_saturation_ratio_percent,
        "low_saturation_duration": low_saturation_duration,

        "first_saturation_time": first_saturation_time,
        "last_saturation_time": last_saturation_time,

        "near_high_saturation_ratio_percent": near_high_saturation_ratio_percent,
        "near_low_saturation_ratio_percent": near_low_saturation_ratio_percent,
    }


def calculate_disturbance_metrics_if_available(df: pd.DataFrame) -> dict:
    """
    disturbance 컬럼이 있는 경우 외란 구간 saturation 및 IAE 계산.
    """

    if "disturbance" not in df.columns:
        return {}

    disturbance = df["disturbance"].to_numpy()
    disturbance_mask = np.abs(disturbance) > 1e-9

    if not disturbance_mask.any():
        return {}

    time = df["time"].to_numpy()
    pwm = df["pwm"].to_numpy()
    target = df["target"].to_numpy()
    current = df["current"].to_numpy()

    if "error" in df.columns:
        error = df["error"].to_numpy()
    else:
        error = target - current

    dt = compute_dt(time)
    abs_error = np.abs(error)

    high_saturation = pwm >= PWM_MAX - SATURATION_TOL
    low_saturation = pwm <= PWM_MIN + SATURATION_TOL
    saturation_mask = high_saturation | low_saturation

    disturbance_saturation_mask = disturbance_mask & saturation_mask

    return {
        "disturbance_start_time": float(time[disturbance_mask][0]),
        "disturbance_end_time": float(time[disturbance_mask][-1]),
        "disturbance_IAE": float(np.sum(abs_error[disturbance_mask] * dt[disturbance_mask])),
        "disturbance_mean_abs_error": float(np.mean(abs_error[disturbance_mask])),
        "disturbance_max_error": float(np.max(abs_error[disturbance_mask])),
        "disturbance_min_current": float(np.min(current[disturbance_mask])),
        "disturbance_mean_pwm": float(np.mean(np.abs(pwm[disturbance_mask]))),
        "disturbance_max_pwm": float(np.max(pwm[disturbance_mask])),
        "disturbance_saturation_count": int(np.sum(disturbance_saturation_mask)),
        "disturbance_saturation_ratio_percent": float(
            np.sum(disturbance_saturation_mask)
            / max(np.sum(disturbance_mask), 1)
            * 100.0
        ),
        "disturbance_saturation_duration": float(
            np.sum(dt[disturbance_saturation_mask])
        ),
    }


def analyze_single_log(log_file: Path, mode_hint: str = None) -> dict:
    """
    단일 로그 파일 분석.
    """

    df = pd.read_csv(log_file)

    if len(df) == 0:
        raise ValueError(f"Empty log file: {log_file}")

    required_cols = ["time", "target", "current", "pwm"]

    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing column {col} in {log_file}")

    mode = mode_hint

    if mode is None:
        if "mode" in df.columns:
            mode = str(df["mode"].iloc[0])
        else:
            mode = "unknown"

    experiment_type = infer_experiment_type(df)

    target_initial = float(df["target"].iloc[0])
    target_final = float(df["target"].iloc[-1])
    target_unique_count = int(df["target"].nunique())

    result = {
        "log_file": str(log_file),
        "file_name": log_file.name,
        "mode": mode,
        "experiment_type": experiment_type,
        "target_initial": target_initial,
        "target_final": target_final,
        "target_unique_count": target_unique_count,
        "duration": float(df["time"].iloc[-1] - df["time"].iloc[0]),
        "num_samples": int(len(df)),
    }

    result.update(calculate_basic_control_metrics(df))
    result.update(calculate_saturation_metrics(df))
    result.update(calculate_disturbance_metrics_if_available(df))

    if "kp" in df.columns:
        result["initial_kp"] = float(df["kp"].iloc[0])
        result["final_kp"] = float(df["kp"].iloc[-1])

    if "ki" in df.columns:
        result["initial_ki"] = float(df["ki"].iloc[0])
        result["final_ki"] = float(df["ki"].iloc[-1])

    if "kd" in df.columns:
        result["initial_kd"] = float(df["kd"].iloc[0])
        result["final_kd"] = float(df["kd"].iloc[-1])

    return result


def load_and_analyze_logs() -> pd.DataFrame:
    """
    LOG_DIR의 fixed/adaptive/simulink 로그를 모두 분석.
    """

    rows = []

    for mode, pattern in LOG_PATTERNS.items():
        log_files = sorted(LOG_DIR.glob(pattern))

        print(f"{mode}: {len(log_files)} files")

        for log_file in log_files:
            try:
                row = analyze_single_log(log_file, mode_hint=mode)
                rows.append(row)
            except Exception as e:
                print(f"Skip: {log_file}")
                print(f"Reason: {e}")

    if not rows:
        raise FileNotFoundError("No valid log files found.")

    df = pd.DataFrame(rows)

    return df


def build_latest_per_condition_summary(all_df: pd.DataFrame) -> pd.DataFrame:
    """
    같은 mode / experiment_type / target_final 조건에서 가장 최신 파일만 남긴다.
    """

    df = all_df.copy()

    df["file_mtime"] = df["log_file"].apply(lambda x: Path(x).stat().st_mtime)

    group_cols = [
        "mode",
        "experiment_type",
        "target_initial",
        "target_final",
    ]

    idx = df.groupby(group_cols)["file_mtime"].idxmax()

    latest_df = (
        df.loc[idx]
        .sort_values(["experiment_type", "target_final", "mode"])
        .reset_index(drop=True)
    )

    latest_df = latest_df.drop(columns=["file_mtime"])

    return latest_df


def build_saturation_risk_table(latest_df: pd.DataFrame) -> pd.DataFrame:
    """
    saturation risk 기준으로 정렬된 테이블 생성.
    """

    risk_df = latest_df.copy()

    risk_df["saturation_risk_score"] = (
        risk_df["saturation_ratio_percent"].fillna(0.0) * 1.0
        + risk_df["near_high_saturation_ratio_percent"].fillna(0.0) * 0.2
        + risk_df["overshoot_percent"].fillna(0.0) * 0.5
    )

    selected_cols = [
        "mode",
        "experiment_type",
        "target_initial",
        "target_final",
        "IAE",
        "final_error",
        "overshoot_percent",
        "settling_time",
        "max_pwm",
        "mean_pwm",
        "total_pwm",
        "saturation_ratio_percent",
        "saturation_duration",
        "near_high_saturation_ratio_percent",
        "first_saturation_time",
        "last_saturation_time",
        "saturation_risk_score",
        "file_name",
    ]

    selected_cols = [col for col in selected_cols if col in risk_df.columns]

    risk_df = (
        risk_df[selected_cols]
        .sort_values("saturation_risk_score", ascending=False)
        .reset_index(drop=True)
    )

    return risk_df


def save_results(
    all_df: pd.DataFrame,
    latest_df: pd.DataFrame,
    risk_df: pd.DataFrame,
):
    """
    분석 결과 저장.
    """

    SATURATION_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_path = SATURATION_DIR / f"saturation_all_logs_{timestamp}.csv"
    latest_path = SATURATION_DIR / f"saturation_latest_per_condition_{timestamp}.csv"
    risk_path = SATURATION_DIR / f"saturation_risk_table_{timestamp}.csv"

    all_df.to_csv(all_path, index=False, encoding="utf-8-sig")
    latest_df.to_csv(latest_path, index=False, encoding="utf-8-sig")
    risk_df.to_csv(risk_path, index=False, encoding="utf-8-sig")

    summary_latest_path = SUMMARY_DIR / "saturation_latest_per_condition.csv"
    summary_risk_path = SUMMARY_DIR / "saturation_risk_table.csv"

    latest_df.to_csv(summary_latest_path, index=False, encoding="utf-8-sig")
    risk_df.to_csv(summary_risk_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {all_path}")
    print(f"Saved: {latest_path}")
    print(f"Saved: {risk_path}")
    print(f"Saved: {summary_latest_path}")
    print(f"Saved: {summary_risk_path}")

    return all_path, latest_path, risk_path


def print_key_summary(risk_df: pd.DataFrame):
    """
    핵심 결과 출력.
    """

    print("\n" + "=" * 80)
    print("Top saturation risk cases")
    print("=" * 80)

    display_cols = [
        "mode",
        "experiment_type",
        "target_final",
        "max_pwm",
        "saturation_ratio_percent",
        "saturation_duration",
        "near_high_saturation_ratio_percent",
        "overshoot_percent",
        "IAE",
        "saturation_risk_score",
    ]

    display_cols = [col for col in display_cols if col in risk_df.columns]

    print(risk_df[display_cols].head(20))

    print("\n" + "=" * 80)
    print("Mean saturation ratio by mode and experiment type")
    print("=" * 80)

    summary = (
        risk_df
        .groupby(["experiment_type", "mode"])[
            [
                "saturation_ratio_percent",
                "near_high_saturation_ratio_percent",
                "max_pwm",
                "IAE",
            ]
        ]
        .mean()
        .reset_index()
    )

    print(summary)


def save_markdown_summary(risk_df: pd.DataFrame):
    """
    Markdown summary 저장.
    """

    SUMMARY_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = SUMMARY_DIR / f"saturation_analysis_summary_{timestamp}.md"

    top_cols = [
        "mode",
        "experiment_type",
        "target_final",
        "IAE",
        "overshoot_percent",
        "max_pwm",
        "saturation_ratio_percent",
        "saturation_duration",
        "near_high_saturation_ratio_percent",
        "saturation_risk_score",
    ]
    top_cols = [col for col in top_cols if col in risk_df.columns]

    lines = []

    lines.append("# Saturation Analysis Summary")
    lines.append("")
    lines.append("## 1. Purpose")
    lines.append("")
    lines.append(
        "This analysis quantifies PWM saturation behavior in previously generated control logs. "
        "The goal is to identify conditions where the controller relies on saturated PWM effort, "
        "which may be problematic for real motor implementation."
    )
    lines.append("")

    lines.append("## 2. Saturation Metrics")
    lines.append("")
    lines.append("- `saturation_ratio_percent`: ratio of samples where PWM reaches the exact upper or lower bound.")
    lines.append("- `saturation_duration`: accumulated duration of exact PWM saturation.")
    lines.append("- `near_high_saturation_ratio_percent`: ratio of samples near the high PWM bound.")
    lines.append("- `saturation_risk_score`: heuristic risk score combining saturation ratio, near-high saturation ratio, and overshoot.")
    lines.append("")

    lines.append("## 3. Top Saturation Risk Cases")
    lines.append("")
    lines.append(risk_df[top_cols].head(20).to_markdown(index=False))
    lines.append("")

    lines.append("## 4. Interpretation")
    lines.append("")
    lines.append(
        "Cases with high saturation ratio or long saturation duration indicate that the controller "
        "achieves tracking performance by using aggressive control effort. "
        "For real-system deployment, these cases should be re-evaluated using saturation-aware gain selection "
        "or a score function that penalizes saturated PWM usage."
    )
    lines.append("")

    lines.append(
        "The next step is to incorporate saturation-related penalties into the gain selection score "
        "and compare the original PID gain DB with a saturation-aware PID gain DB."
    )
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"Saved markdown summary: {save_path}")

    return save_path


def main():
    print("=" * 80)
    print("Analyze PWM saturation from existing logs")
    print("=" * 80)

    all_df = load_and_analyze_logs()
    latest_df = build_latest_per_condition_summary(all_df)
    risk_df = build_saturation_risk_table(latest_df)

    print_key_summary(risk_df)

    save_results(all_df, latest_df, risk_df)
    save_markdown_summary(risk_df)


if __name__ == "__main__":
    main()