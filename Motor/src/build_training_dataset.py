import sys
from pathlib import Path

import pandas as pd
import numpy as np


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))


# ============================================================
# Project paths
# ============================================================

GAIN_DB_DIR = MOTOR_DIR / "results" / "simulink_gain_db"
DATASET_DIR = MOTOR_DIR / "data" / "processed"

DATASET_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Settings
# ============================================================

OVERSHOOT_WEIGHT = 30.0
PWM_WEIGHT = 0.01


def get_latest_gain_sweep_file() -> Path:
    """
    가장 최근 Simulink gain sweep CSV 파일을 찾는다.
    """
    files = sorted(GAIN_DB_DIR.glob("simulink_gain_sweep_*.csv"))

    if not files:
        raise FileNotFoundError(f"No gain sweep files found in: {GAIN_DB_DIR}")

    return files[-1]


def ensure_score_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    score 컬럼이 없으면 계산해서 추가한다.
    """
    df = df.copy()

    if "score" not in df.columns:
        df["score"] = (
            df["IAE"]
            + OVERSHOOT_WEIGHT * df["overshoot_percent"]
            + PWM_WEIGHT * df["total_pwm"]
        )

    return df


def clean_success_cases(df: pd.DataFrame) -> pd.DataFrame:
    """
    성공한 simulation case만 추출하고 결측값을 정리한다.
    """
    df = df.copy()

    if "success" in df.columns:
        df = df[df["success"] == True]

    required_cols = [
        "target",
        "kp",
        "ki",
        "kd",
        "IAE",
        "ISE",
        "overshoot_percent",
        "mean_pwm",
        "total_pwm",
        "rise_time",
        "settling_time",
        "score",
    ]

    existing_required_cols = [col for col in required_cols if col in df.columns]
    df = df.dropna(subset=existing_required_cols)

    return df.reset_index(drop=True)


def add_normalized_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    학습에 사용할 수 있는 정규화 feature를 추가한다.
    """
    df = df.copy()

    target_max = df["target"].max()
    kp_max = df["kp"].max()
    ki_max = df["ki"].max()
    kd_max = df["kd"].max() if df["kd"].max() != 0 else 1.0

    df["target_norm"] = df["target"] / max(target_max, 1e-6)
    df["kp_norm"] = df["kp"] / max(kp_max, 1e-6)
    df["ki_norm"] = df["ki"] / max(ki_max, 1e-6)
    df["kd_norm"] = df["kd"] / max(kd_max, 1e-6)

    return df


def build_performance_prediction_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset A:
    target, kp, ki, kd → performance metrics

    이 데이터셋은 나중에 surrogate model로
    특정 target/gain 조합의 성능을 예측하는 데 사용 가능.
    """
    df = df.copy()

    selected_cols = [
        "target",
        "kp",
        "ki",
        "kd",
        "target_norm",
        "kp_norm",
        "ki_norm",
        "kd_norm",
        "IAE",
        "ISE",
        "overshoot_percent",
        "mean_pwm",
        "total_pwm",
        "rise_time",
        "settling_time",
        "score",
    ]

    selected_cols = [col for col in selected_cols if col in df.columns]

    performance_df = df[selected_cols].copy()

    return performance_df


def build_gain_recommendation_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Dataset B:
    target → best kp, ki, kd

    score 기준으로 target별 best gain을 선택한다.
    """
    df = df.copy()

    best_df = (
        df.sort_values("score")
        .groupby("target", as_index=False)
        .head(1)
        .sort_values("target")
        .reset_index(drop=True)
    )

    best_df = best_df.rename(
        columns={
            "kp": "best_kp",
            "ki": "best_ki",
            "kd": "best_kd",
        }
    )

    selected_cols = [
        "target",
        "target_norm",
        "best_kp",
        "best_ki",
        "best_kd",
        "score",
        "IAE",
        "ISE",
        "overshoot_percent",
        "mean_pwm",
        "total_pwm",
        "rise_time",
        "settling_time",
    ]

    selected_cols = [col for col in selected_cols if col in best_df.columns]

    gain_df = best_df[selected_cols].copy()

    return gain_df


def save_dataset(df: pd.DataFrame, filename: str) -> Path:
    """
    DataFrame을 processed data 폴더에 저장한다.
    """
    save_path = DATASET_DIR / filename
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    print(f"Saved: {save_path}")
    return save_path


def main():
    gain_sweep_file = get_latest_gain_sweep_file()
    print(f"Load gain sweep file: {gain_sweep_file}")

    raw_df = pd.read_csv(gain_sweep_file)

    raw_df = ensure_score_column(raw_df)
    clean_df = clean_success_cases(raw_df)
    clean_df = add_normalized_features(clean_df)

    performance_df = build_performance_prediction_dataset(clean_df)
    gain_recommendation_df = build_gain_recommendation_dataset(clean_df)

    save_dataset(
        performance_df,
        "pid_performance_prediction_dataset.csv",
    )

    save_dataset(
        gain_recommendation_df,
        "pid_gain_recommendation_dataset.csv",
    )

    print("\nPerformance prediction dataset:")
    print(performance_df.head())

    print("\nGain recommendation dataset:")
    print(gain_recommendation_df)

    print("\nDataset summary:")
    print(f"Total successful cases: {len(clean_df)}")
    print(f"Number of target points: {gain_recommendation_df['target'].nunique()}")


if __name__ == "__main__":
    main()