import sys
from pathlib import Path
from datetime import datetime

import joblib
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
    MODEL_DIR,
    FIGURE_DIR,
    RESULTS_DIR,
    TARGET_MIN,
    TARGET_MAX,
)


# ============================================================
# Settings
# ============================================================

OPTIMIZATION_RESULT_DIR = RESULTS_DIR / "model_gain_optimization"

TARGET_LIST = [50, 75, 100, 125, 150, 175, 200]

# surrogate 기반 dense search용 후보
SEARCH_KP_LIST = np.round(np.arange(3.0, 5.01, 0.05), 3)
SEARCH_KI_LIST = np.round(np.arange(1.5, 8.51, 0.05), 3)
SEARCH_KD_LIST = [0.0]

TOP_N = 10


def get_latest_surrogate_model_file() -> Path:
    """
    가장 최근 random forest surrogate model 파일을 찾는다.
    """

    files = sorted(MODEL_DIR.glob("surrogate_random_forest_*.joblib"))

    if not files:
        raise FileNotFoundError(f"No random forest surrogate model found in: {MODEL_DIR}")

    return files[-1]


def load_surrogate_model():
    """
    저장된 surrogate model 로드
    """

    model_path = get_latest_surrogate_model_file()
    print(f"Load surrogate model: {model_path}")

    model_data = joblib.load(model_path)

    model = model_data["model"]
    feature_cols = model_data["feature_cols"]
    target_col = model_data["target_col"]

    print(f"Feature columns: {feature_cols}")
    print(f"Target column: {target_col}")

    return model, feature_cols, model_path


def create_candidate_grid(target: float) -> pd.DataFrame:
    """
    특정 target에 대한 gain candidate grid 생성
    """

    rows = []

    for kp in SEARCH_KP_LIST:
        for ki in SEARCH_KI_LIST:
            for kd in SEARCH_KD_LIST:
                rows.append(
                    {
                        "target": float(target),
                        "kp": float(kp),
                        "ki": float(ki),
                        "kd": float(kd),
                    }
                )

    candidate_df = pd.DataFrame(rows)

    return candidate_df


def predict_score(model, feature_cols, candidate_df: pd.DataFrame) -> pd.DataFrame:
    """
    surrogate model로 candidate score 예측
    """

    df = candidate_df.copy()

    X = df[feature_cols].values
    df["predicted_score"] = model.predict(X)

    return df


def optimize_gain_for_target(model, feature_cols, target: float) -> pd.DataFrame:
    """
    특정 target에 대해 surrogate score 기준 Top-N gain 후보 반환
    """

    candidate_df = create_candidate_grid(target)
    prediction_df = predict_score(model, feature_cols, candidate_df)

    top_df = (
        prediction_df.sort_values("predicted_score", ascending=True)
        .head(TOP_N)
        .reset_index(drop=True)
    )

    return top_df


def optimize_gain_for_targets(model, feature_cols, target_list) -> pd.DataFrame:
    """
    여러 target에 대해 gain 최적화 수행
    """

    all_results = []

    for target in target_list:
        print("-" * 80)
        print(f"Optimize target: {target}")

        top_df = optimize_gain_for_target(
            model=model,
            feature_cols=feature_cols,
            target=target,
        )

        print(top_df)

        top_df["rank"] = np.arange(1, len(top_df) + 1)
        all_results.append(top_df)

    result_df = pd.concat(all_results, axis=0).reset_index(drop=True)

    return result_df


def build_best_gain_table(result_df: pd.DataFrame) -> pd.DataFrame:
    """
    target별 best predicted gain table 생성
    """

    best_df = (
        result_df.sort_values(["target", "predicted_score"])
        .groupby("target", as_index=False)
        .head(1)
        .sort_values("target")
        .reset_index(drop=True)
    )

    best_df = best_df[
        [
            "target",
            "kp",
            "ki",
            "kd",
            "predicted_score",
        ]
    ]

    return best_df


def plot_best_gain_table(best_df: pd.DataFrame):
    """
    target별 추천 gain 시각화
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 5))
    plt.plot(best_df["target"], best_df["kp"], marker="o", label="Kp")
    plt.plot(best_df["target"], best_df["ki"], marker="o", label="Ki")
    plt.plot(best_df["target"], best_df["kd"], marker="o", label="Kd")

    plt.xlabel("Target")
    plt.ylabel("Recommended gain")
    plt.title("Surrogate-based Recommended PID Gains")
    plt.grid(True)
    plt.legend()

    save_path = FIGURE_DIR / "surrogate_recommended_gains.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_predicted_score_map(model, feature_cols, target: float):
    """
    특정 target에 대해 Kp-Ki predicted score map 시각화
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    candidate_df = create_candidate_grid(target)
    prediction_df = predict_score(model, feature_cols, candidate_df)

    pivot_df = prediction_df.pivot_table(
        index="ki",
        columns="kp",
        values="predicted_score",
    )

    plt.figure(figsize=(9, 6))
    contour = plt.contourf(
        pivot_df.columns.values,
        pivot_df.index.values,
        pivot_df.values,
        levels=30,
    )
    plt.colorbar(contour, label="Predicted score")

    best_row = prediction_df.sort_values("predicted_score").iloc[0]

    plt.scatter(
        best_row["kp"],
        best_row["ki"],
        marker="x",
        s=120,
        label=f"Best: Kp={best_row['kp']:.2f}, Ki={best_row['ki']:.2f}",
    )

    plt.xlabel("Kp")
    plt.ylabel("Ki")
    plt.title(f"Predicted Score Map at Target={target}")
    plt.grid(True)
    plt.legend()

    save_path = FIGURE_DIR / f"surrogate_score_map_target_{int(target)}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def save_results(result_df: pd.DataFrame, best_df: pd.DataFrame):
    """
    최적화 결과 저장
    """

    OPTIMIZATION_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    all_result_path = OPTIMIZATION_RESULT_DIR / f"surrogate_gain_candidates_{timestamp}.csv"
    best_result_path = OPTIMIZATION_RESULT_DIR / f"surrogate_best_gain_table_{timestamp}.csv"

    result_df.to_csv(all_result_path, index=False, encoding="utf-8-sig")
    best_df.to_csv(best_result_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {all_result_path}")
    print(f"Saved: {best_result_path}")

    return all_result_path, best_result_path


def print_config_style_gain_db(best_df: pd.DataFrame):
    """
    config.py에 바로 붙여넣을 수 있는 PID_GAIN_DB 형태 출력
    """

    print("\n" + "=" * 80)
    print("PID_GAIN_DB candidate from surrogate model")
    print("=" * 80)

    print("PID_GAIN_DB = {")
    for _, row in best_df.iterrows():
        target = float(row["target"])
        kp = float(row["kp"])
        ki = float(row["ki"])
        kd = float(row["kd"])

        print(
            f"    {target:.1f}: "
            f'{{"kp": {kp:.3f}, "ki": {ki:.3f}, "kd": {kd:.3f}}},'
        )
    print("}")


def main():
    model, feature_cols, model_path = load_surrogate_model()

    print("=" * 80)
    print("Surrogate-based gain optimization")
    print("=" * 80)
    print(f"Target list: {TARGET_LIST}")
    print(f"Kp search range: {SEARCH_KP_LIST[0]} ~ {SEARCH_KP_LIST[-1]}, n={len(SEARCH_KP_LIST)}")
    print(f"Ki search range: {SEARCH_KI_LIST[0]} ~ {SEARCH_KI_LIST[-1]}, n={len(SEARCH_KI_LIST)}")
    print(f"Kd search list: {SEARCH_KD_LIST}")
    print(f"Total candidates per target: {len(SEARCH_KP_LIST) * len(SEARCH_KI_LIST) * len(SEARCH_KD_LIST)}")

    result_df = optimize_gain_for_targets(
        model=model,
        feature_cols=feature_cols,
        target_list=TARGET_LIST,
    )

    best_df = build_best_gain_table(result_df)

    print("\n" + "=" * 80)
    print("Best gain table from surrogate model")
    print("=" * 80)
    print(best_df)

    save_results(result_df, best_df)
    print_config_style_gain_db(best_df)

    plot_best_gain_table(best_df)

    # 대표 target score map 저장
    for target in [50, 100, 150, 200]:
        if target in TARGET_LIST:
            plot_predicted_score_map(model, feature_cols, target)


if __name__ == "__main__":
    main()