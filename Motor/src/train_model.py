import sys
from pathlib import Path
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import MODEL_DIR, FIGURE_DIR


# ============================================================
# Dataset path
# ============================================================

DATASET_PATH = MOTOR_DIR / "data" / "processed" / "pid_performance_prediction_dataset.csv"


# ============================================================
# Settings
# ============================================================

FEATURE_COLS = [
    "target",
    "kp",
    "ki",
    "kd",
]

TARGET_COL = "score"


def load_dataset() -> pd.DataFrame:
    """
    PID performance prediction dataset 로드
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)

    required_cols = FEATURE_COLS + [TARGET_COL]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing columns in dataset: {missing_cols}")

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df


def train_models(X_train, y_train):
    """
    여러 surrogate model 학습
    """

    models = {
        "random_forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
    }

    trained_models = {}

    for name, model in models.items():
        print(f"\nTraining model: {name}")
        model.fit(X_train, y_train)
        trained_models[name] = model

    return trained_models


def evaluate_model(model, X_train, y_train, X_test, y_test) -> dict:
    """
    모델 성능 평가
    """

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)

    train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    return {
        "train_mae": train_mae,
        "test_mae": test_mae,
        "train_rmse": train_rmse,
        "test_rmse": test_rmse,
        "train_r2": train_r2,
        "test_r2": test_r2,
        "y_train_pred": y_train_pred,
        "y_test_pred": y_test_pred,
    }


def plot_prediction_result(y_true, y_pred, model_name: str):
    """
    예측 결과 scatter plot 저장
    """

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.8)

    min_value = min(y_true.min(), y_pred.min())
    max_value = max(y_true.max(), y_pred.max())

    plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

    plt.xlabel("True score")
    plt.ylabel("Predicted score")
    plt.title(f"Surrogate Prediction Result - {model_name}")
    plt.grid(True)

    save_path = FIGURE_DIR / f"surrogate_prediction_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_feature_importance(model, model_name: str):
    """
    Feature importance 저장
    """

    if not hasattr(model, "feature_importances_"):
        print(f"Skip feature importance: {model_name}")
        return

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    importances = model.feature_importances_

    importance_df = pd.DataFrame(
        {
            "feature": FEATURE_COLS,
            "importance": importances,
        }
    ).sort_values("importance", ascending=False)

    print(f"\nFeature importance - {model_name}")
    print(importance_df)

    plt.figure(figsize=(7, 4))
    plt.bar(importance_df["feature"], importance_df["importance"])
    plt.xlabel("Feature")
    plt.ylabel("Importance")
    plt.title(f"Feature Importance - {model_name}")
    plt.grid(True)

    save_path = FIGURE_DIR / f"surrogate_feature_importance_{model_name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def save_model(model, model_name: str):
    """
    학습된 모델 저장
    """

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = MODEL_DIR / f"surrogate_{model_name}_{timestamp}.joblib"

    joblib.dump(
        {
            "model": model,
            "feature_cols": FEATURE_COLS,
            "target_col": TARGET_COL,
        },
        save_path,
    )

    print(f"Saved model: {save_path}")

    return save_path


def main():
    df = load_dataset()

    print("=" * 80)
    print("Load dataset")
    print("=" * 80)
    print(f"Dataset path: {DATASET_PATH}")
    print(f"Dataset shape: {df.shape}")
    print(df.head())

    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
    )

    trained_models = train_models(X_train, y_train)

    results = []

    best_model_name = None
    best_model = None
    best_test_mae = float("inf")

    for model_name, model in trained_models.items():
        metrics = evaluate_model(
            model,
            X_train,
            y_train,
            X_test,
            y_test,
        )

        print("\n" + "=" * 80)
        print(f"Model: {model_name}")
        print("=" * 80)
        print(f"Train MAE : {metrics['train_mae']:.4f}")
        print(f"Test MAE  : {metrics['test_mae']:.4f}")
        print(f"Train RMSE: {metrics['train_rmse']:.4f}")
        print(f"Test RMSE : {metrics['test_rmse']:.4f}")
        print(f"Train R2  : {metrics['train_r2']:.4f}")
        print(f"Test R2   : {metrics['test_r2']:.4f}")

        results.append(
            {
                "model": model_name,
                "train_mae": metrics["train_mae"],
                "test_mae": metrics["test_mae"],
                "train_rmse": metrics["train_rmse"],
                "test_rmse": metrics["test_rmse"],
                "train_r2": metrics["train_r2"],
                "test_r2": metrics["test_r2"],
            }
        )

        plot_prediction_result(
            y_test,
            metrics["y_test_pred"],
            model_name,
        )

        plot_feature_importance(model, model_name)

        if metrics["test_mae"] < best_test_mae:
            best_test_mae = metrics["test_mae"]
            best_model_name = model_name
            best_model = model

    result_df = pd.DataFrame(results)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    result_path = FIGURE_DIR / "surrogate_model_metrics.csv"
    result_df.to_csv(result_path, index=False, encoding="utf-8-sig")

    print("\n" + "=" * 80)
    print("Model comparison")
    print("=" * 80)
    print(result_df)
    print(f"Saved: {result_path}")

    print("\n" + "=" * 80)
    print("Best model")
    print("=" * 80)
    print(f"Best model: {best_model_name}")
    print(f"Best test MAE: {best_test_mae:.4f}")

    save_model(best_model, best_model_name)


if __name__ == "__main__":
    main()