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

from simulink_runner import SimulinkRunner
from config import (
    RESULTS_DIR,
    SIMULINK_MODEL_NAME,
    SIMULINK_MAT_FILE,
    SIMULINK_SWEEP_STOP_TIME,
)


# ============================================================
# Validation settings
# ============================================================

VALIDATION_RESULT_DIR = RESULTS_DIR / "surrogate_validation"

OVERSHOOT_WEIGHT = 30.0
PWM_WEIGHT = 0.01


# Surrogate model 추천 gain
SURROGATE_GAIN_LIST = [
    {"target": 50.0,  "kp": 4.30, "ki": 8.25, "kd": 0.0},
    {"target": 75.0,  "kp": 4.25, "ki": 6.85, "kd": 0.0},
    {"target": 100.0, "kp": 4.00, "ki": 5.55, "kd": 0.0},
    {"target": 125.0, "kp": 3.15, "ki": 4.80, "kd": 0.0},
    {"target": 150.0, "kp": 3.60, "ki": 3.90, "kd": 0.0},
    {"target": 175.0, "kp": 3.70, "ki": 3.25, "kd": 0.0},
    {"target": 200.0, "kp": 3.90, "ki": 2.45, "kd": 0.0},
]


# 기존 실제 sweep best gain
SWEEP_BEST_GAIN_LIST = [
    {"target": 50.0,  "kp": 4.4, "ki": 6.5, "kd": 0.0},
    {"target": 75.0,  "kp": 4.4, "ki": 6.5, "kd": 0.0},
    {"target": 100.0, "kp": 4.2, "ki": 6.0, "kd": 0.0},
    {"target": 125.0, "kp": 4.0, "ki": 5.0, "kd": 0.0},
    {"target": 150.0, "kp": 4.4, "ki": 4.0, "kd": 0.0},
    {"target": 175.0, "kp": 4.4, "ki": 3.0, "kd": 0.0},
    {"target": 200.0, "kp": 4.4, "ki": 2.5, "kd": 0.0},
]


def calculate_score(row: dict) -> float:
    """
    score = IAE + overshoot penalty + PWM penalty
    """
    return (
        row["IAE"]
        + OVERSHOOT_WEIGHT * row["overshoot_percent"]
        + PWM_WEIGHT * row["total_pwm"]
    )


def evaluate_response(df: pd.DataFrame, target: float, kp: float, ki: float, kd: float, source: str) -> dict:
    """
    Simulink 결과 DataFrame에서 성능 지표 계산
    """

    time = df["time"].values
    current = df["current"].values
    pwm = df["pwm"].values

    error = target - current
    abs_error = abs(error)

    dt = time[1] - time[0] if len(time) > 1 else 0.01

    iae = abs_error.sum() * dt
    ise = (error ** 2).sum() * dt

    max_current = current.max()
    overshoot = max(0.0, max_current - target)
    overshoot_percent = overshoot / max(abs(target), 1e-6) * 100.0

    mean_pwm = pwm.mean()
    total_pwm = pwm.sum() * dt

    # rise time: 10% -> 90%
    rise_time = None
    try:
        t10_idx = next(i for i, v in enumerate(current) if v >= 0.1 * target)
        t90_idx = next(i for i, v in enumerate(current) if v >= 0.9 * target)
        rise_time = time[t90_idx] - time[t10_idx]
    except StopIteration:
        rise_time = None

    # settling time: ±2% band
    settling_time = None
    tolerance = 0.02 * abs(target)

    for i in range(len(current)):
        remaining_error = abs(target - current[i:])
        if (remaining_error <= tolerance).all():
            settling_time = time[i]
            break

    result = {
        "source": source,
        "target": target,
        "kp": kp,
        "ki": ki,
        "kd": kd,
        "IAE": iae,
        "ISE": ise,
        "overshoot": overshoot,
        "overshoot_percent": overshoot_percent,
        "mean_pwm": mean_pwm,
        "total_pwm": total_pwm,
        "rise_time": rise_time,
        "settling_time": settling_time,
    }

    result["score"] = calculate_score(result)

    return result


def run_single_case(runner: SimulinkRunner, case: dict, source: str) -> dict:
    """
    gain 하나에 대해 Simulink 실행 후 성능 계산
    """

    target = float(case["target"])
    kp = float(case["kp"])
    ki = float(case["ki"])
    kd = float(case["kd"])

    print(
        f"[{source}] target={target}, "
        f"Kp={kp}, Ki={ki}, Kd={kd}"
    )

    runner.run_simulation(
        stop_time=SIMULINK_SWEEP_STOP_TIME,
        target=target,
        kp=kp,
        ki=ki,
        kd=kd,
    )

    df = runner.get_simulink_dataframe()

    result = evaluate_response(
        df=df,
        target=target,
        kp=kp,
        ki=ki,
        kd=kd,
        source=source,
    )

    return result


def validate_gain_list(runner: SimulinkRunner, gain_list: list, source: str) -> pd.DataFrame:
    """
    gain list 전체 검증
    """

    results = []

    for i, case in enumerate(gain_list):
        print("-" * 80)
        print(f"Run {i + 1}/{len(gain_list)}")

        try:
            result = run_single_case(
                runner=runner,
                case=case,
                source=source,
            )
            results.append(result)

            print(
                f"score={result['score']:.4f}, "
                f"IAE={result['IAE']:.4f}, "
                f"overshoot={result['overshoot_percent']:.4f}%, "
                f"settling={result['settling_time']}"
            )

        except Exception as e:
            print(f"Failed case: {case}")
            print(f"Error: {e}")

            failed_result = {
                "source": source,
                "target": case["target"],
                "kp": case["kp"],
                "ki": case["ki"],
                "kd": case["kd"],
                "IAE": None,
                "ISE": None,
                "overshoot": None,
                "overshoot_percent": None,
                "mean_pwm": None,
                "total_pwm": None,
                "rise_time": None,
                "settling_time": None,
                "score": None,
            }

            results.append(failed_result)

    return pd.DataFrame(results)


def build_comparison_table(validation_df: pd.DataFrame) -> pd.DataFrame:
    """
    target별 surrogate 추천 gain과 기존 sweep best 비교
    """

    surrogate_df = validation_df[validation_df["source"] == "surrogate"].copy()
    sweep_df = validation_df[validation_df["source"] == "sweep_best"].copy()

    merged_df = pd.merge(
        surrogate_df,
        sweep_df,
        on="target",
        suffixes=("_surrogate", "_sweep"),
    )

    comparison_rows = []

    for _, row in merged_df.iterrows():
        score_surrogate = row["score_surrogate"]
        score_sweep = row["score_sweep"]

        if pd.isna(score_surrogate) or pd.isna(score_sweep):
            winner = "invalid"
            score_improvement = None
        elif score_surrogate < score_sweep:
            winner = "surrogate"
            score_improvement = (score_sweep - score_surrogate) / score_sweep * 100.0
        else:
            winner = "sweep_best"
            score_improvement = (score_sweep - score_surrogate) / score_sweep * 100.0

        comparison_rows.append(
            {
                "target": row["target"],

                "surrogate_kp": row["kp_surrogate"],
                "surrogate_ki": row["ki_surrogate"],
                "surrogate_kd": row["kd_surrogate"],
                "surrogate_score": score_surrogate,
                "surrogate_IAE": row["IAE_surrogate"],
                "surrogate_overshoot_percent": row["overshoot_percent_surrogate"],
                "surrogate_settling_time": row["settling_time_surrogate"],

                "sweep_kp": row["kp_sweep"],
                "sweep_ki": row["ki_sweep"],
                "sweep_kd": row["kd_sweep"],
                "sweep_score": score_sweep,
                "sweep_IAE": row["IAE_sweep"],
                "sweep_overshoot_percent": row["overshoot_percent_sweep"],
                "sweep_settling_time": row["settling_time_sweep"],

                "winner": winner,
                "score_improvement_percent": score_improvement,
            }
        )

    return pd.DataFrame(comparison_rows)


def print_final_gain_db(comparison_df: pd.DataFrame):
    """
    비교 결과 기준 최종 gain DB 출력
    """

    print("\n" + "=" * 80)
    print("Final PID_GAIN_DB candidate")
    print("=" * 80)

    print("PID_GAIN_DB = {")

    for _, row in comparison_df.iterrows():
        target = float(row["target"])

        if row["winner"] == "surrogate":
            kp = row["surrogate_kp"]
            ki = row["surrogate_ki"]
            kd = row["surrogate_kd"]
        else:
            kp = row["sweep_kp"]
            ki = row["sweep_ki"]
            kd = row["sweep_kd"]

        print(
            f"    {target:.1f}: "
            f'{{"kp": {kp:.3f}, "ki": {ki:.3f}, "kd": {kd:.3f}}},'
        )

    print("}")


def save_results(validation_df: pd.DataFrame, comparison_df: pd.DataFrame):
    """
    결과 저장
    """

    VALIDATION_RESULT_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    validation_path = VALIDATION_RESULT_DIR / f"surrogate_validation_raw_{timestamp}.csv"
    comparison_path = VALIDATION_RESULT_DIR / f"surrogate_vs_sweep_comparison_{timestamp}.csv"

    validation_df.to_csv(validation_path, index=False, encoding="utf-8-sig")
    comparison_df.to_csv(comparison_path, index=False, encoding="utf-8-sig")

    print(f"Saved: {validation_path}")
    print(f"Saved: {comparison_path}")


def main():
    print("=" * 80)
    print("Surrogate gain validation with Simulink")
    print("=" * 80)

    runner = SimulinkRunner(
        model_name=SIMULINK_MODEL_NAME,
        mat_file_name=SIMULINK_MAT_FILE,
    )

    try:
        surrogate_df = validate_gain_list(
            runner=runner,
            gain_list=SURROGATE_GAIN_LIST,
            source="surrogate",
        )

        sweep_df = validate_gain_list(
            runner=runner,
            gain_list=SWEEP_BEST_GAIN_LIST,
            source="sweep_best",
        )

        validation_df = pd.concat(
            [surrogate_df, sweep_df],
            axis=0,
        ).reset_index(drop=True)

        comparison_df = build_comparison_table(validation_df)

        print("\n" + "=" * 80)
        print("Comparison result")
        print("=" * 80)
        print(comparison_df)

        print_final_gain_db(comparison_df)

        save_results(validation_df, comparison_df)

    finally:
        runner.stop()
        print("MATLAB Engine stopped.")


if __name__ == "__main__":
    main()