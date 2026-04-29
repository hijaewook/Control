import sys
from pathlib import Path
import csv
from datetime import datetime
import time
import argparse

# ============================================================
# Path setting
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"
sys.path.append(str(SRC_DIR))

from pid_controller import PIDController
from gain_scheduler import GainScheduler
from motor_env import SimpleMotorEnv
from simulink_runner import SimulinkRunner
from config import (
    DT,
    LOG_DIR,
    RUN_MODE,
    ENV_TYPE,
    TEST_TARGET,
    TEST_STEPS,
    PWM_MIN,
    PWM_MAX,
    SIMULINK_MODEL_NAME,
    SIMULINK_STOP_TIME,
    SIMULINK_MAT_FILE,
)


def create_log_file(mode: str, env_type: str):
    """
    실시간 기록용 CSV 파일 생성
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{mode}_{env_type}_{timestamp}.csv"

    fieldnames = [
        "mode",
        "env_type",
        "step",
        "time",
        "target",
        "current",
        "error",
        "error_derivative",
        "pwm",
        "prev_pwm",
        "pwm_saturated",
        "high_saturation",
        "low_saturation",
        "kp",
        "ki",
        "kd",
        "integral",
        "gain_update_flag",
    ]

    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    return log_path, fieldnames


def append_log_row(log_path, fieldnames, row):
    """
    CSV 파일에 한 줄씩 실시간 저장
    """

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


def run_simulink_pid_test(stop_time_override: float = None):
    """
    MATLAB Engine을 이용해 Simulink PID 모델을 실행하고 결과를 CSV로 저장한다.
    """

    stop_time = SIMULINK_STOP_TIME if stop_time_override is None else stop_time_override

    print("simulink_pid test start")
    print(f"Model: {SIMULINK_MODEL_NAME}")
    print(f"Stop time: {stop_time}")
    print("-" * 80)

    runner = SimulinkRunner(
        model_name=SIMULINK_MODEL_NAME,
        mat_file_name=SIMULINK_MAT_FILE,
    )

    try:
        runner.run_simulation(stop_time=stop_time)
        df = runner.get_simulink_dataframe()
        save_path = runner.save_simulink_log(df)

        print("-" * 80)
        print("simulink_pid test finished")
        print(f"Log saved to: {save_path}")

    finally:
        runner.stop()


def run_pid_test(
    mode: str = RUN_MODE,
    target_override: float = None,
    steps_override: int = None,
    stop_time_override: float = None,
):
    """
    fixed PID, adaptive PID, simulink PID 실행 함수.
    """

    if mode not in ["fixed_pid", "adaptive_pid", "simulink_pid"]:
        raise ValueError("mode must be 'fixed_pid', 'adaptive_pid', or 'simulink_pid'")

    # ========================================================
    # Simulink PID mode
    # ========================================================

    if mode == "simulink_pid":
        run_simulink_pid_test(stop_time_override=stop_time_override)
        return

    # ========================================================
    # Python PID mode
    # ========================================================

    target = TEST_TARGET if target_override is None else target_override
    test_steps = TEST_STEPS if steps_override is None else steps_override

    env = SimpleMotorEnv(initial_value=0.0)
    current = env.get_state()

    pid = PIDController()
    scheduler = GainScheduler(target=target)

    # Adaptive PID는 gain DB 기반 초기 gain을 PID에 먼저 반영
    if mode == "adaptive_pid":
        kp, ki, kd = scheduler.get_gains()
        pid.set_gains(kp, ki, kd)

    prev_error = target - current
    prev_pwm = 0.0

    log_path, fieldnames = create_log_file(mode=mode, env_type=ENV_TYPE)

    print(f"{mode} realtime test start")
    print(f"Environment: {ENV_TYPE}")
    print(f"Target: {target}")
    print(f"Steps: {test_steps}")
    print(f"Log file: {log_path}")

    if mode == "adaptive_pid":
        init_kp, init_ki, init_kd = scheduler.get_gains()
        print(
            f"Initial adaptive gains from DB: "
            f"Kp={init_kp:.3f}, Ki={init_ki:.3f}, Kd={init_kd:.3f}"
        )

    print("-" * 80)

    for step in range(test_steps):
        error = target - current
        error_derivative = (error - prev_error) / DT

        # ----------------------------------------------------
        # 현재 PID gain으로 PWM 계산
        # ----------------------------------------------------

        pwm = pid.compute(target, current)

        pwm_range = PWM_MAX - PWM_MIN

        high_saturation = pwm > PWM_MAX - 0.1 * pwm_range
        low_saturation = pwm < PWM_MIN + 0.02 * pwm_range

        # 제어 안전 판단에서는 high saturation만 saturation으로 사용
        pwm_saturated = high_saturation

        # ----------------------------------------------------
        # Adaptive gain update
        # ----------------------------------------------------

        if mode == "adaptive_pid":
            kp, ki, kd = scheduler.update(
                target=target,
                error=error,
                error_derivative=error_derivative,
                pwm=pwm,
            )

            pid.set_gains(kp, ki, kd)

            scheduler_state = scheduler.get_scheduler_state()
            gain_update_flag = scheduler_state["gain_update_flag"]

        else:
            pid_state_now = pid.get_state()
            kp = pid_state_now["kp"]
            ki = pid_state_now["ki"]
            kd = pid_state_now["kd"]
            gain_update_flag = False

        # ----------------------------------------------------
        # PID state 저장
        # ----------------------------------------------------

        pid_state = pid.get_state()

        # ----------------------------------------------------
        # Motor environment update
        # ----------------------------------------------------

        current = env.step(pwm)

        # ----------------------------------------------------
        # Logging
        # ----------------------------------------------------

        log_row = {
            "mode": mode,
            "env_type": ENV_TYPE,
            "step": step,
            "time": step * DT,
            "target": target,
            "current": current,
            "error": error,
            "error_derivative": error_derivative,
            "pwm": pwm,
            "prev_pwm": prev_pwm,
            "pwm_saturated": pwm_saturated,
            "high_saturation": high_saturation,
            "low_saturation": low_saturation,
            "kp": kp,
            "ki": ki,
            "kd": kd,
            "integral": pid_state["integral"],
            "gain_update_flag": gain_update_flag,
        }

        append_log_row(log_path, fieldnames, log_row)

        print(
            f"step={step:03d}, "
            f"mode={mode}, "
            f"target={target:.1f}, "
            f"current={current:.2f}, "
            f"error={error:.2f}, "
            f"d_error={error_derivative:.2f}, "
            f"pwm={pwm:.2f}, "
            f"kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f}, "
            f"update={gain_update_flag}"
        )

        prev_pwm = pwm
        prev_error = error

        time.sleep(DT)

    print("-" * 80)
    print(f"{mode} realtime test finished")
    print(f"Log saved to: {log_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="Motor PID control test runner")

    parser.add_argument(
        "--mode",
        type=str,
        default=RUN_MODE,
        choices=["fixed_pid", "adaptive_pid", "simulink_pid"],
        help="Run mode: fixed_pid, adaptive_pid, or simulink_pid",
    )

    parser.add_argument(
        "--target",
        type=float,
        default=None,
        help="Target value for Python motor environment",
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help="Number of simulation steps for Python motor environment",
    )

    parser.add_argument(
        "--stop-time",
        type=float,
        default=None,
        help="Stop time for Simulink simulation",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_pid_test(
        mode=args.mode,
        target_override=args.target,
        steps_override=args.steps,
        stop_time_override=args.stop_time,
    )