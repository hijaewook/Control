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

    # disturbance settings
    USE_DISTURBANCE,
    DISTURBANCE_MODE,
    DISTURBANCE_START_TIME,
    DISTURBANCE_END_TIME,
    DISTURBANCE_MAGNITUDE,
    DISTURBANCE_FREQ,
    EXPERIMENT_TAG,
    USE_SATURATION_AWARE_GAIN,
)


# ============================================================
# Logging
# ============================================================

def create_log_file(mode: str, env_type: str):
    """
    실시간 기록용 CSV 파일 생성.
    """

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"{mode}_{env_type}_{timestamp}.csv"

    fieldnames = [
        "mode",
        "env_type",
        "experiment_tag",
        "use_saturation_aware_gain",
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

        # saturation-aware scheduler state
        "base_kp",
        "base_ki",
        "base_kd",
        "kp_scale",
        "ki_scale",
        "kd_scale",
        "last_update_reason",
        "saturation_counter",
        "saturation_active",

        # disturbance log
        "use_disturbance",
        "disturbance_mode",
        "disturbance",
    ]

    with open(log_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

    return log_path, fieldnames


def append_log_row(log_path, fieldnames, row):
    """
    CSV 파일에 한 줄씩 실시간 저장.
    """

    with open(log_path, mode="a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writerow(row)


# ============================================================
# Simulink PID
# ============================================================

def run_simulink_pid_test(
    target_override: float = None,
    stop_time_override: float = None,
):
    """
    MATLAB Engine을 이용해 Simulink PID 모델을 실행하고 결과를 CSV로 저장한다.
    """

    target = TEST_TARGET if target_override is None else target_override
    stop_time = SIMULINK_STOP_TIME if stop_time_override is None else stop_time_override

    print("simulink_pid test start")
    print(f"Model: {SIMULINK_MODEL_NAME}")
    print(f"Target: {target}")
    print(f"Stop time: {stop_time}")
    print("-" * 80)

    runner = SimulinkRunner(
        model_name=SIMULINK_MODEL_NAME,
        mat_file_name=SIMULINK_MAT_FILE,
    )

    try:
        runner.run_simulation(
            target=target,
            stop_time=stop_time,
            save_log=True,
        )

        print("-" * 80)
        print("simulink_pid test finished")

    finally:
        runner.stop()


# ============================================================
# Python motor environment
# ============================================================

def create_environment():
    """
    Python SimpleMotorEnv 생성 함수.
    disturbance 설정은 config.py에서 관리한다.
    """

    env = SimpleMotorEnv(
        initial_value=0.0,
        dt=DT,
        pwm_min=PWM_MIN,
        pwm_max=PWM_MAX,
        use_disturbance=USE_DISTURBANCE,
        disturbance_mode=DISTURBANCE_MODE,
        disturbance_start_time=DISTURBANCE_START_TIME,
        disturbance_end_time=DISTURBANCE_END_TIME,
        disturbance_magnitude=DISTURBANCE_MAGNITUDE,
        disturbance_freq=DISTURBANCE_FREQ,
    )

    return env


# ============================================================
# PID test
# ============================================================

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
        raise ValueError(
            "mode must be 'fixed_pid', 'adaptive_pid', or 'simulink_pid'"
        )

    # ========================================================
    # Simulink PID mode
    # ========================================================

    if mode == "simulink_pid":
        run_simulink_pid_test(
            target_override=target_override,
            stop_time_override=stop_time_override,
        )
        return

    # ========================================================
    # Python PID mode
    # ========================================================

    target = TEST_TARGET if target_override is None else target_override
    test_steps = TEST_STEPS if steps_override is None else steps_override

    env = create_environment()
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
    print(f"Use disturbance: {USE_DISTURBANCE}")
    print(f"Disturbance mode: {DISTURBANCE_MODE}")
    print(f"Experiment tag: {EXPERIMENT_TAG}")
    print(f"Use saturation-aware gain: {USE_SATURATION_AWARE_GAIN}")

    if USE_DISTURBANCE:
        print(f"Disturbance start time: {DISTURBANCE_START_TIME}")
        print(f"Disturbance end time: {DISTURBANCE_END_TIME}")
        print(f"Disturbance magnitude: {DISTURBANCE_MAGNITUDE}")
        print(f"Disturbance frequency: {DISTURBANCE_FREQ}")

    if mode == "adaptive_pid":
        init_kp, init_ki, init_kd = scheduler.get_gains()
        scheduler_state = scheduler.get_scheduler_state()

        print(
            f"Initial adaptive gains from DB: "
            f"Kp={init_kp:.3f}, Ki={init_ki:.3f}, Kd={init_kd:.3f}"
        )
        print(
            f"Initial gain scales: "
            f"Kp scale={scheduler_state['kp_scale']:.3f}, "
            f"Ki scale={scheduler_state['ki_scale']:.3f}, "
            f"Kd scale={scheduler_state['kd_scale']:.3f}"
        )

    print("-" * 80)

    for step in range(test_steps):
        current_time = step * DT

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

            scheduler_state = {
                "base_kp": kp,
                "base_ki": ki,
                "base_kd": kd,
                "kp_scale": 1.0,
                "ki_scale": 1.0,
                "kd_scale": 1.0,
                "last_update_reason": "fixed_pid",
                "saturation_counter": 0,
                "saturation_active": False,
            }

        # ----------------------------------------------------
        # PID state 저장
        # ----------------------------------------------------

        pid_state = pid.get_state()

        # ----------------------------------------------------
        # Disturbance value before environment update
        # ----------------------------------------------------

        if hasattr(env, "get_disturbance"):
            disturbance = env.get_disturbance()
        else:
            disturbance = 0.0

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
            "experiment_tag": EXPERIMENT_TAG,
            "use_saturation_aware_gain": USE_SATURATION_AWARE_GAIN,
            "step": step,
            "time": current_time,

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

            # saturation-aware scheduler state
            "base_kp": scheduler_state["base_kp"],
            "base_ki": scheduler_state["base_ki"],
            "base_kd": scheduler_state["base_kd"],
            "kp_scale": scheduler_state["kp_scale"],
            "ki_scale": scheduler_state["ki_scale"],
            "kd_scale": scheduler_state["kd_scale"],
            "last_update_reason": scheduler_state["last_update_reason"],
            "saturation_counter": scheduler_state["saturation_counter"],
            "saturation_active": scheduler_state["saturation_active"],

            # disturbance log
            "use_disturbance": USE_DISTURBANCE,
            "disturbance_mode": DISTURBANCE_MODE,
            "disturbance": disturbance,
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
            f"disturbance={disturbance:.2f}, "
            f"kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f}, "
            f"kp_scale={scheduler_state['kp_scale']:.3f}, "
            f"ki_scale={scheduler_state['ki_scale']:.3f}, "
            f"sat_count={scheduler_state['saturation_counter']}, "
            f"reason={scheduler_state['last_update_reason']}, "
            f"update={gain_update_flag}"
        )

        prev_pwm = pwm
        prev_error = error

        time.sleep(DT)

    print("-" * 80)
    print(f"{mode} realtime test finished")
    print(f"Log saved to: {log_path}")


# ============================================================
# Argument parser
# ============================================================

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
        help="Target value for Python motor environment or Simulink model",
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


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    args = parse_args()

    run_pid_test(
        mode=args.mode,
        target_override=args.target,
        steps_override=args.steps,
        stop_time_override=args.stop_time,
    )