import sys
from pathlib import Path
from datetime import datetime
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from motor_interface import ESP32MotorInterface
from pid_controller import PIDController
from config import (
    ESP32_PORT,
    ESP32_BAUDRATE,
    ESP32_TIMEOUT,
    RESULTS_DIR,
    FIGURE_DIR,
)


# ============================================================
# Test settings
# ============================================================

RESULT_DIR = RESULTS_DIR / "esp32_pid_test"

DT = 0.1          # 실물 serial 제어는 20 Hz부터 시작
SIM_TIME = 12.0
N_STEPS = int(SIM_TIME / DT)

TARGET_RPM = 50.0

# 초기 실물 테스트용 보수적 PID gain
TEST_KP = 1.2
TEST_KI = 0.55
TEST_KD = 0.0

# 실물 안전 PWM 제한
REAL_PWM_MIN = 0.0
REAL_PWM_MAX = 140.0

# PWM 변화율 제한: 급격한 명령 변화 방지
PWM_RATE_LIMIT = 20.0

# target 도달 판정
SETTLING_TOLERANCE_RATIO = 0.05


# ============================================================
# Utility
# ============================================================

def apply_pwm_rate_limit(pwm_cmd, prev_pwm, rate_limit):
    """
    PWM 변화량 제한.
    """
    delta = pwm_cmd - prev_pwm
    delta = np.clip(delta, -rate_limit, rate_limit)
    return prev_pwm + delta


def calculate_metrics(df: pd.DataFrame) -> dict:
    time_arr = df["time"].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    rpm = df["rpm"].to_numpy(dtype=float)
    error = df["error"].to_numpy(dtype=float)
    pwm = df["pwm"].to_numpy(dtype=float)

    if len(time_arr) <= 1:
        dt_arr = np.zeros_like(time_arr)
    else:
        dt_arr = np.diff(time_arr, prepend=time_arr[0])
        dt_arr[0] = 0.0

    abs_error = np.abs(error)

    iae = float(np.sum(abs_error * dt_arr))
    mean_abs_error = float(np.mean(abs_error))
    final_error = float(abs_error[-1])

    target_final = float(target[-1])
    max_rpm = float(np.max(rpm))

    overshoot = max(0.0, max_rpm - target_final)
    overshoot_percent = overshoot / max(abs(target_final), 1e-6) * 100.0

    tolerance = SETTLING_TOLERANCE_RATIO * abs(target_final)
    settling_time = np.nan

    for i in range(len(time_arr)):
        if np.all(np.abs(target[i:] - rpm[i:]) <= tolerance):
            settling_time = float(time_arr[i])
            break

    return {
        "target_rpm": target_final,
        "kp": TEST_KP,
        "ki": TEST_KI,
        "kd": TEST_KD,
        "IAE": iae,
        "mean_abs_error": mean_abs_error,
        "final_error": final_error,
        "overshoot_percent": overshoot_percent,
        "settling_time": settling_time,
        "mean_pwm": float(np.mean(np.abs(pwm))),
        "max_pwm": float(np.max(pwm)),
        "min_pwm": float(np.min(pwm)),
    }


def plot_result(df: pd.DataFrame, metrics: dict, timestamp: str):
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    # RPM response
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["rpm"], label="measured RPM")
    plt.plot(df["time"], df["target"], linestyle="--", label="target RPM")
    plt.xlabel("Time [s]")
    plt.ylabel("RPM")
    plt.title("ESP32 Motor PID RPM Response")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_pid_response_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()

    # PWM
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["pwm"], label="PWM")
    plt.axhline(REAL_PWM_MAX, linestyle="--", label="PWM limit")
    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("ESP32 Motor PID PWM Command")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"esp32_pid_pwm_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")
    plt.show()


# ============================================================
# Main test
# ============================================================

def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("ESP32 PID closed-loop test")
    print("=" * 80)
    print(f"Port: {ESP32_PORT}")
    print(f"Baudrate: {ESP32_BAUDRATE}")
    print(f"Target RPM: {TARGET_RPM}")
    print(f"PID: Kp={TEST_KP}, Ki={TEST_KI}, Kd={TEST_KD}")
    print(f"PWM limit: {REAL_PWM_MIN} ~ {REAL_PWM_MAX}")
    print("=" * 80)

    motor = ESP32MotorInterface(
        port=ESP32_PORT,
        baudrate=ESP32_BAUDRATE,
        timeout=ESP32_TIMEOUT,
        pwm_min=REAL_PWM_MIN,
        pwm_max=REAL_PWM_MAX,
    )

    pid = PIDController(
        kp=TEST_KP,
        ki=TEST_KI,
        kd=TEST_KD,
        dt=DT,
        output_min=REAL_PWM_MIN,
        output_max=REAL_PWM_MAX,
    )

    rows = []

    prev_error = 0.0
    prev_pwm = 0.0

    try:
        print("PING:", motor.ping())

        # 시작 전 정지 및 상태 확인
        motor.stop()
        time.sleep(0.5)

        state = motor.get_state()
        current_rpm = state.current

        print(f"Initial RPM: {current_rpm:.3f}")

        start_time = time.time()

        for step in range(N_STEPS):
            loop_start = time.time()
            t = time.time() - start_time

            target = TARGET_RPM
            error = target - current_rpm
            error_derivative = (error - prev_error) / DT

            # PID output
            pwm_cmd = pid.compute(target, current_rpm)

            # rate limit + hard limit
            pwm_cmd = apply_pwm_rate_limit(
                pwm_cmd=pwm_cmd,
                prev_pwm=prev_pwm,
                rate_limit=PWM_RATE_LIMIT,
            )
            pwm_cmd = float(np.clip(pwm_cmd, REAL_PWM_MIN, REAL_PWM_MAX))

            # apply to ESP32 and read new state
            state = motor.step(pwm_cmd)
            current_rpm = state.current

            rows.append(
                {
                    "step": step,
                    "time": t,
                    "target": target,
                    "rpm": current_rpm,
                    "error": error,
                    "error_derivative": error_derivative,
                    "pwm": pwm_cmd,
                    "encoder": state.raw.get("encoder", np.nan) if state.raw else np.nan,
                }
            )

            if step % 10 == 0:
                print(
                    f"step={step:04d}, "
                    f"t={t:.2f}, "
                    f"target={target:.1f}, "
                    f"rpm={current_rpm:.2f}, "
                    f"error={error:.2f}, "
                    f"pwm={pwm_cmd:.2f}"
                )

            prev_error = error
            prev_pwm = pwm_cmd

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, DT - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping motor...")

    finally:
        motor.stop()
        time.sleep(0.3)
        motor.close()

    df = pd.DataFrame(rows)
    metrics = calculate_metrics(df)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    log_path = RESULT_DIR / f"esp32_pid_log_{timestamp}.csv"
    metric_path = RESULT_DIR / f"esp32_pid_metrics_{timestamp}.csv"

    df.to_csv(log_path, index=False, encoding="utf-8-sig")
    pd.DataFrame([metrics]).to_csv(metric_path, index=False, encoding="utf-8-sig")

    print("=" * 80)
    print("ESP32 PID closed-loop test finished")
    print(f"Saved log: {log_path}")
    print(f"Saved metrics: {metric_path}")
    print("=" * 80)
    print(metrics)

    plot_result(df, metrics, timestamp)


if __name__ == "__main__":
    main()