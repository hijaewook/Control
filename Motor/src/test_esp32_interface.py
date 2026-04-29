import sys
from pathlib import Path
import time

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from motor_interface import ESP32MotorInterface
from config import (
    ESP32_PORT,
    ESP32_BAUDRATE,
    ESP32_TIMEOUT,
    REAL_PWM_MIN,
    REAL_PWM_MAX,
)


def main():
    motor = ESP32MotorInterface(
        port=ESP32_PORT,
        baudrate=ESP32_BAUDRATE,
        timeout=ESP32_TIMEOUT,
        pwm_min=REAL_PWM_MIN,
        pwm_max=REAL_PWM_MAX,
    )

    try:
        print("=" * 80)
        print("ESP32 interface test")
        print("=" * 80)

        print("PING:", motor.ping())

        state = motor.get_state()
        print("Initial state:", state)

        test_pwms = [0, 50, 80, 100, 120, 0]

        for pwm in test_pwms:
            print("-" * 80)
            print(f"SET_PWM {pwm}")

            state = motor.step(pwm)
            print("State:", state)

            time.sleep(1.0)

            state = motor.get_state()
            print("State after 1s:", state)

        print("-" * 80)
        print("STOP")
        motor.stop()

        state = motor.get_state()
        print("Final state:", state)

    finally:
        motor.close()


if __name__ == "__main__":
    main()