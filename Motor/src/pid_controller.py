import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import (
    DT,
    PWM_MIN,
    PWM_MAX,
    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,
)


class PIDController:
    def __init__(
        self,
        kp: float = DEFAULT_KP,
        ki: float = DEFAULT_KI,
        kd: float = DEFAULT_KD,
        dt: float = DT,
        output_min: float = PWM_MIN,
        output_max: float = PWM_MAX,
    ):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.dt = dt

        self.output_min = output_min
        self.output_max = output_max

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_output = 0.0

    def set_gains(self, kp: float, ki: float, kd: float):
        self.kp = kp
        self.ki = ki
        self.kd = kd

    def compute(self, target: float, current: float) -> float:
        """
        PID 출력 계산.
        Anti-windup을 포함하여 PWM 포화 상태에서 integral term이 과도하게 누적되는 것을 방지한다.
        """

        error = target - current

        # Derivative term
        derivative = (error - self.prev_error) / self.dt

        # 우선 integral이 업데이트된다고 가정
        tentative_integral = self.integral + error * self.dt

        # PID raw output 계산
        raw_output = (
            self.kp * error
            + self.ki * tentative_integral
            + self.kd * derivative
        )

        # Output saturation
        clipped_output = max(self.output_min, min(self.output_max, raw_output))

        # Anti-windup:
        # 출력이 포화되지 않았으면 integral 업데이트
        # 출력이 상한 포화인데 error가 양수이면 integral 누적 금지
        # 출력이 하한 포화인데 error가 음수이면 integral 누적 금지
        saturated_high = raw_output > self.output_max
        saturated_low = raw_output < self.output_min

        if not saturated_high and not saturated_low:
            self.integral = tentative_integral

        elif saturated_high and error < 0:
            # 상한 포화 상태지만 error가 줄이는 방향이면 integral 허용
            self.integral = tentative_integral

        elif saturated_low and error > 0:
            # 하한 포화 상태지만 error가 줄이는 방향이면 integral 허용
            self.integral = tentative_integral

        # 최종 output
        output = clipped_output

        # Update states
        self.prev_error = error
        self.prev_output = output

        return output

    def get_state(self) -> dict:
        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "integral": self.integral,
            "prev_error": self.prev_error,
            "prev_output": self.prev_output,
        }


if __name__ == "__main__":
    pid = PIDController(kp=0.5, ki=0.01, kd=0.0)

    target = 1000.0
    current = 0.0

    for step in range(10):
        pwm = pid.compute(target, current)
        print(f"step={step}, target={target}, current={current:.2f}, pwm={pwm:.2f}")

        # 임시 모터 응답 가정
        current += pwm * 0.1