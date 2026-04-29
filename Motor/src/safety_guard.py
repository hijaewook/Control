import sys
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import (
    KP_MIN,
    KP_MAX,
    KI_MIN,
    KI_MAX,
    KD_MIN,
    KD_MAX,
    MAX_DELTA_KP,
    MAX_DELTA_KI,
    MAX_DELTA_KD,
    PWM_MIN,
    PWM_MAX,
    MAX_ERROR,
    MAX_ERROR_DERIVATIVE,
    FALLBACK_ENABLED,
    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,
)


def clamp(value: float, min_value: float, max_value: float) -> float:
    """
    값을 최소/최대 범위 안으로 제한
    """
    return max(min_value, min(max_value, value))


class SafetyGuard:
    def __init__(self):
        self.fallback_enabled = FALLBACK_ENABLED

    def limit_pwm(self, pwm: float) -> float:
        """
        PWM 출력 제한
        """
        return clamp(pwm, PWM_MIN, PWM_MAX)

    def limit_gains(self, kp: float, ki: float, kd: float):
        """
        PID gain 범위 제한
        """
        kp = clamp(kp, KP_MIN, KP_MAX)
        ki = clamp(ki, KI_MIN, KI_MAX)
        kd = clamp(kd, KD_MIN, KD_MAX)

        return kp, ki, kd

    def limit_gain_update(
        self,
        prev_kp: float,
        prev_ki: float,
        prev_kd: float,
        new_kp: float,
        new_ki: float,
        new_kd: float,
    ):
        """
        한 번에 변할 수 있는 gain 변화량 제한
        """
        delta_kp = clamp(new_kp - prev_kp, -MAX_DELTA_KP, MAX_DELTA_KP)
        delta_ki = clamp(new_ki - prev_ki, -MAX_DELTA_KI, MAX_DELTA_KI)
        delta_kd = clamp(new_kd - prev_kd, -MAX_DELTA_KD, MAX_DELTA_KD)

        limited_kp = prev_kp + delta_kp
        limited_ki = prev_ki + delta_ki
        limited_kd = prev_kd + delta_kd

        return self.limit_gains(limited_kp, limited_ki, limited_kd)

    def check_fallback(
        self,
        error: float,
        error_derivative: float,
    ) -> bool:
        """
        시스템이 위험 상태인지 판단
        True이면 fallback gain 사용
        """
        if not self.fallback_enabled:
            return False

        if abs(error) > MAX_ERROR:
            return True

        if abs(error_derivative) > MAX_ERROR_DERIVATIVE:
            return True

        return False

    def get_fallback_gains(self):
        """
        기본 PID gain 반환
        """
        return DEFAULT_KP, DEFAULT_KI, DEFAULT_KD


if __name__ == "__main__":
    guard = SafetyGuard()

    prev_gains = (0.5, 0.0, 0.0)
    requested_gains = (3.0, 2.0, 1.0)

    safe_gains = guard.limit_gain_update(
        prev_kp=prev_gains[0],
        prev_ki=prev_gains[1],
        prev_kd=prev_gains[2],
        new_kp=requested_gains[0],
        new_ki=requested_gains[1],
        new_kd=requested_gains[2],
    )

    print("Previous gains:", prev_gains)
    print("Requested gains:", requested_gains)
    print("Safe gains:", safe_gains)

    pwm = 300
    safe_pwm = guard.limit_pwm(pwm)
    print("Requested PWM:", pwm)
    print("Safe PWM:", safe_pwm)

    fallback = guard.check_fallback(
        error=4000,
        error_derivative=100,
    )
    print("Fallback required:", fallback)