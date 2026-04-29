import sys
from pathlib import Path
from typing import Optional

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import (
    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,
    PWM_MIN,
    PWM_MAX,
    PID_GAIN_DB,
    USE_GAIN_DB,
    GAIN_DB_MODE,
    GAIN_UPDATE_INTERVAL,
)

from safety_guard import SafetyGuard


class GainScheduler:
    def __init__(
        self,
        target: Optional[float] = None,
        initial_kp: float = DEFAULT_KP,
        initial_ki: float = DEFAULT_KI,
        initial_kd: float = DEFAULT_KD,
    ):
        """
        GainScheduler

        target이 주어지면 PID_GAIN_DB에서 target에 가장 가까운 gain을 초기값으로 사용.
        target이 없으면 DEFAULT_KP, DEFAULT_KI, DEFAULT_KD를 사용.
        """

        self.guard = SafetyGuard()

        self.update_counter = 0
        self.update_interval = GAIN_UPDATE_INTERVAL
        self.last_update_applied = False

        if target is not None:
            initial_kp, initial_ki, initial_kd = self.get_initial_gains_from_target(target)

        self.kp = initial_kp
        self.ki = initial_ki
        self.kd = initial_kd

    def get_initial_gains_from_target(self, target: float):
        """
        Target 기반 초기 PID gain 선택 함수.

        config.py에서 제어:
        - USE_GAIN_DB
        - GAIN_DB_MODE = "nearest" or "linear"
        - PID_GAIN_DB
        """

        if not USE_GAIN_DB:
            return DEFAULT_KP, DEFAULT_KI, DEFAULT_KD

        if not PID_GAIN_DB:
            return DEFAULT_KP, DEFAULT_KI, DEFAULT_KD

        target = float(target)

        sorted_targets = sorted(float(k) for k in PID_GAIN_DB.keys())

        # --------------------------------------------------------
        # 1. nearest mode
        # --------------------------------------------------------
        if GAIN_DB_MODE == "nearest":
            nearest_target = min(
                sorted_targets,
                key=lambda x: abs(x - target)
            )

            gains = PID_GAIN_DB[nearest_target]
            return gains["kp"], gains["ki"], gains["kd"]

        # --------------------------------------------------------
        # 2. linear interpolation mode
        # --------------------------------------------------------
        elif GAIN_DB_MODE == "linear":

            # target이 DB 범위보다 작으면 가장 작은 target gain 사용
            if target <= sorted_targets[0]:
                gains = PID_GAIN_DB[sorted_targets[0]]
                return gains["kp"], gains["ki"], gains["kd"]

            # target이 DB 범위보다 크면 가장 큰 target gain 사용
            if target >= sorted_targets[-1]:
                gains = PID_GAIN_DB[sorted_targets[-1]]
                return gains["kp"], gains["ki"], gains["kd"]

            # target 사이 구간 찾기
            lower_target = sorted_targets[0]
            upper_target = sorted_targets[-1]

            for i in range(len(sorted_targets) - 1):
                t_low = sorted_targets[i]
                t_high = sorted_targets[i + 1]

                if t_low <= target <= t_high:
                    lower_target = t_low
                    upper_target = t_high
                    break

            lower_gain = PID_GAIN_DB[lower_target]
            upper_gain = PID_GAIN_DB[upper_target]

            ratio = (target - lower_target) / (upper_target - lower_target)

            kp = lower_gain["kp"] + ratio * (upper_gain["kp"] - lower_gain["kp"])
            ki = lower_gain["ki"] + ratio * (upper_gain["ki"] - lower_gain["ki"])
            kd = lower_gain["kd"] + ratio * (upper_gain["kd"] - lower_gain["kd"])

            return kp, ki, kd

        # --------------------------------------------------------
        # 3. fallback
        # --------------------------------------------------------
        else:
            print(f"Warning: Unknown GAIN_DB_MODE = {GAIN_DB_MODE}. Use default gains.")
            return DEFAULT_KP, DEFAULT_KI, DEFAULT_KD

    def rule_based_gain_update(
        self,
        target: float,
        error: float,
        error_derivative: float,
        pwm: float,
    ):
        """
        Relative-error 기반 saturation-aware gain update 함수.

        목적:
        - target scale에 독립적으로 gain scheduling 수행
        - 현재는 메타모델 학습 전 baseline rule 역할
        - 향후 meta-model 예측 gain으로 대체 가능
        """

        new_kp = self.kp
        new_ki = self.ki
        new_kd = self.kd

        target_abs = max(abs(target), 1e-6)
        error_ratio = abs(error) / target_abs

        pwm_range = PWM_MAX - PWM_MIN

        high_saturation = pwm > PWM_MAX - 0.1 * pwm_range
        low_saturation = pwm < PWM_MIN + 0.02 * pwm_range

        # 제어 안전 관점에서는 high saturation만 saturation으로 사용
        pwm_saturated = high_saturation

        # error derivative도 target scale 기준으로 정규화
        normalized_error_derivative = abs(error_derivative) / target_abs

        # 1. 오차가 큰 구간: 목표 추종성을 높이기 위해 Kp 증가
        if error_ratio > 0.5 and not pwm_saturated:
            new_kp += 0.05

        # 2. 중간 오차 구간: Kp 완만 증가
        if 0.2 < error_ratio <= 0.5 and not pwm_saturated:
            new_kp += 0.02

        # 3. 목표 근처에서 PWM이 높은 경우: overshoot 방지를 위해 Kp 감소
        if error_ratio < 0.1 and high_saturation:
            new_kp = max(0.1, new_kp - 0.02)

        # 4. steady-state error 구간: 변화가 느리고 오차가 남아 있으면 Ki 증가
        if 0.05 < error_ratio < 0.3 and normalized_error_derivative < 0.5 and not high_saturation:
            new_ki += 0.01

        # 5. 목표에 충분히 가까운데 PWM이 높은 경우: Ki 감소
        if error_ratio < 0.03 and high_saturation:
            new_ki = max(0.0, new_ki - 0.005)

        # 6. 목표 근처에서 변화율이 크면 진동 가능성 → Kd 증가
        if error_ratio < 0.15 and normalized_error_derivative > 5.0:
            new_kd += 0.02

        return new_kp, new_ki, new_kd

    def update(
        self,
        target: float,
        error: float,
        error_derivative: float,
        pwm: float,
    ):
        """
        현재 상태를 기반으로 gain 업데이트.
        """

        self.last_update_applied = False

        # 위험 상태면 default gain으로 fallback
        if self.guard.check_fallback(error, error_derivative):
            self.kp, self.ki, self.kd = self.guard.get_fallback_gains()
            self.last_update_applied = True
            return self.kp, self.ki, self.kd

        self.update_counter += 1

        # gain은 일정 step마다 한 번만 업데이트
        if self.update_counter % self.update_interval != 0:
            return self.kp, self.ki, self.kd

        requested_kp, requested_ki, requested_kd = self.rule_based_gain_update(
            target=target,
            error=error,
            error_derivative=error_derivative,
            pwm=pwm,
        )

        safe_kp, safe_ki, safe_kd = self.guard.limit_gain_update(
            prev_kp=self.kp,
            prev_ki=self.ki,
            prev_kd=self.kd,
            new_kp=requested_kp,
            new_ki=requested_ki,
            new_kd=requested_kd,
        )

        gain_changed = (
            abs(safe_kp - self.kp) > 1e-12
            or abs(safe_ki - self.ki) > 1e-12
            or abs(safe_kd - self.kd) > 1e-12
        )

        self.kp = safe_kp
        self.ki = safe_ki
        self.kd = safe_kd

        self.last_update_applied = gain_changed

        return self.kp, self.ki, self.kd

    def get_gains(self):
        return self.kp, self.ki, self.kd

    def get_scheduler_state(self) -> dict:
        """
        Gain scheduler 내부 상태 반환
        """

        return {
            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,
            "update_counter": self.update_counter,
            "update_interval": self.update_interval,
            "gain_update_flag": self.last_update_applied,
        }


if __name__ == "__main__":
    scheduler = GainScheduler(target=100)

    test_states = [
        {"target": 100, "error": 100, "error_derivative": 50, "pwm": 50},
        {"target": 100, "error": 50, "error_derivative": 30, "pwm": 40},
        {"target": 100, "error": 20, "error_derivative": 5, "pwm": 25},
        {"target": 100, "error": 5, "error_derivative": 1, "pwm": 20},
        {"target": 200, "error": 100, "error_derivative": 300, "pwm": 200},
    ]

    for i, state in enumerate(test_states):
        kp, ki, kd = scheduler.update(
            target=state["target"],
            error=state["error"],
            error_derivative=state["error_derivative"],
            pwm=state["pwm"],
        )

        print(
            f"step={i}, "
            f"target={state['target']}, "
            f"error={state['error']}, "
            f"error_derivative={state['error_derivative']}, "
            f"pwm={state['pwm']}, "
            f"kp={kp:.3f}, ki={ki:.3f}, kd={kd:.3f}"
        )