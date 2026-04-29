import numpy as np

from config import (
    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,
    KP_MIN,
    KP_MAX,
    KI_MIN,
    KI_MAX,
    KD_MIN,
    KD_MAX,
    GAIN_UPDATE_INTERVAL,
    PWM_MAX,
    PID_GAIN_DB,
    GAIN_DB_MODE,
)

# Optional saturation-aware settings
try:
    from config import (
        USE_SATURATION_AWARE_GAIN,
        PWM_SOFT_LIMIT,
        SATURATION_CONSECUTIVE_STEPS,
        SATURATION_ERROR_THRESHOLD_RATIO,
        SATURATION_KP_DECAY,
        SATURATION_KI_DECAY,
        SATURATION_KD_DECAY,
        SATURATION_MIN_GAIN_SCALE,
        SATURATION_RECOVERY_RATE,
    )
except ImportError:
    USE_SATURATION_AWARE_GAIN = False
    PWM_SOFT_LIMIT = 0.95 * PWM_MAX
    SATURATION_CONSECUTIVE_STEPS = 3
    SATURATION_ERROR_THRESHOLD_RATIO = 0.02
    SATURATION_KP_DECAY = 0.98
    SATURATION_KI_DECAY = 0.95
    SATURATION_KD_DECAY = 1.00
    SATURATION_MIN_GAIN_SCALE = 0.70
    SATURATION_RECOVERY_RATE = 0.002


class GainScheduler:
    """
    Target-based gain scheduler with optional saturation-aware gain scaling.

    Main roles:
    1. Load initial PID gains from PID_GAIN_DB.
    2. Use linear interpolation for unseen targets.
    3. Optionally reduce gain scale when PWM saturation risk is detected.
    4. Slowly recover to the original DB gain when saturation disappears.
    """

    def __init__(self, target=None):
        self.target = target

        self.step_count = 0
        self.gain_update_flag = False
        self.last_update_reason = "init"

        self.saturation_counter = 0
        self.saturation_active = False

        # Base gains from DB
        self.base_kp = DEFAULT_KP
        self.base_ki = DEFAULT_KI
        self.base_kd = DEFAULT_KD

        # Gain scale for saturation-aware adaptation
        self.kp_scale = 1.0
        self.ki_scale = 1.0
        self.kd_scale = 1.0

        # Current gains
        self.kp = DEFAULT_KP
        self.ki = DEFAULT_KI
        self.kd = DEFAULT_KD

        if target is not None:
            self.set_target(target)

    # ========================================================
    # Target-based gain DB
    # ========================================================

    def set_target(self, target):
        """
        Set target and update base gains from PID_GAIN_DB.
        """

        self.target = float(target)

        self.base_kp, self.base_ki, self.base_kd = self._get_gain_from_db(
            self.target
        )

        # New target에서는 gain scale을 원래대로 복구
        self.kp_scale = 1.0
        self.ki_scale = 1.0
        self.kd_scale = 1.0

        self._apply_scaled_gains()

        self.gain_update_flag = True
        self.last_update_reason = "target_gain_db"

    def _get_gain_from_db(self, target):
        """
        PID_GAIN_DB에서 target 기반 gain을 가져온다.
        GAIN_DB_MODE가 linear이면 target 사이를 선형 보간한다.
        """

        if not PID_GAIN_DB:
            return DEFAULT_KP, DEFAULT_KI, DEFAULT_KD

        target = float(target)

        db_targets = sorted([float(k) for k in PID_GAIN_DB.keys()])

        # Exact match
        if target in db_targets:
            gains = PID_GAIN_DB[target]
            return gains["kp"], gains["ki"], gains["kd"]

        # Nearest mode
        if GAIN_DB_MODE == "nearest":
            nearest_target = min(db_targets, key=lambda x: abs(x - target))
            gains = PID_GAIN_DB[nearest_target]
            return gains["kp"], gains["ki"], gains["kd"]

        # Linear interpolation mode
        if GAIN_DB_MODE == "linear":
            # Outside DB range: nearest endpoint
            if target <= db_targets[0]:
                gains = PID_GAIN_DB[db_targets[0]]
                return gains["kp"], gains["ki"], gains["kd"]

            if target >= db_targets[-1]:
                gains = PID_GAIN_DB[db_targets[-1]]
                return gains["kp"], gains["ki"], gains["kd"]

            lower_target = None
            upper_target = None

            for i in range(len(db_targets) - 1):
                t_low = db_targets[i]
                t_high = db_targets[i + 1]

                if t_low <= target <= t_high:
                    lower_target = t_low
                    upper_target = t_high
                    break

            if lower_target is None or upper_target is None:
                nearest_target = min(db_targets, key=lambda x: abs(x - target))
                gains = PID_GAIN_DB[nearest_target]
                return gains["kp"], gains["ki"], gains["kd"]

            ratio = (target - lower_target) / (upper_target - lower_target)

            lower_gain = PID_GAIN_DB[lower_target]
            upper_gain = PID_GAIN_DB[upper_target]

            kp = lower_gain["kp"] + ratio * (upper_gain["kp"] - lower_gain["kp"])
            ki = lower_gain["ki"] + ratio * (upper_gain["ki"] - lower_gain["ki"])
            kd = lower_gain["kd"] + ratio * (upper_gain["kd"] - lower_gain["kd"])

            return kp, ki, kd

        # Fallback
        nearest_target = min(db_targets, key=lambda x: abs(x - target))
        gains = PID_GAIN_DB[nearest_target]
        return gains["kp"], gains["ki"], gains["kd"]

    # ========================================================
    # Saturation-aware gain scaling
    # ========================================================

    def update(self, target, error, error_derivative, pwm):
        """
        Adaptive gain update.

        현재는 rule-based gain search보다 saturation-aware gain scaling에 초점.
        """

        self.step_count += 1
        self.gain_update_flag = False
        self.last_update_reason = "none"

        target = float(target)
        error = float(error)
        pwm = float(pwm)

        # Target이 바뀌면 DB gain 재설정
        if self.target is None or abs(target - self.target) > 1e-9:
            self.set_target(target)

        if USE_SATURATION_AWARE_GAIN:
            self._update_saturation_aware_scale(
                target=target,
                error=error,
                pwm=pwm,
            )

        self._apply_scaled_gains()

        return self.kp, self.ki, self.kd

    def _update_saturation_aware_scale(self, target, error, pwm):
        """
        PWM saturation 위험이 있으면 gain scale을 낮추고,
        saturation이 없으면 천천히 원래 gain으로 복귀.
        """

        error_ratio = abs(error) / max(abs(target), 1e-6)

        high_pwm_risk = pwm >= PWM_SOFT_LIMIT
        meaningful_error = error_ratio >= SATURATION_ERROR_THRESHOLD_RATIO

        if high_pwm_risk and meaningful_error:
            self.saturation_counter += 1
        else:
            self.saturation_counter = 0

        if self.saturation_counter >= SATURATION_CONSECUTIVE_STEPS:
            old_kp_scale = self.kp_scale
            old_ki_scale = self.ki_scale
            old_kd_scale = self.kd_scale

            self.kp_scale = max(
                SATURATION_MIN_GAIN_SCALE,
                self.kp_scale * SATURATION_KP_DECAY,
            )
            self.ki_scale = max(
                SATURATION_MIN_GAIN_SCALE,
                self.ki_scale * SATURATION_KI_DECAY,
            )
            self.kd_scale = max(
                SATURATION_MIN_GAIN_SCALE,
                self.kd_scale * SATURATION_KD_DECAY,
            )

            self.saturation_active = True

            if (
                abs(self.kp_scale - old_kp_scale) > 1e-12
                or abs(self.ki_scale - old_ki_scale) > 1e-12
                or abs(self.kd_scale - old_kd_scale) > 1e-12
            ):
                self.gain_update_flag = True
                self.last_update_reason = "saturation_gain_reduction"

        else:
            # saturation이 없으면 원래 DB gain으로 천천히 복귀
            if self.kp_scale < 1.0 or self.ki_scale < 1.0 or self.kd_scale < 1.0:
                self.kp_scale = min(1.0, self.kp_scale + SATURATION_RECOVERY_RATE)
                self.ki_scale = min(1.0, self.ki_scale + SATURATION_RECOVERY_RATE)
                self.kd_scale = min(1.0, self.kd_scale + SATURATION_RECOVERY_RATE)

                self.gain_update_flag = True
                self.last_update_reason = "saturation_recovery"

            self.saturation_active = False

    def _apply_scaled_gains(self):
        """
        base gain과 scale을 곱해서 현재 gain 계산.
        """

        self.kp = np.clip(self.base_kp * self.kp_scale, KP_MIN, KP_MAX)
        self.ki = np.clip(self.base_ki * self.ki_scale, KI_MIN, KI_MAX)
        self.kd = np.clip(self.base_kd * self.kd_scale, KD_MIN, KD_MAX)

        self.kp = float(self.kp)
        self.ki = float(self.ki)
        self.kd = float(self.kd)

    # ========================================================
    # Getter
    # ========================================================

    def get_gains(self):
        return self.kp, self.ki, self.kd

    def get_scheduler_state(self):
        return {
            "target": self.target,

            "kp": self.kp,
            "ki": self.ki,
            "kd": self.kd,

            "base_kp": self.base_kp,
            "base_ki": self.base_ki,
            "base_kd": self.base_kd,

            "kp_scale": self.kp_scale,
            "ki_scale": self.ki_scale,
            "kd_scale": self.kd_scale,

            "gain_update_flag": self.gain_update_flag,
            "last_update_reason": self.last_update_reason,

            "saturation_counter": self.saturation_counter,
            "saturation_active": self.saturation_active,
            "pwm_soft_limit": PWM_SOFT_LIMIT,
        }