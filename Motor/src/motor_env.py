import numpy as np


class SimpleMotorEnv:
    """
    Simple motor environment for PID control test.

    The environment approximates motor response using a first-order system:

        dx/dt = (K * u - x) / tau

    where:
        x   : current motor output
        u   : PWM input
        K   : motor gain
        tau : time constant

    Disturbance is modeled as an output-side load effect.
    A positive disturbance decreases the effective motor input.
    """

    def __init__(
        self,
        initial_value=0.0,
        dt=0.01,
        k_motor=1.0,
        tau_motor=0.4,
        pwm_min=0.0,
        pwm_max=255.0,
        use_disturbance=False,
        disturbance_mode="none",
        disturbance_start_time=3.0,
        disturbance_end_time=6.0,
        disturbance_magnitude=20.0,
        disturbance_freq=1.0,
    ):
        self.initial_value = float(initial_value)

        self.dt = float(dt)
        self.k_motor = float(k_motor)
        self.tau_motor = float(tau_motor)

        self.pwm_min = float(pwm_min)
        self.pwm_max = float(pwm_max)

        self.use_disturbance = bool(use_disturbance)
        self.disturbance_mode = str(disturbance_mode)
        self.disturbance_start_time = float(disturbance_start_time)
        self.disturbance_end_time = float(disturbance_end_time)
        self.disturbance_magnitude = float(disturbance_magnitude)
        self.disturbance_freq = float(disturbance_freq)

        self.time = 0.0
        self.current = self.initial_value

    def reset(self):
        """
        Environment state reset.
        """
        self.time = 0.0
        self.current = self.initial_value
        return self.current

    def get_state(self) -> float:
        """
        현재 모터 상태 반환.
        """
        return self.current

    def get_disturbance(self) -> float:
        """
        현재 시간에서 disturbance 값을 계산한다.

        disturbance는 output을 감소시키는 부하항으로 해석한다.
        즉, 값이 클수록 같은 PWM에서 motor output이 낮아지는 효과가 있다.
        """

        if not self.use_disturbance:
            return 0.0

        if self.disturbance_mode == "none":
            return 0.0

        t = self.time

        if self.disturbance_mode == "step":
            if t >= self.disturbance_start_time:
                return self.disturbance_magnitude
            return 0.0

        if self.disturbance_mode == "pulse":
            if self.disturbance_start_time <= t <= self.disturbance_end_time:
                return self.disturbance_magnitude
            return 0.0

        if self.disturbance_mode == "sin":
            if t >= self.disturbance_start_time:
                return self.disturbance_magnitude * np.sin(
                    2.0
                    * np.pi
                    * self.disturbance_freq
                    * (t - self.disturbance_start_time)
                )
            return 0.0

        raise ValueError(
            "disturbance_mode must be one of: "
            "'none', 'step', 'pulse', or 'sin'"
        )

    def step(self, pwm: float) -> float:
        """
        Motor state update.
        """

        pwm = float(np.clip(pwm, self.pwm_min, self.pwm_max))

        disturbance = self.get_disturbance()

        # PWM 기반 steady-state input
        motor_input = self.k_motor * pwm

        # disturbance는 출력 방향에서 부하로 작용
        effective_input = motor_input - disturbance

        # 1차 시스템 업데이트
        dx = (effective_input - self.current) / self.tau_motor
        self.current += dx * self.dt

        # 물리적으로 음수 출력 방지
        self.current = max(0.0, self.current)

        self.time += self.dt

        return self.current


if __name__ == "__main__":
    env = SimpleMotorEnv(
        initial_value=0.0,
        use_disturbance=True,
        disturbance_mode="pulse",
        disturbance_start_time=0.03,
        disturbance_end_time=0.06,
        disturbance_magnitude=20.0,
    )

    for step in range(10):
        current = env.step(pwm=255)
        disturbance = env.get_disturbance()
        print(
            f"step={step}, "
            f"time={env.time:.3f}, "
            f"current={current:.2f}, "
            f"disturbance={disturbance:.2f}"
        )