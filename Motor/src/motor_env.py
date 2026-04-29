class SimpleMotorEnv:
    """
    간단한 가상 모터 응답 환경.
    현재는 테스트용이며, 추후 Simulink 모델 또는 실제 모터 응답으로 대체 가능하다.
    """

    def __init__(
        self,
        initial_value: float = 0.0,
        motor_gain: float = 0.08,
        damping: float = 0.02,
    ):
        self.current_value = initial_value
        self.motor_gain = motor_gain
        self.damping = damping

    def reset(self, initial_value: float = 0.0):
        """
        모터 상태 초기화
        """
        self.current_value = initial_value
        return self.current_value

    def step(self, pwm: float) -> float:
        """
        PWM 입력을 받아 다음 current 값을 반환한다.
        """

        next_value = (
            self.current_value
            + self.motor_gain * pwm
            - self.damping * self.current_value
        )

        self.current_value = next_value

        return self.current_value

    def get_state(self) -> float:
        """
        현재 모터 상태 반환
        """
        return self.current_value


if __name__ == "__main__":
    env = SimpleMotorEnv()

    for step in range(10):
        current = env.step(pwm=255)
        print(f"step={step}, current={current:.2f}")