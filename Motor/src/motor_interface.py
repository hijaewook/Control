from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Dict, Any
import time

from motor_env import SimpleMotorEnv


@dataclass
class MotorState:
    """
    Motor interface에서 반환하는 표준 상태.
    나중에 ESP32 실물 모터에서도 같은 형식으로 반환하면 된다.
    """
    timestamp: float
    current: float
    disturbance: float = 0.0
    raw: Optional[Dict[str, Any]] = None


class BaseMotorInterface(ABC):
    """
    모든 motor backend가 따라야 하는 공통 인터페이스.

    Controller는 이 interface만 사용하고,
    실제 backend가 simulation인지 ESP32인지 알 필요가 없도록 한다.
    """

    @abstractmethod
    def reset(self) -> MotorState:
        pass

    @abstractmethod
    def step(self, pwm: float) -> MotorState:
        pass

    @abstractmethod
    def get_state(self) -> MotorState:
        pass

    @abstractmethod
    def close(self):
        pass


class SimMotorInterface(BaseMotorInterface):
    """
    SimpleMotorEnv를 감싼 simulation motor interface.
    기존 local_kafka_controller에서 사용하던 SimpleMotorEnv 역할을 대체한다.
    """

    def __init__(
        self,
        dt: float,
        pwm_min: float,
        pwm_max: float,
        use_disturbance: bool = False,
        disturbance_mode: str = "none",
        disturbance_start_time: float = 3.0,
        disturbance_end_time: float = 6.0,
        disturbance_magnitude: float = 20.0,
        disturbance_freq: float = 1.0,
        initial_value: float = 0.0,
    ):
        self.env = SimpleMotorEnv(
            initial_value=initial_value,
            dt=dt,
            pwm_min=pwm_min,
            pwm_max=pwm_max,
            use_disturbance=use_disturbance,
            disturbance_mode=disturbance_mode,
            disturbance_start_time=disturbance_start_time,
            disturbance_end_time=disturbance_end_time,
            disturbance_magnitude=disturbance_magnitude,
            disturbance_freq=disturbance_freq,
        )

    def reset(self) -> MotorState:
        current = self.env.reset()
        disturbance = self.env.get_disturbance()

        return MotorState(
            timestamp=time.time(),
            current=float(current),
            disturbance=float(disturbance),
            raw={
                "backend": "simulation",
                "env_time": self.env.time,
            },
        )

    def step(self, pwm: float) -> MotorState:
        disturbance = self.env.get_disturbance()
        current = self.env.step(pwm)

        return MotorState(
            timestamp=time.time(),
            current=float(current),
            disturbance=float(disturbance),
            raw={
                "backend": "simulation",
                "env_time": self.env.time,
                "pwm": float(pwm),
            },
        )

    def get_state(self) -> MotorState:
        current = self.env.get_state()
        disturbance = self.env.get_disturbance()

        return MotorState(
            timestamp=time.time(),
            current=float(current),
            disturbance=float(disturbance),
            raw={
                "backend": "simulation",
                "env_time": self.env.time,
            },
        )

    def close(self):
        pass


class ESP32MotorInterface(BaseMotorInterface):
    """
    Serial-based ESP32 motor interface.

    Expected ESP32 protocol:
        PC  -> ESP32: PING
        ESP -> PC:    PONG

        PC  -> ESP32: SET_PWM <pwm>
        ESP -> PC:    OK PWM <pwm>

        PC  -> ESP32: GET_STATE
        ESP -> PC:    STATE rpm=<rpm> pwm=<pwm> encoder=<encoder_count>

        PC  -> ESP32: STOP
        ESP -> PC:    OK STOP
    """

    def __init__(
        self,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.2,
        pwm_min: float = 0.0,
        pwm_max: float = 180.0,
        auto_stop_on_close: bool = True,
    ):
        import serial

        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout

        self.pwm_min = float(pwm_min)
        self.pwm_max = float(pwm_max)

        self.auto_stop_on_close = auto_stop_on_close

        self.last_current = 0.0
        self.last_pwm = 0.0
        self.last_encoder = 0

        self.ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            timeout=self.timeout,
        )

        # ESP32 reset 대기
        time.sleep(2.0)

        self.flush()

    def flush(self):
        """
        Serial buffer 정리.
        """
        if self.ser is None:
            return

        self.ser.reset_input_buffer()
        self.ser.reset_output_buffer()

    def send_command(self, command: str) -> str:
        """
        ESP32에 명령을 보내고 한 줄 응답을 받는다.
        """
        if self.ser is None:
            raise RuntimeError("Serial connection is not open.")

        command = command.strip()

        self.ser.write((command + "\n").encode("utf-8"))
        self.ser.flush()

        response = self.ser.readline().decode(errors="ignore").strip()

        if response == "":
            raise TimeoutError(f"No response from ESP32 for command: {command}")

        return response

    def ping(self) -> bool:
        """
        PING/PONG 통신 확인.
        """
        response = self.send_command("PING")
        return response.strip().upper() == "PONG"

    def parse_state_line(self, line: str) -> MotorState:
        """
        STATE 응답 parsing.

        Expected:
            STATE rpm=123.45 pwm=120 encoder=45678
        """
        line = line.strip()

        if not line.startswith("STATE"):
            raise ValueError(f"Invalid STATE response: {line}")

        values = {}

        parts = line.split()

        for item in parts[1:]:
            if "=" not in item:
                continue

            key, value = item.split("=", 1)
            values[key.strip()] = value.strip()

        rpm = float(values.get("rpm", self.last_current))
        pwm = float(values.get("pwm", self.last_pwm))
        encoder = int(float(values.get("encoder", self.last_encoder)))

        self.last_current = rpm
        self.last_pwm = pwm
        self.last_encoder = encoder

        return MotorState(
            timestamp=time.time(),
            current=float(rpm),
            disturbance=0.0,
            raw={
                "backend": "esp32",
                "line": line,
                "rpm": rpm,
                "pwm": pwm,
                "encoder": encoder,
            },
        )

    def reset(self) -> MotorState:
        """
        모터를 정지시키고 현재 상태를 반환.
        """
        self.stop()
        return self.get_state()

    def step(self, pwm: float) -> MotorState:
        """
        PWM command를 ESP32로 보내고, 이후 상태를 읽는다.
        """
        pwm = float(pwm)
        pwm = max(self.pwm_min, min(self.pwm_max, pwm))

        response = self.send_command(f"SET_PWM {pwm:.2f}")

        if not response.startswith("OK PWM"):
            raise RuntimeError(f"Unexpected SET_PWM response: {response}")

        self.last_pwm = pwm

        return self.get_state()

    def get_state(self) -> MotorState:
        """
        ESP32에서 현재 RPM 상태를 읽는다.
        """
        response = self.send_command("GET_STATE")
        return self.parse_state_line(response)

    def stop(self):
        """
        모터 정지.
        """
        try:
            response = self.send_command("STOP")

            if not response.startswith("OK STOP"):
                print(f"Warning: unexpected STOP response: {response}")

        except Exception as e:
            print(f"Warning: failed to send STOP command: {e}")

    def close(self):
        """
        Serial 연결 종료.
        """
        if self.ser is None:
            return

        if self.auto_stop_on_close:
            self.stop()

        self.ser.close()
        self.ser = None