import sys
from pathlib import Path
import time
import signal
import threading

from kafka import KafkaConsumer, KafkaProducer


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import (
    PID_GAIN_DB,
    GAIN_DB_MODE,
    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,

    ESP32_REAL_PID_GAIN_DB,
    ESP32_REAL_GAIN_DB_MODE,
)

from kafka_config import (
    KAFKA_BOOTSTRAP_SERVERS,
    TOPIC_MOTOR_STATE,
    TOPIC_GAIN_COMMAND,
    GAIN_RECOMMENDER_GROUP_ID,
    GAIN_COMMAND_TTL_SEC,
)

from message_schema import (
    json_serializer,
    json_deserializer,
    make_gain_command_message,
)


# ============================================================
# Server settings
# ============================================================

# 서버 AI 추론 시간을 모사하는 delay
INFERENCE_DELAY_SEC = 0.5

# 너무 많은 state에 대해 매번 command를 보내지 않기 위한 최소 발행 간격
COMMAND_MIN_INTERVAL_SEC = 1.0

# 동일 target에서 너무 작은 변화는 command 반복 발행 방지
TARGET_TOL = 1e-9

# 같은 device라도 backend가 바뀌면 별도로 command frequency를 관리하기 위한 key
USE_MODE_IN_RATE_LIMIT_KEY = True


# ============================================================
# Global stop flag
# ============================================================

stop_event = threading.Event()


def signal_handler(sig, frame):
    print("\nStop signal received. Shutting down gain recommender server...")
    stop_event.set()


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# ============================================================
# Backend / gain DB utility
# ============================================================

def infer_backend_from_state(state: dict) -> str:
    """
    motor_state message에서 backend를 추정한다.

    현재 local_kafka_controller.py는 mode를 다음처럼 보낸다.
        local_kafka_controller_simulation
        local_kafka_controller_esp32

    향후 message에 backend 필드가 추가되면 그것을 우선 사용한다.
    """

    backend = state.get("backend", None)

    if backend is not None:
        backend = str(backend).lower().strip()
        if backend in ["simulation", "esp32"]:
            return backend

    mode = str(state.get("mode", "")).lower()

    if "esp32" in mode:
        return "esp32"

    if "simulation" in mode or "sim" in mode:
        return "simulation"

    # fallback
    return "simulation"


def interpolate_gain_from_db(
    target: float,
    gain_db: dict,
    mode: str,
    fallback_gain: tuple,
):
    """
    공통 gain DB lookup 함수.
    mode='linear'이면 target 사이를 선형 보간한다.
    """

    if not gain_db:
        return fallback_gain

    target = float(target)
    db_targets = sorted([float(k) for k in gain_db.keys()])

    # Exact match
    if target in db_targets:
        gains = gain_db[target]
        return gains["kp"], gains["ki"], gains["kd"]

    # Nearest mode
    if mode == "nearest":
        nearest_target = min(db_targets, key=lambda x: abs(x - target))
        gains = gain_db[nearest_target]
        return gains["kp"], gains["ki"], gains["kd"]

    # Linear interpolation mode
    if mode == "linear":
        if target <= db_targets[0]:
            gains = gain_db[db_targets[0]]
            return gains["kp"], gains["ki"], gains["kd"]

        if target >= db_targets[-1]:
            gains = gain_db[db_targets[-1]]
            return gains["kp"], gains["ki"], gains["kd"]

        for i in range(len(db_targets) - 1):
            t_low = db_targets[i]
            t_high = db_targets[i + 1]

            if t_low <= target <= t_high:
                ratio = (target - t_low) / (t_high - t_low)

                g_low = gain_db[t_low]
                g_high = gain_db[t_high]

                kp = g_low["kp"] + ratio * (g_high["kp"] - g_low["kp"])
                ki = g_low["ki"] + ratio * (g_high["ki"] - g_low["ki"])
                kd = g_low["kd"] + ratio * (g_high["kd"] - g_low["kd"])

                return kp, ki, kd

    # Fallback
    nearest_target = min(db_targets, key=lambda x: abs(x - target))
    gains = gain_db[nearest_target]
    return gains["kp"], gains["ki"], gains["kd"]


def get_gain_from_db(target: float, backend: str):
    """
    Backend별 gain DB에서 target 기반 gain을 가져온다.

    - simulation: PID_GAIN_DB 사용
    - esp32: ESP32_REAL_PID_GAIN_DB 사용
    """

    backend = str(backend).lower().strip()

    if backend == "esp32":
        kp, ki, kd = interpolate_gain_from_db(
            target=target,
            gain_db=ESP32_REAL_PID_GAIN_DB,
            mode=ESP32_REAL_GAIN_DB_MODE,
            fallback_gain=(1.2, 0.7, 0.0),
        )
        db_name = "ESP32_REAL_PID_GAIN_DB"
        return kp, ki, kd, db_name

    kp, ki, kd = interpolate_gain_from_db(
        target=target,
        gain_db=PID_GAIN_DB,
        mode=GAIN_DB_MODE,
        fallback_gain=(DEFAULT_KP, DEFAULT_KI, DEFAULT_KD),
    )
    db_name = "PID_GAIN_DB"
    return kp, ki, kd, db_name


def make_rate_limit_key(device_id: str, backend: str):
    if USE_MODE_IN_RATE_LIMIT_KEY:
        return f"{device_id}:{backend}"
    return device_id


# ============================================================
# Kafka
# ============================================================

def create_consumer():
    consumer = KafkaConsumer(
        TOPIC_MOTOR_STATE,
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        group_id=GAIN_RECOMMENDER_GROUP_ID,
        value_deserializer=json_deserializer,
        auto_offset_reset="latest",
        enable_auto_commit=True,
    )

    return consumer


def create_producer():
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS,
        value_serializer=json_serializer,
    )

    return producer


# ============================================================
# Server loop
# ============================================================

def run_gain_recommender_server():
    print("=" * 80)
    print("Gain Recommender Server Start")
    print("=" * 80)
    print(f"Kafka bootstrap servers: {KAFKA_BOOTSTRAP_SERVERS}")
    print(f"Consume topic: {TOPIC_MOTOR_STATE}")
    print(f"Produce topic: {TOPIC_GAIN_COMMAND}")
    print(f"Inference delay: {INFERENCE_DELAY_SEC} s")
    print(f"Command min interval: {COMMAND_MIN_INTERVAL_SEC} s")
    print("=" * 80)

    consumer = create_consumer()
    producer = create_producer()

    last_command_time_by_key = {}
    last_target_by_key = {}

    try:
        while not stop_event.is_set():
            records = consumer.poll(timeout_ms=100)

            if not records:
                continue

            for _, messages in records.items():
                for msg in messages:
                    state = msg.value

                    try:
                        run_id = state["run_id"]
                        device_id = state["device_id"]
                        seq = int(state["seq"])
                        target = float(state["target"])
                        mode = str(state.get("mode", "unknown"))

                        backend = infer_backend_from_state(state)
                        rate_key = make_rate_limit_key(device_id, backend)

                        now = time.time()

                        # ----------------------------------------------------
                        # command 발행 빈도 제한
                        # ----------------------------------------------------

                        last_time = last_command_time_by_key.get(rate_key, 0.0)
                        last_target = last_target_by_key.get(rate_key, None)

                        same_target = (
                            last_target is not None
                            and abs(float(last_target) - target) <= TARGET_TOL
                        )

                        # 같은 backend + 같은 target에 대해 너무 자주 command 발행하지 않음
                        if same_target and (now - last_time) < COMMAND_MIN_INTERVAL_SEC:
                            continue

                        # ----------------------------------------------------
                        # inference delay 모사
                        # ----------------------------------------------------

                        time.sleep(INFERENCE_DELAY_SEC)

                        # ----------------------------------------------------
                        # gain recommendation
                        # ----------------------------------------------------

                        kp, ki, kd, db_name = get_gain_from_db(
                            target=target,
                            backend=backend,
                        )

                        command = make_gain_command_message(
                            run_id=run_id,
                            device_id=device_id,
                            source_seq=seq,
                            target=target,
                            kp=kp,
                            ki=ki,
                            kd=kd,
                            confidence=1.0,
                            valid_for_sec=GAIN_COMMAND_TTL_SEC,
                            reason=f"{backend}_{db_name}_recommendation",
                        )

                        producer.send(TOPIC_GAIN_COMMAND, value=command)
                        producer.flush()

                        last_command_time_by_key[rate_key] = time.time()
                        last_target_by_key[rate_key] = target

                        print(
                            f"[COMMAND] backend={backend}, "
                            f"mode={mode}, "
                            f"device={device_id}, "
                            f"seq={seq}, target={target:.1f}, "
                            f"DB={db_name}, "
                            f"Kp={kp:.3f}, Ki={ki:.3f}, Kd={kd:.3f}"
                        )

                    except Exception as e:
                        print(f"Failed to process motor_state message: {e}")
                        print(f"Message: {state}")

    finally:
        print("Closing Kafka consumer/producer...")
        consumer.close()
        producer.close()
        print("Gain recommender server stopped.")


if __name__ == "__main__":
    run_gain_recommender_server()