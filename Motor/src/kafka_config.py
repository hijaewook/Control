# Motor/src/kafka_config.py

KAFKA_BOOTSTRAP_SERVERS = "localhost:9092"

TOPIC_MOTOR_STATE = "motor_state"
TOPIC_GAIN_COMMAND = "motor_gain_command"
TOPIC_EVENT_LOG = "motor_event_log"

DEVICE_ID = "sim_motor_01"

# local controller group
LOCAL_CONTROLLER_GROUP_ID = "local_controller_group"

# server recommender group
GAIN_RECOMMENDER_GROUP_ID = "gain_recommender_group"

# Kafka consumer timeout
CONSUMER_POLL_TIMEOUT_MS = 1

# Gain command validity
GAIN_COMMAND_TTL_SEC = 1.0