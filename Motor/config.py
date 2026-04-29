from pathlib import Path

# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
SIMULATION_DATA_DIR = DATA_DIR / "simulation"

RESULTS_DIR = BASE_DIR / "results"
MODEL_DIR = RESULTS_DIR / "models"
FIGURE_DIR = RESULTS_DIR / "figures"
LOG_DIR = RESULTS_DIR / "logs"

SIMULINK_DIR = ROOT_DIR / "Simulink" / "Motor"


# ============================================================
# Control settings
# ============================================================

DT = 0.01  # sampling time [s]

PWM_MIN = 0
PWM_MAX = 255

TARGET_MIN = 0.0
TARGET_MAX = 3000.0  # 예: RPM 기준이면 나중에 수정


# ============================================================
# Default PID gains
# ============================================================

DEFAULT_KP = 0.5
DEFAULT_KI = 0.0
DEFAULT_KD = 0.0


# ============================================================
# PID gain limits
# ============================================================

KP_MIN = 0.0
KP_MAX = 10.0

KI_MIN = 0.0
KI_MAX = 10.0

KD_MIN = 0.0
KD_MAX = 10.0


# ============================================================
# Gain update limits
# ============================================================

MAX_DELTA_KP = 0.5
MAX_DELTA_KI = 0.5
MAX_DELTA_KD = 0.5


# ============================================================
# Safety settings
# ============================================================

MAX_ERROR = 3000.0
MAX_ERROR_DERIVATIVE = 10000.0
OSCILLATION_THRESHOLD = 5
FALLBACK_ENABLED = True


# ============================================================
# Serial communication
# ============================================================

SERIAL_PORT = "COM5"
BAUD_RATE = 115200
SERIAL_TIMEOUT = 2.0

# ============================================================
# Experiment settings
# ============================================================

RUN_MODE = "fixed_pid"  # "fixed_pid", "adaptive_pid", or "simulink_pid"
ENV_TYPE = "simple_motor"

TEST_TARGET = 100.0
TEST_STEPS = 1000

# ============================================================
# Gain scheduler settings
# ============================================================

GAIN_UPDATE_INTERVAL = 10  # update gains every 10 control steps

# ============================================================
# Simulink settings
# ============================================================

SIMULINK_MODEL_NAME = "Motor"
SIMULINK_STOP_TIME = 10.0
SIMULINK_MAT_FILE = "matlab.mat"

# ============================================================
# Simulink gain sweep setting
# ============================================================

TARGET_LIST = [50, 75, 100, 125, 150, 175, 200]

SWEEP_KP_LIST = [3.4, 3.6, 3.8, 4.0, 4.2, 4.4]
SWEEP_KI_LIST = [2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]
SWEEP_KD_LIST = [0.0]

SIMULINK_SWEEP_STOP_TIME = 10.0

# ============================================================
# Target-based PID gain database
# ============================================================

PID_GAIN_DB = {
    50.0:  {"kp": 4.4, "ki": 6.5, "kd": 0.0},
    75.0:  {"kp": 4.4, "ki": 6.5, "kd": 0.0},
    100.0: {"kp": 4.2, "ki": 6.0, "kd": 0.0},
    125.0: {"kp": 4.0, "ki": 5.0, "kd": 0.0},
    150.0: {"kp": 4.4, "ki": 4.0, "kd": 0.0},
    175.0: {"kp": 4.4, "ki": 3.0, "kd": 0.0},
    200.0: {"kp": 4.4, "ki": 2.5, "kd": 0.0},
}

# ============================================================
# Target-based PID gain database
# ============================================================

USE_GAIN_DB = True

# "nearest" 또는 "linear"
# nearest: 가장 가까운 target의 gain 사용
# linear : target 사이를 선형 보간
GAIN_DB_MODE = "linear"

PID_GAIN_DB = {
    50.0:  {"kp": 4.300, "ki": 8.250, "kd": 0.000},
    75.0:  {"kp": 4.250, "ki": 6.850, "kd": 0.000},
    100.0: {"kp": 4.200, "ki": 6.000, "kd": 0.000},
    125.0: {"kp": 4.000, "ki": 5.000, "kd": 0.000},
    150.0: {"kp": 4.400, "ki": 4.000, "kd": 0.000},
    175.0: {"kp": 3.700, "ki": 3.250, "kd": 0.000},
    200.0: {"kp": 4.400, "ki": 2.500, "kd": 0.000},
}