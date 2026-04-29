import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import LOG_DIR, FIGURE_DIR


def get_latest_log_file():
    """
    results/logs 폴더에서 가장 최근 CSV 로그 파일을 찾는다.
    """
    log_files = sorted(LOG_DIR.glob("*.csv"))

    if not log_files:
        raise FileNotFoundError(f"No log files found in {LOG_DIR}")

    return log_files[-1]


def plot_response(df: pd.DataFrame, save: bool = True):
    """
    Target-current 응답 그래프
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["target"], label="Target", linestyle="--")
    plt.plot(df["time"], df["current"], label="Current")
    plt.xlabel("Time [s]")
    plt.ylabel("Value")
    plt.title("Motor Response")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "response_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_pwm(df: pd.DataFrame, save: bool = True):
    """
    PWM 변화 그래프
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["pwm"], label="PWM")
    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("PWM Output")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "pwm_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def plot_gains(df: pd.DataFrame, save: bool = True):
    """
    PID gain 변화 그래프
    """
    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["kp"], label="Kp")
    plt.plot(df["time"], df["ki"], label="Ki")
    plt.plot(df["time"], df["kd"], label="Kd")
    plt.xlabel("Time [s]")
    plt.ylabel("Gain")
    plt.title("PID Gain Scheduling")
    plt.legend()
    plt.grid(True)

    if save:
        FIGURE_DIR.mkdir(parents=True, exist_ok=True)
        save_path = FIGURE_DIR / "gain_plot.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

    plt.show()


def main():
    log_file = get_latest_log_file()
    print(f"Load log file: {log_file}")

    df = pd.read_csv(log_file)

    plot_response(df)
    plot_pwm(df)
    plot_gains(df)


if __name__ == "__main__":
    main()