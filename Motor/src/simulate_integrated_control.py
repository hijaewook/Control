import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from pid_controller import PIDController
from motor_env import SimpleMotorEnv
from config import (
    DT,
    PWM_MIN,
    PWM_MAX,
    RESULTS_DIR,
    FIGURE_DIR,
    PID_GAIN_DB,
    GAIN_DB_MODE,

    DEFAULT_KP,
    DEFAULT_KI,
    DEFAULT_KD,

    KP_MIN,
    KP_MAX,
    KI_MIN,
    KI_MAX,
    KD_MIN,
    KD_MAX,

    PWM_SOFT_LIMIT,
    SATURATION_CONSECUTIVE_STEPS,
    SATURATION_ERROR_THRESHOLD_RATIO,
    SATURATION_KP_DECAY,
    SATURATION_KI_DECAY,
    SATURATION_KD_DECAY,
    SATURATION_MIN_GAIN_SCALE,
    SATURATION_RECOVERY_RATE,
)


# ============================================================
# Experiment settings
# ============================================================

RESULT_DIR = RESULTS_DIR / "integrated_control"

SIM_TIME = 10.0
N_STEPS = int(SIM_TIME / DT)

TARGET_BEFORE = 100.0
TARGET_AFTER = 200.0
TARGET_CHANGE_TIME = 3.0

INFERENCE_DELAY = 0.5

USE_DISTURBANCE = True
DISTURBANCE_MODE = "pulse"
DISTURBANCE_START_TIME = 5.0
DISTURBANCE_END_TIME = 7.0
DISTURBANCE_MAGNITUDE = 20.0

CONTROL_MODES = [
    "fixed_pid",
    "immediate_gain_db",
    "delayed_gain_db",
    "delayed_saturation_aware_gain_db",
]


# ============================================================
# Target / gain utility
# ============================================================

def get_target_at_time(t: float) -> float:
    if t < TARGET_CHANGE_TIME:
        return TARGET_BEFORE
    return TARGET_AFTER


def get_gain_from_db(target: float):
    """
    PID_GAIN_DB에서 target 기반 gain을 가져온다.
    GAIN_DB_MODE='linear'이면 선형 보간한다.
    """

    if not PID_GAIN_DB:
        return DEFAULT_KP, DEFAULT_KI, DEFAULT_KD

    target = float(target)
    db_targets = sorted([float(k) for k in PID_GAIN_DB.keys()])

    if target in db_targets:
        gains = PID_GAIN_DB[target]
        return gains["kp"], gains["ki"], gains["kd"]

    if GAIN_DB_MODE == "nearest":
        nearest_target = min(db_targets, key=lambda x: abs(x - target))
        gains = PID_GAIN_DB[nearest_target]
        return gains["kp"], gains["ki"], gains["kd"]

    if GAIN_DB_MODE == "linear":
        if target <= db_targets[0]:
            gains = PID_GAIN_DB[db_targets[0]]
            return gains["kp"], gains["ki"], gains["kd"]

        if target >= db_targets[-1]:
            gains = PID_GAIN_DB[db_targets[-1]]
            return gains["kp"], gains["ki"], gains["kd"]

        for i in range(len(db_targets) - 1):
            t_low = db_targets[i]
            t_high = db_targets[i + 1]

            if t_low <= target <= t_high:
                ratio = (target - t_low) / (t_high - t_low)

                g_low = PID_GAIN_DB[t_low]
                g_high = PID_GAIN_DB[t_high]

                kp = g_low["kp"] + ratio * (g_high["kp"] - g_low["kp"])
                ki = g_low["ki"] + ratio * (g_high["ki"] - g_low["ki"])
                kd = g_low["kd"] + ratio * (g_high["kd"] - g_low["kd"])

                return kp, ki, kd

    nearest_target = min(db_targets, key=lambda x: abs(x - target))
    gains = PID_GAIN_DB[nearest_target]
    return gains["kp"], gains["ki"], gains["kd"]


# ============================================================
# Saturation-aware gain manager
# ============================================================

class SaturationAwareGainManager:
    def __init__(self, target: float, enabled: bool):
        self.enabled = enabled

        self.target = float(target)
        self.base_kp, self.base_ki, self.base_kd = get_gain_from_db(target)

        self.kp_scale = 1.0
        self.ki_scale = 1.0
        self.kd_scale = 1.0

        self.saturation_counter = 0
        self.saturation_active = False
        self.last_update_reason = "init"
        self.gain_update_flag = False

        self.kp = self.base_kp
        self.ki = self.base_ki
        self.kd = self.base_kd

    def set_target(self, target: float):
        self.target = float(target)
        self.base_kp, self.base_ki, self.base_kd = get_gain_from_db(target)

        self.kp_scale = 1.0
        self.ki_scale = 1.0
        self.kd_scale = 1.0

        self.saturation_counter = 0
        self.saturation_active = False
        self.last_update_reason = "target_gain_db"
        self.gain_update_flag = True

        self._apply_scaled_gains()

    def update(self, target: float, error: float, pwm: float):
        self.gain_update_flag = False
        self.last_update_reason = "none"

        if abs(float(target) - self.target) > 1e-9:
            self.set_target(target)

        if self.enabled:
            self._update_saturation_scale(target=target, error=error, pwm=pwm)

        self._apply_scaled_gains()

        return self.kp, self.ki, self.kd

    def _update_saturation_scale(self, target: float, error: float, pwm: float):
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
            if self.kp_scale < 1.0 or self.ki_scale < 1.0 or self.kd_scale < 1.0:
                self.kp_scale = min(1.0, self.kp_scale + SATURATION_RECOVERY_RATE)
                self.ki_scale = min(1.0, self.ki_scale + SATURATION_RECOVERY_RATE)
                self.kd_scale = min(1.0, self.kd_scale + SATURATION_RECOVERY_RATE)

                self.gain_update_flag = True
                self.last_update_reason = "saturation_recovery"

            self.saturation_active = False

    def _apply_scaled_gains(self):
        self.kp = float(np.clip(self.base_kp * self.kp_scale, KP_MIN, KP_MAX))
        self.ki = float(np.clip(self.base_ki * self.ki_scale, KI_MIN, KI_MAX))
        self.kd = float(np.clip(self.base_kd * self.kd_scale, KD_MIN, KD_MAX))

    def get_gains(self):
        return self.kp, self.ki, self.kd

    def get_state(self):
        return {
            "base_kp": self.base_kp,
            "base_ki": self.base_ki,
            "base_kd": self.base_kd,
            "kp_scale": self.kp_scale,
            "ki_scale": self.ki_scale,
            "kd_scale": self.kd_scale,
            "saturation_counter": self.saturation_counter,
            "saturation_active": self.saturation_active,
            "gain_update_flag": self.gain_update_flag,
            "last_update_reason": self.last_update_reason,
        }
    
    def update_saturation_only(self, error_reference, error, pwm):
        """
        Target DB gain은 바꾸지 않고,
        현재 적용 중인 gain에 대해서만 saturation-aware scale을 조정한다.
        delay window 중 local safety layer로 사용한다.
        """

        self.gain_update_flag = False
        self.last_update_reason = "none"

        if self.enabled:
            self._update_saturation_scale(
                target=error_reference,
                error=error,
                pwm=pwm,
            )

        self._apply_scaled_gains()

        return self.kp, self.ki, self.kd


# ============================================================
# Metrics
# ============================================================

def compute_dt(time: np.ndarray) -> np.ndarray:
    if len(time) <= 1:
        return np.zeros_like(time)

    dt = np.diff(time, prepend=time[0])
    dt[0] = 0.0
    return dt


def calculate_metrics(df: pd.DataFrame) -> dict:
    time = df["time"].to_numpy(dtype=float)
    target = df["target"].to_numpy(dtype=float)
    current = df["current"].to_numpy(dtype=float)
    error = df["error"].to_numpy(dtype=float)
    pwm = df["pwm"].to_numpy(dtype=float)
    disturbance = df["disturbance"].to_numpy(dtype=float)

    dt = compute_dt(time)
    abs_error = np.abs(error)
    abs_pwm = np.abs(pwm)

    iae = float(np.sum(abs_error * dt))
    ise = float(np.sum((error ** 2) * dt))
    mean_abs_error = float(np.mean(abs_error))
    final_error = float(abs_error[-1])

    max_current = float(np.max(current))
    overshoot = max(0.0, max_current - TARGET_AFTER)
    overshoot_percent = overshoot / max(abs(TARGET_AFTER), 1e-6) * 100.0

    mean_pwm = float(np.mean(abs_pwm))
    total_pwm = float(np.sum(abs_pwm * dt))
    max_pwm = float(np.max(pwm))

    high_saturation = pwm >= PWM_MAX - 1e-9
    saturation_ratio_percent = float(np.mean(high_saturation) * 100.0)
    saturation_duration = float(np.sum(dt[high_saturation]))

    near_high_saturation = pwm >= PWM_MAX - 0.1 * (PWM_MAX - PWM_MIN)
    near_high_saturation_ratio_percent = float(np.mean(near_high_saturation) * 100.0)

    after_change_mask = time >= TARGET_CHANGE_TIME

    if after_change_mask.any():
        after_change_IAE = float(np.sum(abs_error[after_change_mask] * dt[after_change_mask]))
        after_change_max_error = float(np.max(abs_error[after_change_mask]))
    else:
        after_change_IAE = np.nan
        after_change_max_error = np.nan

    disturbance_mask = np.abs(disturbance) > 1e-9

    if disturbance_mask.any():
        disturbance_IAE = float(np.sum(abs_error[disturbance_mask] * dt[disturbance_mask]))
        disturbance_max_error = float(np.max(abs_error[disturbance_mask]))
        disturbance_min_current = float(np.min(current[disturbance_mask]))
    else:
        disturbance_IAE = np.nan
        disturbance_max_error = np.nan
        disturbance_min_current = np.nan

    tolerance = 0.02 * abs(TARGET_AFTER)
    settling_time_after_change = np.nan

    after_indices = np.where(after_change_mask)[0]

    for idx in after_indices:
        if np.all(np.abs(target[idx:] - current[idx:]) <= tolerance):
            settling_time_after_change = float(time[idx] - TARGET_CHANGE_TIME)
            break

    return {
        "IAE": iae,
        "ISE": ise,
        "mean_abs_error": mean_abs_error,
        "final_error": final_error,
        "overshoot_percent": overshoot_percent,
        "after_change_IAE": after_change_IAE,
        "after_change_max_error": after_change_max_error,
        "disturbance_IAE": disturbance_IAE,
        "disturbance_max_error": disturbance_max_error,
        "disturbance_min_current": disturbance_min_current,
        "settling_time_after_change": settling_time_after_change,
        "mean_pwm": mean_pwm,
        "total_pwm": total_pwm,
        "max_pwm": max_pwm,
        "saturation_ratio_percent": saturation_ratio_percent,
        "saturation_duration": saturation_duration,
        "near_high_saturation_ratio_percent": near_high_saturation_ratio_percent,
        "gain_reduction_count": int((df["last_update_reason"] == "saturation_gain_reduction").sum()),
        "gain_recovery_count": int((df["last_update_reason"] == "saturation_recovery").sum()),
        "min_kp_scale": float(df["kp_scale"].min()),
        "min_ki_scale": float(df["ki_scale"].min()),
    }


# ============================================================
# Simulation
# ============================================================

def run_single_mode(mode: str) -> pd.DataFrame:
    if mode not in CONTROL_MODES:
        raise ValueError(f"Unknown mode: {mode}")

    env = SimpleMotorEnv(
        initial_value=0.0,
        dt=DT,
        pwm_min=PWM_MIN,
        pwm_max=PWM_MAX,
        use_disturbance=USE_DISTURBANCE,
        disturbance_mode=DISTURBANCE_MODE,
        disturbance_start_time=DISTURBANCE_START_TIME,
        disturbance_end_time=DISTURBANCE_END_TIME,
        disturbance_magnitude=DISTURBANCE_MAGNITUDE,
        disturbance_freq=1.0,
    )

    pid = PIDController()

    target0 = TARGET_BEFORE

    use_gain_db = mode in [
        "immediate_gain_db",
        "delayed_gain_db",
        "delayed_saturation_aware_gain_db",
    ]

    use_saturation_aware = mode == "delayed_saturation_aware_gain_db"

    gain_manager = SaturationAwareGainManager(
        target=target0,
        enabled=use_saturation_aware,
    )

    if use_gain_db:
        kp0, ki0, kd0 = gain_manager.get_gains()
        pid.set_gains(kp0, ki0, kd0)

    current = env.get_state()
    prev_error = target0 - current

    target_change_detected = False
    gain_request_time = np.nan
    gain_arrival_time = np.nan
    pending_target = None
    pending_gain = None
    delayed_gain_applied = False

    rows = []

    for step in range(N_STEPS):
        t = step * DT
        target = get_target_at_time(t)

        target_changed_now = (
            (not target_change_detected)
            and t >= TARGET_CHANGE_TIME
        )

        if target_changed_now:
            target_change_detected = True

            new_kp, new_ki, new_kd = get_gain_from_db(target)

            if mode == "immediate_gain_db":
                gain_manager.set_target(target)
                pid.set_gains(new_kp, new_ki, new_kd)

            elif mode in ["delayed_gain_db", "delayed_saturation_aware_gain_db"]:
                gain_request_time = t
                gain_arrival_time = t + INFERENCE_DELAY
                pending_target = target
                pending_gain = (new_kp, new_ki, new_kd)

        gain_update_event = False

        if mode in ["delayed_gain_db", "delayed_saturation_aware_gain_db"]:
            if (
                target_change_detected
                and not delayed_gain_applied
                and pending_gain is not None
                and t >= gain_arrival_time
            ):
                gain_manager.set_target(pending_target)
                kp_new, ki_new, kd_new = gain_manager.get_gains()
                pid.set_gains(kp_new, ki_new, kd_new)

                delayed_gain_applied = True
                gain_update_event = True

        error = target - current
        error_derivative = (error - prev_error) / DT

        pwm = pid.compute(target, current)

        if use_saturation_aware:
            kp, ki, kd = gain_manager.update_saturation_only(
                error_reference=target,
                error=error,
                pwm=pwm,
            )
            pid.set_gains(kp, ki, kd)

        pid_state = pid.get_state()
        gain_state = gain_manager.get_state()

        if not use_saturation_aware:
            gain_state["gain_update_flag"] = gain_update_event
            if gain_update_event:
                gain_state["last_update_reason"] = "delayed_gain_arrival"

        disturbance = env.get_disturbance()
        current = env.step(pwm)

        rows.append(
            {
                "mode": mode,
                "step": step,
                "time": t,
                "target": target,
                "current": current,
                "error": error,
                "error_derivative": error_derivative,
                "pwm": pwm,

                "kp": pid_state["kp"],
                "ki": pid_state["ki"],
                "kd": pid_state["kd"],
                "integral": pid_state["integral"],

                "base_kp": gain_state["base_kp"],
                "base_ki": gain_state["base_ki"],
                "base_kd": gain_state["base_kd"],
                "kp_scale": gain_state["kp_scale"],
                "ki_scale": gain_state["ki_scale"],
                "kd_scale": gain_state["kd_scale"],
                "saturation_counter": gain_state["saturation_counter"],
                "saturation_active": gain_state["saturation_active"],
                "gain_update_flag": gain_state["gain_update_flag"],
                "last_update_reason": gain_state["last_update_reason"],

                "target_change_time": TARGET_CHANGE_TIME,
                "inference_delay": INFERENCE_DELAY,
                "gain_request_time": gain_request_time,
                "gain_arrival_time": gain_arrival_time,

                "disturbance": disturbance,
                "disturbance_start_time": DISTURBANCE_START_TIME,
                "disturbance_end_time": DISTURBANCE_END_TIME,
                "disturbance_magnitude": DISTURBANCE_MAGNITUDE,
            }
        )

        prev_error = error

    return pd.DataFrame(rows)


def run_all_modes():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logs = {}
    metric_rows = []

    for mode in CONTROL_MODES:
        print("-" * 80)
        print(f"Run mode: {mode}")

        df = run_single_mode(mode)
        logs[mode] = df

        log_path = RESULT_DIR / f"integrated_{mode}_{timestamp}.csv"
        df.to_csv(log_path, index=False, encoding="utf-8-sig")
        print(f"Saved: {log_path}")

        metrics = calculate_metrics(df)
        metrics["mode"] = mode
        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(metric_rows)

    metrics_path = RESULT_DIR / f"integrated_metrics_{timestamp}.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")
    print(f"Saved metrics: {metrics_path}")

    return logs, metrics_df, timestamp


# ============================================================
# Plot
# ============================================================

def shade_event_regions():
    plt.axvline(
        TARGET_CHANGE_TIME,
        linestyle="--",
        linewidth=1.2,
        label="target change",
    )

    plt.axvspan(
        TARGET_CHANGE_TIME,
        TARGET_CHANGE_TIME + INFERENCE_DELAY,
        alpha=0.12,
        label="inference delay",
    )

    plt.axvspan(
        DISTURBANCE_START_TIME,
        DISTURBANCE_END_TIME,
        alpha=0.12,
        label="disturbance",
    )


def plot_response(logs: dict, timestamp: str):
    plt.figure(figsize=(10, 5))

    for mode, df in logs.items():
        plt.plot(df["time"], df["current"], label=mode)

    time_arr = next(iter(logs.values()))["time"].to_numpy()
    target_profile = np.array([get_target_at_time(t) for t in time_arr])
    plt.plot(time_arr, target_profile, linestyle="--", label="target")

    shade_event_regions()

    plt.xlabel("Time [s]")
    plt.ylabel("Current")
    plt.title("Integrated Validation: Target Step + Delay + Disturbance")
    plt.legend(fontsize=8)
    plt.grid(True)

    save_path = FIGURE_DIR / f"integrated_response_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_pwm(logs: dict, timestamp: str):
    plt.figure(figsize=(10, 5))

    for mode, df in logs.items():
        plt.plot(df["time"], df["pwm"], label=mode)

    plt.axhline(PWM_MAX, linestyle="--", label="PWM max")

    shade_event_regions()

    plt.xlabel("Time [s]")
    plt.ylabel("PWM")
    plt.title("Integrated Validation: PWM Comparison")
    plt.legend(fontsize=8)
    plt.grid(True)

    save_path = FIGURE_DIR / f"integrated_pwm_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_gain_scales(logs: dict, timestamp: str):
    mode = "delayed_saturation_aware_gain_db"

    if mode not in logs:
        return

    df = logs[mode]

    plt.figure(figsize=(10, 5))
    plt.plot(df["time"], df["kp_scale"], label="Kp scale")
    plt.plot(df["time"], df["ki_scale"], label="Ki scale")

    shade_event_regions()

    plt.xlabel("Time [s]")
    plt.ylabel("Gain scale")
    plt.title("Integrated Validation: Saturation-aware Gain Scaling")
    plt.legend()
    plt.grid(True)

    save_path = FIGURE_DIR / f"integrated_gain_scales_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {save_path}")

    plt.show()


def plot_metrics(metrics_df: pd.DataFrame, timestamp: str):
    metric_cols = [
        "IAE",
        "after_change_IAE",
        "disturbance_IAE",
        "settling_time_after_change",
        "saturation_ratio_percent",
        "saturation_duration",
        "near_high_saturation_ratio_percent",
    ]

    for metric in metric_cols:
        plt.figure(figsize=(9, 5))
        plt.bar(metrics_df["mode"], metrics_df[metric])
        plt.xticks(rotation=20, ha="right")

        plt.xlabel("Controller")
        plt.ylabel(metric)
        plt.title(f"Integrated Validation: {metric}")
        plt.grid(True, axis="y")

        save_path = FIGURE_DIR / f"integrated_metric_{metric}_{timestamp}.png"
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {save_path}")

        plt.show()


# ============================================================
# Summary
# ============================================================

def save_markdown_summary(metrics_df: pd.DataFrame, timestamp: str):
    summary_dir = RESULTS_DIR / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)

    save_path = summary_dir / f"integrated_control_summary_{timestamp}.md"

    lines = []

    lines.append("# Integrated Control Validation Summary")
    lines.append("")
    lines.append("## 1. Experiment Setting")
    lines.append("")
    lines.append("- Target step: 100 to 200 at 3.0 s")
    lines.append("- Inference delay: 0.5 s")
    lines.append("- Pulse disturbance: 5.0 to 7.0 s")
    lines.append("- Disturbance magnitude: 20.0")
    lines.append("- Compared controllers: fixed PID, immediate gain DB, delayed gain DB, delayed saturation-aware gain DB")
    lines.append("")

    lines.append("## 2. Metrics")
    lines.append("")
    lines.append(metrics_df.to_markdown(index=False))
    lines.append("")

    lines.append("## 3. Interpretation")
    lines.append("")
    lines.append(
        "This integrated experiment evaluates target tracking, delayed gain update, disturbance rejection, "
        "and saturation-aware gain scaling in a single scenario. "
        "The delayed gain DB controller is expected to maintain stable tracking despite inference delay, "
        "while the saturation-aware version aims to reduce PWM saturation during aggressive control intervals."
    )
    lines.append("")

    save_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved markdown summary: {save_path}")


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 80)
    print("Integrated control validation")
    print("=" * 80)
    print(f"Target: {TARGET_BEFORE} -> {TARGET_AFTER}")
    print(f"Target change time: {TARGET_CHANGE_TIME}")
    print(f"Inference delay: {INFERENCE_DELAY}")
    print(f"Disturbance: {DISTURBANCE_START_TIME} ~ {DISTURBANCE_END_TIME}, magnitude={DISTURBANCE_MAGNITUDE}")
    print("=" * 80)

    logs, metrics_df, timestamp = run_all_modes()

    print("\n" + "=" * 80)
    print("Integrated metrics")
    print("=" * 80)
    print(metrics_df)

    plot_response(logs, timestamp)
    plot_pwm(logs, timestamp)
    plot_gain_scales(logs, timestamp)
    plot_metrics(metrics_df, timestamp)
    save_markdown_summary(metrics_df, timestamp)


if __name__ == "__main__":
    main()