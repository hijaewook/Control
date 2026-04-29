import time
from pathlib import Path

import pandas as pd
import streamlit as st

from config import LOG_DIR


st.set_page_config(
    page_title="Adaptive PID Monitoring",
    layout="wide",
)

st.title("PID Real-time Monitoring")


def get_latest_log_file():
    log_files = sorted(LOG_DIR.glob("*.csv"))

    if not log_files:
        return None

    return log_files[-1]


refresh_interval = st.sidebar.slider(
    "Refresh interval [s]",
    min_value=0.5,
    max_value=5.0,
    value=1.0,
    step=0.5,
)

log_file = get_latest_log_file()

if log_file is None:
    st.warning("No realtime log file found.")
    time.sleep(refresh_interval)
    st.rerun()

st.caption(f"Current log file: {log_file}")

try:
    df = pd.read_csv(log_file)
except Exception as e:
    st.error(f"Failed to read log file: {e}")
    time.sleep(refresh_interval)
    st.rerun()

if len(df) == 0:
    st.warning("Log file is empty.")
    time.sleep(refresh_interval)
    st.rerun()

# ============================================================
# Latest state
# ============================================================

latest = df.iloc[-1]

mode = latest["mode"] if "mode" in df.columns else "unknown"
env_type = latest["env_type"] if "env_type" in df.columns else "unknown"

st.title("PID Real-time Monitoring")
st.caption(f"Mode: {mode} | Environment: {env_type}")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Target", f"{latest['target']:.2f}")
col2.metric("Current", f"{latest['current']:.2f}")
col3.metric("Error", f"{latest['error']:.2f}")
col4.metric("PWM", f"{latest['pwm']:.2f}")

col5, col6, col7 = st.columns(3)

col5.metric("Kp", f"{latest['kp']:.3f}")
col6.metric("Ki", f"{latest['ki']:.3f}")
col7.metric("Kd", f"{latest['kd']:.3f}")

col8, col9, col10 = st.columns(3)

if "pwm_saturated" in df.columns:
    col8.metric("PWM Saturated", str(latest["pwm_saturated"]))

if "high_saturation" in df.columns:
    col9.metric("High Saturation", str(latest["high_saturation"]))

if "gain_update_flag" in df.columns:
    col10.metric("Gain Updated", str(latest["gain_update_flag"]))

col11, col12 = st.columns(2)

if "integral" in df.columns:
    col11.metric("Integral", f"{latest['integral']:.3f}")

if "prev_pwm" in df.columns:
    col12.metric("Previous PWM", f"{latest['prev_pwm']:.2f}")

if "integral" in df.columns:
    st.subheader("Integral Term")
    st.line_chart(df.set_index("time")[["integral"]])

if "gain_update_flag" in df.columns:
    st.subheader("Gain Update Flag")
    st.line_chart(df.set_index("time")[["gain_update_flag"]])

if "pwm_saturated" in df.columns:
    st.subheader("PWM Saturation")
    st.line_chart(df.set_index("time")[["pwm_saturated", "high_saturation", "low_saturation"]])

# ============================================================
# Plots
# ============================================================

st.subheader("Target vs Current")
st.line_chart(df.set_index("time")[["target", "current"]])

st.subheader("PWM Output")
st.line_chart(df.set_index("time")[["pwm"]])

st.subheader("PID Gains")
st.line_chart(df.set_index("time")[["kp", "ki", "kd"]])

st.subheader("Latest Data")
st.dataframe(df.tail(20), use_container_width=True)

# ============================================================
# Auto refresh
# ============================================================

time.sleep(refresh_interval)
st.rerun()