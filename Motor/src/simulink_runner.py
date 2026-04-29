import sys
from pathlib import Path
from datetime import datetime

import pandas as pd

# ============================================================
# Path setting
# ============================================================

CURRENT_DIR = Path(__file__).resolve().parent
MOTOR_DIR = CURRENT_DIR.parent
sys.path.append(str(MOTOR_DIR))

from config import SIMULINK_DIR, LOG_DIR


# ============================================================
# Utility functions
# ============================================================

def matlab_to_list(value):
    """
    MATLAB Engine에서 받은 matlab.double, list, scalar 등을 Python list로 변환한다.
    """

    # matlab.double 계열
    if hasattr(value, "_data"):
        return [float(v) for v in value._data]

    # 일반 list 또는 tuple
    if isinstance(value, (list, tuple)):
        result = []
        for v in value:
            if isinstance(v, (list, tuple)):
                result.append(float(v[0]))
            else:
                result.append(float(v))
        return result

    # scalar
    try:
        return [float(value)]
    except Exception:
        return list(value)


def trim_to_min_length(*arrays):
    """
    여러 배열의 길이를 가장 짧은 길이에 맞춘다.
    Simulink signal 간 길이 1칸 mismatch 방지용.
    """
    min_len = min(len(arr) for arr in arrays)
    return [arr[:min_len] for arr in arrays]


# ============================================================
# Simulink runner
# ============================================================

class SimulinkRunner:
    """
    MATLAB Engine을 이용해 Simulink 모델을 실행하고,
    결과를 pandas DataFrame 및 CSV로 변환하는 클래스.
    """

    def __init__(
        self,
        model_name: str = "Motor",
        simulink_dir: Path = SIMULINK_DIR,
        mat_file_name: str = "matlab.mat",
        start_matlab: bool = True,
    ):
        self.model_name = model_name
        self.simulink_dir = Path(simulink_dir)
        self.mat_file_name = mat_file_name
        self.eng = None

        if start_matlab:
            self.start()

    def start(self):
        """
        MATLAB Engine 시작, Simulink 폴더 이동, 변수 로드, 모델 로드.
        """

        import matlab.engine

        print("Starting MATLAB Engine...")
        self.eng = matlab.engine.start_matlab()

        print(f"Changing MATLAB directory to: {self.simulink_dir}")
        self.eng.cd(str(self.simulink_dir), nargout=0)

        mat_file = self.simulink_dir / self.mat_file_name

        if mat_file.exists():
            print(f"Loading MATLAB variables: {mat_file}")
            self.eng.load(str(mat_file), nargout=0)
        else:
            print(f"Warning: MATLAB variable file not found: {mat_file}")

        print(f"Loading Simulink model: {self.model_name}")
        self.eng.load_system(self.model_name, nargout=0)

        print("MATLAB Engine is ready.")

    def stop(self):
        """
        MATLAB Engine 종료.
        """

        if self.eng is not None:
            print("Stopping MATLAB Engine...")
            self.eng.quit()
            self.eng = None

    def run_simulation(
        self,
        kp=None,
        ki=None,
        kd=None,
        target=None,
        stop_time=10.0,
        save_log=True,
    ):
        """
        Simulink 모델을 실행하고 결과를 pandas DataFrame으로 반환한다.
        """

        # MATLAB Engine이 아직 시작되지 않았으면 시작
        if self.eng is None:
            self.start()

        # Simulink 폴더 이동
        self.eng.cd(str(self.simulink_dir), nargout=0)

        # 모델 로드
        self.eng.load_system(self.model_name, nargout=0)

        # 외부 입력 gain/target이 들어오면 MATLAB workspace에 반영
        if kp is not None:
            self.eng.workspace["Kp"] = float(kp)

        if ki is not None:
            self.eng.workspace["Ki"] = float(ki)

        if kd is not None:
            self.eng.workspace["Kd"] = float(kd)

        if target is not None:
            self.eng.workspace["target_rpm"] = float(target)

        # Stop time 설정
        self.eng.set_param(self.model_name, "StopTime", str(stop_time), nargout=0)

        print(
            f"Run Simulink: model={self.model_name}, "
            f"Kp={kp}, Ki={ki}, Kd={kd}, target={target}, stop_time={stop_time}"
        )

        sim_out = self.eng.sim(self.model_name, nargout=1)

        # MATLAB workspace에 out 이름으로 저장
        self.eng.workspace["out"] = sim_out

        df = self.get_simulink_dataframe()

        # sweep 결과에 gain 정보가 명확히 남도록 보정
        if kp is not None:
            df["kp"] = float(kp)
        if ki is not None:
            df["ki"] = float(ki)
        if kd is not None:
            df["kd"] = float(kd)

        if save_log:
            self.save_simulink_log(df)

        return df

    def _get_signal_data(self, signal_name: str):
        """
        out.<signal_name>.Data를 가져온다.

        예:
        signal_name = "sim_rpm"
        → out.sim_rpm.Data
        """

        expr = f"out.{signal_name}.Data"

        try:
            value = self.eng.eval(expr, nargout=1)
            return matlab_to_list(value)

        except Exception:
            # 혹시 To Workspace 저장 형식이 timeseries가 아니라 array일 경우 대비
            expr_fallback = f"out.{signal_name}"
            value = self.eng.eval(expr_fallback, nargout=1)
            return matlab_to_list(value)

    def _get_signal_time(self, signal_name: str):
        """
        out.<signal_name>.Time을 가져온다.
        """

        expr = f"out.{signal_name}.Time"

        try:
            value = self.eng.eval(expr, nargout=1)
            return matlab_to_list(value)

        except Exception:
            return None

    def get_simulink_dataframe(self) -> pd.DataFrame:
        """
        MATLAB workspace의 out 객체에서 Simulink 결과를 가져와 DataFrame으로 변환한다.

        Simulink output mapping:
        - out.sim_time   → time
        - out.sim_target → target
        - out.sim_rpm    → current
        - out.sim_error  → error
        - out.sim_pwm    → pwm
        """

        if self.eng is None:
            raise RuntimeError("MATLAB Engine is not started.")

        print("Extracting Simulink results from MATLAB workspace...")

        # Clock 블록을 To Workspace로 저장한 값
        time = self._get_signal_data("sim_time")

        # 주요 출력값
        target = self._get_signal_data("sim_target")
        current = self._get_signal_data("sim_rpm")
        error = self._get_signal_data("sim_error")
        pwm = self._get_signal_data("sim_pwm")

        kp = float(self.eng.eval("Kp", nargout=1))
        ki = float(self.eng.eval("Ki", nargout=1))
        kd = float(self.eng.eval("Kd", nargout=1))

        # 길이 mismatch 방지
        time, target, current, error, pwm = trim_to_min_length(
            time,
            target,
            current,
            error,
            pwm,
        )

        df = pd.DataFrame(
            {
                "mode": "simulink_pid",
                "env_type": "simulink_motor",
                "time": time,
                "target": target,
                "current": current,
                "error": error,
                "pwm": pwm,
                "kp": kp,
                "ki": ki,
                "kd": kd,
            }
        )

        df["time"] = df["time"].round(6)
        df = df.drop_duplicates(subset=["time"], keep="last").reset_index(drop=True)
        
        print("DataFrame created.")
        print(df.head())
        print(df.tail())

        return df

    def save_simulink_log(self, df: pd.DataFrame) -> Path:
        """
        Simulink 결과 DataFrame을 CSV로 저장한다.
        """

        LOG_DIR.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = LOG_DIR / f"simulink_motor_{timestamp}.csv"

        df.to_csv(save_path, index=False, encoding="utf-8-sig")

        print(f"Saved Simulink log: {save_path}")

        return save_path


# ============================================================
# Main test
# ============================================================

if __name__ == "__main__":
    runner = SimulinkRunner(model_name="Motor")

    try:
        runner.run_simulation(stop_time=10.0, save_log=True)

    finally:
        runner.stop()