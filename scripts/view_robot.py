"""
Visualizer tool to inspect the balancing bot and print live sensor telemetry.

Launches the MuJoCo passive viewer, runs the physics stepping, and prints IMU
accelerometer, gyroscope, and wheel encoder readings.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import mujoco
import mujoco.viewer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "mjcf" / "balancing_bot.xml"


def read_sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    """Read full sensor vector from MuJoCo data by sensor name."""
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise ValueError(f"Sensor not found: {name}")

    address = model.sensor_adr[sensor_id]
    dimension = model.sensor_dim[sensor_id]
    return data.sensordata[address : address + dimension].copy()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the robot in MuJoCo viewer and stream sensor telemetry."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to MuJoCo XML model",
    )
    parser.add_argument(
        "--report-interval",
        type=float,
        default=0.2,
        help="Sensor reporting interval in seconds",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found at: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    next_report_time = 0.0

    print("====================================================")
    print(f"Loading robot model: {model_path.name}")
    print("Close the MuJoCo viewer window to exit.")
    print("====================================================")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_started = time.perf_counter()
            mujoco.mj_step(model, data)
            viewer.sync()

            if data.time >= next_report_time:
                left_position = read_sensor(model, data, "left_wheel_position")[0]
                left_velocity = read_sensor(model, data, "left_wheel_velocity")[0]
                right_position = read_sensor(model, data, "right_wheel_position")[0]
                right_velocity = read_sensor(model, data, "right_wheel_velocity")[0]
                gyro = read_sensor(model, data, "imu_gyro")
                acceleration = read_sensor(model, data, "imu_accel")
                print(
                    f"t={data.time:.2f}s | "
                    f"encoders: left={left_position:.3f} rad, {left_velocity:.3f} rad/s; "
                    f"right={right_position:.3f} rad, {right_velocity:.3f} rad/s | "
                    f"gyro={gyro.round(3)} rad/s | "
                    f"accel={acceleration.round(3)} m/s^2"
                )
                next_report_time = data.time + args.report_interval

            remaining = model.opt.timestep - (time.perf_counter() - step_started)
            if remaining > 0:
                time.sleep(remaining)


if __name__ == "__main__":
    main()
