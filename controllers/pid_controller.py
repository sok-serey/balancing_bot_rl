"""
Classical PID controller baseline for the two-wheeled self-balancing robot.

Implements a cascaded controller:
  1. Outer loop: Position-hold / velocity control predicting desired tilt angle
  2. Inner loop: Pitch angle & pitch-rate PID controller commanding wheel torques
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time

import mujoco
import mujoco.viewer

# In the CAD model, the center of mass is 21.1 mm behind the wheel axle.
# At +17.69 degrees pitch, the center of mass aligns vertically directly over the wheels.
NATURAL_PITCH: float = math.radians(17.69)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "mjcf" / "balancing_bot.xml"


def clamp(value: float, lower: float, upper: float) -> float:
    """Clamp value between lower and upper bounds."""
    return max(lower, min(upper, value))


def get_id(model: mujoco.MjModel, object_type: mujoco.mjtObj, name: str) -> int:
    """Look up an object ID in MuJoCo model, raising ValueError if not found."""
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return object_id


def read_sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str):
    """Read full sensor vector from MuJoCo data."""
    sensor_id = get_id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    address = model.sensor_adr[sensor_id]
    dimension = model.sensor_dim[sensor_id]
    return data.sensordata[address : address + dimension]


@dataclass
class PIDController:
    """PID controller for pitch error and damping."""

    kp: float = 18.0  # Proportional gain for pitch error
    ki: float = 0.1   # Integral gain for steady-state bias
    kd: float = 1.2   # Derivative gain for damping oscillations
    integral_limit: float = 0.5
    integral_error: float = 0.0

    def update(self, error: float, error_rate: float, dt: float) -> float:
        self.integral_error += error * dt
        self.integral_error = clamp(
            self.integral_error, -self.integral_limit, self.integral_limit
        )
        return (
            self.kp * error
            + self.ki * self.integral_error
            + self.kd * error_rate
        )


@dataclass
class PositionHoldController:
    """Outer loop controller translating horizontal position drift into target pitch."""

    position_gain: float = 0.8
    position_integral_gain: float = 0.01
    velocity_gain: float = 0.25
    max_velocity: float = 0.3
    max_pitch: float = math.radians(5.0)
    max_pitch_rate: float = math.radians(15.0)
    integral_limit: float = 0.2
    position_integral: float = 0.0
    target_pitch: float = NATURAL_PITCH

    def update(self, position_error: float, x_velocity: float, dt: float) -> float:
        requested_velocity = (
            self.position_gain * position_error
            + self.position_integral_gain * self.position_integral
        )
        target_velocity = clamp(
            requested_velocity,
            -self.max_velocity,
            self.max_velocity,
        )
        if abs(requested_velocity) < self.max_velocity:
            self.position_integral = clamp(
                self.position_integral + position_error * dt,
                -self.integral_limit,
                self.integral_limit,
            )
        velocity_error = target_velocity - x_velocity
        desired_offset = clamp(
            self.velocity_gain * velocity_error,
            -self.max_pitch,
            self.max_pitch,
        )
        desired_pitch = NATURAL_PITCH + desired_offset
        max_pitch_change = self.max_pitch_rate * dt
        self.target_pitch += clamp(
            desired_pitch - self.target_pitch,
            -max_pitch_change,
            max_pitch_change,
        )
        return self.target_pitch


def run_pid_simulation(
    model_path: Path | str = DEFAULT_MODEL_PATH,
    headless: bool = False,
    duration: float = 10.0,
) -> None:
    """Run simulation using the PID controller."""
    resolved_path = Path(model_path).resolve()
    if not resolved_path.exists():
        raise FileNotFoundError(f"Model file not found: {resolved_path}")

    model = mujoco.MjModel.from_xml_path(str(resolved_path))
    data = mujoco.MjData(model)

    base_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    root_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    left_motor_id = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_motor")
    right_motor_id = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_motor")

    # Set default reset pose to natural balanced lean angle
    theta_0 = NATURAL_PITCH
    root_qpos_adr = model.jnt_qposadr[root_joint_id]
    model.qpos0[root_qpos_adr + 2] = 0.103
    model.qpos0[root_qpos_adr + 3 : root_qpos_adr + 7] = [
        math.cos(theta_0 / 2),
        0.0,
        math.sin(theta_0 / 2),
        0.0,
    ]
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    controller = PIDController()
    position_controller = PositionHoldController()
    next_report_time = 0.0
    target_x = 0.0
    root_x_dof = int(model.jnt_dofadr[root_joint_id])

    def step_controller():
        nonlocal next_report_time
        report = None
        qw, qx, qy, qz = data.xquat[base_id]
        sin_pitch = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(clamp(float(sin_pitch), -1.0, 1.0))
        pitch_rate = float(read_sensor(model, data, "imu_gyro")[1])
        base_x = float(data.xpos[base_id][0])
        base_x_velocity = float(data.qvel[root_x_dof])

        target_pitch = position_controller.update(
            target_x - base_x,
            base_x_velocity,
            model.opt.timestep,
        )

        error = target_pitch - pitch
        torque = controller.update(error, -pitch_rate, model.opt.timestep)

        left_range = model.actuator_ctrlrange[left_motor_id]
        right_range = model.actuator_ctrlrange[right_motor_id]

        data.ctrl[left_motor_id] = clamp(
            torque, float(left_range[0]), float(left_range[1])
        )
        data.ctrl[right_motor_id] = clamp(
            torque, float(right_range[0]), float(right_range[1])
        )

        mujoco.mj_step(model, data)
        if data.time >= next_report_time:
            report = (
                data.time,
                base_x,
                math.degrees(pitch),
                math.degrees(target_pitch),
                torque,
            )
            next_report_time = data.time + 0.5
        return report

    if headless:
        print(f"Running PID simulation headless for {duration:.1f}s...")
        while data.time < duration:
            report = step_controller()
            if report is not None:
                sim_time, base_x, pitch_deg, target_deg, torque = report
                print(
                    f"t={sim_time:.2f}s | x={base_x:.3f}m | "
                    f"pitch={pitch_deg:.2f}° (target {target_deg:.2f}°) | torque={torque:.2f}Nm"
                )
        print("Headless PID simulation completed successfully.")
        return

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running() and (duration <= 0 or data.time < duration):
            step_started = time.perf_counter()
            with viewer.lock():
                report = step_controller()
            viewer.sync()

            if report is not None:
                sim_time, base_x, pitch_deg, target_deg, torque = report
                print(
                    f"t={sim_time:.2f}s | x={base_x:.3f}m | "
                    f"pitch={pitch_deg:.2f}° (target {target_deg:.2f}°) | torque={torque:.2f}Nm"
                )

            remaining = model.opt.timestep - (time.perf_counter() - step_started)
            if remaining > 0:
                time.sleep(remaining)


def main() -> None:
    parser = argparse.ArgumentParser(description="Test PID balance controller.")
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to MuJoCo XML model",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without opening MuJoCo viewer window",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=0.0,
        help="Run duration in seconds (0 = run indefinitely until viewer is closed)",
    )
    args = parser.parse_args()
    run_pid_simulation(args.model, headless=args.headless, duration=args.duration)


if __name__ == "__main__":
    main()
