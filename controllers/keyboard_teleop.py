"""
Keyboard teleoperation for the balancing robot in MuJoCo simulation.

Use arrow keys to drive the robot forward/backward and turn in place.
Releasing arrow keys triggers position-hold mode.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import threading
import time

import mujoco
import mujoco.viewer

try:
    from pynput import keyboard
except ImportError:
    keyboard = None

from controllers.pid_controller import (
    NATURAL_PITCH,
    PIDController,
    PositionHoldController,
    clamp,
    get_id,
    read_sensor,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "assets" / "mjcf" / "balancing_bot.xml"

DRIVE_SPEED = 0.9
TURN_RATE = 1.5
YAW_RATE_GAIN = 0.15
MAX_TURN_TORQUE = 0.1


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Keyboard teleoperation for the balancing robot."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Path to MuJoCo XML model",
    )
    args = parser.parse_args()

    if keyboard is None:
        raise ImportError(
            "pynput is required for keyboard teleoperation. Install it with: pip install pynput"
        )

    model_path = args.model.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model not found at: {model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    base_id = get_id(model, mujoco.mjtObj.mjOBJ_BODY, "base_link")
    root_joint_id = get_id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    left_motor_id = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "left_wheel_motor")
    right_motor_id = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "right_wheel_motor")
    root_qpos = int(model.jnt_qposadr[root_joint_id])
    root_dof = int(model.jnt_dofadr[root_joint_id])

    theta_0 = NATURAL_PITCH
    model.qpos0[root_qpos + 2] = 0.103
    model.qpos0[root_qpos + 3 : root_qpos + 7] = [
        math.cos(theta_0 / 2),
        0.0,
        math.sin(theta_0 / 2),
        0.0,
    ]
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    balance_controller = PIDController()
    position_controller = PositionHoldController(
        position_gain=3.0,
        velocity_gain=0.9,
        max_velocity=0.9,
        max_pitch=math.radians(7.0),
        max_pitch_rate=math.radians(60.0),
    )
    target_x = 0.0
    target_y = 0.0
    next_report_time = 0.0

    pressed_keys = set()
    key_lock = threading.Lock()
    drive_keys = {
        keyboard.Key.up,
        keyboard.Key.down,
        keyboard.Key.left,
        keyboard.Key.right,
    }

    def on_press(key) -> None:
        if key in drive_keys:
            with key_lock:
                pressed_keys.add(key)

    def on_release(key) -> None:
        with key_lock:
            pressed_keys.discard(key)

    print("====================================================")
    print("Balancing Bot Keyboard Teleoperation")
    print("Hold Arrow Keys to Drive (UP/DOWN: drive, LEFT/RIGHT: turn)")
    print("Release keys to activate position-hold balancing.")
    print("====================================================")

    with keyboard.Listener(on_press=on_press, on_release=on_release):
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running():
                step_started = time.perf_counter()
                report = None

                with key_lock:
                    forward = int(keyboard.Key.up in pressed_keys) - int(
                        keyboard.Key.down in pressed_keys
                    )
                    turn = int(keyboard.Key.left in pressed_keys) - int(
                        keyboard.Key.right in pressed_keys
                    )

                with viewer.lock():
                    qw, qx, qy, qz = data.xquat[base_id]
                    sin_pitch = 2.0 * (qw * qy - qz * qx)
                    pitch = math.asin(clamp(float(sin_pitch), -1.0, 1.0))
                    yaw = math.atan2(
                        2.0 * (qw * qz + qx * qy),
                        1.0 - 2.0 * (qy * qy + qz * qz),
                    )

                    dt = model.opt.timestep
                    target_x += forward * DRIVE_SPEED * dt * math.cos(yaw)
                    target_y += forward * DRIVE_SPEED * dt * math.sin(yaw)

                    base_x, base_y = data.xpos[base_id][:2]
                    x_velocity, y_velocity = data.qvel[root_dof : root_dof + 2]
                    forward_position_error = (
                        (target_x - base_x) * math.cos(yaw)
                        + (target_y - base_y) * math.sin(yaw)
                    )
                    forward_velocity = (
                        x_velocity * math.cos(yaw) + y_velocity * math.sin(yaw)
                    )
                    target_pitch = position_controller.update(
                        float(forward_position_error),
                        float(forward_velocity),
                        dt,
                    )

                    imu_gyro = read_sensor(model, data, "imu_gyro")
                    balance_torque = balance_controller.update(
                        pitch - target_pitch,
                        float(imu_gyro[1]),
                        dt,
                    )
                    desired_yaw_rate = turn * TURN_RATE
                    turn_torque = clamp(
                        YAW_RATE_GAIN * (desired_yaw_rate - float(imu_gyro[2])),
                        -MAX_TURN_TORQUE,
                        MAX_TURN_TORQUE,
                    )

                    left_range = model.actuator_ctrlrange[left_motor_id]
                    right_range = model.actuator_ctrlrange[right_motor_id]
                    data.ctrl[left_motor_id] = clamp(
                        balance_torque + turn_torque,
                        float(left_range[0]),
                        float(left_range[1]),
                    )
                    data.ctrl[right_motor_id] = clamp(
                        -balance_torque + turn_torque,
                        float(right_range[0]),
                        float(right_range[1]),
                    )

                    mujoco.mj_step(model, data)
                    if data.time >= next_report_time:
                        report = (
                            data.time,
                            base_x,
                            base_y,
                            math.degrees(pitch),
                            math.degrees(target_pitch),
                            balance_torque,
                            turn_torque,
                        )
                        next_report_time = data.time + 0.2

                viewer.sync()

                if report is not None:
                    (
                        sim_time,
                        base_x,
                        base_y,
                        pitch_degrees,
                        target_degrees,
                        requested_balance,
                        requested_turn,
                    ) = report
                    print(
                        f"t={sim_time:.2f}s | pos=({base_x:.3f}, {base_y:.3f}) | "
                        f"pitch={pitch_degrees:.2f}° (target {target_degrees:.2f}°) | "
                        f"balance_torque={requested_balance:.2f} Nm | "
                        f"turn_torque={requested_turn:.2f} Nm"
                    )

                remaining = model.opt.timestep - (time.perf_counter() - step_started)
                if remaining > 0:
                    time.sleep(remaining)


if __name__ == "__main__":
    main()
