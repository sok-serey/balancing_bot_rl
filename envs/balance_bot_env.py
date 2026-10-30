"""
Gymnasium environment for a two-wheeled self-balancing robot simulated in MuJoCo.

This environment wraps MuJoCo physics, modeling a two-wheeled inverted-pendulum
robot with continuous torque control, noisy/filtered IMU pitch estimation, and
curriculum-friendly reward terms.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import mujoco
import mujoco.viewer
import numpy as np

# Default model location relative to the project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MJCF_PATH = PROJECT_ROOT / "assets" / "mjcf" / "balancing_bot.xml"


class BalanceBotEnv(gym.Env):
    """
    Two-wheeled self-balancing robot environment for Gymnasium.

    Observation Space (Box(4,)):
        0: Pitch angle (rad) - estimated via complementary filter (accel + gyro)
        1: Pitch rate / angular velocity around Y axis (rad/s)
        2: Left wheel rotational velocity (rad/s)
        3: Right wheel rotational velocity (rad/s)

    Action Space (Box(2,)):
        Continuous motor effort normalized to [-1.0, 1.0], mapped linearly to
        the MuJoCo actuator control ranges for [left_motor, right_motor].
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        mjcf_path: str | Path | None = None,
        max_steps: int = 10000,
        render_mode: str | None = None,
        alpha: float = 0.99,
        sensor_imu_accel: str = "imu_accel",
        sensor_imu_gyro: str = "imu_gyro",
        sensor_left_wheel_vel: str = "left_wheel_velocity",
        sensor_right_wheel_vel: str = "right_wheel_velocity",
        actuator_left_motor: str = "left_wheel_motor",
        actuator_right_motor: str = "right_wheel_motor",
        initial_pitch_rate_range: float = 0.15,
        alive_bonus: float = 1.0,
        pitch_penalty_coef: float = 5.0,
        pitch_rate_penalty_coef: float = 0.05,
        action_penalty_coef: float = 0.01,
        position_penalty_coef: float = 0.01,
        yaw_penalty_coef: float = 0.1,
        tip_threshold_deg: float = 30.0,
    ) -> None:
        super().__init__()

        # Resolve model file path flexibly
        if mjcf_path is None:
            resolved_path = DEFAULT_MJCF_PATH
        else:
            p = Path(mjcf_path)
            if p.is_absolute() and p.is_file():
                resolved_path = p
            elif (PROJECT_ROOT / p).is_file():
                resolved_path = PROJECT_ROOT / p
            elif (PROJECT_ROOT / "assets" / "mjcf" / p).is_file():
                resolved_path = PROJECT_ROOT / "assets" / "mjcf" / p
            elif (Path(__file__).resolve().parent / p).is_file():
                resolved_path = Path(__file__).resolve().parent / p
            else:
                resolved_path = p

        if not resolved_path.exists():
            raise FileNotFoundError(
                f"MuJoCo MJCF model file not found at: {resolved_path}"
            )

        self.mjcf_path = str(resolved_path)
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)

        self.render_mode = render_mode
        self.alpha = alpha

        # Sensor names
        self.sensor_imu_accel = sensor_imu_accel
        self.sensor_imu_gyro = sensor_imu_gyro
        self.sensor_left_wheel_vel = sensor_left_wheel_vel
        self.sensor_right_wheel_vel = sensor_right_wheel_vel

        # Actuator identifiers
        self.left_motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_left_motor
        )
        self.right_motor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_right_motor
        )

        if self.left_motor_id < 0:
            raise ValueError(f"Actuator not found in MJCF: {actuator_left_motor}")
        if self.right_motor_id < 0:
            raise ValueError(f"Actuator not found in MJCF: {actuator_right_motor}")

        self.left_motor_ctrlrange = self.model.actuator_ctrlrange[self.left_motor_id].copy()
        self.right_motor_ctrlrange = self.model.actuator_ctrlrange[self.right_motor_id].copy()

        # Observation Space: [pitch (rad), pitch_rate (rad/s), left_vel (rad/s), right_vel (rad/s)]
        obs_low = np.array([-np.pi, -20.0, -50.0, -50.0], dtype=np.float32)
        obs_high = np.array([np.pi, 20.0, 50.0, 50.0], dtype=np.float32)
        self.observation_space = spaces.Box(obs_low, obs_high, dtype=np.float32)

        # Action Space: [left_motor_torque, right_motor_torque] normalized to [-1, 1]
        actions_low = np.array([-1.0, -1.0], dtype=np.float32)
        actions_high = np.array([1.0, 1.0], dtype=np.float32)
        self.action_space = spaces.Box(actions_low, actions_high, dtype=np.float32)

        # Reward coefficients
        self.alive_bonus = alive_bonus
        self.pitch_penalty_coef = pitch_penalty_coef
        self.pitch_rate_penalty_coef = pitch_rate_penalty_coef
        self.action_penalty_coef = action_penalty_coef
        self.position_penalty_coef = position_penalty_coef
        self.yaw_penalty_coef = yaw_penalty_coef

        # Episode termination parameters
        self.tip_threshold_deg = tip_threshold_deg
        self.max_steps = max_steps
        self.initial_pitch_rate_range = initial_pitch_rate_range

        # Runtime state
        self._pitch = 0.0
        self._viewer: mujoco.viewer.Handle | None = None
        self._step = 0

    def _get_obs(self) -> np.ndarray:
        """Read sensor data and estimate pitch using a complementary filter."""
        accel_x, _, accel_z = self.data.sensor(self.sensor_imu_accel).data
        pitch_rate = float(self.data.sensor(self.sensor_imu_gyro).data[1])

        # Accelerometer-derived pitch angle
        accel_pitch = -1.0 * math.atan2(accel_x, accel_z)

        # Complementary filter fusion (gyro integration + accel correction)
        self._pitch = self.alpha * (
            self._pitch + pitch_rate * self.model.opt.timestep
        ) + (1.0 - self.alpha) * accel_pitch

        wheel_vel_left = float(self.data.sensor(self.sensor_left_wheel_vel).data[0])
        wheel_vel_right = float(self.data.sensor(self.sensor_right_wheel_vel).data[0])

        return np.array(
            [self._pitch, pitch_rate, wheel_vel_left, wheel_vel_right],
            dtype=np.float32,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Reset the simulator to initial state with a small random pitch disturbance."""
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self._pitch = 0.0

        # Inject initial pitch rate perturbation around the Y axis
        self.data.qvel[4] += self.np_random.uniform(
            -self.initial_pitch_rate_range,
            self.initial_pitch_rate_range,
        )

        mujoco.mj_forward(self.model, self.data)
        self._step = 0

        return self._get_obs(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """Advance the physics by one timestep with commanded normalized torques."""
        action = np.clip(action, self.action_space.low, self.action_space.high)

        # Map normalized [-1, 1] actions into actuator control limits
        self.data.ctrl[self.left_motor_id] = float(
            np.interp(action[0], [-1.0, 1.0], self.left_motor_ctrlrange)
        )
        self.data.ctrl[self.right_motor_id] = float(
            np.interp(action[1], [-1.0, 1.0], self.right_motor_ctrlrange)
        )

        mujoco.mj_step(self.model, self.data)
        self._step += 1

        obs = self._get_obs()
        pitch = obs[0]
        pitch_rate = obs[1]

        # Reward structure:
        #   + alive_bonus for keeping upright
        #   - pitch^2 penalty
        #   - pitch_rate^2 penalty
        #   - action^2 penalty (discourage erratic motor chattering)
        #   - position penalty (drift from origin)
        #   - yaw penalty (rotation around Z axis)
        pitch_penalty = self.pitch_penalty_coef * (pitch**2)
        pitch_rate_penalty = self.pitch_rate_penalty_coef * (pitch_rate**2)
        action_penalty = self.action_penalty_coef * float(np.sum(action**2))

        x_pos = float(self.data.qpos[0])
        y_pos = float(self.data.qpos[1])
        position_penalty = self.position_penalty_coef * (x_pos**2 + y_pos**2)

        yaw_rate = float(self.data.qvel[5])
        yaw_penalty = self.yaw_penalty_coef * abs(yaw_rate)

        reward = (
            self.alive_bonus
            - pitch_penalty
            - pitch_rate_penalty
            - action_penalty
            - position_penalty
            - yaw_penalty
        )

        terminated = bool(abs(pitch) > math.radians(self.tip_threshold_deg))
        truncated = bool(self._step >= self.max_steps)

        return obs, float(reward), terminated, truncated, {}

    def render(self) -> None:
        """Render the current simulation state via MuJoCo passive viewer."""
        if self.render_mode != "human":
            return

        if self._viewer is None:
            self._viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self._viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            self._viewer.cam.trackbodyid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link"
            )
            self._viewer.cam.lookat[:] = [0.0, 0.0, 0.05]
            self._viewer.cam.distance = 0.8
            self._viewer.cam.azimuth = 45.0
            self._viewer.cam.elevation = -25.0

        self._viewer.sync()

    def close(self) -> None:
        """Clean up viewer resources."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

