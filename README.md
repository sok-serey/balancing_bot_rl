# Two-Wheeled Balancing Robot — Reinforcement Learning with PPO & MuJoCo

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![MuJoCo](https://img.shields.io/badge/physics-MuJoCo%203.x-orange.svg)](https://mujoco.org/)
[![Gymnasium](https://img.shields.io/badge/env-Gymnasium-brightgreen.svg)](https://gymnasium.farama.org/)
[![PyTorch](https://img.shields.io/badge/framework-PyTorch-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A complete, production-ready framework for training, evaluating, and deploying a continuous-action Proximal Policy Optimization (**PPO**) agent to balance a custom two-wheeled inverted-pendulum robot in **MuJoCo**.

Includes full SolidWorks CAD sources, exported URDF/STL models, a Gymnasium environment with complementary filter IMU estimation, classical PID baselines, keyboard teleoperation, curriculum training notebooks, and pure C-header export for microcontroller deployment.

---

## Architecture & Folder Structure

```
balancing_bot_rl/
├── assets/
│   ├── cad/                          # SolidWorks CAD sources (.SLDASM, .SLDPRT)
│   │   ├── balancing_bot_assembly.SLDASM
│   │   ├── Body.SLDPRT
│   │   ├── wheel.SLDPRT
│   │   └── ros_package/             # ROS URDF export package (CMakeLists, package.xml, launch)
│   ├── meshes/                       # High-resolution STL 3D meshes
│   │   ├── base_link.STL
│   │   ├── imu_link.STL
│   │   ├── left_wheel_link.STL
│   │   └── right_wheel_link.STL
│   ├── mjcf/                         # MuJoCo XML simulation descriptions
│   │   ├── balancing_bot.xml         # Primary simulation model with actuators & sensors
│   │   └── test_patch.xml            # Debug / collision patch model
│   └── urdf/                         # Standard ROS URDF model & joint specifications
│       ├── bot_balancing_urdf.urdf
│       └── bot_balancing_urdf.csv
│
├── checkpoints/                      # Pretrained model weights & exported policies
│   ├── best_model.cleanrl_model      # Saved PyTorch actor-critic weights
│   └── README.md
│
├── controllers/                      # Baseline and interactive controllers
│   ├── __init__.py
│   ├── pid_controller.py             # Cascaded PID baseline (Position + Pitch stabilization)
│   └── keyboard_teleop.py            # Real-time keyboard teleoperation in MuJoCo viewer
│
├── envs/                             # Gymnasium environment
│   ├── __init__.py                   # Registers 'BalancingBot-v0'
│   └── balance_bot_env.py            # Physics wrapper, IMU filtering, and reward calculation
│
├── notebooks/                        # Interactive Jupyter notebooks
│   ├── train_curriculum.ipynb        # 2-Phase curriculum training workflow
│   └── env_testing.ipynb             # Environment sanity checks and space validation
│
├── rl/                               # PPO algorithm implementation & export tools
│   ├── __init__.py
│   ├── ppo.py                        # CleanRL PPO agent, vectorized rollouts, GAE, and logger
│   └── onnx_to_c.py                  # Standalone compiler: ONNX actor -> zero-dependency C header
│
├── scripts/                          # Command-line entrypoints
│   ├── train.py                      # CLI training script with hyperparameter flags
│   ├── evaluate.py                   # Evaluate checkpoints/ONNX models with metrics
│   └── view_robot.py                 # Interactive MuJoCo viewer & live telemetry stream
│
├── .gitignore                        # Git rules (ignores runs/, logs/, venv/, caches)
├── pyproject.toml                    # Standard pip packaging config (`pip install -e .`)
├── requirements.txt                  # Python dependencies
└── README.md                         # Documentation and user guide
```

---

## Robot Specifications

The physical characteristics are derived directly from the SolidWorks CAD model:

| Parameter | Value | Notes |
| :--- | :--- | :--- |
| **Total Mass** | ~0.665 kg | Base link: 0.475 kg, Left/Right wheels: 0.095 kg each |
| **Center of Mass (CoM)** | -21.1 mm (X offset) | Center of mass is located behind the wheel axle |
| **Natural Balanced Pitch** | **+17.69°** (+0.3087 rad) | Pitch angle where CoM sits directly above wheel axle |
| **Actuators** | Continuous Torque | Left & right wheel motors with normalized control limits |
| **Sensors** | 6-Axis IMU + Encoders | Accelerometer (m/s²), Gyroscope (rad/s), Wheel velocity & position |
| **Simulation Step** | 0.002 s (500 Hz) | Fast implicit integrator (`implicitfast`) |

---

## Quickstart & Installation

### 1. Clone the repository
```bash
git clone https://github.com/sokserey215/balancing_bot_rl.git
cd balancing_bot_rl
```

### 2. Set up virtual environment
```bash
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
# Optional: install project in editable mode
pip install -e .
```

---

## Gymnasium Environment (`BalancingBot-v0`)

The custom Gymnasium environment simulates the robot and fuses noisy raw sensor readings into state estimates using a complementary filter:

### Observation Space (`Box(4,)`)
1. **$\theta$ (Pitch)**: Body lean angle in radians, fused from accelerometer and gyroscope ($\alpha = 0.99$).
2. **$\dot{\theta}$ (Pitch Rate)**: Angular velocity about the Y-axis from the IMU gyroscope (rad/s).
3. **$\dot{\phi}_{\text{left}}$ (Left Wheel Velocity)**: Rotational speed of the left wheel (rad/s).
4. **$\dot{\phi}_{\text{right}}$ (Right Wheel Velocity)**: Rotational speed of the right wheel (rad/s).

### Action Space (`Box(2,)`)
Normalized continuous actions in `[-1.0, 1.0]`, mapped to the physical motor torque ranges of the left and right wheel actuators.

### Reward Function
$$R_t = R_{\text{alive}} - c_1 \theta^2 - c_2 \dot{\theta}^2 - c_3 \|a\|^2 - c_4 (x^2 + y^2) - c_5 |\dot{\psi}|$$

- $R_{\text{alive}}$: Bonus for every step the robot stays upright.
- $c_1 \theta^2$: Quadratic penalty on deviation from balance angle.
- $c_2 \dot{\theta}^2$: Damping penalty to minimize oscillations.
- $c_3 \|a\|^2$: Penalty on excessive control effort to prevent motor chattering.
- $c_4 (x^2 + y^2)$: Spatial penalty to discourage drifting from the origin.
- $c_5 |\dot{\psi}|$: Yaw penalty to discourage spinning around the vertical axis.

---

## Training with PPO

### Method 1: Interactive Curriculum Notebook (Recommended)
Open `notebooks/train_curriculum.ipynb` in Jupyter or VS Code:
- **Phase 1 (Upright Balance)**: Trains for 500,000 steps focusing on upright stabilization with recovery from random pitch disturbances.
- **Phase 2 (Position & Yaw Hold)**: Warms up from Phase 1 and adds penalties for translation drift and yaw spin.
- Automatically exports checkpoints, ONNX models, and TensorBoard CSV curves.

```bash
jupyter notebook notebooks/train_curriculum.ipynb
```

### Method 2: Command-Line Interface
Train directly with customizable hyperparameters:
```bash
# Fast training with 4 parallel environments
python scripts/train.py --total-timesteps 200000 --num-envs 4

# Train with live 3D MuJoCo viewer enabled
python scripts/train.py --total-timesteps 100000 --num-envs 2 --viewer
```

Monitor training progress with TensorBoard:
```bash
tensorboard --logdir runs
```

---

## Evaluation & Testing

Evaluate a saved checkpoint in the MuJoCo viewer:

```bash
# Evaluate PyTorch checkpoint (opens 3D viewer)
python scripts/evaluate.py --model checkpoints/best_model.cleanrl_model --viewer --episodes 5

# Fast headless benchmark
python scripts/evaluate.py --model checkpoints/best_model.cleanrl_model --episodes 10
```

---

## Baselines & Teleoperation

### 1. Interactive 3D Viewer & Sensor Telemetry
Inspect robot kinematics, contacts, and real-time sensor streams:
```bash
python scripts/view_robot.py
```

### 2. Classical Cascaded PID Controller
Runs a deterministic PID baseline using position-hold and pitch damping:
```bash
python controllers/pid_controller.py
# Or run headless:
python controllers/pid_controller.py --headless --duration 5.0
```

### 3. Keyboard Teleoperation
Drive the balancing robot in the simulation using the arrow keys:
```bash
python controllers/keyboard_teleop.py
```
- **UP / DOWN**: Drive forward / backward
- **LEFT / RIGHT**: Turn in place
- **Release keys**: Automatic position-hold stabilization

---

## Embedded Hardware Deployment

Once trained, the policy can be compiled directly into a self-contained C header file for embedded microcontrollers (ESP32, Teensy, STM32, Arduino) with **zero external inference dependencies**:

```bash
# Convert trained ONNX actor to pure C header
python rl/onnx_to_c.py runs/<run_name>/actor.onnx firmware/actor.h
```

The resulting `actor.h` contains static weight matrices and an optimized float-precision forward pass function ready to run at high frequencies (>200 Hz) on embedded hardware.

---

## Contributing & Development

Contributions, bug reports, and hardware testing feedback are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

Distributed under the MIT License. See `LICENSE` for more information.
