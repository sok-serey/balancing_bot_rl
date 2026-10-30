"""
Evaluate a trained PPO policy in the MuJoCo balancing bot simulation.

Supports evaluating:
  - PyTorch checkpoints (.cleanrl_model or .pt)
  - Exported ONNX actors (.onnx) via onnxruntime
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import torch

# Add repository root to python search path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.balance_bot_env import BalanceBotEnv
from rl.ppo import build_mlp


def find_latest_checkpoint() -> Path:
    """Find the best or most recent checkpoint under checkpoints/ or runs/."""
    candidates = []
    checkpoints_dir = PROJECT_ROOT / "checkpoints"
    runs_dir = PROJECT_ROOT / "runs"

    if checkpoints_dir.exists():
        candidates.extend(checkpoints_dir.glob("*.cleanrl_model"))
        candidates.extend(checkpoints_dir.glob("*.onnx"))

    if runs_dir.exists():
        candidates.extend(runs_dir.glob("**/*.cleanrl_model"))
        candidates.extend(runs_dir.glob("**/actor.onnx"))

    if not candidates:
        raise FileNotFoundError(
            f"No checkpoints or exported models found in {checkpoints_dir} or {runs_dir}."
        )

    # Prefer 'best_model.cleanrl_model' or newest file
    best_models = [c for c in candidates if "best" in c.name]
    if best_models:
        return max(best_models, key=lambda p: p.stat().st_mtime)
    return max(candidates, key=lambda p: p.stat().st_mtime)


def evaluate_onnx(model_path: Path, env: BalanceBotEnv, episodes: int, seed: int, realtime: bool):
    try:
        import onnxruntime as ort
    except ImportError:
        raise ImportError(
            "onnxruntime is required to run .onnx models. Install with: pip install onnxruntime"
        )

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    episode_returns = []
    print(f"Loaded ONNX model: {model_path}")
    print("Beginning evaluation...")

    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        ep_return = 0.0
        ep_steps = 0

        while True:
            step_start = time.perf_counter()
            action = session.run([output_name], {input_name: obs.reshape(1, -1)})[0][0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            ep_steps += 1

            if env.render_mode == "human":
                env.render()
                if realtime:
                    slack = env.model.opt.timestep - (time.perf_counter() - step_start)
                    if slack > 0:
                        time.sleep(slack)

            if terminated or truncated:
                print(
                    f"Episode {ep + 1}/{episodes}: "
                    f"Return = {ep_return:.2f}, Steps = {ep_steps}, "
                    f"Tipped = {terminated}"
                )
                episode_returns.append(ep_return)
                break

    return episode_returns


def evaluate_pytorch(
    model_path: Path,
    env: BalanceBotEnv,
    episodes: int,
    seed: int,
    realtime: bool,
    hidden_layers: int = 2,
    hidden_size: int = 16,
):
    obs_size = int(np.prod(env.observation_space.shape))
    action_size = int(np.prod(env.action_space.shape))

    actor = build_mlp(
        input_size=obs_size,
        output_size=action_size,
        num_hidden_layers=hidden_layers,
        hidden_layer_size=hidden_size,
        output_std=1.0,
    )

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    actor_weights = {
        k.replace("actor_mean.", ""): v
        for k, v in state_dict.items()
        if k.startswith("actor_mean.")
    }
    if not actor_weights:
        # State dict might be the actor weights directly
        actor_weights = state_dict

    actor.load_state_dict(actor_weights)
    actor.eval()

    episode_returns = []
    print(f"Loaded PyTorch checkpoint: {model_path}")
    print("Beginning evaluation...")

    with torch.no_grad():
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            ep_return = 0.0
            ep_steps = 0

            while True:
                step_start = time.perf_counter()
                obs_tensor = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = actor(obs_tensor).squeeze(0).numpy()
                obs, reward, terminated, truncated, _ = env.step(action)
                ep_return += reward
                ep_steps += 1

                if env.render_mode == "human":
                    env.render()
                    if realtime:
                        slack = env.model.opt.timestep - (time.perf_counter() - step_start)
                        if slack > 0:
                            time.sleep(slack)

                if terminated or truncated:
                    print(
                        f"Episode {ep + 1}/{episodes}: "
                        f"Return = {ep_return:.2f}, Steps = {ep_steps}, "
                        f"Tipped = {terminated}"
                    )
                    episode_returns.append(ep_return)
                    break

    return episode_returns


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO policy in MuJoCo simulation."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=None,
        help="Path to .cleanrl_model or .onnx (default: automatically finds newest best checkpoint)",
    )
    parser.add_argument("--episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Open interactive 3D MuJoCo viewer window",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        default=True,
        help="Pace simulation to real-time timestep",
    )
    parser.add_argument(
        "--actor-hidden-layers",
        type=int,
        default=2,
        help="Number of hidden layers (default: 2)",
    )
    parser.add_argument(
        "--actor-hidden-size",
        type=int,
        default=16,
        help="Hidden layer dimension (default: 16)",
    )
    args = parser.parse_args()

    model_path = args.model.resolve() if args.model else find_latest_checkpoint()
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    env = BalanceBotEnv(
        render_mode="human" if args.viewer else None,
        initial_pitch_rate_range=0.08,
    )

    try:
        if model_path.suffix == ".onnx":
            returns = evaluate_onnx(
                model_path, env, args.episodes, args.seed, args.realtime
            )
        else:
            returns = evaluate_pytorch(
                model_path,
                env,
                args.episodes,
                args.seed,
                args.realtime,
                args.actor_hidden_layers,
                args.actor_hidden_size,
            )

        print("====================================================")
        print(f"Evaluation finished over {len(returns)} episodes.")
        print(f"Mean Return: {np.mean(returns):.2f} ± {np.std(returns):.2f}")
        print("====================================================")
    finally:
        env.close()


if __name__ == "__main__":
    main()

