"""
Train the balancing bot using Proximal Policy Optimization (PPO).

Command-line entrypoint supporting curriculum phases, hyperparameter overrides,
and automatic checkpointing.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import envs
from envs.balance_bot_env import BalanceBotEnv
from rl.ppo import PPOConfig, export_actor_onnx, export_tb_plots_as_csv, train


def make_env_thunk(render: bool = False, **env_kwargs):
    def thunk():
        env = BalanceBotEnv(render_mode="human" if render else None, **env_kwargs)
        return gym.wrappers.RecordEpisodeStatistics(env)

    return thunk


def build_vector_envs(num_envs: int, render: bool = False, **env_kwargs):
    return gym.vector.SyncVectorEnv(
        [
            make_env_thunk(render=(render and idx == 0), **env_kwargs)
            for idx in range(num_envs)
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train balancing bot with PPO.")
    parser.add_argument("--exp-name", type=str, default="balancing-bot-ppo", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--total-timesteps", type=int, default=100000, help="Total environment steps")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Adam learning rate")
    parser.add_argument("--num-steps", type=int, default=2048, help="Rollout steps per env")
    parser.add_argument("--num-minibatches", type=int, default=32, help="Number of minibatches")
    parser.add_argument("--update-epochs", type=int, default=10, help="PPO update epochs per rollout")
    parser.add_argument("--checkpoint-interval", type=int, default=25, help="Iterations between checkpoints")
    parser.add_argument("--viewer", action="store_true", help="Render first env in live viewer")
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available")
    parser.add_argument("--export-onnx", action="store_true", default=True, help="Export actor to ONNX after training")
    args = parser.parse_args()

    config = PPOConfig(
        exp_name=args.exp_name,
        env_id="BalancingBot-v0",
        seed=args.seed,
        cuda=args.cuda,
        num_envs=args.num_envs,
        actor_hidden_layers=2,
        actor_hidden_size=16,
        critic_hidden_layers=2,
        critic_hidden_size=16,
        total_timesteps=args.total_timesteps,
        num_steps=args.num_steps,
        num_minibatches=args.num_minibatches,
        update_epochs=args.update_epochs,
        anneal_lr=True,
        learning_rate=args.learning_rate,
        gamma=0.99,
        gae_lambda=0.95,
        clip_coef=0.2,
        value_clip=1.0,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        checkpoint_interval=args.checkpoint_interval,
        save_model=True,
        timestep=0.002 if args.viewer else None,
    )

    print("====================================================")
    print(f"Starting training run: {config.exp_name}")
    print(f"Total timesteps: {config.total_timesteps} across {config.num_envs} environments")
    print("====================================================")

    envs_vec = build_vector_envs(
        args.num_envs,
        render=args.viewer,
        initial_pitch_rate_range=0.08,
        pitch_penalty_coef=5.0,
        pitch_rate_penalty_coef=0.2,
        action_penalty_coef=0.01,
        position_penalty_coef=0.0,
        yaw_penalty_coef=0.0,
    )

    result = train(config, envs=envs_vec)

    print("====================================================")
    print("Training finished!")
    print(f"Checkpoint directory: {result.checkpoint_dir}")
    print(f"Best model path: {result.best_model_path}")
    print(f"Final model path: {result.final_model_path}")
    print("====================================================")

    if args.export_onnx:
        model_to_export = result.best_model_path or result.final_model_path
        if model_to_export is not None:
            onnx_path = result.checkpoint_dir / "actor.onnx"
            obs_size = int(np.prod(envs_vec.single_observation_space.shape))
            action_size = int(np.prod(envs_vec.single_action_space.shape))
            export_actor_onnx(
                model_path=model_to_export,
                output_path=onnx_path,
                obs_size=obs_size,
                action_size=action_size,
                num_hidden_layers=config.actor_hidden_layers,
                hidden_layer_size=config.actor_hidden_size,
            )
            print(f"Exported ONNX actor to: {onnx_path}")

    export_tb_plots_as_csv(result.checkpoint_dir)
    envs_vec.close()


if __name__ == "__main__":
    main()
