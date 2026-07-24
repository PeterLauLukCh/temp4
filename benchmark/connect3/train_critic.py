#!/usr/bin/env python3
"""Train an OpenSpiel AlphaZero critic for 4x4 Connect-3."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


GAME = "connect_four(rows=4,columns=4,x_in_row=3)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--checkpoint-freq", type=int, default=20)
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--evaluators", type=int, default=0)
    parser.add_argument("--max-simulations", type=int, default=64)
    parser.add_argument("--uct-c", type=float, default=1.41)
    parser.add_argument("--train-batch-size", type=int, default=512)
    parser.add_argument("--replay-buffer-size", type=int, default=8192)
    parser.add_argument("--replay-buffer-reuse", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--policy-epsilon", type=float, default=0.25)
    parser.add_argument("--policy-alpha", type=float, default=1.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--temperature-drop", type=int, default=4)
    parser.add_argument("--nn-model", choices=("mlp", "conv2d", "resnet"), default="resnet")
    parser.add_argument("--nn-width", type=int, default=64)
    parser.add_argument("--nn-depth", type=int, default=3)
    parser.add_argument("--nn-api", choices=("linen", "nnx"), default="linen")
    parser.add_argument("--eval-levels", type=int, default=3)
    parser.add_argument("--evaluation-window", type=int, default=50)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Permit CPU execution. Intended only for a tiny smoke test.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "max_steps": args.max_steps,
        "checkpoint_freq": args.checkpoint_freq,
        "actors": args.actors,
        "max_simulations": args.max_simulations,
        "train_batch_size": args.train_batch_size,
        "replay_buffer_size": args.replay_buffer_size,
        "replay_buffer_reuse": args.replay_buffer_reuse,
        "nn_width": args.nn_width,
        "nn_depth": args.nn_depth,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.evaluators < 0:
        raise ValueError("--evaluators must be nonnegative")
    if args.eval_levels < 2:
        raise ValueError("--eval-levels must be at least 2")
    if args.max_steps % args.checkpoint_freq != 0:
        raise ValueError(
            "--max-steps must be divisible by --checkpoint-freq so the final "
            "checkpoint is retained under a stable numeric name"
        )
    if args.replay_buffer_size < args.train_batch_size:
        raise ValueError("--replay-buffer-size must be at least --train-batch-size")
    if args.replay_buffer_size // args.replay_buffer_reuse < args.train_batch_size:
        raise ValueError(
            "replay_buffer_size // replay_buffer_reuse must be at least one batch"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)

    run_dir = args.run_dir.expanduser().resolve()
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(
            f"{run_dir} is not empty; use a fresh directory for each training run"
        )
    run_dir.mkdir(parents=True, exist_ok=True)

    # AlphaZero launches one JAX process per actor plus the learner. Disabling
    # preallocation prevents every process from reserving most of the GPU.
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(run_dir / ".jax_cache"))
    os.environ.setdefault("PYTHONUNBUFFERED", "1")

    import jax
    import pyspiel
    from open_spiel.python.algorithms.alpha_zero import alpha_zero

    devices = jax.devices()
    has_gpu = any(device.platform == "gpu" for device in devices)
    if not has_gpu and not args.allow_cpu:
        raise RuntimeError(
            f"No JAX GPU device detected: {devices}. "
            "Use --allow-cpu only for a tiny smoke test."
        )

    game = pyspiel.load_game(GAME)
    game_type = game.get_type()
    preflight = {
        "game": GAME,
        "devices": [str(device) for device in devices],
        "jax_backend": jax.default_backend(),
        "observation_shape": game.observation_tensor_shape(),
        "num_actions": game.num_distinct_actions(),
        "max_game_length": game.max_game_length(),
        "dynamics": str(game_type.dynamics),
        "chance_mode": str(game_type.chance_mode),
        "information": str(game_type.information),
        "utility": str(game_type.utility),
    }
    print(json.dumps({"preflight": preflight}, sort_keys=True), flush=True)

    config = alpha_zero.Config(
        game=GAME,
        path=str(run_dir),
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        decouple_weight_decay=False,
        train_batch_size=args.train_batch_size,
        replay_buffer_size=args.replay_buffer_size,
        replay_buffer_reuse=args.replay_buffer_reuse,
        max_steps=args.max_steps,
        checkpoint_freq=args.checkpoint_freq,
        actors=args.actors,
        evaluators=args.evaluators,
        evaluation_window=args.evaluation_window,
        eval_levels=args.eval_levels,
        uct_c=args.uct_c,
        max_simulations=args.max_simulations,
        policy_alpha=args.policy_alpha,
        policy_epsilon=args.policy_epsilon,
        temperature=args.temperature,
        temperature_drop=args.temperature_drop,
        nn_model=args.nn_model,
        nn_width=args.nn_width,
        nn_depth=args.nn_depth,
        observation_shape=None,
        output_size=None,
        verbose=False,
        quiet=True,
        nn_api_version=args.nn_api,
    )
    alpha_zero.alpha_zero(config)

    final_checkpoint = run_dir / f"checkpoint-{args.max_steps}"
    if not final_checkpoint.is_dir():
        raise RuntimeError(f"Expected final checkpoint was not written: {final_checkpoint}")
    print(
        json.dumps(
            {
                "status": "complete",
                "run_dir": str(run_dir),
                "final_checkpoint": str(final_checkpoint),
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
