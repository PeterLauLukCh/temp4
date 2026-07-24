#!/usr/bin/env python3
"""Load a Connect-3 AlphaZero checkpoint and compare it with exact minimax."""

from __future__ import annotations

import argparse
import functools
import json
import math
import random
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPECTED_GAME = "connect_four(rows=4,columns=4,x_in_row=3)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, default=None)
    parser.add_argument("--plies", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    parser.add_argument("--max-states", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260723)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def choose_checkpoint(run_dir: Path, requested: int | None) -> int:
    if requested is not None:
        path = run_dir / f"checkpoint-{requested}"
        if not path.is_dir():
            raise FileNotFoundError(path)
        return requested

    steps = []
    pattern = re.compile(r"checkpoint-(\d+)$")
    for path in run_dir.glob("checkpoint-*"):
        match = pattern.fullmatch(path.name)
        if path.is_dir() and match:
            steps.append(int(match.group(1)))
    if not steps:
        raise FileNotFoundError(f"No numeric checkpoint under {run_dir}")
    return max(steps)


def mirror_board(board: str) -> str:
    return "\n".join(row[::-1] for row in board.splitlines()) + "\n"


def quantile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return math.nan
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def enumerate_states(game: Any) -> tuple[list[list[Any]], dict[str, Any]]:
    initial = game.new_initial_state()
    layers = [[initial]]
    states = {str(initial): initial}
    for _ in range(game.max_game_length()):
        next_layer: dict[str, Any] = {}
        for state in layers[-1]:
            if state.is_terminal():
                continue
            for action in state.legal_actions():
                child = state.child(action)
                key = str(child)
                next_layer.setdefault(key, child)
                states.setdefault(key, child)
        layers.append(list(next_layer.values()))
    return layers, states


def build_model(run_dir: Path, config: dict[str, Any]) -> Any:
    from open_spiel.python.algorithms.alpha_zero import utils

    model_lib = utils.api_selector(config.get("nn_api_version", "linen"))
    return model_lib.Model.build_model(
        config["nn_model"],
        config["observation_shape"],
        config["output_size"],
        config["nn_width"],
        config["nn_depth"],
        config["weight_decay"],
        config["learning_rate"],
        str(run_dir),
        decouple_weight_decay=config.get("decouple_weight_decay", False),
    )


def main() -> None:
    args = parse_args()
    if args.max_states <= 0:
        raise ValueError("--max-states must be positive")

    run_dir = args.run_dir.expanduser().resolve()
    config = json.loads((run_dir / "config.json").read_text())
    if config["game"] != EXPECTED_GAME:
        raise ValueError(f"Expected {EXPECTED_GAME!r}, got {config['game']!r}")
    checkpoint_step = choose_checkpoint(run_dir, args.checkpoint_step)

    import jax
    import pyspiel

    game = pyspiel.load_game(config["game"])
    layers, states = enumerate_states(game)

    @functools.lru_cache(maxsize=None)
    def exact_value(key: str) -> float:
        state = states[key]
        if state.is_terminal():
            return float(state.returns()[0])
        child_values = [
            exact_value(str(state.child(action))) for action in state.legal_actions()
        ]
        if state.current_player() == 0:
            return max(child_values)
        return min(child_values)

    candidates = []
    seen_canonical = set()
    for ply in args.plies:
        if ply < 0 or ply >= len(layers):
            raise ValueError(f"Invalid ply: {ply}")
        for state in layers[ply]:
            if state.is_terminal():
                continue
            key = str(state)
            canonical = min(key, mirror_board(key))
            if canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            candidates.append((ply, state))

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    candidates = candidates[: args.max_states]

    model = build_model(run_dir, config)
    model.load_checkpoint(checkpoint_step)

    records = []
    for ply, state in candidates:
        prediction, policy = model.inference(
            state.observation_tensor(), state.legal_actions_mask()
        )
        predicted_value = float(prediction)
        truth = exact_value(str(state))
        records.append(
            {
                "ply": ply,
                "remaining_depth": game.max_game_length() - ply,
                "prediction": predicted_value,
                "truth": truth,
                "absolute_error": abs(predicted_value - truth),
                "policy_sum": float(policy.sum()),
            }
        )

    errors = [record["absolute_error"] for record in records]
    squared_errors = [error * error for error in errors]
    class_correct = 0
    by_depth: dict[int, list[float]] = defaultdict(list)
    for record in records:
        predicted_class = -1 if record["prediction"] < -1 / 3 else (
            1 if record["prediction"] > 1 / 3 else 0
        )
        class_correct += int(predicted_class == int(record["truth"]))
        by_depth[record["remaining_depth"]].append(record["absolute_error"])

    depth_envelopes = {
        str(depth): {
            "count": len(depth_errors),
            "max": max(depth_errors),
            "q95": quantile(depth_errors, 0.95),
            "q99": quantile(depth_errors, 0.99),
        }
        for depth, depth_errors in sorted(by_depth.items())
    }
    payload = {
        "status": "ok",
        "run_dir": str(run_dir),
        "checkpoint_step": checkpoint_step,
        "device": [str(device) for device in jax.devices()],
        "num_states": len(records),
        "plies": args.plies,
        "mae": statistics.fmean(errors),
        "rmse": math.sqrt(statistics.fmean(squared_errors)),
        "max_absolute_error": max(errors),
        "value_class_accuracy": class_correct / len(records),
        "policy_sum_max_deviation": max(
            abs(record["policy_sum"] - 1.0) for record in records
        ),
        "error_by_remaining_depth": depth_envelopes,
    }

    out_path = args.out
    if out_path is None:
        out_path = run_dir / f"verification-checkpoint-{checkpoint_step}.json"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
