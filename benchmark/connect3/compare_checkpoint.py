#!/usr/bin/env python3
"""Exploratory 2FFS versus BAI-MCTS comparison on 4x4 Connect-3.

This adapter keeps the method implementations unchanged apart from their
slow-sample hook.  It builds shallow explicit planning trees from real
Connect-3 states, uses a learned AlphaZero checkpoint as the fast oracle,
finite-budget MCTS with random rollouts as the slow oracle, and full-game
memoized minimax as ground truth.

The output is deliberately marked exploratory.  It is a checkpoint sanity
comparison, not the final benchmark split.
"""

from __future__ import annotations

import argparse
import functools
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
METHOD_ROOT = REPO_ROOT / "code" / "method"
sys.path.insert(0, str(METHOD_ROOT))

from mcts_bai import MCTSBAIConfig, MCTSBAIRunner  # noqa: E402
from twoffs import TwoFFSConfig, TwoFFSRunner  # noqa: E402


EXPECTED_GAME = "connect_four(rows=4,columns=4,x_in_row=3)"
EXPECTED_OPEN_SPIEL_COMMIT = "112b77704631fc2ce7ad8e4581f6ca09798ce15a"
ADAPTER_REVISION = 6


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--checkpoint-step", type=int, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--root-plies", type=int, nargs="+", default=[6, 7, 8, 9, 10])
    parser.add_argument("--root-count", type=int, default=3)
    parser.add_argument("--planning-depth", type=int, default=3)
    parser.add_argument("--max-terminal-leaf-fraction", type=float, default=1.0)
    parser.add_argument("--min-nonterminal-leaves", type=int, default=0)
    parser.add_argument("--calibration-plies", type=int, nargs="+", default=list(range(6, 14)))
    parser.add_argument("--calibration-states", type=int, default=1000)
    parser.add_argument("--envelope-quantile", type=float, default=1.0)
    parser.add_argument("--envelope-margin", type=float, default=0.05)
    parser.add_argument(
        "--proxy-envelope",
        "--proxy_envelop",
        dest="proxy_envelope",
        choices=("linear",),
        default=None,
        help=(
            "Reference-free envelope proxy. 'linear' uses "
            "B_proxy(h)=2*h/game.max_game_length() and skips residual "
            "calibration entirely."
        ),
    )
    parser.add_argument("--mirror-average", action="store_true")
    parser.add_argument("--slow-simulations", type=int, default=16)
    parser.add_argument("--uct-c", type=float, default=1.41)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--replicates", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260724)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive = {
        "checkpoint_step": args.checkpoint_step,
        "root_count": args.root_count,
        "planning_depth": args.planning_depth,
        "calibration_states": args.calibration_states,
        "slow_simulations": args.slow_simulations,
        "uct_c": args.uct_c,
        "replicates": args.replicates,
        "max_rounds": args.max_rounds,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if not 0 < args.delta < 1:
        raise ValueError("--delta must lie in (0, 1)")
    if args.epsilon < 0:
        raise ValueError("--epsilon must be nonnegative")
    if args.envelope_margin < 0:
        raise ValueError("--envelope-margin must be nonnegative")
    if not 0 < args.envelope_quantile <= 1:
        raise ValueError("--envelope-quantile must lie in (0, 1]")
    if not 0 <= args.max_terminal_leaf_fraction <= 1:
        raise ValueError("--max-terminal-leaf-fraction must lie in [0, 1]")
    if args.min_nonterminal_leaves < 0:
        raise ValueError("--min-nonterminal-leaves must be nonnegative")
    for label, plies in (
        ("root", args.root_plies),
        ("calibration", args.calibration_plies),
    ):
        if not plies:
            raise ValueError(f"--{label}-plies must be nonempty")
        if any(ply < 0 or ply > 15 for ply in plies):
            raise ValueError(f"--{label}-plies must lie in [0, 15]")


def prepare_out_dir(path: Path) -> Path:
    out_dir = path.expanduser().resolve()
    if out_dir.exists() and any(out_dir.iterdir()):
        raise FileExistsError(f"{out_dir} is not empty; use a fresh output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "trees").mkdir()
    return out_dir


def verify_open_spiel_source() -> str:
    marker = REPO_ROOT / ".open_spiel_commit"
    if marker.is_file():
        actual = marker.read_text().strip()
    else:
        git_marker = REPO_ROOT / "third_party" / "open_spiel" / ".git"
        if not git_marker.exists():
            raise FileNotFoundError(
                "Cannot verify OpenSpiel source: .open_spiel_commit and "
                "third_party/open_spiel/.git are both absent"
            )
        import subprocess

        actual = subprocess.run(
            [
                "git",
                "-C",
                str(REPO_ROOT / "third_party" / "open_spiel"),
                "rev-parse",
                "HEAD",
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    if actual != EXPECTED_OPEN_SPIEL_COMMIT:
        raise RuntimeError(
            f"OpenSpiel commit mismatch: expected {EXPECTED_OPEN_SPIEL_COMMIT}, "
            f"got {actual}"
        )
    return actual


def mirror_board(board: str) -> str:
    return "\n".join(row[::-1] for row in board.splitlines()) + "\n"


def canonical_board(board: str) -> str:
    return min(board, mirror_board(board))


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


def stratified_sample(
    by_ply: dict[int, list[Any]], limit: int, rng: random.Random
) -> list[tuple[int, Any]]:
    """Sample nearly equally by ply so every requested depth is represented."""
    plies = sorted(ply for ply, values in by_ply.items() if values)
    if not plies:
        return []
    for ply in plies:
        rng.shuffle(by_ply[ply])
    base, extra = divmod(limit, len(plies))
    selected: list[tuple[int, Any]] = []
    leftovers: list[tuple[int, Any]] = []
    for index, ply in enumerate(plies):
        quota = base + int(index < extra)
        values = by_ply[ply]
        selected.extend((ply, state) for state in values[:quota])
        leftovers.extend((ply, state) for state in values[quota:])
    if len(selected) < limit:
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def descendants_canonical(roots: Iterable[Any]) -> set[str]:
    excluded: set[str] = set()
    seen: set[str] = set()
    stack = list(roots)
    while stack:
        state = stack.pop()
        key = str(state)
        if key in seen:
            continue
        seen.add(key)
        excluded.add(canonical_board(key))
        if not state.is_terminal():
            stack.extend(state.child(action) for action in state.legal_actions())
    return excluded


class CheckpointPredictor:
    def __init__(self, model: Any, game: Any, mirror_average: bool) -> None:
        self.model = model
        self.game = game
        self.mirror_average = mirror_average
        self.raw_cache: dict[str, float] = {}
        self.cache: dict[str, float] = {}

    def raw_prediction(self, state: Any) -> float:
        key = str(state)
        if key not in self.raw_cache:
            value, _ = self.model.inference(
                state.observation_tensor(), state.legal_actions_mask()
            )
            prediction = float(value)
            if not math.isfinite(prediction):
                raise FloatingPointError(f"Non-finite critic value for state:\n{key}")
            self.raw_cache[key] = prediction
        return self.raw_cache[key]

    def mirrored_state(self, state: Any) -> Any:
        mirrored = self.game.new_initial_state()
        max_action = self.game.num_distinct_actions() - 1
        for action in state.history():
            mirrored.apply_action(max_action - int(action))
        return mirrored

    def __call__(self, state: Any) -> float:
        key = str(state)
        if key not in self.cache:
            prediction = self.raw_prediction(state)
            if self.mirror_average:
                mirrored = self.mirrored_state(state)
                prediction = 0.5 * (
                    prediction + self.raw_prediction(mirrored)
                )
            self.cache[key] = prediction
        return self.cache[key]


def select_roots(
    layers: list[list[Any]],
    states: dict[str, Any],
    exact_player0: Any,
    plies: list[int],
    count: int,
    planning_depth: int,
    max_terminal_leaf_fraction: float,
    min_nonterminal_leaves: int,
    rng: random.Random,
) -> tuple[list[Any], dict[str, Any]]:
    @functools.lru_cache(maxsize=None)
    def frontier_profile(key: str, remaining: int) -> tuple[int, int]:
        state = states[key]
        if state.is_terminal():
            return 1, 1
        if remaining == 0:
            return 1, 0
        leaf_count = 0
        terminal_leaf_count = 0
        for action in state.legal_actions():
            child_leaves, child_terminals = frontier_profile(
                str(state.child(action)), remaining - 1
            )
            leaf_count += child_leaves
            terminal_leaf_count += child_terminals
        return leaf_count, terminal_leaf_count

    candidates: list[tuple[Any, dict[str, Any]]] = []
    seen = set()
    counters = {
        "nonterminal_with_multiple_actions": 0,
        "mirror_deduplicated": 0,
        "passed_structure_filter": 0,
        "unique_best_action": 0,
    }
    for ply in plies:
        for state in layers[ply]:
            if state.is_terminal() or len(state.legal_actions()) < 2:
                continue
            counters["nonterminal_with_multiple_actions"] += 1
            canonical = canonical_board(str(state))
            if canonical in seen:
                continue
            seen.add(canonical)
            counters["mirror_deduplicated"] += 1
            leaf_count, terminal_leaf_count = frontier_profile(
                str(state), planning_depth
            )
            nonterminal_leaf_count = leaf_count - terminal_leaf_count
            terminal_fraction = terminal_leaf_count / leaf_count
            if terminal_fraction > max_terminal_leaf_fraction:
                continue
            if nonterminal_leaf_count < min_nonterminal_leaves:
                continue
            counters["passed_structure_filter"] += 1
            root_player = state.current_player()
            sign = 1.0 if root_player == 0 else -1.0
            child_values = [
                sign * exact_player0(str(state.child(action)))
                for action in state.legal_actions()
            ]
            best = max(child_values)
            if child_values.count(best) != 1:
                continue
            ordered = sorted(child_values, reverse=True)
            if ordered[0] <= ordered[1]:
                continue
            counters["unique_best_action"] += 1
            candidates.append(
                (
                    state,
                    {
                        "leaf_count": leaf_count,
                        "terminal_leaf_count": terminal_leaf_count,
                        "nonterminal_leaf_count": nonterminal_leaf_count,
                        "terminal_leaf_fraction": terminal_fraction,
                    },
                )
            )
    rng.shuffle(candidates)
    if len(candidates) < count:
        raise RuntimeError(
            f"Only {len(candidates)} roots passed the structural filters and "
            f"had a unique best move; requested {count}. Selection counters: "
            f"{counters}"
        )
    selected = candidates[:count]
    return [state for state, _ in selected], {
        "root_plies": plies,
        "planning_depth": planning_depth,
        "max_terminal_leaf_fraction": max_terminal_leaf_fraction,
        "min_nonterminal_leaves": min_nonterminal_leaves,
        "candidate_counters": counters,
        "eligible_roots": len(candidates),
        "selected_profiles": [profile for _, profile in selected],
    }


def calibrate_envelope(
    game: Any,
    layers: list[list[Any]],
    exact_player0: Any,
    predictor: CheckpointPredictor,
    plies: list[int],
    max_states: int,
    excluded: set[str],
    quantile_probability: float,
    margin: float,
    rng: random.Random,
) -> tuple[dict[int, float], dict[str, Any]]:
    candidates: dict[int, list[Any]] = defaultdict(list)
    seen = set()
    for ply in plies:
        for state in layers[ply]:
            if state.is_terminal():
                continue
            canonical = canonical_board(str(state))
            if canonical in excluded or canonical in seen:
                continue
            seen.add(canonical)
            candidates[ply].append(state)

    sampled = stratified_sample(candidates, max_states, rng)
    if not sampled:
        raise RuntimeError("No calibration states remain after descendant exclusion")

    by_depth: dict[int, list[float]] = defaultdict(list)
    all_errors: list[float] = []
    for ply, state in sampled:
        error = abs(predictor(state) - exact_player0(str(state)))
        remaining = game.max_game_length() - ply
        by_depth[remaining].append(error)
        all_errors.append(error)

    def empirical_quantile(values: list[float]) -> float:
        ordered = sorted(values)
        index = round((len(ordered) - 1) * quantile_probability)
        return ordered[index]

    global_bound = (
        min(2.0, max(all_errors) + margin)
        if quantile_probability == 1.0
        else 2.0
    )
    bounds = {
        depth: min(2.0, empirical_quantile(errors) + margin)
        for depth, errors in by_depth.items()
    }
    details = {
        "construction": "empirical per-depth quantile plus fixed margin",
        "quantile": quantile_probability,
        "margin": margin,
        "num_states": len(sampled),
        "excluded_descendant_canonical_states": len(excluded),
        "global_fallback_bound": global_bound,
        "mae": statistics.fmean(all_errors),
        "rmse": math.sqrt(statistics.fmean(error * error for error in all_errors)),
        "max_absolute_error": max(all_errors),
        "by_remaining_depth": {
            str(depth): {
                "count": len(errors),
                "max_absolute_error": max(errors),
                "selected_quantile": empirical_quantile(errors),
                "bound": bounds[depth],
                "empirical_coverage": statistics.fmean(
                    float(error <= bounds[depth] + 1e-12) for error in errors
                ),
            }
            for depth, errors in sorted(by_depth.items())
        },
    }
    bounds[-1] = global_bound
    return bounds, details


def linear_proxy_envelope(game: Any) -> tuple[dict[int, float], dict[str, Any]]:
    """Build a reference-free depth proxy from only range and horizon.

    Connect-3 values lie in [-1, 1], so their maximum possible absolute
    difference is 2. The proxy follows the synthetic benchmark's linear
    remaining-depth shape without consulting critic residuals or minimax
    values. It is a proxy, not a guaranteed nonterminal error bound.
    """

    horizon = int(game.max_game_length())
    if horizon <= 0:
        raise ValueError("game.max_game_length() must be positive")
    value_range_diameter = 2.0
    bounds = {
        remaining: value_range_diameter * remaining / horizon
        for remaining in range(horizon + 1)
    }
    bounds[-1] = value_range_diameter
    return bounds, {
        "construction": "reference-free linear remaining-horizon proxy",
        "proxy": "linear",
        "formula": f"B_proxy(h)=2*h/{horizon}",
        "horizon": horizon,
        "value_range": [-1.0, 1.0],
        "value_range_diameter": value_range_diameter,
        "uses_reference_values_for_envelope": False,
        "num_states": 0,
        "global_fallback_bound": value_range_diameter,
    }


def build_planning_tree(
    game: Any,
    root_state: Any,
    root_index: int,
    planning_depth: int,
    exact_player0: Any,
    predictor: CheckpointPredictor,
    envelope: dict[int, float],
    slow_simulations: int,
    fast_query_cost: float,
) -> dict[str, Any]:
    root_player = root_state.current_player()
    root_sign = 1.0 if root_player == 0 else -1.0
    nodes: list[dict[str, Any]] = []

    def add_node(state: Any, node_id: str, depth: int) -> None:
        terminal = state.is_terminal()
        frontier = depth >= planning_depth
        player = (
            "leaf"
            if terminal or frontier
            else ("max" if state.current_player() == root_player else "min")
        )
        truth = root_sign * exact_player0(str(state))
        remaining_game_depth = game.max_game_length() - len(state.history())
        if terminal:
            fast_value = truth
            fast_bound = 0.0
        else:
            fast_value = root_sign * predictor(state)
            fast_bound = envelope.get(remaining_game_depth, envelope[-1])

        node = {
            "id": node_id,
            "depth": depth,
            "remaining_depth": 0 if player == "leaf" else planning_depth - depth,
            "game_remaining_depth": remaining_game_depth,
            "player": player,
            "children": [],
            "value": truth,
            "slow_mean": truth,
            "fast_bound": fast_bound,
            "fast_value": fast_value,
            "history": [int(action) for action in state.history()],
            "terminal": terminal,
        }
        nodes.append(node)
        if player == "leaf":
            return
        for action in state.legal_actions():
            child_id = f"{node_id}/{int(action)}"
            node["children"].append(child_id)
            add_node(state.child(action), child_id, depth + 1)

    add_node(root_state, "root", 0)
    node_map = {node["id"]: node for node in nodes}
    root_children = node_map["root"]["children"]
    child_values = {child: float(node_map[child]["value"]) for child in root_children}
    ordered = sorted(child_values.items(), key=lambda item: item[1], reverse=True)
    if len(ordered) < 2 or ordered[0][1] <= ordered[1][1]:
        raise RuntimeError("Planning root must have a unique optimal child")

    return {
        "tree_id": f"connect3_root_{root_index:03d}",
        "source": "open_spiel_connect3_truncated_planning_tree",
        "game": EXPECTED_GAME,
        "root": "root",
        "root_children": root_children,
        "depth": planning_depth,
        "branching_factor": max(
            (len(node["children"]) for node in nodes), default=0
        ),
        "leaf_count": sum(node["player"] == "leaf" for node in nodes),
        "node_count": len(nodes),
        "root_history": [int(action) for action in root_state.history()],
        "root_board": str(root_state),
        "root_player": int(root_player),
        "optimal_root_child": ordered[0][0],
        "root_gap": ordered[0][1] - ordered[1][1],
        "fast_oracle": {
            "kind": "alpha_zero_checkpoint",
            "cost": fast_query_cost,
            "envelope": "empirical_depth_quantile_plus_margin",
        },
        "slow_oracle": {
            "kind": "finite_budget_open_spiel_mcts",
            "cost": float(slow_simulations),
            "sigma": 1.0,
            "simulations": slow_simulations,
        },
        "nodes": nodes,
    }


class FiniteBudgetMCTSOracle:
    """One independent stream of finite-budget MCTS slow samples."""

    def __init__(
        self,
        game: Any,
        tree: dict[str, Any],
        simulations: int,
        uct_c: float,
        seed: int,
    ) -> None:
        from open_spiel.python.algorithms import mcts

        self.game = game
        self.tree = tree
        self.nodes = {node["id"]: node for node in tree["nodes"]}
        self.root_player = int(tree["root_player"])
        self.simulations = simulations
        self.uct_c = uct_c
        self.random_state = np.random.RandomState(seed)
        self.mcts = mcts
        self.num_calls = 0
        self.wall_seconds = 0.0

    def state_for(self, node_id: str) -> Any:
        state = self.game.new_initial_state()
        for action in self.nodes[node_id]["history"]:
            state.apply_action(action)
        return state

    def sample(self, node_id: str) -> float:
        started = time.perf_counter()
        state = self.state_for(node_id)
        if state.is_terminal():
            raise RuntimeError(
                "Terminal payoffs are known exactly and must not be sent to "
                "the finite-budget MCTS oracle"
            )
        evaluator = self.mcts.RandomRolloutEvaluator(
            n_rollouts=1, random_state=self.random_state
        )
        bot = self.mcts.MCTSBot(
            self.game,
            self.uct_c,
            self.simulations,
            evaluator,
            solve=False,
            random_state=self.random_state,
        )
        root = bot.mcts_search(state)
        if root.explore_count != self.simulations:
            raise RuntimeError(
                f"MCTS ran {root.explore_count} simulations, expected "
                f"{self.simulations}"
            )
        current_player_value = float(root.total_reward / root.explore_count)
        value = (
            current_player_value
            if state.current_player() == self.root_player
            else -current_player_value
        )
        if not -1.000001 <= value <= 1.000001:
            raise FloatingPointError(f"MCTS value outside [-1, 1]: {value}")
        self.num_calls += 1
        self.wall_seconds += time.perf_counter() - started
        return value


class Connect3TwoFFSRunner(TwoFFSRunner):
    def __init__(
        self,
        tree: dict[str, Any],
        config: TwoFFSConfig,
        oracle: FiniteBudgetMCTSOracle,
    ) -> None:
        self.oracle = oracle
        super().__init__(tree, config)
        nonterminal_nonroot = sum(
            node["id"] != self.root and not node["terminal"]
            for node in tree["nodes"]
        )
        self.node_delta = self.config.delta / max(1, nonterminal_nonroot)

    def draw_slow_sample(self, node_id: str) -> float:
        return self.oracle.sample(node_id)

    def expose_node(self, node_id: str) -> None:
        node = self.nodes[node_id]
        if not node["terminal"]:
            super().expose_node(node_id)
            return
        if node_id in self.explored:
            return
        self.explored.add(node_id)
        value = float(node["value"])
        st = self.state[node_id]
        st.fast_l = value
        st.fast_u = value
        self.interval_write_touches()


class Connect3MCTSBAIRunner(MCTSBAIRunner):
    def __init__(
        self,
        tree: dict[str, Any],
        config: MCTSBAIConfig,
        oracle: FiniteBudgetMCTSOracle,
    ) -> None:
        self.oracle = oracle
        super().__init__(tree, config)
        self.exact_terminal_critical_fallbacks = 0
        nonterminal_leaves = sum(
            not self.nodes[leaf]["terminal"] for leaf in self.leaves
        )
        self.leaf_delta = self.config.delta / max(1, nonterminal_leaves)

    def draw_slow_sample(self, leaf: str) -> float:
        return self.oracle.sample(leaf)

    def sample_leaf(self, leaf: str) -> None:
        node = self.nodes[leaf]
        if not node["terminal"]:
            super().sample_leaf(leaf)
            return
        st = self.stats[leaf]
        value = float(node["value"])
        st.low = value
        st.high = value
        self.interval_write_touches()

    @functools.lru_cache(maxsize=None)
    def nonterminal_descendant_leaves(self, node_id: str) -> tuple[str, ...]:
        node = self.nodes[node_id]
        if node["player"] == "leaf":
            return () if node["terminal"] else (node_id,)
        leaves: list[str] = []
        for child in node["children"]:
            leaves.extend(self.nonterminal_descendant_leaves(child))
        return tuple(leaves)

    def critical_leaf(self, node_id: str, side: str) -> str:
        leaf = super().critical_leaf(node_id, side)
        if not self.nodes[leaf]["terminal"]:
            return leaf
        candidates = self.nonterminal_descendant_leaves(node_id)
        if not candidates:
            return leaf
        self.exact_terminal_critical_fallbacks += 1
        self.scan_touches(len(candidates))
        return max(
            candidates,
            key=lambda candidate: (
                self.stats[candidate].high - self.stats[candidate].low
            ),
        )


def envelope_diagnostics(trees: list[dict[str, Any]]) -> dict[str, Any]:
    records = []
    seen = set()
    for tree in trees:
        root_sign = 1.0 if tree["root_player"] == 0 else -1.0
        for node in tree["nodes"]:
            if node["id"] == "root" or node["terminal"]:
                continue
            key = (tree["tree_id"], node["id"])
            if key in seen:
                continue
            seen.add(key)
            error = abs(float(node["fast_value"]) - float(node["value"]))
            records.append(
                {
                    "tree_id": tree["tree_id"],
                    "node_id": node["id"],
                    "remaining_depth": node["game_remaining_depth"],
                    "error": error,
                    "bound": float(node["fast_bound"]),
                    "covered": error <= float(node["fast_bound"]) + 1e-12,
                    "root_sign": root_sign,
                }
            )
    violations = [record for record in records if not record["covered"]]
    by_depth: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_depth[record["remaining_depth"]].append(record)
    return {
        "num_comparison_nodes": len(records),
        "num_violations": len(violations),
        "violation_rate": len(violations) / max(1, len(records)),
        "max_absolute_error": max((record["error"] for record in records), default=0.0),
        "max_excess_over_bound": max(
            (record["error"] - record["bound"] for record in violations),
            default=0.0,
        ),
        "by_remaining_depth": {
            str(depth): {
                "count": len(depth_records),
                "violations": sum(
                    not record["covered"] for record in depth_records
                ),
                "coverage": statistics.fmean(
                    float(record["covered"]) for record in depth_records
                ),
                "bound": depth_records[0]["bound"],
                "max_absolute_error": max(
                    record["error"] for record in depth_records
                ),
            }
            for depth, depth_records in sorted(by_depth.items())
        },
    }


def aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_method[record["method"]].append(record)
    summaries = {}
    for method, rows in sorted(by_method.items()):
        summaries[method] = {
            "runs": len(rows),
            "stopped_rate": statistics.fmean(float(row["stopped"]) for row in rows),
            "accuracy": statistics.fmean(float(row["correct"]) for row in rows),
            "mean_total_cost": statistics.fmean(row["total_cost"] for row in rows),
            "mean_fast_queries": statistics.fmean(
                row["num_fast_queries"] for row in rows
            ),
            "mean_slow_queries": statistics.fmean(
                row["num_slow_queries"] for row in rows
            ),
            "mean_wall_seconds": statistics.fmean(
                row["wall_seconds"] for row in rows
            ),
            "mean_slow_oracle_seconds": statistics.fmean(
                row["slow_oracle_seconds"] for row in rows
            ),
            "mean_exact_terminal_critical_fallbacks": statistics.fmean(
                row["exact_terminal_critical_fallbacks"] for row in rows
            ),
        }
    return summaries


def main() -> None:
    args = parse_args()
    validate_args(args)
    started_all = time.perf_counter()
    run_dir = args.run_dir.expanduser().resolve()
    checkpoint_dir = run_dir / f"checkpoint-{args.checkpoint_step}"
    if not checkpoint_dir.is_dir():
        raise FileNotFoundError(checkpoint_dir)
    out_dir = prepare_out_dir(args.out_dir)
    open_spiel_commit = verify_open_spiel_source()

    config = json.loads((run_dir / "config.json").read_text())
    if config["game"] != EXPECTED_GAME:
        raise ValueError(f"Expected {EXPECTED_GAME!r}, got {config['game']!r}")

    import jax
    import pyspiel

    game = pyspiel.load_game(config["game"])
    layers, states = enumerate_states(game)

    @functools.lru_cache(maxsize=None)
    def exact_player0(key: str) -> float:
        state = states[key]
        if state.is_terminal():
            return float(state.returns()[0])
        values = [
            exact_player0(str(state.child(action))) for action in state.legal_actions()
        ]
        return max(values) if state.current_player() == 0 else min(values)

    rng = random.Random(args.seed)
    roots, root_selection = select_roots(
        layers,
        states,
        exact_player0,
        args.root_plies,
        args.root_count,
        args.planning_depth,
        args.max_terminal_leaf_fraction,
        args.min_nonterminal_leaves,
        rng,
    )
    excluded = descendants_canonical(roots)

    model = build_model(run_dir, config)
    model.load_checkpoint(args.checkpoint_step)
    predictor = CheckpointPredictor(model, game, args.mirror_average)
    if args.proxy_envelope == "linear":
        envelope, calibration = linear_proxy_envelope(game)
    else:
        envelope, calibration = calibrate_envelope(
            game,
            layers,
            exact_player0,
            predictor,
            args.calibration_plies,
            args.calibration_states,
            excluded,
            args.envelope_quantile,
            args.envelope_margin,
            rng,
        )
    fast_query_cost = 2.0 if args.mirror_average else 1.0

    trees = [
        build_planning_tree(
            game,
            root,
            index,
            args.planning_depth,
            exact_player0,
            predictor,
            envelope,
            args.slow_simulations,
            fast_query_cost,
        )
        for index, root in enumerate(roots)
    ]
    for tree in trees:
        tree_path = out_dir / "trees" / f"{tree['tree_id']}.json"
        tree_path.write_text(json.dumps(tree, indent=2, sort_keys=True) + "\n")

    diagnostics = envelope_diagnostics(trees)
    records: list[dict[str, Any]] = []
    for tree_index, tree in enumerate(trees):
        for replicate in range(args.replicates):
            base_seed = args.seed + 100_000 * tree_index + 1_000 * replicate
            method_specs = (
                (
                    "2ffs",
                    Connect3TwoFFSRunner,
                    TwoFFSConfig(
                        delta=args.delta,
                        epsilon=args.epsilon,
                        slow_cost=float(args.slow_simulations),
                        slow_sigma=1.0,
                        max_outer_rounds=args.max_rounds,
                        seed=base_seed,
                    ),
                ),
                (
                    "bai_mcts",
                    Connect3MCTSBAIRunner,
                    MCTSBAIConfig(
                        delta=args.delta,
                        epsilon=args.epsilon,
                        slow_cost=float(args.slow_simulations),
                        slow_sigma=1.0,
                        max_rounds=args.max_rounds,
                        seed=base_seed,
                    ),
                ),
            )
            for method_index, (method, runner_type, method_config) in enumerate(
                method_specs
            ):
                oracle = FiniteBudgetMCTSOracle(
                    game,
                    tree,
                    args.slow_simulations,
                    args.uct_c,
                    base_seed + 10_000 * method_index,
                )
                started = time.perf_counter()
                runner = runner_type(tree, method_config, oracle)
                result = runner.run().to_dict()
                elapsed = time.perf_counter() - started
                if oracle.num_calls != result["num_slow_queries"]:
                    raise RuntimeError(
                        "Slow-query accounting mismatch: "
                        f"oracle={oracle.num_calls}, result="
                        f"{result['num_slow_queries']}"
                    )
                record = {
                    **result,
                    "method": method,
                    "replicate": replicate,
                    "checkpoint_step": args.checkpoint_step,
                    "root_history": tree["root_history"],
                    "root_board": tree["root_board"],
                    "root_player": tree["root_player"],
                    "planning_depth": args.planning_depth,
                    "slow_simulations_per_query": args.slow_simulations,
                    "wall_seconds": elapsed,
                    "slow_oracle_seconds": oracle.wall_seconds,
                    "exact_terminal_critical_fallbacks": getattr(
                        runner, "exact_terminal_critical_fallbacks", 0
                    ),
                    "exploratory": True,
                }
                records.append(record)
                print(
                    json.dumps(
                        {
                            "tree_id": tree["tree_id"],
                            "method": method,
                            "replicate": replicate,
                            "stopped": result["stopped"],
                            "correct": result["correct"],
                            "fast_queries": result["num_fast_queries"],
                            "slow_queries": result["num_slow_queries"],
                            "total_cost": result["total_cost"],
                            "wall_seconds": round(elapsed, 3),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )

    result_path = out_dir / "results.jsonl"
    result_path.write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in records)
    )
    payload = {
        "status": "complete",
        "exploratory": True,
        "adapter_revision": ADAPTER_REVISION,
        "warning": (
            "This method comparison is exploratory and is not the final "
            "training-overlap-filtered benchmark split."
        ),
        "run_dir": str(run_dir),
        "checkpoint_step": args.checkpoint_step,
        "out_dir": str(out_dir),
        "game": EXPECTED_GAME,
        "open_spiel_commit": open_spiel_commit,
        "jax_backend": jax.default_backend(),
        "jax_devices": [str(device) for device in jax.devices()],
        "config": {
            "root_plies": args.root_plies,
            "root_count": args.root_count,
            "planning_depth": args.planning_depth,
            "max_terminal_leaf_fraction": args.max_terminal_leaf_fraction,
            "min_nonterminal_leaves": args.min_nonterminal_leaves,
            "calibration_plies": (
                None if args.proxy_envelope else args.calibration_plies
            ),
            "calibration_states": (
                None if args.proxy_envelope else args.calibration_states
            ),
            "envelope_quantile": (
                None if args.proxy_envelope else args.envelope_quantile
            ),
            "envelope_margin": (
                None if args.proxy_envelope else args.envelope_margin
            ),
            "proxy_envelope": args.proxy_envelope,
            "envelope_uses_reference_values": bool(
                calibration.get("uses_reference_values_for_envelope", True)
            ),
            "mirror_average": args.mirror_average,
            "slow_simulations": args.slow_simulations,
            "uct_c": args.uct_c,
            "delta": args.delta,
            "epsilon": args.epsilon,
            "replicates": args.replicates,
            "max_rounds": args.max_rounds,
            "seed": args.seed,
            "slow_cost_units": "MCTS simulations",
            "fast_cost_per_query": fast_query_cost,
            "slow_sigma": 1.0,
            "terminal_payoffs": "known exactly at zero query cost for both methods",
        },
        "calibration": calibration,
        "root_selection": root_selection,
        "envelope_diagnostics_on_comparison_nodes": diagnostics,
        "roots": [
            {
                "tree_id": tree["tree_id"],
                "history": tree["root_history"],
                "player": tree["root_player"],
                "root_gap": tree["root_gap"],
                "node_count": tree["node_count"],
                "leaf_count": tree["leaf_count"],
                "terminal_leaf_count": sum(
                    node["player"] == "leaf" and node["terminal"]
                    for node in tree["nodes"]
                ),
            }
            for tree in trees
        ],
        "summary": aggregate(records),
        "wall_seconds": time.perf_counter() - started_all,
        "environment": {
            "JAX_PLATFORMS": os.environ.get("JAX_PLATFORMS"),
            "python": sys.version,
        },
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
