#!/usr/bin/env python3
"""Print a compact, auditable summary of a completed Connect-3 run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    expected = list(range(0, 201, 20))
    missing = [
        step for step in expected if not (run_dir / f"checkpoint-{step}").is_dir()
    ]
    print(f"run_dir: {run_dir}")
    print(f"checkpoints_0_to_200_every_20: {'complete' if not missing else 'missing ' + str(missing)}")

    for step in (20, 100, 200):
        path = run_dir / f"verification-checkpoint-{step}.json"
        payload = json.loads(path.read_text())
        print(
            f"checkpoint={step} states={payload['num_states']} "
            f"mae={payload['mae']:.8f} rmse={payload['rmse']:.8f} "
            f"class_accuracy={payload['value_class_accuracy']:.8f} "
            f"max_abs_error={payload['max_absolute_error']:.8f}"
        )
        for depth, metrics in payload["error_by_remaining_depth"].items():
            print(
                f"  remaining_depth={depth} count={metrics['count']} "
                f"q95={metrics['q95']:.8f} q99={metrics['q99']:.8f} "
                f"max={metrics['max']:.8f}"
            )

    training = json.loads((run_dir / "training_states.json").read_text())
    print(f"training_games: {training['num_games']}")
    print(
        "unique_canonical_training_states: "
        f"{training['num_unique_canonical_training_states']}"
    )
    print(f"unique_states_by_ply: {training['unique_states_by_ply']}")

    resource_path = run_dir.parent / f"{run_dir.name}.resource.csv"
    if resource_path.is_file():
        with resource_path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        gpu_util = []
        gpu_mem = []
        for row in rows:
            try:
                util = float(row["gpu_util_percent"])
                memory = float(row["memory_used_mib"])
            except (KeyError, TypeError, ValueError):
                continue
            if math.isfinite(util) and math.isfinite(memory):
                gpu_util.append(util)
                gpu_mem.append(memory)
        if gpu_util:
            print(
                f"gpu_samples={len(gpu_util)} "
                f"util_mean_percent={statistics.fmean(gpu_util):.3f} "
                f"util_max_percent={max(gpu_util):.3f} "
                f"memory_max_mib={max(gpu_mem):.3f}"
            )


if __name__ == "__main__":
    main()

