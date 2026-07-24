#!/usr/bin/env python3
"""Extract canonical board states visited by OpenSpiel AlphaZero actors."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path


GAME = "connect_four(rows=4,columns=4,x_in_row=3)"
GAME_LINE = re.compile(r"Game \d+: Returns: .*; Actions: ?(.*)$")
ACTION = re.compile(r"([xo])(\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def mirror_board(board: str) -> str:
    return "\n".join(row[::-1] for row in board.splitlines()) + "\n"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()

    import pyspiel

    game = pyspiel.load_game(GAME)
    records: dict[str, dict[str, object]] = {}
    games = 0
    actor_logs = sorted(run_dir.glob("log-actor-*.txt"))
    if not actor_logs:
        raise FileNotFoundError(f"No actor logs under {run_dir}")

    for log_path in actor_logs:
        for line in log_path.read_text().splitlines():
            match = GAME_LINE.search(line)
            if not match:
                continue
            state = game.new_initial_state()
            tokens = match.group(1).split()
            for ply, token in enumerate(tokens):
                action_match = ACTION.fullmatch(token)
                if not action_match:
                    raise ValueError(f"Unrecognized action token {token!r} in {log_path}")
                key = str(state)
                canonical = min(key, mirror_board(key))
                records.setdefault(
                    canonical,
                    {
                        "canonical_board": canonical,
                        "ply": ply,
                        "current_player": state.current_player(),
                    },
                )
                action = int(action_match.group(2))
                if action not in state.legal_actions():
                    raise ValueError(f"Illegal action {token!r} in {log_path}")
                state.apply_action(action)
            games += 1

    by_ply = Counter(int(record["ply"]) for record in records.values())
    payload = {
        "game": GAME,
        "run_dir": str(run_dir),
        "actor_logs": [path.name for path in actor_logs],
        "num_games": games,
        "num_unique_canonical_training_states": len(records),
        "unique_states_by_ply": {str(ply): count for ply, count in sorted(by_ply.items())},
        "states": sorted(records.values(), key=lambda record: (
            int(record["ply"]), str(record["canonical_board"])
        )),
    }

    out_path = args.out
    if out_path is None:
        out_path = run_dir / "training_states.json"
    out_path = out_path.expanduser().resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                key: payload[key]
                for key in (
                    "game",
                    "num_games",
                    "num_unique_canonical_training_states",
                    "unique_states_by_ply",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
