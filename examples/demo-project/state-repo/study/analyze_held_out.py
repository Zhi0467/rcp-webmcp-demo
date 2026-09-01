"""Deterministic reference analysis for the public held-out fixture."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

ARMS = ("value_only", "search_assisted")
UPDATES = tuple(range(5))
MATCH_TOLERANCE = 0.02


def _slope(points: list[tuple[int, float]]) -> float:
    x_mean = sum(update for update, _value in points) / len(points)
    y_mean = sum(value for _update, value in points) / len(points)
    numerator = sum((update - x_mean) * (value - y_mean) for update, value in points)
    denominator = sum((update - x_mean) ** 2 for update, _value in points)
    return numerator / denominator


def analyze(path: Path) -> dict[str, object]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8").splitlines()))
    required = {
        "seed",
        "arm",
        "update",
        "first_shift_return",
        "first_shift_kl",
        "second_shift_return",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError("The held-out CSV has an unexpected schema.")

    trajectories: dict[tuple[int, str], list[dict[str, float | int | str]]] = defaultdict(list)
    for raw in rows:
        seed = int(raw["seed"])
        arm = raw["arm"]
        if arm not in ARMS:
            raise ValueError(f"Unknown arm {arm!r}.")
        trajectories[(seed, arm)].append(
            {
                "seed": seed,
                "arm": arm,
                "update": int(raw["update"]),
                "first_shift_return": float(raw["first_shift_return"]),
                "first_shift_kl": float(raw["first_shift_kl"]),
                "second_shift_return": float(raw["second_shift_return"]),
            }
        )

    seeds = sorted({seed for seed, _arm in trajectories})
    if len(seeds) != 3 or set(trajectories) != {(seed, arm) for seed in seeds for arm in ARMS}:
        raise ValueError("Expected exactly three complete seeds per arm.")
    for key, trajectory in trajectories.items():
        trajectory.sort(key=lambda row: int(row["update"]))
        if tuple(int(row["update"]) for row in trajectory) != UPDATES:
            raise ValueError(
                f"Trajectory {key!r} does not contain updates 0 through 4 exactly once."
            )

    mean_paths: dict[str, dict[str, list[float]]] = {}
    for arm in ARMS:
        mean_paths[arm] = {
            metric: [
                sum(float(trajectories[(seed, arm)][update][metric]) for seed in seeds) / len(seeds)
                for update in UPDATES
            ]
            for metric in ("first_shift_return", "first_shift_kl")
        }
    return_gap = max(
        abs(
            mean_paths[ARMS[0]]["first_shift_return"][update]
            - mean_paths[ARMS[1]]["first_shift_return"][update]
        )
        for update in UPDATES
    )
    kl_gap = max(
        abs(
            mean_paths[ARMS[0]]["first_shift_kl"][update]
            - mean_paths[ARMS[1]]["first_shift_kl"][update]
        )
        for update in UPDATES
    )
    slopes = {
        arm: {
            str(seed): _slope(
                [
                    (int(row["update"]), float(row["second_shift_return"]))
                    for row in trajectories[(seed, arm)]
                ]
            )
            for seed in seeds
        }
        for arm in ARMS
    }
    mean_slopes = {arm: sum(per_seed.values()) / len(per_seed) for arm, per_seed in slopes.items()}
    return {
        "source": str(path),
        "synthetic": True,
        "rows": len(rows),
        "seeds": seeds,
        "updates": list(UPDATES),
        "matching": {
            "tolerance": MATCH_TOLERANCE,
            "max_first_shift_return_gap": round(return_gap, 6),
            "max_first_shift_kl_gap": round(kl_gap, 6),
            "passed": return_gap <= MATCH_TOLERANCE and kl_gap <= MATCH_TOLERANCE,
        },
        "second_shift_slope": {
            "per_seed": {
                arm: {seed: round(value, 6) for seed, value in per_seed.items()}
                for arm, per_seed in slopes.items()
            },
            "mean": {arm: round(value, 6) for arm, value in mean_slopes.items()},
            "search_minus_value": round(
                mean_slopes["search_assisted"] - mean_slopes["value_only"], 6
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    args = parser.parse_args()
    print(json.dumps(analyze(args.csv_path), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
