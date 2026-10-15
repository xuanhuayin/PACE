"""Compute fixed-corpus fulfillment estimates from two blind audit labels."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics

LABELS = {"GENUINE", "LOOPHOLE", "GARBLED"}


def estimate(data, *, resamples=5000, seed=0):
    if resamples < 100:
        raise ValueError("Use at least 100 bootstrap resamples")
    rng = random.Random(seed)
    arms = {}
    samples = {}
    for name, arm in data["arms"].items():
        passed, total = arm["official_passes"], arm["official_total"]
        if type(passed) is not int or type(total) is not int or total <= 0 or not 0 <= passed <= total:
            raise ValueError(f"Invalid official instruction counts for {name}")
        draws = arm["draws"]
        if not draws:
            raise ValueError(f"No audit draws for {name}")
        sample = []
        for draw in draws:
            labels = draw["labels"]
            if len(labels) != 2 or any(label not in LABELS for label in labels):
                raise ValueError("Each retained draw must contain two valid labels")
            sample.append(sum(label == "GENUINE" for label in labels) / 2)
        if passed == 0:
            raise ValueError("An arm with no official passes cannot have official-pass audit draws")
        rate = 100 * passed / total
        mean = statistics.mean(sample)
        bootstrap = sorted(statistics.mean(rng.choices(sample, k=len(sample))) for _ in range(resamples))
        lower = bootstrap[math.floor(0.025 * resamples)]
        upper = bootstrap[math.ceil(0.975 * resamples) - 1]
        arms[name] = dict(
            official_passes=passed, official_total=total, official_instruction_accuracy_pct=rate,
            audit_draws=len(sample), genuine_share_pct=100 * mean,
            fulfillment_pct=rate * mean, fulfillment_ci95_pct=[rate * lower, rate * upper],
        )
        samples[name] = sample
    contrasts = {}
    for first, second in data.get("comparisons", []):
        a, b = arms[first], arms[second]
        va, vb = samples[first], samples[second]
        bootstrap = sorted(
            a["official_instruction_accuracy_pct"] * statistics.mean(rng.choices(va, k=len(va)))
            - b["official_instruction_accuracy_pct"] * statistics.mean(rng.choices(vb, k=len(vb)))
            for _ in range(resamples)
        )
        contrasts[f"{first} - {second}"] = dict(
            difference_points=a["fulfillment_pct"] - b["fulfillment_pct"],
            ci95_points=[bootstrap[math.floor(0.025 * resamples)], bootstrap[math.ceil(0.975 * resamples) - 1]],
        )
    return dict(
        protocol="Two binary labels averaged per retained instruction draw; official counts fixed; independent within-arm bootstrap.",
        bootstrap_resamples=resamples, seed=seed, arms=arms, comparisons=contrasts,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resamples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    encoded = json.dumps(estimate(json.loads(args.input.read_text()), resamples=args.resamples, seed=args.seed), indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()

