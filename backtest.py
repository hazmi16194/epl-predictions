"""Walk-forward test: every match is predicted using only earlier matches.

Settings are tuned on 2025/26 (from match 101). The 80 matches of 2026/27 are
held out and scored once with the chosen settings.
"""
import itertools

import numpy as np
import pandas as pd

from model import OUTCOMES, fit, load_results, remove_margin

HALF_LIVES = [60, 120, 240, 480, 960]
RIDGES = [0.5, 1, 2, 4, 8]
FIRST_TEST_MATCH = 100
SEED = 7


def walk_forward(results, half_life, ridge):
    probs = np.full((len(results), 3), np.nan)
    for date, group in results.iloc[FIRST_TEST_MATCH:].groupby("Date"):
        model = fit(results, date, half_life, ridge)
        for i, row in group.iterrows():
            probs[i] = model.outcome_probs(row["HomeTeam"], row["AwayTeam"])
    return probs


def base_rate_probs(results):
    """Prediction from how often H/D/A happened in earlier matches only."""
    probs = np.full((len(results), 3), np.nan)
    for date, group in results.iloc[FIRST_TEST_MATCH:].groupby("Date"):
        past = results.loc[results["Date"] < date, "FTR"]
        probs[group.index] = [(past == o).mean() for o in OUTCOMES]
    return probs


def per_match_log_loss(probs, y):
    return -np.log(probs[np.arange(len(y)), y])


def scores(probs, y):
    one_hot = np.eye(3)[y]
    # Ranked probability score respects the order home win, draw, away win.
    rps = ((np.cumsum(probs, 1) - np.cumsum(one_hot, 1))[:, :2] ** 2).sum(1) / 2
    return {
        "log_loss": per_match_log_loss(probs, y).mean(),
        "rps": rps.mean(),
        "accuracy": (probs.argmax(1) == y).mean(),
    }


def main():
    results = load_results()
    y = results["FTR"].map({o: i for i, o in enumerate(OUTCOMES)}).to_numpy()
    tune = (results["Season"] == "2025/26") & (results.index >= FIRST_TEST_MATCH)
    test = results["Season"] == "2026/27"

    grid = []
    for half_life, ridge in itertools.product(HALF_LIVES, RIDGES):
        probs = walk_forward(results, half_life, ridge)
        grid.append((scores(probs[tune], y[tune])["log_loss"], half_life, ridge, probs))
    grid.sort(key=lambda g: g[0])
    best_loss, half_life, ridge, model_probs = grid[0]
    print(f"Chosen on tuning set: half_life={half_life}, ridge={ridge}, log_loss={best_loss:.4f}")

    odds = remove_margin(results[["AvgCH", "AvgCD", "AvgCA"]].to_numpy())
    base = base_rate_probs(results)

    table = []
    for label, mask in [("Tuning: 2025/26", tune), ("Held out: 2026/27", test)]:
        for name, probs in [("Base rates", base), ("Model", model_probs), ("Closing odds", odds)]:
            table.append({"set": label, "matches": int(mask.sum()), "predictor": name, **scores(probs[mask], y[mask])})
    table = pd.DataFrame(table).round(4)
    print(table.to_string(index=False))

    rng = np.random.default_rng(SEED)
    diff = per_match_log_loss(model_probs[test], y[test]) - per_match_log_loss(odds[test], y[test])
    draws = rng.choice(diff, (10000, len(diff))).mean(1)
    low, high = np.percentile(draws, [2.5, 97.5])
    print(f"Held-out log loss, model minus odds: {diff.mean():.4f} (95% CI {low:.4f} to {high:.4f})")

    table.to_csv("backtest_results.csv", index=False)
    print(f"RESULT half_life={half_life} ridge={ridge}")


if __name__ == "__main__":
    main()
