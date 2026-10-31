"""Predict upcoming EPL fixtures and save a dated file for later scoring."""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model import fit, load_fixtures, load_results, remove_margin

# Chosen in backtest.py.
HALF_LIFE = 120
RIDGE = 8


def main():
    results = load_results()
    fixtures = load_fixtures()
    as_of = fixtures["Date"].min()
    model = fit(results, as_of, HALF_LIFE, RIDGE)
    market = remove_margin(fixtures[["AvgH", "AvgD", "AvgA"]].to_numpy())

    rows = []
    for i, f in fixtures.iterrows():
        home, away = f["HomeTeam"], f["AwayTeam"]
        probs = model.outcome_probs(home, away)
        lam, mu = model.rates(home, away)
        grid = model.score_matrix(home, away)
        hg, ag = np.unravel_index(grid.argmax(), grid.shape)
        rows.append({
            "date": f["Date"].date(), "time_uk": f["Time"], "home": home, "away": away,
            "home_win": probs[0], "draw": probs[1], "away_win": probs[2],
            "home_xg": lam, "away_xg": mu, "likely_score": f"{hg}-{ag}", "score_prob": grid[hg, ag],
            "odds_home": market[i, 0], "odds_draw": market[i, 1], "odds_away": market[i, 2],
        })
    out = pd.DataFrame(rows).round(3)
    out["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M")

    folder = Path(__file__).parent / "predictions"
    folder.mkdir(exist_ok=True)
    path = folder / f"{as_of.date()}.csv"
    if path.exists():
        raise SystemExit(f"{path} already exists; published predictions are not overwritten.")
    out.to_csv(path, index=False)
    print(f"rho={model.rho:.2f} trained on {int((results['Date'] < as_of).sum())} matches, saved {path}")
    print(out.drop(columns="generated_utc").to_string(index=False))


if __name__ == "__main__":
    main()
