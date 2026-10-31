"""Time-weighted Dixon-Coles match model, fitted with numpy only.

Home goals ~ Poisson(exp(base + home + attack[home_team] + defence[away_team]))
Away goals ~ Poisson(exp(base + attack[away_team] + defence[home_team]))

A low-score correction (rho) adjusts the 0-0, 1-0, 0-1 and 1-1 probabilities,
which independent Poisson distributions get slightly wrong.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).parent / "data" / "raw"
MAX_GOALS = 10
RHO_GRID = np.arange(-0.20, 0.0501, 0.01)
OUTCOMES = ["H", "D", "A"]


def load_results():
    frames = []
    for season, name in [("2025/26", "E0_2526.csv"), ("2026/27", "E0_2627.csv")]:
        df = pd.read_csv(RAW / name, encoding="utf-8-sig")
        frames.append(df.copy().assign(Season=season))
    df = pd.concat(frames, ignore_index=True)
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    return df.sort_values(["Date", "Time"], ignore_index=True)


def load_fixtures():
    df = pd.read_csv(RAW / "fixtures.csv", encoding="utf-8-sig")
    df = df[df["Div"] == "E0"].copy()
    df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y")
    return df.reset_index(drop=True)


def remove_margin(odds):
    """Convert decimal H/D/A odds to probabilities that sum to 1."""
    implied = 1 / np.asarray(odds, dtype=float)
    return implied / implied.sum(axis=-1, keepdims=True)


def tau(home_goals, away_goals, lam, mu, rho):
    """Dixon-Coles low-score adjustment factor."""
    t = np.ones_like(lam, dtype=float)
    t = np.where((home_goals == 0) & (away_goals == 0), 1 - lam * mu * rho, t)
    t = np.where((home_goals == 0) & (away_goals == 1), 1 + lam * rho, t)
    t = np.where((home_goals == 1) & (away_goals == 0), 1 + mu * rho, t)
    t = np.where((home_goals == 1) & (away_goals == 1), 1 - rho, t)
    return t


@dataclass
class Model:
    teams: dict
    beta: np.ndarray
    rho: float

    def rates(self, home, away):
        """Expected goals for each side. A team with no data rates as average."""
        n = len(self.teams)
        attack, defence = self.beta[2:2 + n], self.beta[2 + n:]
        h, a = self.teams.get(home), self.teams.get(away)
        att_h, def_h = (attack[h], defence[h]) if h is not None else (0.0, 0.0)
        att_a, def_a = (attack[a], defence[a]) if a is not None else (0.0, 0.0)
        lam = np.exp(self.beta[0] + self.beta[1] + att_h + def_a)
        mu = np.exp(self.beta[0] + att_a + def_h)
        return lam, mu

    def score_matrix(self, home, away):
        lam, mu = self.rates(home, away)
        goals = np.arange(MAX_GOALS + 1)
        logfact = np.cumsum(np.r_[0.0, np.log(goals[1:])])
        p_home = np.exp(goals * np.log(lam) - lam - logfact)
        p_away = np.exp(goals * np.log(mu) - mu - logfact)
        grid = np.outer(p_home, p_away)
        hg, ag = np.meshgrid(goals, goals, indexing="ij")
        grid = grid * tau(hg, ag, np.full_like(grid, lam), np.full_like(grid, mu), self.rho)
        return grid / grid.sum()

    def outcome_probs(self, home, away):
        grid = self.score_matrix(home, away)
        return np.array([np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()])


def fit(results, as_of, half_life=120.0, ridge=2.0):
    """Fit on matches played before as_of, weighting recent matches more."""
    played = results[results["Date"] < as_of]
    names = sorted(set(played["HomeTeam"]) | set(played["AwayTeam"]))
    teams = {t: i for i, t in enumerate(names)}
    n, m = len(teams), len(played)
    home = played["HomeTeam"].map(teams).to_numpy()
    away = played["AwayTeam"].map(teams).to_numpy()
    hg = played["FTHG"].to_numpy(float)
    ag = played["FTAG"].to_numpy(float)
    age = (pd.Timestamp(as_of) - played["Date"]).dt.days.to_numpy(float)
    w = 0.5 ** (age / half_life)

    # Two rows per match: one for each side's goals.
    X = np.zeros((2 * m, 2 + 2 * n))
    rows = np.arange(m)
    X[:, 0] = 1
    X[rows, 1] = 1
    X[rows, 2 + home] = 1
    X[rows, 2 + n + away] = 1
    X[m + rows, 2 + away] = 1
    X[m + rows, 2 + n + home] = 1
    y = np.r_[hg, ag]
    weights = np.r_[w, w]

    penalty = np.r_[0.0, 0.0, np.full(2 * n, ridge)]
    beta = np.zeros(X.shape[1])
    beta[0] = np.log(np.average(y, weights=weights))
    for _ in range(25):
        eta = X @ beta
        rate = np.exp(eta)
        grad = X.T @ (weights * (y - rate)) - penalty * beta
        hess = (X * (weights * rate)[:, None]).T @ X + np.diag(penalty)
        step = np.linalg.solve(hess, grad)
        beta += step
        if np.abs(step).max() < 1e-7:
            break

    # The goal rates are held fixed while rho is estimated by grid search.
    lam = np.exp(X[:m] @ beta)
    mu = np.exp(X[m:] @ beta)
    scores = [
        (w * np.log(np.clip(tau(hg, ag, lam, mu, r), 1e-9, None))).sum()
        for r in RHO_GRID
    ]
    return Model(teams, beta, float(RHO_GRID[int(np.argmax(scores))]))
