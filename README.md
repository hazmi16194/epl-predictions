# EPL match predictions

Win, draw and loss probabilities for Premier League matches, published before kickoff and scored afterwards. Predictions are for analysis, not betting.

## Latest predictions
Matchweek of 31 Oct to 2 Nov 2026, published before the first kickoff. These matches have not been scored yet. Full numbers are in [`predictions/2026-10-31.csv`](predictions/2026-10-31.csv).

| Match | Model pick | Chance | Bookmakers' pick |
|---|---|---:|---|
| Chelsea v Man United | Chelsea | 44% | Chelsea |
| Bournemouth v Leeds | Bournemouth | 41% | Bournemouth |
| Brentford v Nott'm Forest | Brentford | 49% | Brentford |
| Coventry v Sunderland | Coventry | 38% | Sunderland |
| Hull v Ipswich | Hull | 48% | Hull |
| Man City v Brighton | Man City | 46% | Man City |
| Tottenham v Crystal Palace | Tottenham | 50% | Tottenham |
| Aston Villa v Fulham | Aston Villa | 46% | Aston Villa |
| Liverpool v Arsenal | Arsenal | 40% | Arsenal |
| Newcastle v Everton | Newcastle | 42% | Newcastle |

## Model
A time-weighted Dixon-Coles model, which is a Poisson model built for football scores:

1. Estimate how many goals each team should score.
2. Convert those estimates into a probability for each scoreline, with a correction for 0-0, 1-0, 0-1 and 1-1.
3. Add the scorelines up into home win, draw and away win probabilities.

```text
Home goals rate = baseline x home advantage x home attack x away defence
Away goals rate = baseline x away attack x home defence
```

## Data
- **Source:** [Football-Data.co.uk](https://www.football-data.co.uk/), covering 460 EPL matches from 15 Aug 2025 to 25 Oct 2026.
- **Model inputs:** date, home team, away team and final score.
- **Benchmark only:** bookmaker odds, which are never a model input.
- **Not used:** xG, shots, injuries, lineups, form streaks, rest days, transfers and manager changes.

Raw data files are not stored in this repo.

## Features
Values below come from the fit used for the 31 Oct 2026 predictions.

| Feature | Fitted value | Meaning |
|---|---:|---|
| Baseline | 1.205 goals | Goal rate for an average team, before home advantage |
| Home advantage | x1.224 | The home team scores about 22% more |
| Team attack | Man City x1.439, Wolves x0.764 | Above 1 means scoring more than average |
| Team defence | Arsenal x0.671, Crystal Palace x1.264 | Below 1 means conceding less than average |
| Low-score correction | -0.10 | Adjusts the probabilities of low scores |

No feature has a hand-set weight. The values are learned from the data.

## Settings
- **Recency weighting:** a match's weight halves every 120 days. A match is weighted 0.96 at 7 days old, 0.84 at 30 days, 0.50 at 120 days, 0.25 at 240 days and 0.12 at 365 days.
- **Shrinkage:** a ridge penalty of 8 pulls team ratings towards average, so a few results do not swing them too far.

## Training and testing
The model is tested walk-forward. For each matchday, it trains on all earlier matches and predicts that day's games, so it never learns from future results.

| Matches | Season | Role | Count |
|---|---|---|---:|
| 1-100 | 2025/26 | Training only, never scored | 100 |
| 101-380 | 2025/26 | Tuning, used to choose settings from 25 combinations | 280 |
| 381-460 | 2026/27 | Final test, not used to choose settings | 80 |

The model is then retrained on all 460 matches to predict upcoming games.

## Backtest results
Lower log loss and RPS are better.

| Set | Matches | Predictor | Log loss | RPS | Accuracy |
|---|---:|---|---:|---:|---:|
| 2025/26 (tuning) | 280 | Base rates | 1.1034 | 0.2291 | 39.6% |
| 2025/26 (tuning) | 280 | Model | 1.0501 | 0.2105 | 45.7% |
| 2025/26 (tuning) | 280 | Closing odds | 1.0276 | 0.2040 | 47.9% |
| 2026/27 (final test) | 80 | Base rates | 1.0967 | 0.2299 | 38.8% |
| 2026/27 (final test) | 80 | Model | 1.0506 | 0.2150 | 41.3% |
| 2026/27 (final test) | 80 | Closing odds | 1.0159 | 0.2024 | 47.5% |

The model beats base rates and trails closing odds. On the final test, its log loss is 0.0347 worse than the odds (95% CI -0.0286 to 0.0943), so 80 matches cannot settle the size of that gap.

## Known limitations
- The model only sees scorelines, so it cannot react to injuries or lineups.
- The chosen ridge penalty is the largest value tested, so a stronger setting might score better. Strong shrinkage can also underrate top teams.
- Promoted teams have no earlier data in these files and start the season rated as average.
- Settings were tuned on a single season, and the final test is small.
- The final test results have now been seen, so future model changes need newly played matches for a clean test.

## Run
Requires Python with numpy and pandas. Download these files from Football-Data.co.uk into `data/raw/`:

| Source path | Save as |
|---|---|
| `mmz4281/2526/E0.csv` | `E0_2526.csv` |
| `mmz4281/2627/E0.csv` | `E0_2627.csv` |
| `fixtures.csv` | `fixtures.csv` |

Then run:

```bash
python backtest.py
```

```bash
python predict.py
```

`predict.py` writes `predictions/<first match date>.csv` and refuses to overwrite an existing file, which keeps the record honest.
