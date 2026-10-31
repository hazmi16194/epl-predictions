# EPL match predictions

Win, draw and loss probabilities for Premier League matches, saved before kickoff and scored afterwards.

## Method
A time-weighted Dixon-Coles model estimates each team's attack and defence from past scorelines, with home advantage and a correction for low scores. It uses results only, so it knows nothing about injuries, lineups or transfers. Odds are used as a benchmark, never as an input.

## Backtest
Every match was predicted using only earlier matches. Settings were tuned on 2025/26, then scored on 2026/27. Lower log loss and RPS are better.

| Set | Matches | Predictor | Log loss | RPS | Accuracy |
|---|---:|---|---:|---:|---:|
| 2025/26 (tuning) | 280 | Base rates | 1.1034 | 0.2291 | 39.6% |
| 2025/26 (tuning) | 280 | Model | 1.0501 | 0.2105 | 45.7% |
| 2025/26 (tuning) | 280 | Closing odds | 1.0276 | 0.2040 | 47.9% |
| 2026/27 (held out) | 80 | Base rates | 1.0967 | 0.2299 | 38.8% |
| 2026/27 (held out) | 80 | Model | 1.0506 | 0.2150 | 41.3% |
| 2026/27 (held out) | 80 | Closing odds | 1.0159 | 0.2024 | 47.5% |

The model beats base rates and trails closing odds. On the held-out matches, its log loss is 0.0347 worse than the odds (95% CI -0.0286 to 0.0943), so 80 matches cannot settle the size of that gap.

## Known limitations
- The chosen shrinkage (ridge = 8) is the largest value tested, so a stronger setting might score better, and strong shrinkage pulls teams toward average.
- Promoted teams have no earlier data in these files and start the season rated as average.
- The data starts in 2025/26, so the first 100 matches are used only for training.

## Run
Requires Python with numpy and pandas. Refresh the three files in `data/raw/` from [Football-Data.co.uk](https://www.football-data.co.uk/), then:

```bash
python backtest.py
```

```bash
python predict.py
```

`predict.py` writes `predictions/<first match date>.csv` and refuses to overwrite an existing file, which keeps the record honest.
