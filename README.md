# RR IPL 2027 Auction Intelligence – Pressure Performers

Static dashboard (HTML + ECharts) fed by a Python pipeline that reads the original data pack.

## Run
```bash
pip install pandas numpy
python pipeline/build_data.py "path/to/Data Pack"   # writes dashboard/data.js
cd dashboard && python -m http.server 8000           # open http://localhost:8000
```
`index.html` also opens directly from disk. ECharts is bundled (`dashboard/echarts.min.js`), so it works offline.

## What the data does and does not contain
* IPL 2022–26 and T20Is from 2022 on, plus two emerging-team Asia Cups. **There is no true domestic T20 data**, and no auction, price, age, nationality or availability fields. The dashboard states this and never asserts a player will be at the 2027 auction.
* 883 matches listed, 867 with ball data (16 abandoned). Super-over balls (38) are excluded.

## Semantics validated from the CSVs
* `runs` = bat runs only. Wides/byes/leg-byes carry 0 bat runs; extras sit in `extras1_runs`/`extras_2_runs`.
* Legal ball = not wide and not no-ball (`Overstep`, `other`) – this gives 120 legal balls in 1,011 full innings.
* Bowler runs = bat runs + wide/no-ball extras (byes and leg-byes not charged). Bowler wickets = Caught, Bowled, LBW, Stump, Hit Wicket (run-outs, retired, obstruction excluded).
* Checks (in the app's data note): 0 duplicate balls, 0 unmapped players, ball totals reproduce the recorded winner in 99.8% of 809 decisive matches.

## Pressure definition
* **Death**: overs 16–20. **High-pressure chase**: 2nd innings, over ≥7, required rate ≥ 9.5 (rain-revised matches excluded).
* **Pressure ball** = either. **Ordinary ball** = baseline for each player's own comparison.
* **Close match** (derived): chase lost/tied by ≤10 runs or won with ≤6 balls left. Reported as context only.
* Thresholds (9.5 RRR, 10 runs, 6 balls) are judgement calls, not data-derived.

## Rating
Every index = shrunk player rate ÷ league pressure rate (1.00 = league average); shrinkage prior 60 balls (batters) / 30 balls (bowlers). Score = 100 × weighted mean (weights editable in the Rankings tab).
* Batters: pressure SR, dismissal avoidance, boundary %, consistency (share of pressure innings ≥4 balls with SR ≥ league), pressure-vs-own-baseline.
* Bowlers: economy, wicket rate, dot %, consistency (spells ≥6 balls with economy ≤ league or a wicket), pressure-vs-own-baseline.
* Roles: Finisher = avg batting position ≥5 & ≥40 death balls; All-rounder = ≥150 balls faced & ≥120 bowled; pace/spin from `pace_or_spin`.

## Limitations
Associations in a small sample, not proof of temperament. Competitions are pooled unless filtered (IPL / T20I filter available). Opposition and venue strength are not adjusted. Overseas/Indian status is not in the data (squad builder uses an editable India-T20I proxy). "Illustrative need" levels in the Auction and Squad tabs are planning assumptions. Public-source verification of 2027 auction lists was not performed.
