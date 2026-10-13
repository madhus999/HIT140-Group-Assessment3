# Model 2.2 – Team-match dataset (Suman part)

**File:** `model22_team_matches.csv` — 208 rows (2 per WC 2026 match, one from each team's side), 0 missing values.
**Rebuild:** `python build_model22_dataset.py` (reads KB's and Nabin's `raw/` folders, no new downloads; writes the CSV).
**Student ID:** s397846 (Suman Silwal)

## Target
| column | meaning |
|---|---|
| `team_goals` | goals scored by `team` against `opponent` at **90 min** (extra time and penalties excluded, same rule as Model 2.1) |

## 8 explanatory variables (all pre-match)
All history-based values use only internationals played **before the match date** (state frozen at start of match day, same method as KB).

| # | column | shared with 2.1? | definition | why pre-match valid |
|---|---|---|---|---|
| 1 | `elo_diff` | yes | team Elo − opponent Elo (KB's Elo method) | rating frozen at start of match day |
| 2 | `top5_share_diff` | yes | team's share of 26-man squad at ENG/ESP/GER/ITA/FRA clubs − opponent's | squads registered before opening match |
| 3 | `host_adv` | yes (counted to be safe) | +1 team plays in own country, −1 opponent does, 0 neither | venues fixed before tournament |
| 4 | `attack_l10` | no | team's avg goals **scored** in its last 10 internationals | only prior results |
| 5 | `opp_defence_l10` | no | opponent's avg goals **conceded** in its last 10 internationals | only prior results |
| 6 | `h2h_goals_pg` | no | team's avg goals per game in previous meetings with this opponent; 1.36 if never met (avg goals per team per game, all internationals 2018 → 10 Jun 2026) | only prior meetings |
| 7 | `elo_change_1y` | no | team Elo at start of match day − team Elo 365 days earlier (momentum) | only prior results |
| 8 | `knockout` | no | 1 = Round of 32 onward, 0 = group stage | schedule fixed before tournament |

**Overlap rule:** 3 of 8 shared with Model 2.1 (limit 4). `host_adv` is the team-view version of `host_adv_diff`; counted as shared to stay safe.

Helper columns (not predictors): `match_id` (same ids as KB's file), `date`, `stage`, `side` (1 = listed first in fixture), `team`, `opponent`, `venue`, `opp_goals`, `h2h_meetings`.
`opp_goals` is in-match → **never use as a predictor.**

## Checks done (asserts in script)
- 208 rows, every `match_id` exactly twice, no missing values.
- Two rows of each match mirror each other (`team_goals` ↔ `opp_goals`, `elo_diff` and `host_adv` sign-flipped).
- `match_id`, team order, 90-min goals and `elo_diff` agree with `../kb_model21/model21_matches_KB.csv`.
- Overlap with Model 2.1 ≤ 4.

## Quick results
`team_goals`: mean 1.44, range 0–7, skew 1.23 (56 zeros).

| variable | r with team_goals | VIF |
|---|---|---|
| elo_diff | 0.46 | 2.53 |
| top5_share_diff | 0.44 | 1.93 |
| opp_defence_l10 | 0.28 | 1.46 |
| attack_l10 | 0.25 | 1.67 |
| h2h_goals_pg | 0.20 | 1.15 |
| host_adv | 0.15 | 1.05 |
| elo_change_1y | −0.02 | 1.28 |
| knockout | −0.06 | 1.22 |

Highest inter-correlation: `elo_diff` ↔ `top5_share_diff` r = 0.66. All VIF < 3. Full output: `outputs_summary.txt`.

## Notes for Prabin (model building)
- **Independence:** 2 rows per match are linked (same game). Report cluster-robust SEs by `match_id` (`fit(cov_type="cluster", cov_kwds={"groups": df.match_id})`) and mention in report. Train/test split by `match_id`, not by row, so a match is never in both sets.
- **Target is a count, right-skewed** → residuals may fail normality / equal variance. Check Shapiro-Wilk + Breusch-Pagan; if failed, report HC3/cluster SEs. Optional comparison: `log(team_goals + 1)` or Poisson GLM.
- `elo_change_1y` and `knockout` weak alone (|r| < 0.06); kept because brief needs 8 and they add different info (low VIF), same reasoning Nabin used.
- `attack_l10` / `opp_defence_l10` include earlier WC 2026 games played before that match day — still pre-match.
- Limitation: `h2h_goals_pg` uses fallback 1.36 for 33 pairs that never met; `h2h_meetings` column lets you test a "never met" flag.

## Sources
- openfootball/worldcup.json – https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json (+ squads, stadiums)
- martj42/international_results – https://raw.githubusercontent.com/martj42/international_results/master/results.csv
