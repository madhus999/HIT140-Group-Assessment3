# Model 2.1 – Match dataset (KB part)

**File:** `model21_matches_KB.csv` — 104 rows (every WC 2026 match), 0 missing values except `group` (blank for knockouts).
**Rebuild:** `python build_model21_dataset.py` (reads `raw/`, writes the CSV).

## Target
| column | meaning |
|---|---|
| `goal_diff` | `team1_goals − team2_goals` at **90 min** (extra time and penalties excluded, so every match is measured over the same duration) |

`extra_time` / `penalties` flags kept for reference (Final: 0–0 at 90', Spain won 1–0 AET).

## 4 explanatory variables (all pre-match)
Every value is computed only from internationals played **before the match date**.

| column | definition | why pre-match valid |
|---|---|---|
| `elo_diff` | World Football Elo rating team1 − team2 (eloratings.net method: K = 60 WC / 50 continental / 40 qualifiers / 30 other / 20 friendly; goal-margin multiplier; +100 home advantage in non-neutral games; start 1500; full history since 1872) | rating frozen at start of match day |
| `h2h_score_rate` | team1's historical record vs team2: (wins + 0.5·draws) / meetings. 0.5 if never met (33 pairs) | only prior meetings |
| `rest_days_diff` | days since team1's last match − days since team2's last match, each capped at 14 | fixture dates known in advance |
| `form_gd_diff` | team1 avg goal difference over its last 10 internationals − team2's | only prior results |

Helper columns (not explanatory vars, don't count toward the 8): `match_id`, `date`, `stage`, `group`, `team1`, `team2`, `venue`, `team1_goals`, `team2_goals`, `team1_elo`, `team2_elo`, `h2h_meetings`.

## Quick results
Correlation with `goal_diff`: elo_diff **0.60**, form_gd_diff **0.51**, h2h_score_rate **0.32**, rest_days_diff **0.10**.

## Notes for Nabin (model building)
- **Multicollinearity:** `elo_diff` ↔ `form_gd_diff` r = 0.76. Check VIF; if > 5, mention in report or drop/replace one.
- **Rest cap:** uncapped, Saudi Arabia v Uruguay was −70 (Uruguay's last game 31 Mar). Cap at 14 removes that leverage point — justify in the methodology.
- **Sign convention:** team1 = team listed first in FIFA fixture. All diffs are team1 − team2, matching the target.
- **Remaining 4 variables** must also be pre-match. Ideas: host-nation flag diff, squad market value diff (Transfermarkt), avg squad age diff, FIFA ranking diff (watch collinearity with Elo), travel distance, knockout-stage dummy.
- **Overlap rule:** max 4 vars shared with Model 2.2. Tell Suman which 8 we use.

## Sources
- openfootball/worldcup.json – https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json
- martj42/international_results – https://raw.githubusercontent.com/martj42/international_results/master/results.csv
