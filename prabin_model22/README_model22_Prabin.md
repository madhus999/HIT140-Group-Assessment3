# Model 2.2 – Prabin Shrestha

This folder contains the final linear regression for Model 2.2, which predicts `team_goals` (the number of goals a team scores in a single World Cup match at 90 minutes). The model uses the cleaned 208-row dataset created by Suman in `../suman_model22/model22_team_matches.csv` and is estimated in Python with `statsmodels` using cluster-robust standard errors grouped by `match_id`.

## Model specification

The regression uses the approved 8 pre-match predictors in the team-match dataset:

- Shared with Model 2.1: `elo_diff`, `top5_share_diff`, `host_adv`
- New unique team variables: `attack_l10`, `opp_defence_l10`, `h2h_goals_pg`, `elo_change_1y`, `knockout`

This respects the assignment rule because the overlap with Model 2.1 is only 3 variables, which is below the maximum 4-variable cap.

## Regression results

| metric | value |
|---|---:|
| R² | 0.272 |
| Adjusted R² | 0.242 |
| RMSE | 1.180 |
| F-statistic | 8.70 |
| p-value (F) | 5.17e-09 |

## Coefficients

| variable | coefficient | p-value |
|---|---:|---:|
| const | 0.8416 | 0.0775 |
| elo_diff | 0.0012 | 0.0257 |
| top5_share_diff | 0.8012 | 0.0010 |
| host_adv | 0.4821 | 0.0485 |
| attack_l10 | 0.1104 | 0.5446 |
| opp_defence_l10 | 0.3371 | 0.1444 |
| h2h_goals_pg | 0.0927 | 0.3562 |
| elo_change_1y | -0.0012 | 0.5063 |
| knockout | -0.0993 | 0.6107 |

## Short report paragraph

Model 2.2 predicts the number of goals a team is expected to score in a single World Cup match using 8 pre-match team-level indicators. The strongest predictors are the team’s Elo advantage and the share of its squad playing for the top five European clubs, both of which are positively associated with more goals. The model explains roughly 27% of the variation in team goals scored (R² = 0.272), with a root mean squared error of 1.18 goals, suggesting useful but limited predictive power for a low-scoring football outcome. Because two rows come from the same fixture, the regression uses cluster-robust standard errors by `match_id` to avoid overstating the certainty of the coefficients. Overall, Model 2.2 is interpretable and consistent with football logic, but it is not strong enough to be a highly precise scoring forecast on its own.

## How to run

```bash
cd prabin_model22
python model22_regression.py
```

This writes the regression summary to `outputs/model22_results.txt`.
