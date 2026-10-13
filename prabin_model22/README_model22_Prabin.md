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

## Model comparison and diagnostics

A simple baseline model was compared against OLS, Ridge and Lasso using 5-fold cross-validation grouped by `match_id`, so the same match does not appear in both training and validation folds.

| model | CV RMSE |
|---|---:|
| Baseline (mean) | 1.382 |
| Linear Regression | 1.214 |
| Ridge | 1.213 |
| Lasso | 1.233 |

The OLS model performs best among the simple alternatives and is therefore the main model reported. The improvement over the mean baseline is modest, which is consistent with the fact that goals are inherently noisy and relatively low-scoring events.

## Diagnostic checks

- Residual diagnostics were reviewed for heteroskedasticity using the Breusch-Pagan test. The p-value is 0.0023, indicating some evidence of non-constant variance.
- VIF values were checked to assess multicollinearity; no variable showed a severe inflation issue.
- The target variable (`team_goals`) is right-skewed and count-like, so the model is best interpreted as a moderate predictive model rather than a highly precise scoring forecast.
- Because each match contributes two rows (one per team), cluster-robust standard errors by `match_id` were used in the reported OLS model.

## Output files

- `outputs/model22_results.txt` — full results summary
- `outputs/team_goals_distribution.png` — distribution of team goals
- `outputs/elo_diff_vs_goals.png` — scatterplot for Elo advantage vs goals
- `outputs/correlation_heatmap.png` — variable correlation heatmap

## How to run

```bash
cd prabin_model22
python model22_regression.py
```

This writes the regression summary to `outputs/model22_results.txt`.
