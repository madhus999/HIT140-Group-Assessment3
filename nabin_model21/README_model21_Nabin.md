# Model 2.1 – Remaining 4 variables + regression (Nabin part)

**Student ID:** s397551 (Nabin Thapa)

Builds on KB's file `../kb_model21/model21_matches_KB.csv` (104 matches, target `goal_diff` at 90 min, 4 variables).

## How to run

    pip install pandas numpy statsmodels matplotlib scipy scikit-learn
    cd nabin_model21
    python build_model21_full_dataset.py    # writes model21_matches_full.csv + team_lookup.csv
    python model21_regression.py            # prints results, writes ./outputs

## The 8 explanatory variables (all team1 − team2, all known before kick-off)

| # | column | by | definition | why pre-match valid |
|---|---|---|---|---|
| 1 | `elo_diff` | KB | World Football Elo rating difference | rating frozen at start of match day |
| 2 | `h2h_score_rate` | KB | team1's historical head-to-head score rate (0.5 if never met) | only prior meetings |
| 3 | `rest_days_diff` | KB | difference in days since last match (capped at 14) | fixture dates known in advance |
| 4 | `form_gd_diff` | KB | difference in average goal difference, last 10 internationals | only prior results |
| 5 | `host_adv_diff` | Nabin | +1 if team1 plays in its own country, −1 if team2 does, else 0 | venues fixed before the tournament |
| 6 | `squad_age_diff` | Nabin | difference in mean squad age (years, at 11 June 2026) | squads registered before the opening match |
| 7 | `top5_share_diff` | Nabin | difference in share of the 26-man squad at clubs in ENG/ESP/GER/ITA/FRA | squads registered before the opening match |
| 8 | `wc_exp_diff` | Nabin | difference in number of previous World Cup finals tournaments (1930–2022) | uses tournaments up to 2022 only |

`top5_share_diff` is a free, reproducible proxy for squad quality (instead of Transfermarkt market values, which cannot be downloaded as open data).

## Results (n = 104)

| | value |
|---|---|
| R² / adjusted R² | 0.471 / 0.427 |
| F-test | F(8, 95) = 10.58, p < 0.001 |
| RMSE in-sample | 1.46 goals |
| Test set (21 unseen matches) | R² 0.444, RMSE 1.60, MAE 1.22 |
| Baseline on same test set (predict the mean) | R² 0.00, RMSE 2.15 |
| 5-fold cross-validation ×20 | R² 0.336, RMSE 1.63 |
| KB's 4 variables alone | R² 0.399, adjusted 0.375 |

Adding the 4 new variables is a significant improvement over KB's 4 alone (partial F = 3.24, p = 0.015).

| variable | coef | p (OLS) | p (robust HC3) | VIF |
|---|---|---|---|---|
| const | −0.362 | 0.273 | 0.365 | |
| elo_diff | 0.0031 | 0.015 | 0.032 | 3.53 |
| h2h_score_rate | 0.974 | 0.113 | 0.192 | 1.11 |
| rest_days_diff | 0.082 | 0.415 | 0.478 | 1.03 |
| form_gd_diff | −0.020 | 0.917 | 0.921 | 2.68 |
| host_adv_diff | 0.920 | 0.049 | 0.135 | 1.14 |
| squad_age_diff | 0.153 | 0.233 | 0.227 | 1.58 |
| top5_share_diff | 2.060 | 0.003 | 0.005 | 3.38 |
| wc_exp_diff | −0.017 | 0.564 | 0.619 | 3.14 |

## Exploratory analysis (why these 8 variables)

Variables were first chosen on football logic, then checked against the target with scatterplots (`outputs/fig4_eda_scatterplots.png`) and Pearson correlation.

| variable | Pearson r with goal_diff | p-value |
|---|---|---|
| elo_diff | 0.60 | < 0.001 |
| top5_share_diff | 0.58 | < 0.001 |
| wc_exp_diff | 0.52 | < 0.001 |
| form_gd_diff | 0.51 | < 0.001 |
| h2h_score_rate | 0.32 | 0.001 |
| host_adv_diff | 0.16 | 0.095 |
| rest_days_diff | 0.10 | 0.337 |
| squad_age_diff | −0.07 | 0.497 |

Five variables are clearly related to goal difference on their own. `rest_days_diff` and `squad_age_diff` are not; they were kept because the brief requires exactly 8 and they carry different information from the strength variables (low VIF).

## Competing algorithms

Same 8 variables, same repeated 5-fold cross-validation (×10) for every model. Baseline (always predict the mean): RMSE 2.02.

| model | CV R² | CV RMSE | CV MAE |
|---|---|---|---|
| Random forest | 0.413 | 1.534 | 1.220 |
| Gradient boosting | 0.386 | 1.569 | 1.252 |
| Linear regression (2 significant variables) | 0.380 | 1.578 | 1.237 |
| Ridge regression | 0.363 | 1.600 | 1.262 |
| Lasso regression | 0.349 | 1.617 | 1.268 |
| Linear regression (8 variables) | 0.340 | 1.628 | 1.278 |

All six models beat the baseline and sit within 0.1 goals of each other. The random forest is slightly best, and the 8-variable linear regression is last because six of its variables add noise rather than signal. Recommendation: linear regression remains the model to report because it is interpretable and the assignment requires it; the comparison shows a more flexible model gains only about 0.09 goals of accuracy.

## Assumption checks

| assumption | check | result |
|---|---|---|
| Linearity | residuals vs fitted; Ramsey RESET p = 0.174 | met |
| Normal residuals | Q-Q plot; Shapiro-Wilk W = 0.981, p = 0.141 | met |
| Equal variance | Breusch-Pagan p = 0.028 | **not met** → robust (HC3) standard errors reported |
| Independence | Durbin-Watson = 1.99 | met |
| No multicollinearity | all VIF < 5 (max 3.53) | met |
| Influential points | max Cook's distance 0.117, no standardised residual above 3 | none removed |

## Notes for Prabin (report)

- KB's warning about `elo_diff` ↔ `form_gd_diff` (r = 0.76): VIF is 3.53 and 2.68, both under 5, so both were kept. Once Elo is in the model, form adds nothing (p = 0.92).
- Robust standard errors change one conclusion: `host_adv_diff` is significant with ordinary errors (p = 0.049) but not with robust errors (p = 0.135). Only 13 of 104 matches involved a host, so report it as weak evidence.
- The diagonal stripes in the residuals-vs-fitted plot appear because goal difference is a whole number; they are not a problem.
- Limitation: `wc_exp_diff` counts teams under their current name (Czech Republic does not inherit Czechoslovakia's appearances).

## Note for Suman (Model 2.2 overlap rule)

Model 2.1 uses the 8 variables above. Model 2.2 may share **at most 4** of them. Team-level versions (e.g. a team's own Elo instead of the difference) are different variables, but check with the lecturer if unsure.

## Files

- `build_model21_full_dataset.py` – adds variables 5–8
- `model21_matches_full.csv` – final 104-row dataset
- `team_lookup.csv` – per-team squad age, top-5 share, World Cup appearances
- `model21_regression.py` – model, assumption checks, evaluation
- `outputs/` – results text, coefficient, EDA and model-comparison tables, 5 figures
- `raw/` – squads and stadiums from openfootball/worldcup.json (2026)
