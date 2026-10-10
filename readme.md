# HIT140 Foundations of Data Science – Assessment 3 (Objective 2)

Charles Darwin University · Semester 2, 2026

Two linear regression models on FIFA World Cup 2026 data, built in Python.

| Model | Predicts | Rows | Explanatory vars |
|---|---|---|---|
| 2.1 | Match goal difference (team1 − team2, 90 min) | 104 (one per match) | 8 |
| 2.2 | Goals scored by a team in a match | 208 (two per match) | 8 |

**Rules followed**
- All explanatory variables are known before kick-off (no in-match stats).
- At most 4 variables are shared between Model 2.1 and Model 2.2.

## Team

| Member | Student ID | Role | Folder |
|---|---|---|---|
| KB (Khadka Bohora) | <ID> | Model 2.1 dataset: 104 matches + 4 pre-match vars | kb_model21/ |
| Nabin | <ID> | Model 2.1: remaining 4 vars + regression | nabin_model21/ |
| Suman | <ID> | Model 2.2 dataset: 208 team-match rows + 8 vars | suman_model22/ |
| Prabin | <ID> | Model 2.2 regression, assumption checks, report | prabin_model22/ |

## Repository structure

- **kb_model21/**
  - raw/ – source data (untouched downloads)
  - build_model21_dataset.py – builds the 104-row dataset
  - model21_matches_KB.csv – output dataset
  - README_model21_KB.md – data dictionary + notes
- **nabin_model21/**
  - build_model21_full_dataset.py – adds 4 more pre-match variables
  - model21_matches_full.csv – final 104-row dataset (8 variables)
  - model21_regression.py – OLS model, assumption checks, evaluation
  - outputs/ – results, tables, figures
  - README_model21_Nabin.md – variables, results, notes
- **suman_model22/** – to be added
- **prabin_model22/** – to be added

## How to run

Requires Python 3.10+.

    pip install pandas numpy statsmodels matplotlib seaborn scipy
    cd kb_model21
    python3 build_model21_dataset.py
    cd ../nabin_model21
    python3 build_model21_full_dataset.py
    python3 model21_regression.py

## Data sources

- openfootball/worldcup.json – WC 2026 match results: https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json
- martj42/international_results – all men's internationals since 1872: https://raw.githubusercontent.com/martj42/international_results/master/results.csv

## Variables

| Variable | Model 2.1 | Model 2.2 |
|---|---|---|
| elo_diff | ✅ | |
| h2h_score_rate | ✅ | |
| rest_days_diff | ✅ | |
| form_gd_diff | ✅ | |
| host_adv_diff | ✅ | |
| squad_age_diff | ✅ | |
| top5_share_diff | ✅ | |
| wc_exp_diff | ✅ | |
| (Suman adds 8) | | ✅ |
