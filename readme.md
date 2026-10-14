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
| KB (Khadka Bohora) |  | Model 2.1 dataset: 104 matches + 4 pre-match vars | kb_model21/ |
| Nabin | s397551 | Model 2.1: remaining 4 vars + regression | nabin_model21/ |
| Suman Silwal | s397846 | Model 2.2 dataset: 208 team-match rows + 8 vars | suman_model22/ |
| Prabin |  | Model 2.2 regression, assumption checks, report | prabin_model22/ |

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
- **suman_model22/**
  - build_model22_dataset.py – builds the 208-row team-match dataset (reads kb/nabin raw/)
  - model22_team_matches.csv – output dataset (8 variables + target team_goals)
  - outputs_summary.txt – descriptive stats, correlations, VIF
  - README_model22_Suman.md – data dictionary + notes for Prabin
- **prabin_model22/**
  - model22_regression.py – OLS with cluster-robust errors, diagnostics, grouped-CV model comparison
  - outputs/ – results, comparison table, 5 figures
  - README_model22_Prabin.md – results and report notes
- **report_analysis/**
  - optimise_models.py – feature selection, Poisson and log models, nested check (both models)
  - compare_algorithms.py – OLS vs ridge, lasso, random forest, boosting, Poisson under one validation design
  - report_figures.py – the 4 figures used in the Word report
  - outputs/ – results and figures
  - README_report_analysis.md – method and key results

## How to run

Requires Python 3.10+.

    pip install pandas numpy statsmodels matplotlib seaborn scipy scikit-learn
    cd kb_model21
    python3 build_model21_dataset.py
    cd ../nabin_model21
    python3 build_model21_full_dataset.py
    python3 model21_regression.py
    cd ../suman_model22
    python3 build_model22_dataset.py
    cd ../prabin_model22
    python3 model22_regression.py
    cd ..
    python3 report_analysis/optimise_models.py
    python3 report_analysis/compare_algorithms.py
    python3 report_analysis/report_figures.py

## Data sources

- openfootball/worldcup.json – WC 2026 match results: https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json
- martj42/international_results – all men's internationals since 1872: https://raw.githubusercontent.com/martj42/international_results/master/results.csv

## Variables

| Variable | Model 2.1 | Model 2.2 |
|---|---|---|
| elo_diff | ✅ | ✅ |
| h2h_score_rate | ✅ | |
| rest_days_diff | ✅ | |
| form_gd_diff | ✅ | |
| host_adv_diff (2.1) / host_adv (2.2, team view) | ✅ | ✅ |
| squad_age_diff | ✅ | |
| top5_share_diff | ✅ | ✅ |
| wc_exp_diff | ✅ | |
| attack_l10 | | ✅ |
| opp_defence_l10 | | ✅ |
| h2h_goals_pg | | ✅ |
| elo_change_1y | | ✅ |
| knockout | | ✅ |

Shared between models: 3 (limit 4).
