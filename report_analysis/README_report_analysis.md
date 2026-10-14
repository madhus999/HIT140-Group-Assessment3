# Report analysis – optimisation, competing algorithms and report figures

Shared analysis for the group Word report, covering both models under one validation design.
Reads the final datasets from `../nabin_model21/` and `../suman_model22/`; no new data.

## How to run (from the repo root, ~8 minutes in total)

    pip install pandas numpy statsmodels scikit-learn matplotlib scipy
    python report_analysis/optimise_models.py      # feature selection, Poisson, log model, nested check (~6 min)
    python report_analysis/compare_algorithms.py   # OLS vs ridge, lasso, random forest, boosting, Poisson (~2 min)
    python report_analysis/report_figures.py       # the 4 figures used in the report

## Validation design

- Model 2.1: 5-fold cross-validation repeated 10–20 times.
- Model 2.2: grouped 5-fold cross-validation; both rows of a match always stay in the same fold.
- Ridge and lasso penalties are tuned inside each training fold only (nested), with grouped inner folds for Model 2.2.
- Best-subset selection (all 255 subsets of 8 variables) is checked with a nested run, where the whole search
  is repeated inside each training fold, to measure selection optimism.

## Key results

| | Model 2.1 CV RMSE | Model 2.2 CV RMSE |
|---|---|---|
| Mean baseline | 2.024 | 1.389 |
| OLS, 8 variables | 1.628 | 1.237 |
| Ridge (tuned, nested) | 1.605 (+1.4%) | 1.229 (+0.7%) |
| Random forest | 1.541 (+5.3%) | 1.261 (−1.9%) |
| Best subset, selected on all data | 1.569 (+3.9%) | 1.219 (+1.9%) |
| Best subset, selection nested | 1.627 (+0.3%) | 1.238 (+0.3%) |

- The apparent gain from best-subset selection is mostly selection optimism.
- Elo and top-5 share are kept in over 90% of folds for both models.
- Ridge is the only method that improves both models under honest validation, so it is the recommended optimised model.

## Outputs (`outputs/`)

- `optimisation_results.txt`, `model21_optimisation.csv`, `model22_optimisation.csv`, `selection_stability.csv`
- `algorithm_comparison.txt`, `model21_algorithms.csv`, `model22_algorithms.csv`
- `fig1_eda.png`, `fig2_effects.png`, `fig3_diagnostics.png`, `fig4_optimisation.png`
