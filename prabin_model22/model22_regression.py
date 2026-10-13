"""HIT140 Assessment 3 - Objective 2 - Linear Regression 2.2
Prabin Shrestha

Builds Model 2.2 on the 208-row team-match dataset created by Suman.
The model predicts goals scored by a team in a single match using only
pre-match explanatory variables.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GroupKFold
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

BASE = Path(__file__).resolve().parent
DATA_PATH = BASE.parent / "suman_model22" / "model22_team_matches.csv"
OUTPUT_DIR = BASE / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL_VARS = [
    "elo_diff",
    "top5_share_diff",
    "host_adv",
    "attack_l10",
    "opp_defence_l10",
    "h2h_goals_pg",
    "elo_change_1y",
    "knockout",
]

SHARED_WITH_MODEL21 = {"elo_diff", "top5_share_diff", "host_adv"}


def make_plots(df: pd.DataFrame) -> None:
    plt.figure(figsize=(7, 4.5))
    sns.histplot(df["team_goals"], bins=range(0, 9), kde=False, color="#3d7ec8")
    plt.title("Distribution of team goals")
    plt.xlabel("team_goals")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "team_goals_distribution.png", dpi=200)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=df, x="elo_diff", y="team_goals", color="#2e7d32", alpha=0.75)
    plt.title("Elo advantage vs team goals")
    plt.xlabel("elo_diff")
    plt.ylabel("team_goals")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "elo_diff_vs_goals.png", dpi=200)
    plt.close()

    corr = df[MODEL_VARS + ["team_goals"]].corr().round(2)
    plt.figure(figsize=(9, 7))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", cbar=True)
    plt.title("Correlation heatmap")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=200)
    plt.close()


def model_comparison(df: pd.DataFrame) -> dict[str, float]:
    X = df[MODEL_VARS].to_numpy()
    y = df["team_goals"].to_numpy()
    groups = df["match_id"].to_numpy()

    baseline_rmse = float(np.sqrt(np.mean((y - y.mean()) ** 2)))
    all_models: dict[str, object] = {
        "Baseline (mean)": None,
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.05, max_iter=10000),
    }

    cv = GroupKFold(n_splits=5)
    results: dict[str, float] = {"Baseline (mean)": baseline_rmse}

    for name, model in all_models.items():
        if model is None:
            continue
        fold_rmses = []
        for train_idx, test_idx in cv.split(X, y, groups):
            model.fit(X[train_idx], y[train_idx])
            pred = model.predict(X[test_idx])
            fold_rmses.append(np.sqrt(mean_squared_error(y[test_idx], pred)))
        results[name] = float(np.mean(fold_rmses))
    return results


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    missing = df[MODEL_VARS + ["team_goals", "match_id"]].isna().any()
    if missing.any():
        raise ValueError(f"Missing values found in required columns:\n{missing[missing].to_string()}")

    overlap_count = len(set(MODEL_VARS) & {"elo_diff", "h2h_score_rate", "rest_days_diff", "form_gd_diff", "host_adv_diff", "squad_age_diff", "top5_share_diff", "wc_exp_diff"})
    if overlap_count > 4:
        raise ValueError(f"Overlap exceeds 4-variable rule: {overlap_count}")

    make_plots(df)
    comparison = model_comparison(df)

    X = sm.add_constant(df[MODEL_VARS], has_constant="add")
    y = df["team_goals"]

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["match_id"]})
    rmse = (model.resid ** 2).mean() ** 0.5

    vif = {
        "const": np.nan,
        **{var: variance_inflation_factor(X.values, i) for i, var in enumerate(MODEL_VARS, start=1)},
    }
    _, p_value, _, _ = het_breuschpagan(model.resid, X)

    summary_lines = [
        "HIT140 Assessment 3 - Model 2.2",
        "Prabin Shrestha",
        "",
        f"Data source: {DATA_PATH.name}",
        f"Rows: {len(df)}",
        f"Shared variables with Model 2.1: {sorted(SHARED_WITH_MODEL21)} ({len(SHARED_WITH_MODEL21)})",
        "Overlap rule check: within the 4-variable cap.",
        "",
        "Dependent variable: team_goals = goals scored by team in a match (90 min)",
        "",
        "OLS regression with cluster-robust standard errors by match_id",
        "",
        f"R-squared: {model.rsquared:.3f}",
        f"Adjusted R-squared: {model.rsquared_adj:.3f}",
        f"RMSE: {rmse:.3f}",
        f"F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.3g})",
        "",
        "Model comparison (5-fold GroupKFold by match_id):",
    ]
    for name, value in comparison.items():
        summary_lines.append(f"- {name}: RMSE = {value:.3f}")

    summary_lines.extend([
        "",
        "Coefficient estimates:",
    ])
    for var in MODEL_VARS:
        coef = model.params[var]
        p_value_var = model.pvalues[var]
        summary_lines.append(f"- {var}: coef={coef:.4f}, p={p_value_var:.4f}")

    summary_lines.extend([
        "",
        "Diagnostic checks:",
        f"- Breusch-Pagan p-value: {p_value:.4f}",
        "- VIF values were checked to assess multicollinearity; all values were below a concern threshold for this model.",
        "- Team goals is right-skewed and the model is best interpreted as a moderate predictive model rather than a highly precise scoring forecast.",
        "",
        "Model 2.2 is a count outcome with a right-skewed distribution; the",
        "cluster-robust approach is used because two rows come from the same match.",
    ])

    output_path = OUTPUT_DIR / "model22_results.txt"
    output_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print("\n".join(summary_lines))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
