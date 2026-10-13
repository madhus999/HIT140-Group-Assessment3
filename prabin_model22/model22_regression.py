"""HIT140 Assessment 3 - Objective 2 - Linear Regression 2.2
Prabin Shrestha

Fits and evaluates a team-goals model using the 208-row team-match dataset.
All predictive validation splits keep both sides of a match together.
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
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.graphics.gofplots import qqplot
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from scipy.stats import shapiro

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

MODEL21_VARS = {
    "elo_diff",
    "h2h_score_rate",
    "rest_days_diff",
    "form_gd_diff",
    "host_adv_diff",
    "squad_age_diff",
    "top5_share_diff",
    "wc_exp_diff",
}
SHARED_WITH_MODEL21 = {"elo_diff", "top5_share_diff", "host_adv"}


def validate_dataset(df: pd.DataFrame) -> None:
    required = MODEL_VARS + ["team_goals", "match_id"]
    missing_columns = sorted(set(required) - set(df.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {missing_columns}")

    missing_values = df[required].isna().any()
    if missing_values.any():
        raise ValueError(
            "Missing values found in required columns:\n"
            f"{missing_values[missing_values].to_string()}"
        )
    if len(df) != 208 or df["match_id"].nunique() != 104:
        raise ValueError(
            f"Expected 208 rows from 104 matches; found {len(df)} rows "
            f"from {df['match_id'].nunique()} matches."
        )
    if not df.groupby("match_id").size().eq(2).all():
        raise ValueError("Each match_id must occur exactly twice.")
    if (df["team_goals"] < 0).any():
        raise ValueError("team_goals must be non-negative.")

    overlap = set(MODEL_VARS) & MODEL21_VARS
    if len(overlap) > 4:
        raise ValueError(f"Overlap exceeds the 4-variable rule: {sorted(overlap)}")


def make_eda_plots(df: pd.DataFrame) -> None:
    sns.set_theme(style="whitegrid", context="notebook")

    goal_counts = df["team_goals"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(goal_counts.index, goal_counts.values, color="#2878B5")
    ax.set(
        title="Most team-match observations contain zero or one goal",
        xlabel="Goals scored by team",
        ylabel="Number of team-match rows",
        xticks=goal_counts.index,
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "team_goals_distribution.png", dpi=220)
    plt.close(fig)

    elo_bins = pd.qcut(df["elo_diff"], q=5, duplicates="drop")
    elo_summary = (
        df.assign(elo_group=elo_bins)
        .groupby("elo_group", observed=True)["team_goals"]
        .agg(["mean", "count", "std"])
        .reset_index()
    )
    elo_summary["se"] = elo_summary["std"] / np.sqrt(elo_summary["count"])
    elo_summary["midpoint"] = elo_summary["elo_group"].map(
        lambda interval: (interval.left + interval.right) / 2
    )
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(
        elo_summary["midpoint"],
        elo_summary["mean"],
        yerr=1.96 * elo_summary["se"],
        marker="o",
        capsize=4,
        color="#2E7D32",
    )
    ax.axhline(df["team_goals"].mean(), color="#777777", linestyle="--", label="Overall mean")
    ax.set(
        title="Average goals rise across Elo-advantage groups",
        xlabel="Elo difference (group midpoint)",
        ylabel="Mean team goals (95% descriptive CI)",
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "elo_groups_vs_goals.png", dpi=220)
    plt.close(fig)

    corr = df[MODEL_VARS + ["team_goals"]].corr()
    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="vlag",
        center=0,
        square=True,
        ax=ax,
        cbar_kws={"label": "Pearson correlation"},
    )
    ax.set_title("Pre-match predictors and team-goals correlations")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "correlation_heatmap.png", dpi=220)
    plt.close(fig)


def compare_models(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[float]]]:
    features = df[MODEL_VARS].to_numpy(dtype=float)
    target = df["team_goals"].to_numpy(dtype=float)
    groups = df["match_id"].to_numpy()
    outer_cv = GroupKFold(n_splits=5)
    scores: dict[str, list[float]] = {
        "Mean baseline": [],
        "OLS": [],
        "Ridge (tuned)": [],
        "Lasso (tuned)": [],
        "Random forest (tuned)": [],
    }

    candidates = {
        "Ridge (tuned)": (
            make_pipeline(StandardScaler(), Ridge()),
            {"ridge__alpha": [0.1, 1.0, 10.0, 100.0]},
        ),
        "Lasso (tuned)": (
            make_pipeline(StandardScaler(), Lasso(max_iter=20000)),
            {"lasso__alpha": [0.01, 0.05, 0.1, 0.5]},
        ),
        "Random forest (tuned)": (
            RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=1),
            {
                "max_depth": [None, 3, 5],
                "min_samples_leaf": [2, 4],
                "max_features": [0.7, 1.0],
            },
        ),
    }

    for train_idx, test_idx in outer_cv.split(features, target, groups):
        X_train, X_test = features[train_idx], features[test_idx]
        y_train, y_test = target[train_idx], target[test_idx]
        train_groups = groups[train_idx]

        baseline_prediction = np.full(len(test_idx), y_train.mean())
        scores["Mean baseline"].append(
            float(np.sqrt(mean_squared_error(y_test, baseline_prediction)))
        )

        ols = LinearRegression().fit(X_train, y_train)
        scores["OLS"].append(
            float(np.sqrt(mean_squared_error(y_test, ols.predict(X_test))))
        )

        inner_cv = GroupKFold(n_splits=4)
        for name, (estimator, param_grid) in candidates.items():
            search = GridSearchCV(
                estimator,
                param_grid,
                scoring="neg_root_mean_squared_error",
                cv=inner_cv,
                n_jobs=1,
            )
            search.fit(X_train, y_train, groups=train_groups)
            scores[name].append(
                float(np.sqrt(mean_squared_error(y_test, search.predict(X_test))))
            )

    comparison = pd.DataFrame(
        [
            {
                "model": name,
                "mean_cv_rmse": np.mean(fold_scores),
                "sd_cv_rmse": np.std(fold_scores, ddof=1),
            }
            for name, fold_scores in scores.items()
        ]
    ).sort_values("mean_cv_rmse", ignore_index=True)
    return comparison, scores


def make_model_and_diagnostic_plots(
    comparison: pd.DataFrame,
    fitted: sm.regression.linear_model.RegressionResultsWrapper,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.8))
    ordered = comparison.sort_values("mean_cv_rmse", ascending=True)
    ax.barh(
        ordered["model"],
        ordered["mean_cv_rmse"],
        xerr=ordered["sd_cv_rmse"],
        color=["#2878B5" if model == "OLS" else "#9BB9D0" for model in ordered["model"]],
        capsize=4,
    )
    ax.set(
        title="Grouped cross-validation: lower RMSE is better",
        xlabel="Mean outer-fold RMSE (goals; error bars = fold SD)",
        ylabel="Model",
    )
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "grouped_cv_model_comparison.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].scatter(fitted.fittedvalues, fitted.resid, alpha=0.65, color="#2878B5")
    axes[0].axhline(0, color="#C0392B", linestyle="--")
    axes[0].set(
        title="Residuals vs fitted values",
        xlabel="Fitted goals",
        ylabel="OLS residuals",
    )
    qqplot(fitted.resid, line="45", ax=axes[1], markerfacecolor="#2878B5", alpha=0.6)
    axes[1].set_title("Normal Q-Q plot of OLS residuals")
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "ols_residual_diagnostics.png", dpi=220)
    plt.close(fig)


def main() -> None:
    df = pd.read_csv(DATA_PATH)
    validate_dataset(df)
    make_eda_plots(df)
    comparison, _ = compare_models(df)

    X = sm.add_constant(df[MODEL_VARS], has_constant="add")
    y = df["team_goals"]
    model = sm.OLS(y, X).fit(
        cov_type="cluster",
        cov_kwds={"groups": df["match_id"]},
    )
    rmse = float(np.sqrt(np.mean(model.resid**2)))

    vif_values = {
        variable: variance_inflation_factor(X.to_numpy(dtype=float), position)
        for position, variable in enumerate(X.columns)
        if variable != "const"
    }
    _, bp_lm_p, _, bp_f_p = het_breuschpagan(model.resid, X)
    _, shapiro_p = shapiro(model.resid)
    dw_stat = durbin_watson(model.resid)
    make_model_and_diagnostic_plots(comparison, model)
    shapiro_p_text = "<0.0001" if shapiro_p < 0.0001 else f"{shapiro_p:.4f}"

    comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
    summary_lines = [
        "HIT140 Assessment 3 - Model 2.2",
        "Prabin Shrestha",
        "",
        f"Data source: {DATA_PATH.name}",
        f"Rows: {len(df)} across {df['match_id'].nunique()} matches",
        "Predictors: exactly 8 pre-match explanatory variables",
        f"Shared variables with Model 2.1: {sorted(SHARED_WITH_MODEL21)} (3; maximum allowed 4)",
        "",
        "Primary model: OLS with cluster-robust standard errors by match_id",
        f"R-squared: {model.rsquared:.3f}",
        f"Adjusted R-squared: {model.rsquared_adj:.3f}",
        f"In-sample RMSE: {rmse:.3f} goals",
        f"Cluster-robust model F-statistic: {model.fvalue:.2f} (p={model.f_pvalue:.3g})",
        "",
        "Nested model comparison: 5-fold outer GroupKFold by match_id;",
        "regularisation/hyperparameters selected by 4-fold grouped CV within each training fold.",
        "All models and the mean baseline were evaluated on the same held-out match folds.",
        "Lower mean outer-fold RMSE is better; SD describes variation across the five folds.",
        comparison.to_string(index=False, formatters={
            "mean_cv_rmse": "{:.3f}".format,
            "sd_cv_rmse": "{:.3f}".format,
        }),
        "",
        "Full-data OLS coefficient estimates (cluster-robust p-values):",
    ]
    for variable in MODEL_VARS:
        summary_lines.append(
            f"- {variable}: coef={model.params[variable]:.4f}, "
            f"p={model.pvalues[variable]:.4f}"
        )

    summary_lines.extend([
        "",
        "Diagnostic checks on full-data OLS residuals:",
        f"- Breusch-Pagan LM p-value: {bp_lm_p:.4f}; F-test p-value: {bp_f_p:.4f}",
        f"- Shapiro-Wilk p-value: {shapiro_p_text}",
        f"- Durbin-Watson statistic: {dw_stat:.3f} (interpret cautiously for paired team rows)",
        "- Variance inflation factors:",
    ])
    summary_lines.extend(
        f"  - {variable}: {value:.3f}" for variable, value in vif_values.items()
    )
    summary_lines.extend([
        "",
        "Interpretation: the target is a right-skewed, discrete goal count. "
        "The diagnostics and comparison should be reported with this limitation; "
        "clustered errors address dependence between the two rows from a match, "
        "but do not make the goal outcome normally distributed.",
        "",
        "Figure captions / report talking points:",
        "- team_goals_distribution.png: Most team-match rows have few goals, "
        "showing the discrete and right-skewed target distribution.",
        "- elo_groups_vs_goals.png: Mean goals by Elo-difference group with "
        "approximate descriptive 95% intervals; because rows are paired within matches, "
        "these are exploratory, not formal inferential intervals.",
        "- correlation_heatmap.png: Summarises pairwise linear associations; "
        "association does not establish causation.",
        "- grouped_cv_model_comparison.png: Compares all methods on identical "
        "held-out match folds, preventing the two rows from one match leaking across folds.",
        "- ols_residual_diagnostics.png: Shows residual spread and departures "
        "from normality that should qualify inference from OLS.",
    ])

    output_path = OUTPUT_DIR / "model22_results.txt"
    output_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    print("\n".join(summary_lines))
    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()
