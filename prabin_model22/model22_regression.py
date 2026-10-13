"""HIT140 Assessment 3 - Objective 2 - Linear Regression 2.2
Prabin Shrestha

Builds Model 2.2 on the 208-row team-match dataset created by Suman.
The model predicts goals scored by a team in a single match using only
pre-match explanatory variables.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import statsmodels.api as sm

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


def main() -> None:
    df = pd.read_csv(DATA_PATH)

    missing = df[MODEL_VARS + ["team_goals", "match_id"]].isna().any()
    if missing.any():
        raise ValueError(f"Missing values found in required columns:\n{missing[missing].to_string()}")

    overlap_count = len(set(MODEL_VARS) & {"elo_diff", "h2h_score_rate", "rest_days_diff", "form_gd_diff", "host_adv_diff", "squad_age_diff", "top5_share_diff", "wc_exp_diff"})
    if overlap_count > 4:
        raise ValueError(f"Overlap exceeds 4-variable rule: {overlap_count}")

    X = sm.add_constant(df[MODEL_VARS], has_constant="add")
    y = df["team_goals"]

    model = sm.OLS(y, X).fit(cov_type="cluster", cov_kwds={"groups": df["match_id"]})
    rmse = (model.resid ** 2).mean() ** 0.5

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
        "Coefficient estimates:",
    ]

    for var in MODEL_VARS:
        coef = model.params[var]
        p_value = model.pvalues[var]
        summary_lines.append(f"- {var}: coef={coef:.4f}, p={p_value:.4f}")

    summary_lines.extend([
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
