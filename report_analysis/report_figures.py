"""
HIT140 Assessment 3 - the four figures used in the group Word report.
Run from the repo root after optimise_models.py:  python report_analysis/report_figures.py
"""
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, numpy as np, pandas as pd, statsmodels.api as sm
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"; OUT.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
C1, C2, GREY, LIGHT = "#2563a8", "#c2571a", "#666666", "#c9c9c9"

d1 = pd.read_csv(ROOT / "nabin_model21/model21_matches_full.csv")
X1 = ["elo_diff","h2h_score_rate","rest_days_diff","form_gd_diff","host_adv_diff","squad_age_diff","top5_share_diff","wc_exp_diff"]
d2 = pd.read_csv(ROOT / "suman_model22/model22_team_matches.csv")
X2 = ["elo_diff","top5_share_diff","host_adv","attack_l10","opp_defence_l10","h2h_goals_pg","elo_change_1y","knockout"]
m1 = sm.OLS(d1.goal_diff, sm.add_constant(d1[X1])).fit(cov_type="HC3")
m2 = sm.OLS(d2.team_goals, sm.add_constant(d2[X2])).fit(cov_type="cluster", cov_kwds={"groups": d2.match_id})
NICE = {"elo_diff": "Elo rating gap", "top5_share_diff": "Squad at top-5 league clubs", "host_adv_diff": "Host nation", "host_adv": "Host nation",
        "h2h_score_rate": "Head-to-head record", "h2h_goals_pg": "Head-to-head goals", "rest_days_diff": "Rest days", "form_gd_diff": "Recent form",
        "squad_age_diff": "Squad age", "wc_exp_diff": "World Cup experience", "attack_l10": "Own recent scoring",
        "opp_defence_l10": "Opponent recent conceding", "elo_change_1y": "Elo momentum (1 yr)", "knockout": "Knockout stage"}

# Figure 1 - EDA: strongest predictors against each target
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.85))
jit = np.random.default_rng(1).uniform(-.13, .13, len(d2))
for a, (df, x, y, lab, c) in zip(ax, [(d1,"elo_diff","goal_diff","2.1",C1),(d1,"top5_share_diff","goal_diff","2.1",C1),
                                      (d2,"elo_diff","team_goals","2.2",C2),(d2,"top5_share_diff","team_goals","2.2",C2)]):
    r, _ = stats.pearsonr(df[x], df[y])
    a.scatter(df[x], df[y] + (jit if y == "team_goals" else 0), s=6, alpha=.45, color=c, lw=0)
    s, i = np.polyfit(df[x], df[y], 1); xs = np.array([df[x].min(), df[x].max()]); a.plot(xs, i + s*xs, color="black", lw=1)
    a.set_title(f"Model {lab}: {NICE[x]}\nr = {r:.2f}, p < 0.001", fontsize=7.5)
    a.set_xlabel(x, fontsize=7); a.set_ylabel(y, fontsize=7, labelpad=1)
fig.tight_layout(pad=.5, w_pad=1.0); fig.savefig(OUT / "fig1_eda.png", dpi=300); plt.close(fig)

# Figure 2 - what a typical pre-match advantage is worth, in goals, with 95% CI
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.35), sharex=False)
for a, (m, df, X, lab, c, unit) in zip(ax, [(m1, d1, X1, "Model 2.1: effect on goal difference", C1, "goal difference"),
                                           (m2, d2, X2, "Model 2.2: effect on goals scored", C2, "goals scored")]):
    sd = df[X].std(); sd[[v for v in X if v.startswith("host") or v == "knockout"]] = 1   # binary-type: one step
    ci = m.conf_int().loc[X]; eff = (m.params[X] * sd); lo, hi = ci[0] * sd, ci[1] * sd
    order = eff.sort_values().index
    for j, v in enumerate(order):
        sig = m.pvalues[v] < 0.05
        a.plot([lo[v], hi[v]], [j, j], color=c if sig else LIGHT, lw=2.2, solid_capstyle="round")
        a.plot(eff[v], j, "o", color=c if sig else GREY, ms=4)
    a.axvline(0, color="black", lw=.8)
    a.set_yticks(range(len(order)), [NICE[v] for v in order], fontsize=7)
    a.set_title(lab, fontsize=8); a.set_xlabel(f"change in {unit} (95% CI)", fontsize=7)
fig.tight_layout(pad=.5, w_pad=2.0, rect=(0, 0, 1, 1)); fig.savefig(OUT / "fig2_effects.png", dpi=300); plt.close(fig)

# Figure 3 - residual diagnostics, both models
o1 = sm.OLS(d1.goal_diff, sm.add_constant(d1[X1])).fit(); o2 = sm.OLS(d2.team_goals, sm.add_constant(d2[X2])).fit()
fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.8))
for k, (m, lab, c) in enumerate([(o1, "2.1", C1), (o2, "2.2", C2)]):
    a = ax[2*k]; a.scatter(m.fittedvalues, m.resid, s=6, alpha=.45, color=c, lw=0); a.axhline(0, color="black", lw=.8)
    a.set_title(f"Model {lab}: residuals vs fitted", fontsize=7.5); a.set_xlabel("fitted", fontsize=7); a.set_ylabel("residual", fontsize=7, labelpad=1)
    a = ax[2*k+1]; (osm, osr), (sl, ic, _) = stats.probplot(m.resid); a.scatter(osm, osr, s=6, alpha=.55, color=c, lw=0); a.plot(osm, ic + sl*osm, color="black", lw=.8)
    a.set_title(f"Model {lab}: normal Q-Q", fontsize=7.5); a.set_xlabel("theoretical quantile", fontsize=7); a.set_ylabel("residual", fontsize=7, labelpad=1)
fig.tight_layout(pad=.5, w_pad=.8); fig.savefig(OUT / "fig3_diagnostics.png", dpi=300); plt.close(fig)

# Figure 4 - optimisation: (a) algorithms vs OLS, (b) which variables survive selection
a1 = pd.read_csv(OUT / "model21_algorithms.csv", index_col=0); a2 = pd.read_csv(OUT / "model22_algorithms.csv", index_col=0)
st = pd.read_csv(OUT / "selection_stability.csv", index_col=0)
fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.3), gridspec_kw={"width_ratios": [1, 1.15]})
algs = ["Random forest", "Gradient boosting", "Ridge (tuned)", "Lasso (tuned)", "Poisson GLM"]
y = np.arange(len(algs)); a = ax[0]
v1 = [a1.change_vs_ols_pct.get(n, np.nan) for n in algs]; v2 = [a2.change_vs_ols_pct.get(n, np.nan) for n in algs]
a.barh(y - .19, v1, .36, color=C1, label="Model 2.1"); a.barh(y + .19, v2, .36, color=C2, label="Model 2.2")
for j, (p, q) in enumerate(zip(v1, v2)):
    if np.isnan(p): a.text(.3, j - .19, "n/a (2.1)", fontsize=6, va="center", color=GREY)
a.axvline(0, color="black", lw=.8); a.set_yticks(y, algs, fontsize=7); a.invert_yaxis()
a.set_xlabel("% lower prediction error than OLS (right = better)", fontsize=7); a.set_title("(a) Competing algorithms vs OLS", fontsize=8)
a.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.62))
a = ax[1]
rows = [("Elo rating gap", "elo_diff", "elo_diff"), ("Squad at top-5 clubs", "top5_share_diff", "top5_share_diff"), ("Host nation", "host_adv_diff", "host_adv"),
        ("Head-to-head", "h2h_score_rate", "h2h_goals_pg"), ("Recent form / scoring", "form_gd_diff", "attack_l10"),
        ("Opponent conceding", None, "opp_defence_l10"), ("Rest days", "rest_days_diff", None), ("Squad age", "squad_age_diff", None),
        ("World Cup experience", "wc_exp_diff", None), ("Elo momentum", None, "elo_change_1y"), ("Knockout stage", None, "knockout")]
y = np.arange(len(rows))
s1 = [100 * st["Model 2.1"].get(r[1], np.nan) if r[1] else np.nan for r in rows]; s2 = [100 * st["Model 2.2"].get(r[2], np.nan) if r[2] else np.nan for r in rows]
a.barh(y - .19, s1, .36, color=C1); a.barh(y + .19, s2, .36, color=C2)
a.axvline(50, color=GREY, ls=":", lw=.8); a.set_yticks(y, [r[0] for r in rows], fontsize=6.8); a.invert_yaxis(); a.set_xlim(0, 100)
a.set_xlabel("% of training folds keeping the variable", fontsize=7); a.set_title("(b) Which variables survive selection", fontsize=8)
fig.tight_layout(pad=.5, w_pad=1.5); fig.savefig(OUT / "fig4_optimisation.png", dpi=300); plt.close(fig)
print("figures written to", OUT)
