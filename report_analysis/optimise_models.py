"""
HIT140 Assessment 3 - model optimisation experiments for both models.
Every candidate is scored with the same repeated cross-validation, so improvements are comparable.
Model 2.1: 5-fold CV x20.  Model 2.2: grouped 5-fold CV x20 (both rows of a match stay in one fold).
Run from the repo root:  python report_analysis/optimise_models.py
"""
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from itertools import combinations

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"; OUT.mkdir(exist_ok=True)
REPS, K, SEED = 20, 5, 140

def folds(n_units, rep):
    return np.array_split(np.random.default_rng(SEED + rep).permutation(n_units), K)

def cv_rmse(df, y, cols, kind="ols", groups=None, transform=None):
    """mean RMSE over REPS x K-fold CV; groups = array of group ids (keeps groups together)"""
    units = np.unique(groups) if groups is not None else np.arange(len(df))
    out = []
    for rep in range(REPS):
        pred = np.empty(len(df))
        for te_u in folds(len(units), rep):
            te = np.isin(groups, units[te_u]) if groups is not None else np.isin(np.arange(len(df)), te_u)
            Xtr, Xte = sm.add_constant(df.loc[~te, cols], has_constant="add"), sm.add_constant(df.loc[te, cols], has_constant="add")
            ytr = df.loc[~te, y]
            if kind == "poisson":
                m = sm.GLM(ytr, Xtr, family=sm.families.Poisson()).fit(); pred[te] = m.predict(Xte)
            elif transform == "log1p":
                m = sm.OLS(np.log1p(ytr), Xtr).fit(); s2 = m.scale
                pred[te] = np.exp(m.predict(Xte) + s2 / 2) - 1          # smearing-type back-transform
            else:
                pred[te] = sm.OLS(ytr, Xtr).fit().predict(Xte)
        out.append(np.sqrt(np.mean((df[y] - pred) ** 2)))
    return float(np.mean(out)), float(np.std(out))

def best_subset(df, y, X, groups=None, kind="ols"):
    """exhaustive search over all 255 non-empty subsets of the 8 variables, scored by CV RMSE"""
    res = []
    for k in range(1, len(X) + 1):
        for c in combinations(X, k):
            res.append((cv_rmse(df, y, list(c), kind, groups)[0], c))
    res.sort()
    return res

lines = []
def say(s=""): print(s); lines.append(s)

# ---------------------------------------------------------------- Model 2.1
d1 = pd.read_csv(ROOT / "nabin_model21/model21_matches_full.csv")
X1 = ["elo_diff","h2h_score_rate","rest_days_diff","form_gd_diff","host_adv_diff","squad_age_diff","top5_share_diff","wc_exp_diff"]
r1 = d1[X1].corrwith(d1.goal_diff).abs()
eda1 = [c for c in X1 if r1[c] >= 0.2]
base1 = np.sqrt(((d1.goal_diff - d1.goal_diff.mean()) ** 2).mean())
REPS_SEARCH = REPS; REPS = 5
subs1 = best_subset(d1, "goal_diff", X1)
REPS = REPS_SEARCH
rows1 = [("Mean baseline", None),
         ("Full OLS (8 variables)", X1),
         (f"EDA-screened OLS (|r| >= 0.2, {len(eda1)} vars)", eda1),
         (f"Best-subset OLS ({len(subs1[0][1])} vars)", list(subs1[0][1]))]
t1 = []
for name, cols in rows1:
    if cols is None:
        t1.append((name, "-", base1, 0.0)); continue
    m, s = cv_rmse(d1, "goal_diff", cols)
    t1.append((name, ", ".join(cols), m, s))
t1 = pd.DataFrame(t1, columns=["candidate", "variables", "cv_rmse", "sd"])

say("MODEL 2.1 - goal difference (5-fold CV x20)"); say(t1.round(3).to_string(index=False))
say(f"Top 5 subsets in the exhaustive search: " + "; ".join(f"{', '.join(c)} ({r:.3f})" for r, c in subs1[:5]))
t1.to_csv(OUT / "model21_optimisation.csv", index=False)

# ---------------------------------------------------------------- Model 2.2
d2 = pd.read_csv(ROOT / "suman_model22/model22_team_matches.csv")
X2 = ["elo_diff","top5_share_diff","host_adv","attack_l10","opp_defence_l10","h2h_goals_pg","elo_change_1y","knockout"]
g = d2.match_id.values
r2 = d2[X2].corrwith(d2.team_goals).abs()
eda2 = [c for c in X2 if r2[c] >= 0.2]
REPS = 5; subs2 = best_subset(d2, "team_goals", X2, g); REPS = REPS_SEARCH
best2 = list(subs2[0][1])
rows2 = [("Full OLS (8 variables)", X2, "ols", None),
         (f"EDA-screened OLS (|r| >= 0.2, {len(eda2)} vars)", eda2, "ols", None),
         (f"Best-subset OLS ({len(best2)} vars)", best2, "ols", None),
         ("Log-transformed OLS, log(goals+1) (8 vars)", X2, "ols", "log1p"),
         ("Poisson GLM (8 variables)", X2, "poisson", None),
         (f"Poisson GLM, best subset ({len(best2)} vars)", best2, "poisson", None)]
t2 = []
for name, cols, kind, tr in rows2:
    m, s = cv_rmse(d2, "team_goals", cols, kind, g, tr)
    t2.append((name, ", ".join(cols), m, s))
t2 = pd.DataFrame(t2, columns=["candidate", "variables", "cv_rmse", "sd"])
say("\nMODEL 2.2 - team goals (grouped 5-fold CV x20)"); say(t2.round(3).to_string(index=False))
say(f"Top 5 subsets in the exhaustive search: " + "; ".join(f"{', '.join(c)} ({r:.3f})" for r, c in subs2[:5]))
t2.to_csv(OUT / "model22_optimisation.csv", index=False)

# ---------------------------------------------------------------- inference on the optimised models
say("\nOPTIMISED MODELS - coefficients with 95% confidence intervals")
best1 = list(subs1[0][1])
m1 = sm.OLS(d1.goal_diff, sm.add_constant(d1[best1])).fit(cov_type="HC3")
say("Model 2.1 best subset (HC3):"); say(pd.concat([m1.params, m1.conf_int(), m1.pvalues], axis=1).set_axis(["coef","ci_low","ci_high","p"], axis=1).round(4).to_string())
say(f"R2 = {m1.rsquared:.3f}, adj R2 = {m1.rsquared_adj:.3f}")
m2 = sm.GLM(d2.team_goals, sm.add_constant(d2[best2]), family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": g})
ir = pd.concat([np.exp(m2.params), np.exp(m2.conf_int()), m2.pvalues], axis=1).set_axis(["rate_ratio","ci_low","ci_high","p"], axis=1)
say("Model 2.2 Poisson best subset (cluster-robust), as rate ratios:"); say(ir.round(4).to_string())
dev_r2 = 1 - m2.deviance / m2.null_deviance
say(f"Deviance explained = {dev_r2:.3f}; Pearson chi2/df = {m2.pearson_chi2 / m2.df_resid:.2f} (near 1 = no overdispersion)")
(OUT / "optimisation_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

# ---------------------------------------------------------------- nested check of the selection procedure
# The best subset above was chosen using all the data, which flatters it. Here the whole search is
# repeated inside each outer training fold only, and the chosen subset is scored on the untouched outer fold.
def nested_subset_rmse(df, y, X, groups=None, outer_reps=5):
    units = np.unique(groups) if groups is not None else np.arange(len(df))
    gid = groups if groups is not None else np.arange(len(df))
    subsets = [list(c) for k in range(1, len(X) + 1) for c in combinations(X, k)]
    scores, chosen = [], []
    for rep in range(outer_reps):
        pred = np.empty(len(df))
        for te_u in folds(len(units), rep):
            te = np.isin(gid, units[te_u]); tr_df = df.loc[~te].reset_index(drop=True); tr_g = gid[~te]
            tr_units = np.unique(tr_g); inner = np.array_split(np.random.default_rng(999 + rep).permutation(len(tr_units)), K)
            best, best_cols = np.inf, None
            for cols in subsets:
                p = np.empty(len(tr_df))
                for iu in inner:
                    it = np.isin(tr_g, tr_units[iu])
                    p[it] = sm.OLS(tr_df.loc[~it, y], sm.add_constant(tr_df.loc[~it, cols])).fit().predict(sm.add_constant(tr_df.loc[it, cols], has_constant="add"))
                r = np.sqrt(np.mean((tr_df[y] - p) ** 2))
                if r < best: best, best_cols = r, cols
            chosen.append(tuple(best_cols))
            pred[te] = sm.OLS(tr_df[y], sm.add_constant(tr_df[best_cols])).fit().predict(sm.add_constant(df.loc[te, best_cols], has_constant="add"))
        scores.append(np.sqrt(np.mean((df[y] - pred) ** 2)))
    freq = pd.Series([v for c in chosen for v in c]).value_counts() / len(chosen)
    return float(np.mean(scores)), float(np.std(scores)), freq

stab = {}
say("\nNESTED CHECK - best-subset selection repeated inside each training fold (5 x 5-fold)")
for lab, df, y, X, gg, full in [("Model 2.1", d1, "goal_diff", X1, None, t1.loc[1, "cv_rmse"]),
                                ("Model 2.2", d2, "team_goals", X2, g, t2.loc[0, "cv_rmse"])]:
    m, s, freq = nested_subset_rmse(df, y, X, gg)
    say(f"{lab}: nested RMSE = {m:.3f} (sd {s:.3f}) vs full OLS {full:.3f} -> improvement {100*(full-m)/full:.1f}%")
    say("  share of outer folds selecting each variable: " + ", ".join(f"{k} {v:.0%}" for k, v in freq.items()))
    stab[lab] = freq.reindex(X).fillna(0)
pd.concat(stab, axis=1).rename_axis("variable").round(3).to_csv(OUT / "selection_stability.csv")
(OUT / "optimisation_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
