"""
HIT140 Assessment 3 - competing algorithms for both models under ONE validation scheme.
Model 2.1: 5-fold CV x10.  Model 2.2: grouped 5-fold CV x10 (both rows of a match in the same fold).
Hyperparameters (ridge/lasso alpha) are tuned inside each training fold only (nested), using grouped
inner folds for Model 2.2, so no held-out information is used for tuning.
Run from the repo root:  python report_analysis/compare_algorithms.py
"""
from pathlib import Path
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, statsmodels.api as sm
from sklearn.linear_model import LinearRegression, Ridge, Lasso, PoissonRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "outputs"; OUT.mkdir(exist_ok=True)
REPS, K, SEED = 10, 5, 140
ALPHAS = np.logspace(-3, 2, 25)

def tuned(est, grid, grouped):
    return ("tune", est, grid, grouped)

def make(name, grouped, n):
    if name == "OLS (8 variables)": return LinearRegression()
    if name == "Ridge (tuned)": return tuned(make_pipeline(StandardScaler(), Ridge()), {"ridge__alpha": ALPHAS * 100}, grouped)
    if name == "Lasso (tuned)": return tuned(make_pipeline(StandardScaler(), Lasso(max_iter=20000)), {"lasso__alpha": ALPHAS}, grouped)
    if name == "Poisson GLM": return make_pipeline(StandardScaler(), PoissonRegressor(alpha=0, max_iter=1000))
    if name == "Random forest": return RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=SEED, n_jobs=-1)
    if name == "Gradient boosting": return GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2, random_state=SEED)

def evaluate(df, y, X, names, groups=None):
    units = np.unique(groups) if groups is not None else np.arange(len(df))
    gid = groups if groups is not None else np.arange(len(df))
    res = {n: [] for n in names + ["Mean baseline"]}
    for rep in range(REPS):
        preds = {n: np.empty(len(df)) for n in res}
        for te_u in np.array_split(np.random.default_rng(SEED + rep).permutation(len(units)), K):
            te = np.isin(gid, units[te_u])
            Xtr, ytr, Xte = df.loc[~te, X].values, df.loc[~te, y].values, df.loc[te, X].values
            preds["Mean baseline"][te] = ytr.mean()
            for n in names:
                est = make(n, groups is not None, len(ytr))
                if isinstance(est, tuple):
                    _, base, grid, grouped = est
                    cv = GroupKFold(n_splits=4) if grouped else KFold(4, shuffle=True, random_state=rep)
                    gs = GridSearchCV(base, grid, cv=cv, scoring="neg_root_mean_squared_error")
                    gs.fit(Xtr, ytr, **({"groups": gid[~te]} if grouped else {}))
                    preds[n][te] = gs.predict(Xte)
                else:
                    preds[n][te] = est.fit(Xtr, ytr).predict(Xte)
        for n in res:
            res[n].append(np.sqrt(np.mean((df[y].values - preds[n]) ** 2)))
    t = pd.DataFrame({"cv_rmse": {n: np.mean(v) for n, v in res.items()}, "sd": {n: np.std(v) for n, v in res.items()}})
    ols = t.loc["OLS (8 variables)", "cv_rmse"]
    t["change_vs_ols_pct"] = 100 * (ols - t.cv_rmse) / ols
    return t.sort_values("cv_rmse")

d1 = pd.read_csv(ROOT / "nabin_model21/model21_matches_full.csv")
X1 = ["elo_diff","h2h_score_rate","rest_days_diff","form_gd_diff","host_adv_diff","squad_age_diff","top5_share_diff","wc_exp_diff"]
d2 = pd.read_csv(ROOT / "suman_model22/model22_team_matches.csv")
X2 = ["elo_diff","top5_share_diff","host_adv","attack_l10","opp_defence_l10","h2h_goals_pg","elo_change_1y","knockout"]

common = ["OLS (8 variables)", "Ridge (tuned)", "Lasso (tuned)", "Random forest", "Gradient boosting"]
t1 = evaluate(d1, "goal_diff", X1, common)
t2 = evaluate(d2, "team_goals", X2, common + ["Poisson GLM"], d2.match_id.values)
t1.round(3).to_csv(OUT / "model21_algorithms.csv"); t2.round(3).to_csv(OUT / "model22_algorithms.csv")
txt = "MODEL 2.1 (5-fold CV x10)\n" + t1.round(3).to_string() + "\n\nMODEL 2.2 (grouped 5-fold CV x10)\n" + t2.round(3).to_string() + "\n"
(OUT / "algorithm_comparison.txt").write_text(txt, encoding="utf-8"); print(txt)
