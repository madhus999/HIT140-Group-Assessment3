"""
HIT140 Assessment 3 - Objective 2 - Linear Regression 2.1
Part: Nabin - build and evaluate the model

Question : can information known before kick-off predict the goal difference
           (team1 goals - team2 goals at 90 minutes) of a World Cup 2026 match?
Data     : model21_matches_full.csv  (104 matches, 8 explanatory variables)
Method   : multiple linear regression (ordinary least squares) with statsmodels;
           scikit-learn is used only for the competing algorithms in step 6

Steps
  1. load data and describe it
  2. exploratory analysis: scatterplots + Pearson correlation, then multicollinearity (VIF)
  3. fit the model on all 104 matches (coefficients, p-values, R-squared)
  4. check the regression assumptions on the residuals (robust errors if variance is unequal)
  5. evaluate prediction accuracy on unseen matches (train/test split + 5-fold CV)
  6. compare linear regression with competing algorithms (ridge, lasso, random forest,
     gradient boosting) using the same cross-validation

"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan, linear_reset
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson

BASE = Path(__file__).resolve().parent
OUT = BASE / "outputs"
OUT.mkdir(exist_ok=True)
SEED = 140
ALPHA = 0.05

Y = "goal_diff"
X_COLS = ["elo_diff", "h2h_score_rate", "rest_days_diff", "form_gd_diff",
          "host_adv_diff", "squad_age_diff", "top5_share_diff", "wc_exp_diff"]

lines = []


def say(text=""):
    """print and also keep a copy for outputs/model21_results.txt"""
    print(text)
    lines.append(str(text))


def metrics(actual, predicted):
    err = np.asarray(actual) - np.asarray(predicted)
    ss_res = (err ** 2).sum()
    ss_tot = ((actual - np.mean(actual)) ** 2).sum()
    return {"R2": 1 - ss_res / ss_tot, "RMSE": np.sqrt((err ** 2).mean()), "MAE": np.abs(err).mean()}


# ------------------------------------------------------------ 1. data
df = pd.read_csv(BASE / "model21_matches_full.csv")
assert len(df) == 104 and len(X_COLS) == 8
assert df[X_COLS + [Y]].notna().all().all()

say("=" * 70)
say("1. DESCRIPTIVE STATISTICS (n = 104 matches)")
say("=" * 70)
say(df[X_COLS + [Y]].describe().T[["mean", "std", "min", "50%", "max"]].round(2).to_string())

# ------------------------------------------------------------ 2. multicollinearity
corr = df[X_COLS + [Y]].corr()
X = sm.add_constant(df[X_COLS])
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(1, X.shape[1])], index=X_COLS)

say("\n" + "=" * 70)
say("2. MULTICOLLINEARITY")
say("=" * 70)
say("Correlation of each variable with goal_diff:")
say(corr[Y].drop(Y).round(2).to_string())
say("\nVariance inflation factor (rule of thumb: above 5 is a problem):")
say(vif.round(2).to_string())

fig, ax = plt.subplots(figsize=(7.5, 6))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)), corr.columns, rotation=45, ha="right")
ax.set_yticks(range(len(corr)), corr.columns)
for i in range(len(corr)):
    for j in range(len(corr)):
        v = corr.iloc[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.55 else "black")
fig.colorbar(im, ax=ax, shrink=0.8, label="Pearson r")
ax.set_title("Model 2.1 - correlation matrix")
fig.tight_layout()
fig.savefig(OUT / "fig1_correlation_matrix.png", dpi=200)
plt.close(fig)

# EDA: one scatterplot per explanatory variable against the target, with Pearson r and its p-value.
# This is the evidence that each variable chosen on football logic really relates to goal difference.
eda_rows = []
fig, axes = plt.subplots(2, 4, figsize=(14, 6.5), sharey=True)
for ax, col in zip(axes.ravel(), X_COLS):
    r, p = stats.pearsonr(df[col], df[Y])
    eda_rows.append({"variable": col, "pearson_r": r, "p_value": p})
    ax.scatter(df[col], df[Y], alpha=0.6, s=18)
    slope, intercept = np.polyfit(df[col], df[Y], 1)
    xs = np.array([df[col].min(), df[col].max()])
    ax.plot(xs, intercept + slope * xs, color="black", lw=1.2)
    ax.set_title(f"{col}\nr = {r:.2f}, p = {p:.3f}", fontsize=10)
    ax.set_xlabel(col)
for ax in axes[:, 0]:
    ax.set_ylabel("goal_diff")
fig.suptitle("Model 2.1 - each explanatory variable against goal difference")
fig.tight_layout()
fig.savefig(OUT / "fig4_eda_scatterplots.png", dpi=200)
plt.close(fig)
eda = pd.DataFrame(eda_rows).set_index("variable")
eda.round(4).to_csv(OUT / "model21_eda_correlations.csv")
say("\nPearson correlation with goal_diff and its p-value (EDA):")
say(eda.round(4).to_string())

# ------------------------------------------------------------ 3. fit on all 104 matches
model = sm.OLS(df[Y], X).fit()

say("\n" + "=" * 70)
say("3. FULL MODEL (OLS, all 104 matches)")
say("=" * 70)
say(model.summary().as_text())

coef = pd.DataFrame({
    "coef": model.params, "std_err": model.bse, "t": model.tvalues, "p_value": model.pvalues,
    "ci_low": model.conf_int()[0], "ci_high": model.conf_int()[1],
})
# standardised coefficients: effect of a 1-standard-deviation change, so variables are comparable
coef["std_coef"] = model.params * df[X_COLS].std().reindex(model.params.index) / df[Y].std()
coef["VIF"] = vif.reindex(model.params.index)
coef.round(4).to_csv(OUT / "model21_coefficients.csv")

say("\nStandardised coefficients (largest effect first):")
say(coef["std_coef"].drop("const").sort_values(key=abs, ascending=False).round(3).to_string())
sig = coef.index[(coef.p_value < ALPHA) & (coef.index != "const")].tolist()
say(f"\nSignificant at the {ALPHA:.0%} level: {sig if sig else 'none'}")

# ------------------------------------------------------------ 4. assumption checks
resid, fitted = model.resid, model.fittedvalues
student = model.get_influence().resid_studentized_internal
cooks = model.get_influence().cooks_distance[0]

sw_stat, sw_p = stats.shapiro(resid)
bp_stat, bp_p, _, _ = het_breuschpagan(resid, X)
dw = durbin_watson(resid)
reset = linear_reset(model, power=2, use_f=True)

say("\n" + "=" * 70)
say("4. REGRESSION ASSUMPTIONS")
say("=" * 70)
say(f"Linearity          Ramsey RESET F = {reset.fvalue:.3f}, p = {reset.pvalue:.3f}"
    f"  -> {'no evidence of non-linearity' if reset.pvalue > ALPHA else 'possible non-linearity'}")
say(f"Normal residuals   Shapiro-Wilk W = {sw_stat:.3f}, p = {sw_p:.3f}"
    f"  -> {'consistent with normality' if sw_p > ALPHA else 'residuals not normal'}")
say(f"                   skew = {stats.skew(resid):.2f}, excess kurtosis = {stats.kurtosis(resid):.2f}")
say(f"Homoscedasticity   Breusch-Pagan LM = {bp_stat:.3f}, p = {bp_p:.3f}"
    f"  -> {'constant variance' if bp_p > ALPHA else 'heteroscedastic'}")
say(f"Independence       Durbin-Watson = {dw:.2f}  (close to 2 means no autocorrelation)")
say(f"Multicollinearity  largest VIF = {vif.max():.2f} ({vif.idxmax()})")
say(f"Influential points largest Cook's distance = {cooks.max():.3f}; "
    f"{int((cooks > 4 / len(df)).sum())} matches above 4/n; "
    f"{int((np.abs(student) > 3).sum())} with |standardised residual| > 3")

# Remedy for unequal variance: keep the same coefficients but use heteroscedasticity-robust
# (HC3) standard errors, so the p-values and confidence intervals stay valid.
robust = model.get_robustcov_results(cov_type="HC3")
rob = pd.DataFrame({"coef": model.params, "p_ols": model.pvalues, "p_robust_HC3": robust.pvalues,
                    "ci_low_HC3": robust.conf_int()[:, 0], "ci_high_HC3": robust.conf_int()[:, 1]})
rob.round(4).to_csv(OUT / "model21_robust_se.csv")
say("\nCoefficients with ordinary vs robust (HC3) p-values:")
say(rob.round(4).to_string())
sig_rob = rob.index[(rob.p_robust_HC3 < ALPHA) & (rob.index != "const")].tolist()
say(f"Significant at the {ALPHA:.0%} level with robust errors: {sig_rob if sig_rob else 'none'}")

big = df.assign(fitted=fitted.round(2), resid=resid.round(2), cooks=cooks.round(3))
say("\nFive largest residuals (matches the model got most wrong):")
say(big.reindex(resid.abs().sort_values(ascending=False).index)
    [["date", "team1", "team2", "team1_goals", "team2_goals", "fitted", "resid", "cooks"]]
    .head(5).to_string(index=False))

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
ax = axes[0, 0]
ax.scatter(fitted, resid, alpha=0.7)
ax.axhline(0, color="black", lw=1)
ax.set(xlabel="Fitted goal difference", ylabel="Residual", title="Residuals vs fitted (linearity, equal variance)")
ax = axes[0, 1]
stats.probplot(resid, dist="norm", plot=ax)
ax.set_title("Normal Q-Q plot (normality)")
ax = axes[1, 0]
ax.hist(resid, bins=12, density=True, alpha=0.7, edgecolor="white")
grid = np.linspace(resid.min(), resid.max(), 200)
ax.plot(grid, stats.norm.pdf(grid, resid.mean(), resid.std()), color="black")
ax.set(xlabel="Residual", ylabel="Density", title="Histogram of residuals")
ax = axes[1, 1]
ax.scatter(fitted, np.sqrt(np.abs(student)), alpha=0.7)
ax.set(xlabel="Fitted goal difference", ylabel="sqrt(|standardised residual|)",
       title="Scale-location (equal variance)")
fig.suptitle("Model 2.1 - residual diagnostics")
fig.tight_layout()
fig.savefig(OUT / "fig2_residual_diagnostics.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------ 5. predictive evaluation
# 5a. one 80/20 split: fit on 83 matches, predict the 21 the model has never seen
rng = np.random.default_rng(SEED)
order = rng.permutation(len(df))
n_test = round(0.2 * len(df))
test_idx, train_idx = order[:n_test], order[n_test:]
train, test = df.iloc[train_idx], df.iloc[test_idx]

m_train = sm.OLS(train[Y], sm.add_constant(train[X_COLS])).fit()
pred_test = m_train.predict(sm.add_constant(test[X_COLS], has_constant="add"))
split_train = metrics(train[Y], m_train.fittedvalues)
split_test = metrics(test[Y], pred_test)
split_base = metrics(test[Y], np.full(len(test), train[Y].mean()))   # baseline: always predict the mean

# 5b. 5-fold cross-validation repeated 20 times (more stable than one split with n = 104)
cv_rows, base_rows = [], []
for rep in range(20):
    folds = np.array_split(np.random.default_rng(SEED + rep).permutation(len(df)), 5)
    pred = np.empty(len(df))
    base = np.empty(len(df))
    for k in range(5):
        te = folds[k]
        tr = np.concatenate([folds[j] for j in range(5) if j != k])
        fit = sm.OLS(df[Y].iloc[tr], sm.add_constant(df[X_COLS].iloc[tr])).fit()
        pred[te] = fit.predict(sm.add_constant(df[X_COLS].iloc[te], has_constant="add"))
        base[te] = df[Y].iloc[tr].mean()
    cv_rows.append(metrics(df[Y], pred))
    base_rows.append(metrics(df[Y], base))
cv = pd.DataFrame(cv_rows).mean()
cv_base = pd.DataFrame(base_rows).mean()

# share of matches where the model picks the right result direction (win / loss), draws excluded
decided = test[Y] != 0
direction = (np.sign(pred_test[decided]) == np.sign(test[Y][decided])).mean()

eval_table = pd.DataFrame({
    "in-sample (all 104)": metrics(df[Y], fitted),
    "train (83)": split_train,
    "test (21)": split_test,
    "test baseline (mean)": split_base,
    "5-fold CV x20": cv,
    "CV baseline (mean)": cv_base,
}).T
eval_table.round(3).to_csv(OUT / "model21_evaluation.csv")

say("\n" + "=" * 70)
say("5. PREDICTIVE EVALUATION")
say("=" * 70)
say(eval_table.round(3).to_string())
say(f"\nAdjusted R-squared (all 104): {model.rsquared_adj:.3f}")
say(f"F-test of the whole model: F = {model.fvalue:.2f}, p = {model.f_pvalue:.2e}")
say(f"Correct winner on decided test matches: {direction:.0%} ({int(decided.sum())} matches)")

fig, ax = plt.subplots(figsize=(6, 5.5))
ax.scatter(m_train.fittedvalues, train[Y], alpha=0.5, label="train (83)")
ax.scatter(pred_test, test[Y], alpha=0.9, marker="s", label="test (21)")
lim = [df[Y].min() - 0.5, df[Y].max() + 0.5]
ax.plot(lim, lim, color="black", lw=1, label="perfect prediction")
ax.set(xlabel="Predicted goal difference", ylabel="Actual goal difference",
       title="Model 2.1 - actual vs predicted", xlim=lim, ylim=lim)
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "fig3_actual_vs_predicted.png", dpi=200)
plt.close(fig)

# ------------------------------------------------------------ 6. competing algorithms
# Every model gets the same 8 variables and the same repeated 5-fold cross-validation as 5b,
# so the scores are directly comparable. Ridge and lasso choose their penalty inside each
# training fold only (no information from the held-out fold leaks in).
SIG_COLS = ["elo_diff", "top5_share_diff"]          # the two variables significant with robust errors
candidates = {
    "Linear regression (8 vars)": (LinearRegression(), X_COLS),
    "Linear regression (2 significant vars)": (LinearRegression(), SIG_COLS),
    "Ridge regression": (make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 30))), X_COLS),
    "Lasso regression": (make_pipeline(StandardScaler(), LassoCV(cv=5, random_state=SEED)), X_COLS),
    "Random forest": (RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=SEED), X_COLS),
    "Gradient boosting": (GradientBoostingRegressor(n_estimators=150, learning_rate=0.05, max_depth=2,
                                                    random_state=SEED), X_COLS),
}
N_REP = 10
comp_rows = []
for name, (est, cols) in candidates.items():
    rep_scores = []
    for rep in range(N_REP):
        folds = np.array_split(np.random.default_rng(SEED + rep).permutation(len(df)), 5)
        pred = np.empty(len(df))
        for k in range(5):
            te = folds[k]
            tr = np.concatenate([folds[j] for j in range(5) if j != k])
            est.fit(df[cols].iloc[tr], df[Y].iloc[tr])
            pred[te] = est.predict(df[cols].iloc[te])
        rep_scores.append(metrics(df[Y], pred))
    row = pd.DataFrame(rep_scores).mean()
    comp_rows.append({"model": name, "CV_R2": row.R2, "CV_RMSE": row.RMSE, "CV_MAE": row.MAE})
comp = pd.DataFrame(comp_rows).set_index("model").sort_values("CV_RMSE")
comp.round(3).to_csv(OUT / "model21_model_comparison.csv")

say("\n" + "=" * 70)
say(f"6. COMPETING ALGORITHMS (5-fold CV x{N_REP}, lower RMSE is better)")
say("=" * 70)
say(comp.round(3).to_string())
say(f"\nBaseline (always predict the mean): RMSE = {cv_base.RMSE:.3f}")
say(f"Best model by cross-validated RMSE: {comp.index[0]}")

fig, ax = plt.subplots(figsize=(7.5, 4))
ax.barh(comp.index[::-1], comp.CV_RMSE[::-1])
ax.axvline(cv_base.RMSE, color="black", ls="--", lw=1, label="baseline (predict the mean)")
for i, v in enumerate(comp.CV_RMSE[::-1]):
    ax.text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
ax.set(xlabel="Cross-validated RMSE (goals)", title="Model 2.1 - competing algorithms",
       xlim=(0, cv_base.RMSE * 1.15))
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8, frameon=False)
fig.tight_layout()
fig.savefig(OUT / "fig5_model_comparison.png", dpi=200)
plt.close(fig)

(OUT / "model21_results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nsaved figures and tables to {OUT}")
