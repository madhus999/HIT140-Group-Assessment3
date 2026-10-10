"""
HIT140 Assessment 3 - Objective 2 - Linear Regression 2.1
Part: KB - Match dataset compilation (104 rows) + 4 pre-match explanatory variables

Target  : goal_diff = team1 goals - team2 goals at 90 minutes (full time, excl. extra time & pens)
Features (all computed ONLY from matches played strictly BEFORE the match date):
  1. elo_diff        - World Football Elo rating of team1 minus team2, before kick-off
  2. h2h_score_rate  - team1's historical head-to-head score vs team2 ((W + 0.5*D) / meetings); 0.5 if never met
  3. rest_days_diff  - days since team1's previous match minus team2's (each capped at 14 days)
  4. form_gd_diff    - team1 avg goal difference in last 10 internationals minus team2's

Sources (raw files in ./raw):
  - openfootball/worldcup.json 2026  (104 WC 2026 matches, FT / ET / pens scores)
      https://raw.githubusercontent.com/openfootball/worldcup.json/master/2026/worldcup.json
  - martj42/international_results     (all men's internationals since 1872)
      https://raw.githubusercontent.com/martj42/international_results/master/results.csv

Run: python build_model21_dataset.py   -> writes model21_matches_KB.csv
"""
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
OUT = BASE / "model21_matches_KB.csv"

NAME_FIX = {"USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina"}
FORM_N = 10
REST_CAP = 14   # days; rest beyond 2 weeks treated as "fully rested" (limits outlier leverage)

# ---------------------------------------------------------------- 1. WC 2026 matches
wc = json.loads((RAW / "worldcup2026.json").read_text(encoding="utf-8"))["matches"]
rows = []
for i, m in enumerate(wc, start=1):
    s = m["score"]
    rows.append({
        "match_id": i,
        "date": m["date"],
        "stage": "Group" if m["round"].startswith("Matchday") else m["round"],
        "group": m.get("group", ""),
        "team1": NAME_FIX.get(m["team1"], m["team1"]),
        "team2": NAME_FIX.get(m["team2"], m["team2"]),
        "team1_goals": s["ft"][0],
        "team2_goals": s["ft"][1],
        "extra_time": "et" in s,
        "penalties": "p" in s,
        "venue": m["ground"],
    })
matches = pd.DataFrame(rows)
matches["date"] = pd.to_datetime(matches["date"])
matches = matches.sort_values(["date", "match_id"]).reset_index(drop=True)
matches["match_id"] = range(1, len(matches) + 1)
matches["goal_diff"] = matches["team1_goals"] - matches["team2_goals"]
assert len(matches) == 104

# ---------------------------------------------------------------- 2. historical results
hist = pd.read_csv(RAW / "intl_results.csv", parse_dates=["date"])
hist = hist.dropna(subset=["home_score", "away_score"])
hist[["home_score", "away_score"]] = hist[["home_score", "away_score"]].astype(int)
hist["neutral"] = hist["neutral"].astype(str).str.upper().eq("TRUE")
hist = hist.sort_values("date", kind="stable").reset_index(drop=True)
hist = hist[hist["date"] <= matches["date"].max()]   # nothing after the final is needed

for t in pd.concat([matches.team1, matches.team2]).unique():
    assert ((hist.home_team == t) | (hist.away_team == t)).any(), f"name mismatch: {t}"


# ---------------------------------------------------------------- 3. Elo (eloratings.net method)
def k_factor(tournament: str) -> int:
    t = tournament.lower()
    if t == "fifa world cup":
        return 60
    if t in {"uefa euro", "copa américa", "african cup of nations", "afc asian cup",
             "gold cup", "confederations cup", "concacaf championship", "oceania nations cup"}:
        return 50
    if "qualification" in t:
        return 40
    if t == "friendly":
        return 20
    return 30


def gd_mult(gd: int) -> float:
    gd = abs(gd)
    return 1.0 if gd <= 1 else 1.5 if gd == 2 else (11 + gd) / 8


elo = defaultdict(lambda: 1500.0)
last_date = {}
form = defaultdict(lambda: deque(maxlen=FORM_N))
h2h = defaultdict(lambda: [0, 0, 0])          # key (a, b) -> [a wins, draws, b wins]

match_dates = set(matches["date"])
snapshots = {}                                  # date -> state BEFORE any match that day


def snapshot():
    return (dict(elo), dict(last_date), {k: list(v) for k, v in form.items()},
            {k: list(v) for k, v in h2h.items()})


current = None
for r in hist.itertuples(index=False):
    # freeze state at the start of each WC match day (= strictly pre-match info)
    if r.date != current:
        current = r.date
        if current in match_dates:
            snapshots[current] = snapshot()
    a, b, ga, gb = r.home_team, r.away_team, r.home_score, r.away_score
    ra, rb = elo[a], elo[b]
    home_adv = 0 if r.neutral else 100
    we = 1 / (10 ** (-(ra + home_adv - rb) / 400) + 1)
    w = 1.0 if ga > gb else 0.5 if ga == gb else 0.0
    delta = k_factor(r.tournament) * gd_mult(ga - gb) * (w - we)
    elo[a], elo[b] = ra + delta, rb - delta
    last_date[a] = last_date[b] = r.date
    form[a].append(ga - gb)
    form[b].append(gb - ga)
    key = tuple(sorted((a, b)))
    rec = h2h[key]
    if ga == gb:
        rec[1] += 1
    elif (ga > gb) == (a == key[0]):
        rec[0] += 1
    else:
        rec[2] += 1

missing = match_dates - snapshots.keys()
assert not missing, f"WC dates missing from history file: {missing}"

# ---------------------------------------------------------------- 4. features per match
feat = []
for m in matches.itertuples(index=False):
    E, L, F, H = snapshots[m.date]
    t1, t2 = m.team1, m.team2
    key = tuple(sorted((t1, t2)))
    w0, d, w1 = H.get(key, [0, 0, 0])
    n = w0 + d + w1
    t1_w = w0 if t1 == key[0] else w1
    feat.append({
        "team1_elo": round(E[t1], 1),
        "team2_elo": round(E[t2], 1),
        "elo_diff": round(E[t1] - E[t2], 1),
        "h2h_meetings": n,
        "h2h_score_rate": round((t1_w + 0.5 * d) / n, 3) if n else 0.5,
        "rest_days_diff": min((m.date - L[t1]).days, REST_CAP) - min((m.date - L[t2]).days, REST_CAP),
        "form_gd_diff": round(np.mean(F[t1]) - np.mean(F[t2]), 2),
    })

df = pd.concat([matches, pd.DataFrame(feat)], axis=1)

# sanity: 90-min scores agree with martj42 where no extra time was played
hk = hist.assign(k=[frozenset(p) for p in zip(hist.home_team, hist.away_team)])
bad = []
for m in df[~df.extra_time].itertuples():
    h = hk[(hk.date == m.date) & (hk.k == frozenset((m.team1, m.team2)))].iloc[0]
    g = {h.home_team: h.home_score, h.away_team: h.away_score}
    if (g[m.team1], g[m.team2]) != (m.team1_goals, m.team2_goals):
        bad.append(m.match_id)
assert not bad, f"score mismatch: {bad}"

cols = ["match_id", "date", "stage", "group", "team1", "team2", "venue",
        "team1_goals", "team2_goals", "extra_time", "penalties",
        "team1_elo", "team2_elo", "h2h_meetings",
        "elo_diff", "h2h_score_rate", "rest_days_diff", "form_gd_diff",   # <- 4 explanatory vars
        "goal_diff"]                                                      # <- target
df[cols].assign(date=df.date.dt.date).to_csv(OUT, index=False)
print(f"wrote {OUT.name}: {df.shape[0]} rows")
print(df[["elo_diff", "h2h_score_rate", "rest_days_diff", "form_gd_diff", "goal_diff"]]
      .describe().round(2).to_string())
print("\ncorrelation with goal_diff:")
print(df[["elo_diff", "h2h_score_rate", "rest_days_diff", "form_gd_diff"]]
      .corrwith(df.goal_diff).round(3).to_string())
print("\nfeature inter-correlation:")
print(df[["elo_diff", "h2h_score_rate", "rest_days_diff", "form_gd_diff"]].corr().round(2).to_string())
