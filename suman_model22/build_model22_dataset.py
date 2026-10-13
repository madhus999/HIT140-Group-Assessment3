"""
HIT140 Assessment 3 - Objective 2 - Linear Regression 2.2
Part: Suman - Team-match dataset (208 rows = 2 per match) + 8 pre-match explanatory variables

Target  : team_goals = goals scored by `team` against `opponent` at 90 minutes
          (extra time & penalties excluded, same rule as Model 2.1)

Each match appears twice: once from team1's side, once from team2's side.
All variables are "team minus opponent" or "team / opponent" from the row team's view.

8 explanatory variables (all known before kick-off):
  Shared with Model 2.1 (3 of max 4):
    1. elo_diff           team Elo - opponent Elo (start of match day)
    2. top5_share_diff    team share of 26-man squad at ENG/ESP/GER/ITA/FRA clubs - opponent's
    3. host_adv           +1 team plays in own country, -1 opponent does, 0 neither
                          (team-view version of host_adv_diff; counted as shared to be safe)
  New for Model 2.2 (5):
    4. attack_l10         team avg goals SCORED in its last 10 internationals
    5. opp_defence_l10    opponent avg goals CONCEDED in its last 10 internationals
    6. h2h_goals_pg       team avg goals scored per game in previous meetings with opponent
                          (never met -> 1.36, avg goals per team per game in all
                           internationals 2018-01-01 to 2026-06-10; see h2h_meetings)
    7. elo_change_1y      team Elo at start of match day - team Elo 365 days earlier (momentum)
    8. knockout           1 = knockout match (Round of 32 onward), 0 = group stage
                          (fixture schedule fixed before the tournament)

Inputs (no new downloads; reuses group raw files):
  ../kb_model21/raw/worldcup2026.json        openfootball/worldcup.json 2026
  ../kb_model21/raw/intl_results.csv         martj42/international_results
  ../nabin_model21/raw/worldcup2026_squads.json, worldcup2026_stadiums.json

Run: python build_model22_dataset.py  -> writes model22_team_matches.csv
"""
import json
from bisect import bisect_left
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
KB_RAW = BASE.parent / "kb_model21" / "raw"
NB_RAW = BASE.parent / "nabin_model21" / "raw"
OUT = BASE / "model22_team_matches.csv"

NAME_FIX = {"USA": "United States", "Bosnia & Herzegovina": "Bosnia and Herzegovina"}
FORM_N = 10
TOP5 = {"ENG", "ESP", "GER", "ITA", "FRA"}
HOST_COUNTRY = {"us": "United States", "mx": "Mexico", "ca": "Canada"}
TOURNAMENT_START = pd.Timestamp("2026-06-11")

# ---------------------------------------------------------------- 1. WC 2026 matches
wc = json.loads((KB_RAW / "worldcup2026.json").read_text(encoding="utf-8"))["matches"]
rows = []
for m in wc:
    s = m["score"]
    rows.append({
        "date": m["date"],
        "stage": "Group" if m["round"].startswith("Matchday") else m["round"],
        "team1": NAME_FIX.get(m["team1"], m["team1"]),
        "team2": NAME_FIX.get(m["team2"], m["team2"]),
        "team1_goals": s["ft"][0],
        "team2_goals": s["ft"][1],
        "venue": m["ground"],
    })
matches = pd.DataFrame(rows)
matches["date"] = pd.to_datetime(matches["date"])
# same ordering + match_id as KB's file so rows can be joined back to Model 2.1
matches = matches.reset_index().sort_values(["date", "index"]).drop(columns="index")
matches = matches.reset_index(drop=True)
matches["match_id"] = range(1, len(matches) + 1)
assert len(matches) == 104

# ---------------------------------------------------------------- 2. historical results
hist = pd.read_csv(KB_RAW / "intl_results.csv", parse_dates=["date"])
hist = hist.dropna(subset=["home_score", "away_score"])
hist[["home_score", "away_score"]] = hist[["home_score", "away_score"]].astype(int)
hist["neutral"] = hist["neutral"].astype(str).str.upper().eq("TRUE")
hist = hist.sort_values("date", kind="stable").reset_index(drop=True)
hist = hist[hist["date"] <= matches["date"].max()]

# fallback for teams that never met: avg goals per team per game, 2018 to day before WC
recent = hist[(hist.date >= "2018-01-01") & (hist.date < TOURNAMENT_START)]
H2H_FALLBACK = round((recent.home_score.sum() + recent.away_score.sum()) / (2 * len(recent)), 2)


# ---------------------------------------------------------------- 3. Elo (same method as KB)
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
elo_hist = defaultdict(lambda: ([], []))        # team -> (dates, elo AFTER that match)
scored = defaultdict(lambda: deque(maxlen=FORM_N))
conceded = defaultdict(lambda: deque(maxlen=FORM_N))
h2h_goals = defaultdict(list)                   # (team, opp) -> goals team scored each meeting

match_dates = set(matches["date"])
snapshots = {}


def snapshot():
    return (dict(elo),
            {k: list(v) for k, v in scored.items()},
            {k: list(v) for k, v in conceded.items()},
            {k: list(v) for k, v in h2h_goals.items()})


current = None
for r in hist.itertuples(index=False):
    if r.date != current:                       # freeze state BEFORE each WC match day
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
    for t in (a, b):
        elo_hist[t][0].append(r.date)
        elo_hist[t][1].append(elo[t])
    scored[a].append(ga); conceded[a].append(gb)
    scored[b].append(gb); conceded[b].append(ga)
    h2h_goals[(a, b)].append(ga)
    h2h_goals[(b, a)].append(gb)

assert not (match_dates - snapshots.keys()), "WC dates missing from history file"


def elo_before(team: str, day: pd.Timestamp) -> float:
    """Team Elo after its last match strictly before `day` (1500 if none)."""
    dates, vals = elo_hist[team]
    i = bisect_left(dates, day)
    return vals[i - 1] if i else 1500.0


# ---------------------------------------------------------------- 4. squads + venues
squads = json.loads((NB_RAW / "worldcup2026_squads.json").read_text(encoding="utf-8"))
top5 = {}
for s in squads:
    clubs = pd.Series([p["club"]["country"] for p in s["players"]])
    assert len(clubs) == 26
    top5[s["name"]] = clubs.isin(TOP5).mean()

stadiums = json.loads((NB_RAW / "worldcup2026_stadiums.json").read_text(encoding="utf-8"))["stadiums"]
venue_country = {s["city"]: HOST_COUNTRY[s["cc"]] for s in stadiums}

teams = set(matches.team1) | set(matches.team2)
assert len(teams) == 48 and teams <= set(top5) and set(matches.venue) <= set(venue_country)

# ---------------------------------------------------------------- 5. 208 team-match rows
out = []
for m in matches.itertuples(index=False):
    E, S, C, H = snapshots[m.date]
    host = venue_country[m.venue]
    for side, team, opp, gf, ga in [(1, m.team1, m.team2, m.team1_goals, m.team2_goals),
                                    (2, m.team2, m.team1, m.team2_goals, m.team1_goals)]:
        prev = H.get((team, opp), [])
        out.append({
            "match_id": m.match_id,
            "date": m.date.date(),
            "stage": m.stage,
            "side": side,                                   # 1 = listed first in fixture
            "team": team,
            "opponent": opp,
            "venue": m.venue,
            "opp_goals": ga,                                # helper only, NOT a predictor
            "h2h_meetings": len(prev),                      # helper
            # ---- shared with Model 2.1
            "elo_diff": round(E[team] - E[opp], 1),
            "top5_share_diff": round(top5[team] - top5[opp], 3),
            "host_adv": int(team == host) - int(opp == host),
            # ---- new
            "attack_l10": round(np.mean(S[team]), 2),
            "opp_defence_l10": round(np.mean(C[opp]), 2),
            "h2h_goals_pg": round(np.mean(prev), 2) if prev else H2H_FALLBACK,
            "elo_change_1y": round(E[team] - elo_before(team, m.date - pd.Timedelta(days=365)), 1),
            "knockout": int(m.stage != "Group"),
            # ---- target
            "team_goals": gf,
        })
df = pd.DataFrame(out)

X_COLS = ["elo_diff", "top5_share_diff", "host_adv",
          "attack_l10", "opp_defence_l10", "h2h_goals_pg", "elo_change_1y", "knockout"]

# ---------------------------------------------------------------- 6. checks
assert len(df) == 208 and df.match_id.value_counts().eq(2).all()
assert df[X_COLS + ["team_goals"]].notna().all().all(), "missing values"
assert (df.groupby("team").size().sum() == 208)
# each match: the two rows mirror each other
pair = df.sort_values(["match_id", "side"])
a, b = pair[pair.side == 1].reset_index(), pair[pair.side == 2].reset_index()
assert (a.team_goals == b.opp_goals).all() and (a.elo_diff == -b.elo_diff).all()
assert (a.host_adv == -b.host_adv).all()
# Elo and 90-min goals agree with KB's Model 2.1 file
kb_path = BASE.parent / "kb_model21" / "model21_matches_KB.csv"
if kb_path.exists():
    kb = pd.read_csv(kb_path)
    chk = a.merge(kb, on="match_id", suffixes=("", "_kb"))
    assert (chk.team == chk.team1).all(), "match_id order differs from KB"
    assert (chk.team_goals == chk.team1_goals).all(), "goals differ from KB"
    assert np.allclose(chk.elo_diff, chk.elo_diff_kb, atol=0.11), "Elo differs from KB"
# overlap rule: Model 2.1 variables reused here
M21 = {"elo_diff", "h2h_score_rate", "rest_days_diff", "form_gd_diff",
       "host_adv_diff", "squad_age_diff", "top5_share_diff", "wc_exp_diff"}
shared = (set(X_COLS) & M21) | {"host_adv"}          # host_adv counted as shared (conservative)
assert len(shared) <= 4, shared

cols = ["match_id", "date", "stage", "side", "team", "opponent", "venue",
        "opp_goals", "h2h_meetings"] + X_COLS + ["team_goals"]
df[cols].to_csv(OUT, index=False)

# ---------------------------------------------------------------- 7. summary for Prabin
print(f"wrote {OUT.name}: {len(df)} rows, {len(X_COLS)} explanatory variables")
print(f"shared with Model 2.1 ({len(shared)}): {sorted(shared)}")
print(f"h2h fallback = {H2H_FALLBACK} goals/game; pairs never met: {(df.h2h_meetings == 0).sum() // 2}")
print("\n" + df[X_COLS + ["team_goals"]].describe().round(2).to_string())
print("\ncorrelation with team_goals:")
print(df[X_COLS].corrwith(df.team_goals).round(3).sort_values(ascending=False).to_string())
print("\nteam_goals counts:\n" + df.team_goals.value_counts().sort_index().to_string())
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    Xc = sm.add_constant(df[X_COLS].astype(float))
    vif = {c: variance_inflation_factor(Xc.values, i) for i, c in enumerate(Xc.columns) if c != "const"}
    print("\nVIF:\n" + pd.Series(vif).round(2).to_string())
except ImportError:
    pass
print("\nfeature inter-correlation:\n" + df[X_COLS].corr().round(2).to_string())
