"""
Model 2.1 - Nabin's part

What this file does:
Takes KB's file of 104 World Cup matches and adds 4 more pre-match columns,
then saves everything as model21_matches_full.csv.

The 4 new columns (team 1 minus team 2):
  host_adv_diff    - is a team playing in its own country?
  squad_age_diff   - difference in the average age of the two squads
  top5_share_diff  - difference in how many players play in the top 5 European leagues
  wc_exp_diff      - difference in how many World Cups each team has played before

All 4 are known before the match starts.
Data comes from openfootball (squads and stadiums) and KB's results file.
"""
import json
from pathlib import Path

import pandas as pd

BASE = Path(__file__).resolve().parent
RAW = BASE / "raw"
KB = BASE.parent / "kb_model21"
OUT = BASE / "model21_matches_full.csv"

TOURNAMENT_START = pd.Timestamp("2026-06-11")
TOP5 = {"ENG", "ESP", "GER", "ITA", "FRA"}
HOST_COUNTRY = {"us": "United States", "mx": "Mexico", "ca": "Canada"}

# ------------------------------------------------------------ 1. KB's dataset
df = pd.read_csv(KB / "model21_matches_KB.csv", parse_dates=["date"])
assert len(df) == 104, "expected 104 matches"
teams = sorted(set(df.team1) | set(df.team2))
assert len(teams) == 48

# ------------------------------------------------------------ 2. host advantage
stadiums = json.loads((RAW / "worldcup2026_stadiums.json").read_text(encoding="utf-8"))["stadiums"]
venue_country = {s["city"]: HOST_COUNTRY[s["cc"]] for s in stadiums}
assert set(df.venue) <= set(venue_country), "venue name mismatch"
host = df.venue.map(venue_country)
df["host_adv_diff"] = (df.team1 == host).astype(int) - (df.team2 == host).astype(int)

# ------------------------------------------------------------ 3. squad variables
squads = json.loads((RAW / "worldcup2026_squads.json").read_text(encoding="utf-8"))
squad_rows = []
for s in squads:
    players = pd.DataFrame(s["players"])
    dob = pd.to_datetime(players["date_of_birth"])
    age = (TOURNAMENT_START - dob).dt.days / 365.25
    club_country = players["club"].apply(lambda c: c["country"])
    squad_rows.append({
        "team": s["name"],
        "squad_size": len(players),
        "squad_age": age.mean(),
        "top5_share": club_country.isin(TOP5).mean(),
    })
squad = pd.DataFrame(squad_rows).set_index("team")
assert set(teams) == set(squad.index), "squad team name mismatch"
assert (squad.squad_size == 26).all()

# ------------------------------------------------------------ 4. World Cup experience
hist = pd.read_csv(KB / "raw" / "intl_results.csv", parse_dates=["date"])
wc = hist[(hist.tournament == "FIFA World Cup") & (hist.date.dt.year < 2026)]
long = pd.concat([
    wc[["date", "home_team"]].rename(columns={"home_team": "team"}),
    wc[["date", "away_team"]].rename(columns={"away_team": "team"}),
])
wc_exp = long.assign(year=long.date.dt.year).groupby("team").year.nunique()
squad["wc_exp"] = wc_exp.reindex(squad.index).fillna(0).astype(int)   # 0 = first World Cup

# ------------------------------------------------------------ 5. team1 - team2 differences
for col, new in [("squad_age", "squad_age_diff"), ("top5_share", "top5_share_diff"),
                 ("wc_exp", "wc_exp_diff")]:
    df[new] = df.team1.map(squad[col]) - df.team2.map(squad[col])
df["squad_age_diff"] = df["squad_age_diff"].round(2)
df["top5_share_diff"] = df["top5_share_diff"].round(3)

X_COLS = ["elo_diff", "h2h_score_rate", "rest_days_diff", "form_gd_diff",        # KB
          "host_adv_diff", "squad_age_diff", "top5_share_diff", "wc_exp_diff"]   # Nabin
assert df[X_COLS + ["goal_diff"]].notna().all().all(), "missing values"

helper = ["match_id", "date", "stage", "group", "team1", "team2", "venue",
          "team1_goals", "team2_goals", "extra_time", "penalties"]
df[helper + X_COLS + ["goal_diff"]].assign(date=df.date.dt.date).to_csv(OUT, index=False)
squad.round(3).sort_index().to_csv(BASE / "team_lookup.csv")

print(f"wrote {OUT.name}: {len(df)} rows, {len(X_COLS)} explanatory variables")
print(df[X_COLS[4:]].describe().round(2).to_string())
print("\nhost_adv_diff counts:\n", df.host_adv_diff.value_counts().to_string())
