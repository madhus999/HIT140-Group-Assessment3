"""
HIT140 Assessment 3 - Objective 2 - Linear Regression 2.1
Part: Nabin - add the remaining 4 pre-match explanatory variables to KB's dataset

Input : ../kb_model21/model21_matches_KB.csv   (104 matches, target + KB's 4 variables)
Output: model21_matches_full.csv               (104 matches, target + all 8 variables)

The 4 variables added here (all team1 minus team2, same sign convention as KB):
  5. host_adv_diff     +1 if team1 plays in its own country, -1 if team2 does, else 0
  6. squad_age_diff    mean squad age (years, at 11 June 2026) of team1 minus team2
  7. top5_share_diff   share of the 26-man squad playing club football in a top-5
                       European league country (ENG, ESP, GER, ITA, FRA), team1 minus team2
  8. wc_exp_diff       number of previous World Cup finals tournaments (1930-2022)
                       team1 appeared in, minus team2's

Why each one is known before kick-off:
  - venues were fixed by FIFA long before the tournament
  - final squads (dates of birth, clubs) were registered before the opening match
  - World Cup history only uses tournaments up to 2022

Sources (raw files in ./raw and ../kb_model21/raw):
  - openfootball/worldcup.json 2026: worldcup.squads.json, worldcup.stadiums.json
      https://github.com/openfootball/worldcup.json/tree/master/2026
  - martj42/international_results: results.csv (KB's copy, intl_results.csv)

Run: python build_model21_full_dataset.py
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
