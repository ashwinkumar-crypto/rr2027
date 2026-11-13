"""RR pressure-analytics pipeline. Usage: python pipeline/build_data.py "<Data Pack dir>"
Reads the original CSVs, validates, computes player stats, writes dashboard/data.js."""
import sys, json, numpy as np, pandas as pd
D = sys.argv[1] if len(sys.argv) > 1 else "Data Pack"
c = pd.read_csv(f"{D}/competitions.csv"); m = pd.read_csv(f"{D}/matches.csv")
mp = pd.read_csv(f"{D}/competition_team_player_mapping.csv")
b = pd.read_csv(f"{D}/ball_by_ball.csv", low_memory=False)

# ---- scope: innings 1-2 only (super overs excluded), competition group ----
c["grp"] = np.where(c.comp_name.str.startswith("IPL"), "IPL", np.where(c.match_format == "T20I", "T20I", "Other T20"))
m = m.merge(c[["comp_id", "comp_name", "grp"]], on="comp_id")
m["year"] = m.match_date.str[:4].astype(int)
b = b[b.innings.isin([1, 2])].merge(m[["match_id", "grp", "year", "revised_target", "first_inn_revised_overs", "second_inn_revised_overs"]], on="match_id")
b = b.sort_values(["match_id", "innings", "innings_ball_num"]).reset_index(drop=True)

# ---- delivery semantics (validated): runs = bat runs; extras1/2 separate ----
et = b.extras_type.fillna("")
b["wide"] = et.eq("wide"); b["nb"] = et.isin(["Overstep", "other"])
b["legal"] = ~(b.wide | b.nb)
b["faced"] = ~b.wide
chg = et.isin(["wide", "Overstep", "other"])           # extras charged to bowler (not byes/leg-byes/penalty)
b["conc"] = b.runs + np.where(chg, b.extras1_runs + b.extras_2_runs, 0)
b["team_runs"] = b.runs + b.extras1_runs + b.extras_2_runs
BOWL_W = ["Caught", "Bowled", "LBW", "Stump", "Hit Wicket"]
b["bwk"] = b.is_wicket & b.dismissal_type.isin(BOWL_W)   # run-outs/retired excluded
b["out_id"] = np.where(b.is_wicket & (b.dismissal_type != "Retire Out"), b.dismissed_player_id.fillna(b.striker_id), np.nan)
b["bound"] = b.is_four | b.is_six

# ---- match state: cumulative runs/legal balls/wickets BEFORE each ball ----
g = b.groupby(["match_id", "innings"])
b["cum_runs"] = g.team_runs.cumsum() - b.team_runs
b["cum_legal"] = g.legal.cumsum() - b.legal.astype(int)
b["cum_wk"] = g.is_wicket.cumsum() - b.is_wicket.astype(int)
tot = g.team_runs.sum().unstack().rename(columns={1: "t1", 2: "t2"})
used = b[b.innings == 2].groupby("match_id").legal.sum().rename("balls2")
wk2 = b[b.innings == 2].groupby("match_id").is_wicket.sum().rename("wk2")
mt = tot.join(used).join(wk2).join(m.set_index("match_id")[["win_team_id", "revised_target", "first_inn_revised_overs", "second_inn_revised_overs"]])
mt = mt.dropna(subset=["t1", "t2"])
clean = mt.revised_target.isna() & mt.first_inn_revised_overs.isna() & mt.second_inn_revised_overs.isna()
# Close match (derived from deliveries): chase fell short/tied by <=10, or chase won with <=6 balls left
mt["close"] = clean & (((mt.t2 <= mt.t1) & (mt.t1 + 1 - mt.t2 <= 10)) | ((mt.t2 > mt.t1) & (mt.balls2 >= 114)))
b = b.merge(mt[["t1", "close"]], left_on="match_id", right_index=True, how="left")
b["close"] = b.close.fillna(False).astype(bool)

# ---- pressure definitions ----
b["phase"] = np.where(b.overid <= 6, "PP", np.where(b.overid <= 15, "MID", "DEATH"))
need = b.t1 + 1 - b.cum_runs; left = 120 - b.cum_legal
b["rrr"] = np.where((b.innings == 2) & (left > 0), need / left.clip(lower=1) * 6, np.nan)
cleanball = b.match_id.isin(mt.index[clean])
b["chaseP"] = (b.innings == 2) & (b.overid >= 7) & (b.rrr >= 9.5) & cleanball
b["pressure"] = (b.phase == "DEATH") | b.chaseP
b["base"] = ~b.pressure

# ---- aggregation helpers ----
def bat_full(sub):
    f = sub[sub.faced]
    a = f.groupby("striker_id").agg(r=("runs", "sum"), b=("faced", "sum"), bd=("bound", "sum"), d=("runs", lambda x: (x == 0).sum()), inn=("match_id", "nunique"))
    o = sub[sub.out_id.notna()].groupby("out_id").size().rename("o")
    return a.join(o).fillna({"o": 0})
def bowl(sub):
    a = sub.groupby("bowler_id").agg(b=("legal", "sum"), r=("conc", "sum"), w=("bwk", "sum"), d=("conc", lambda x: (x == 0).sum()), inn=("match_id", "nunique"))
    return a
def pack(df): return {int(i): [int(x) for x in r] for i, r in df.iterrows()}

masks = {"all": b.index == b.index, "pp": b.phase == "PP", "mid": b.phase == "MID", "death": b.phase == "DEATH",
         "chase": b.innings == 2, "chaseP": b.chaseP, "close": b.close, "press": b.pressure, "base": b.base}
BC, WC = ["r", "b", "bd", "d", "inn", "o"], ["b", "r", "w", "d", "inn"]
out = {"bat": {}, "bowl": {}}
for grp in ["ALL", "IPL", "T20I"]:
    gm = np.ones(len(b), bool) if grp == "ALL" else (b.grp == grp).values
    for k, mk in masks.items():
        sub = b[gm & np.asarray(mk)]
        out["bat"][f"{grp}|{k}"] = pack(bat_full(sub)[BC]); out["bowl"][f"{grp}|{k}"] = pack(bowl(sub)[WC])

# league benchmarks (pressure vs ordinary) for context + shrinkage priors
def lg_bat(s): s = s[s.faced]; return [int(s.runs.sum()), int(len(s)), int(s.bound.sum()), int((s.runs == 0).sum())]
def lg_bowl(s): return [int(s.legal.sum()), int(s.conc.sum()), int(s.bwk.sum()), int((s.conc[s.legal] == 0).sum())]
league = {k: {"bat": lg_bat(b[np.asarray(mk)]), "bowl": lg_bowl(b[np.asarray(mk)])} for k, mk in masks.items()}
league["byGrp"] = {gp: {ph: {"bat": lg_bat(b[(b.grp == gp) & (b.phase == ph)]), "bowl": lg_bowl(b[(b.grp == gp) & (b.phase == ph)])} for ph in ["PP", "MID", "DEATH"]} for gp in ["IPL", "T20I"]}

# per-innings / per-spell consistency (pressure balls only)
P = b[b.pressure]
lgsr = P[P.faced].runs.sum() / P.faced.sum() * 100; lgec = P.conc.sum() / P.legal.sum() * 6
pi = P[P.faced].groupby(["match_id", "striker_id"]).agg(r=("runs", "sum"), n=("faced", "sum")).reset_index()
pi = pi[pi.n >= 4]; pi["ok"] = pi.r / pi.n * 100 >= lgsr
bc = pi.groupby("striker_id").agg(q=("ok", "size"), ok=("ok", "sum"))
ps = P.groupby(["match_id", "bowler_id"]).agg(n=("legal", "sum"), r=("conc", "sum"), w=("bwk", "sum")).reset_index()
ps = ps[ps.n >= 6]; ps["ok"] = (ps.r / ps.n * 6 <= lgec) | (ps.w > 0)
wc = ps.groupby("bowler_id").agg(q=("ok", "size"), ok=("ok", "sum"))
cons = {"bat": {int(i): [int(r.q), int(r.ok)] for i, r in bc.iterrows()}, "bowl": {int(i): [int(r.q), int(r.ok)] for i, r in wc.iterrows()}}

# trend by year (batting: runs,balls; bowling: balls,runs,wkts) on pressure balls
tb = P[P.faced].groupby(["striker_id", "year"]).agg(r=("runs", "sum"), b=("faced", "sum"))
tw = P.groupby(["bowler_id", "year"]).agg(b=("legal", "sum"), r=("conc", "sum"), w=("bwk", "sum"))
trend = {"bat": {}, "bowl": {}}
for (i, y), r in tb.iterrows(): trend["bat"].setdefault(int(i), {})[int(y)] = [int(r.r), int(r.b)]
for (i, y), r in tw.iterrows(): trend["bowl"].setdefault(int(i), {})[int(y)] = [int(r.b), int(r.r), int(r.w)]
# opposition (bowlers, all balls)
tm = b.groupby(["bowler_id", "batting_team_id"]).agg(b=("legal", "sum"), r=("conc", "sum"), w=("bwk", "sum"))
tname = pd.concat([m[["home_team_id", "home_team"]].set_axis(["id", "n"], axis=1), m[["away_team_id", "away_team"]].set_axis(["id", "n"], axis=1)]).drop_duplicates("id").set_index("id").n
opp = {}
for (i, t), r in tm[tm.b >= 12].iterrows(): opp.setdefault(int(i), {})[tname.get(t, str(t))] = [int(r.b), int(r.r), int(r.w)]

# ---- player meta ----
nm = mp.drop_duplicates("player_id").set_index("player_id").player_name
ipl26 = mp[mp.comp_name == "IPL 2026"].drop_duplicates("player_id").set_index("player_id").team_name
rr26 = set(mp[(mp.comp_name == "IPL 2026") & mp.team_name.str.contains("Rajasthan")].player_id)
ind_t20i = set(mp[(mp.team_name == "India") & mp.comp_name.str.contains("T20I")].player_id)
bt = b.drop_duplicates("striker_id").set_index("striker_id").batting_type
ps_ = b.groupby("bowler_id").pace_or_spin.agg(lambda x: x.value_counts().index[0])
bo = b[b.faced].groupby("striker_id").batting_order.mean()
death_b = b[(b.phase == "DEATH") & b.faced].groupby("striker_id").size()
ids = sorted(set(b.striker_id) | set(b.bowler_id))
players = []
for i in ids:
    players.append({"id": int(i), "n": nm.get(i, str(i)), "hand": bt.get(i, ""), "ps": ps_.get(i, ""), "bo": round(float(bo.get(i, 0)), 1),
                    "t26": ipl26.get(i, ""), "rr": int(i in rr26), "ind": int(i in ind_t20i)})
# ---- validation ----
win_by_total = np.where(mt.t2 > mt.t1, 2, np.where(mt.t2 < mt.t1, 1, 0))
chk = {"matches_in_bbb": int(b.match_id.nunique()), "matches_total": int(len(m)), "balls": int(len(b)),
       "super_over_balls_excluded": int(pd.read_csv(f"{D}/ball_by_ball.csv", usecols=["innings"]).innings.gt(2).sum()),
       "dup_balls": int(b.duplicated(["match_id", "innings", "innings_ball_num"]).sum()),
       "bat_runs_equal_sum": int(b.runs.sum()), "extras_total": int((b.extras1_runs + b.extras_2_runs).sum()),
       "bowler_wkts": int(b.bwk.sum()), "run_outs_excluded": int((b.dismissal_type == "RunOut").sum()),
       "close_matches": int(mt.close.sum()), "clean_matches": int(clean.sum()),
       "totals_agree_with_result_pct": round(float((pd.Series(win_by_total, index=mt.index)[mt.win_team_id.notna()] > 0).mean() * 100), 1),
       "missing_player_mappings": int(len(set(ids) - set(mp.player_id))),
       "lg_pressure_sr": round(lgsr, 2), "lg_pressure_econ": round(lgec, 2)}
# result vs totals agreement (home/away mapping)
mm = m.set_index("match_id").loc[mt.index]
fb = b[b.innings == 1].groupby("match_id").batting_team_id.first()
w_tot = np.where(mt.t2 > mt.t1, 2, 1); sec = b[b.innings == 2].groupby("match_id").batting_team_id.first()
pred = pd.Series(np.where(mt.t2 > mt.t1, sec.reindex(mt.index), fb.reindex(mt.index)), index=mt.index)
dec = (mt.t2 != mt.t1) & mm.win_team_id.notna() & clean
chk["winner_matches_totals_pct"] = round(float((pred[dec] == mm.win_team_id[dec]).mean() * 100), 1); chk["winner_check_n"] = int(dec.sum())
chk["data_note"] = "Only IPL 2022-26, T20Is and two emerging-team Asia Cups exist in the pack: no true domestic T20 data. No auction, price, age, nationality or availability fields."
yrs = sorted(int(y) for y in m.year.unique())
res = {"checks": chk, "league": league, "players": players, "stat": out, "cons": cons, "trend": trend, "opp": opp, "years": yrs}
open("dashboard/data.js", "w").write("window.DATA=" + json.dumps(res, separators=(",", ":")) + ";")
print(json.dumps(chk, indent=1))
