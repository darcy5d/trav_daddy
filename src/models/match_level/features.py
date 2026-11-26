"""Strictly pre-match features, one row per match, plus the rolling state
needed to score an upcoming fixture.

Every feature for match M uses only matches dated before M (same-day matches
are excluded too, because Cricsheet dates do not order double-headers). The
tiered ELO is `elo - elo_change` on the match's own history row, which is the
rating the production calculator held going into that match.

`MatchFeatureBuilder.run_history()` walks every match in date order. The
builder can then be pickled and `features_for_upcoming()` scores fixtures that
are not in the database yet, from the state after the last processed date.
"""
from __future__ import annotations

import math
import sqlite3
from collections import defaultdict, deque
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

QUERIES = {
    "matches": """
        SELECT m.match_id, m.match_type, m.date, m.venue_id, m.team1_id, m.team2_id,
               m.toss_winner_id, m.toss_decision, m.winner_id, m.event_name, m.gender,
               m.match_category, m.cricsheet_id
        FROM matches m
    """,
    "venues": "SELECT venue_id, country FROM venues",
    "teams": "SELECT team_id, name, team_type, tier, canonical_team_id FROM teams",
    "innings": """SELECT match_id, innings_number, batting_team_id, total_runs FROM innings""",
    "team_elo": """
        SELECT team_id, match_id, elo, elo_change
        FROM team_elo_history
        WHERE is_monthly_snapshot = 0 AND match_id IS NOT NULL
    """,
    "player_elo": """
        SELECT player_id, match_id, batting_elo, bowling_elo,
               batting_elo_change, bowling_elo_change
        FROM player_elo_history
        WHERE is_monthly_snapshot = 0 AND match_id IS NOT NULL
    """,
    "pms": """
        SELECT match_id, player_id, team_id, batting_innings, runs_scored, balls_faced,
               overs_bowled, runs_conceded, wickets_taken
        FROM player_match_stats
    """,
}


def load_tables(db_path: str) -> Dict[str, pd.DataFrame]:
    """Read the tables the builder needs. Opens the database read-only."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120)
    try:
        return {name: pd.read_sql_query(sql, conn) for name, sql in QUERIES.items()}
    finally:
        conn.close()


class SimpleElo:
    """Per (format, gender) Elo with a margin-free update."""

    def __init__(self, k: float = 24.0, base: float = 1500.0):
        self.k = k
        self.base = base
        self.r: dict = {}

    def get(self, key):
        return self.r.get(key, self.base)

    def update(self, k1, k2, s1: float, k_mult: float = 1.0):
        r1, r2 = self.get(k1), self.get(k2)
        e1 = 1.0 / (1.0 + 10 ** ((r2 - r1) / 400.0))
        d = self.k * k_mult * (s1 - e1)
        self.r[k1] = r1 + d
        self.r[k2] = r2 - d


def _deque20():
    return deque(maxlen=20)


def _deque60():
    return deque(maxlen=60)


def _int_dict():
    return defaultdict(int)


def _pair():
    return [0, 0]


def _venue_rec():
    return [0, 0, 0.0, 0]


PF_ALPHA = 0.15
UNKNOWN_COUNTRY = "?"


class MatchFeatureBuilder:
    def __init__(self, tables: Dict[str, pd.DataFrame], declared_xi: Dict[Tuple[int, int], List[int]]):
        teams = tables["teams"].set_index("team_id")
        venues = tables["venues"].set_index("venue_id")
        self.canon = {
            int(t): int(c) if pd.notna(c) else int(t)
            for t, c in teams["canonical_team_id"].items()
        }
        # Venues without a country share one "unknown" country key. The
        # walk-forward evaluation was run that way; distinct NaN objects would
        # not compare equal as dict keys.
        self.venue_country = {k: (v if isinstance(v, str) and v else UNKNOWN_COUNTRY)
                              for k, v in venues["country"].items()}
        self.team_name = {k: (v if isinstance(v, str) and v else None)
                          for k, v in teams["name"].items()}
        self.team_type = teams["team_type"].to_dict()
        self.team_tier = teams["tier"].to_dict()

        self.elo_simple = SimpleElo(k=24.0)
        self.elo_fast = SimpleElo(k=40.0)
        self.team_results = defaultdict(_deque20)
        self.team_dates = defaultdict(list)
        self.team_elo_trail = defaultdict(_deque60)
        self.team_country_counts = defaultdict(_int_dict)
        self.team_venue_counts = defaultdict(int)
        self.h2h = defaultdict(_pair)
        self.venue_stats = defaultdict(_venue_rec)
        self.toss_stats = defaultdict(_pair)
        self.last_xi = {}
        self.last_pelo = {}
        self.last_tiered_elo = {}
        self.pform = {}
        self.last_date: Optional[pd.Timestamp] = None

        self._tables = tables
        self._declared_xi = declared_xi

    def canonical(self, team_id: int) -> int:
        return self.canon.get(int(team_id), int(team_id))

    # ------------------------------------------------------------------
    def _features(self, r: dict, t1_pre: float, t2_pre: float,
                  xi1: List[int], xi2: List[int]) -> dict:
        fmt, gender = r["match_type"], r["gender"]
        pool = (fmt, gender)
        c1, c2 = self.canonical(r["team1_id"]), self.canonical(r["team2_id"])
        date = r["date"]
        feat = {
            "match_id": r.get("match_id"), "date": date, "format": fmt, "gender": gender,
            "team1_id": int(r["team1_id"]), "team2_id": int(r["team2_id"]), "c1": c1, "c2": c2,
            "event_name": r.get("event_name"), "match_category": r.get("match_category"),
            "venue_id": r.get("venue_id"),
            "tiered_elo_pre_1": t1_pre, "tiered_elo_pre_2": t2_pre,
            "simple_elo_1": self.elo_simple.get((pool, c1)), "simple_elo_2": self.elo_simple.get((pool, c2)),
            "fast_elo_1": self.elo_fast.get((pool, c1)), "fast_elo_2": self.elo_fast.get((pool, c2)),
            "is_t20": float(fmt == "T20"), "is_female": float(gender == "female"),
        }
        tt1, tt2 = self.team_type.get(c1), self.team_type.get(c2)
        feat["intl"] = float(tt1 == "international" and tt2 == "international")
        feat["franchise"] = float(tt1 == "franchise" or tt2 == "franchise")
        feat["tier_1"] = float(self.team_tier.get(c1) or 3)
        feat["tier_2"] = float(self.team_tier.get(c2) or 3)

        venue_id = r.get("venue_id")
        vcountry = self.venue_country.get(venue_id)
        for side, c in (("1", c1), ("2", c2)):
            res = self.team_results[(pool, c)]
            n = len(res)
            if n:
                ws = np.array([x[1] for x in res], dtype=float)
                wts = 0.8 ** np.arange(n)[::-1]
                feat[f"form_ewm_{side}"] = float((ws * wts).sum() / wts.sum())
                feat[f"form5_{side}"] = float(ws[-5:].mean())
            else:
                feat[f"form_ewm_{side}"] = 0.5
                feat[f"form5_{side}"] = 0.5
            dts = self.team_dates[(pool, c)]
            feat[f"n_prior_{side}"] = float(len(dts))
            feat[f"rest_days_{side}"] = float(min(60, (date - dts[-1]).days)) if dts else 60.0
            feat[f"games_30d_{side}"] = float(sum(1 for d in dts[-15:] if (date - d).days <= 30))
            trail = self.team_elo_trail[(pool, c)]
            old = [e for d, e in trail if e is not None and (date - d).days >= 90]
            cur_pre = t1_pre if side == "1" else t2_pre
            feat[f"elo_mom90_{side}"] = (float(cur_pre - old[-1]) if old and not np.isnan(cur_pre) else 0.0)
            cc = self.team_country_counts[(gender, c)]
            tot = sum(cc.values())
            if tot and vcountry:
                feat[f"country_share_{side}"] = cc.get(vcountry, 0) / tot
                feat[f"is_home_{side}"] = float(max(cc, key=cc.get) == vcountry and tot >= 3)
            else:
                feat[f"country_share_{side}"] = np.nan
                feat[f"is_home_{side}"] = 0.0
            if tt1 == "international" and self.team_name.get(c) and vcountry:
                if str(self.team_name.get(c)).split(" ")[0] == str(vcountry).split(" ")[0]:
                    feat[f"is_home_{side}"] = 1.0
            feat[f"venue_games_{side}"] = float(min(30, self.team_venue_counts[(gender, c, venue_id)]))

        a, b = sorted((c1, c2))
        rec = self.h2h[(pool, a, b)]
        wins_c1 = rec[0] if a == c1 else rec[1] - rec[0]
        feat["h2h_n"] = float(rec[1])
        feat["h2h_rate_1"] = (wins_c1 + 1.0) / (rec[1] + 2.0)

        vs = self.venue_stats[(pool, venue_id)]
        feat["venue_n"] = float(vs[1])
        feat["venue_bat1_rate"] = (vs[0] + 2.0) / (vs[1] + 4.0)
        par = 160.0 if fmt == "T20" else 260.0
        feat["venue_par_rel"] = ((vs[2] + par * 5) / (vs[3] + 5)) / par
        ts = self.toss_stats[(pool, venue_id)]
        feat["venue_toss_win_rate"] = (ts[0] + 2.0) / (ts[1] + 4.0)

        for side, c, players in (("1", c1, xi1), ("2", c2, xi2)):
            bats, bowls, runs, srs, bwks, econs, exps = [], [], [], [], [], [], []
            for p in players:
                pre = self.last_pelo.get((pool, p), (1500.0, 1500.0))
                bats.append(pre[0]); bowls.append(pre[1])
                pf = self.pform.get((pool, p))
                if pf is not None:
                    runs.append(pf["runs"]); srs.append(pf["sr"]); exps.append(pf["n"])
                    if pf["nb"] >= 3:
                        bwks.append(pf["bwk"]); econs.append(pf["econ"])
                else:
                    exps.append(0)
            bats = sorted(bats, reverse=True)[:7]
            bowls = sorted(bowls, reverse=True)[:5]
            feat[f"xi_bat_elo_{side}"] = float(np.mean(bats)) if bats else np.nan
            feat[f"xi_bowl_elo_{side}"] = float(np.mean(bowls)) if bowls else np.nan
            feat[f"xi_runs_form_{side}"] = float(np.sum(sorted(runs, reverse=True)[:7])) if runs else np.nan
            feat[f"xi_sr_form_{side}"] = float(np.mean(sorted(srs, reverse=True)[:7])) if srs else np.nan
            feat[f"xi_bwk_form_{side}"] = float(np.sum(sorted(bwks, reverse=True)[:5])) if bwks else np.nan
            feat[f"xi_econ_form_{side}"] = float(np.mean(sorted(econs)[:5])) if econs else np.nan
            feat[f"xi_exp_{side}"] = float(np.median(exps)) if exps else 0.0
            feat[f"xi_debut_{side}"] = float(sum(1 for e in exps if e == 0))
            prev = self.last_xi.get((pool, c))
            feat[f"xi_overlap_{side}"] = (len(prev & set(players)) / max(1, len(players))) if prev and players else np.nan
            feat[f"xi_n_{side}"] = float(len(players))
        return feat

    def _apply_updates(self, batch: Iterable[dict]) -> None:
        for b in batch:
            pool, c1, c2 = b["pool"], b["c1"], b["c2"]
            y = b["y"]
            if y is not None:
                self.elo_simple.update((pool, c1), (pool, c2), y)
                self.elo_fast.update((pool, c1), (pool, c2), y)
                self.team_results[(pool, c1)].append((b["date"], y))
                self.team_results[(pool, c2)].append((b["date"], 1 - y))
                a, bb = sorted((c1, c2))
                rec = self.h2h[(pool, a, bb)]
                rec[0] += (y if a == c1 else 1 - y)
                rec[1] += 1
                if b["bat1"] is not None:
                    vs = self.venue_stats[(pool, b["venue"])]
                    vs[0] += b["bat1"]
                    vs[1] += 1
                if b["toss_won"] is not None:
                    ts = self.toss_stats[(pool, b["venue"])]
                    ts[0] += b["toss_won"]
                    ts[1] += 1
            if b["fi"] is not None:
                vs = self.venue_stats[(pool, b["venue"])]
                vs[2] += b["fi"]
                vs[3] += 1
            for c in (c1, c2):
                self.team_dates[(pool, c)].append(b["date"])
                self.team_elo_trail[(pool, c)].append((b["date"], b["elo_pre"].get(c)))
                post = b["elo_post"].get(c)
                if post is not None:
                    self.last_tiered_elo[(pool, c)] = post
                if b["vcountry"]:
                    self.team_country_counts[(b["gender"], c)][b["vcountry"]] += 1
                self.team_venue_counts[(b["gender"], c, b["venue"])] += 1
            for c, ps in b["xi_sets"].items():
                self.last_xi[(pool, c)] = ps
            for pid, stats in b["pstats"].items():
                key = (pool, pid)
                post = b["pelo_post"].get(pid)
                if post is not None:
                    self.last_pelo[key] = post
                cur = self.pform.get(key)
                if cur is None:
                    cur = {"n": 0, "runs": 20.0, "sr": 120.0 if b["fmt"] == "T20" else 80.0,
                           "bwk": 1.0, "econ": 8.0 if b["fmt"] == "T20" else 5.5, "nb": 0}
                cur["n"] += 1
                if stats["bat"]:
                    cur["runs"] = (1 - PF_ALPHA) * cur["runs"] + PF_ALPHA * stats["runs"]
                    if stats["balls"] > 0:
                        cur["sr"] = (1 - PF_ALPHA) * cur["sr"] + PF_ALPHA * min(400.0, 100.0 * stats["runs"] / stats["balls"])
                if stats["overs"] > 0:
                    cur["nb"] += 1
                    cur["bwk"] = (1 - PF_ALPHA) * cur["bwk"] + PF_ALPHA * stats["wk"]
                    cur["econ"] = (1 - PF_ALPHA) * cur["econ"] + PF_ALPHA * min(36.0, stats["rc"] / stats["overs"])
                self.pform[key] = cur

    # ------------------------------------------------------------------
    def run_history(self, before: Optional[pd.Timestamp] = None) -> pd.DataFrame:
        """Walk every match (optionally only those dated before `before`)."""
        t = self._tables
        m = t["matches"].copy()
        m["date"] = pd.to_datetime(m["date"].astype(str).str[:10])
        if before is not None:
            m = m[m.date < before]
        m = m.sort_values(["date", "match_id"]).reset_index(drop=True)

        telo = t["team_elo"].copy()
        telo["pre"] = telo["elo"] - telo["elo_change"].fillna(0.0)
        te = {(int(r.match_id), int(r.team_id)): (r.pre, r.elo) for r in telo.itertuples()}
        pelo = t["player_elo"]
        pe = {
            (int(r.match_id), int(r.player_id)): (r.batting_elo, r.bowling_elo)
            for r in pelo.itertuples()
        }
        inn = t["innings"]
        first_inn = inn[inn.innings_number == 1].set_index("match_id")
        xi = {mid: g for mid, g in t["pms"].groupby("match_id")}
        dxi = self._declared_xi

        rows = []
        pending: List[dict] = []
        cur_date = None
        for r in m.to_dict("records"):
            if cur_date is not None and r["date"] != cur_date:
                self._apply_updates(pending)
                pending = []
            cur_date = r["date"]
            fmt, gender = r["match_type"], r["gender"]
            pool = (fmt, gender)
            c1, c2 = self.canonical(r["team1_id"]), self.canonical(r["team2_id"])
            mid = int(r["match_id"])
            if pd.isna(r["winner_id"]):
                y = None
            else:
                w = self.canonical(r["winner_id"])
                y = 1 if w == c1 else (0 if w == c2 else None)
            t1_pre, t1_post = te.get((mid, c1), (np.nan, np.nan))
            t2_pre, t2_post = te.get((mid, c2), (np.nan, np.nan))

            players = {}
            for c, raw in ((c1, int(r["team1_id"])), (c2, int(r["team2_id"]))):
                players[c] = dxi.get((mid, raw)) or dxi.get((mid, c)) or []
            feat = self._features(r, t1_pre, t2_pre, players[c1], players[c2])
            feat["y"] = y
            feat["toss_t1"] = (np.nan if pd.isna(r["toss_winner_id"])
                               else float(self.canonical(r["toss_winner_id"]) == c1))
            feat["toss_bat"] = (np.nan if r["toss_decision"] is None else float(r["toss_decision"] == "bat"))
            feat["tiered_elo_post_1"] = t1_post
            feat["tiered_elo_post_2"] = t2_post

            g = xi.get(mid)
            pstats = {}
            if g is not None:
                for raw, c in ((int(r["team1_id"]), c1), (int(r["team2_id"]), c2)):
                    gg = g[(g.team_id == raw) | (g.team_id == c)]
                    for s in gg.itertuples():
                        ov = float(s.overs_bowled or 0)
                        ov = math.floor(ov) + (ov - math.floor(ov)) * 10 / 6
                        pstats[int(s.player_id)] = {
                            "bat": bool(s.batting_innings), "runs": float(s.runs_scored or 0),
                            "balls": float(s.balls_faced or 0), "overs": ov,
                            "wk": float(s.wickets_taken or 0), "rc": float(s.runs_conceded or 0),
                        }

            fi = first_inn.loc[mid] if mid in first_inn.index else None
            bat1_won = None
            fi_runs = None
            if fi is not None and not isinstance(fi, pd.DataFrame):
                fi_runs = float(fi.total_runs) if pd.notna(fi.total_runs) else None
                bt = self.canonical(fi.batting_team_id)
                if y is not None:
                    bat1_won = int((bt == c1) == (y == 1))
                feat["bat_first_t1"] = float(bt == c1)
            else:
                feat["bat_first_t1"] = np.nan
            toss_won = None
            if y is not None and not np.isnan(feat["toss_t1"]):
                toss_won = int((feat["toss_t1"] == 1.0) == (y == 1))

            rows.append(feat)
            pending.append({
                "pool": pool, "c1": c1, "c2": c2, "y": y, "date": r["date"], "venue": r["venue_id"],
                "vcountry": self.venue_country.get(r["venue_id"]), "gender": gender, "fmt": fmt,
                "bat1": bat1_won, "fi": fi_runs, "toss_won": toss_won,
                "elo_pre": {c1: None if np.isnan(t1_pre) else t1_pre,
                            c2: None if np.isnan(t2_pre) else t2_pre},
                "elo_post": {c1: None if np.isnan(t1_post) else t1_post,
                             c2: None if np.isnan(t2_post) else t2_post},
                "xi_sets": {c1: set(players[c1]), c2: set(players[c2])},
                "pstats": pstats,
                "pelo_post": {pid: pe[(mid, pid)] for pid in pstats if (mid, pid) in pe},
            })
        self._apply_updates(pending)
        self.last_date = cur_date
        return pd.DataFrame(rows)

    def compact(self) -> "MatchFeatureBuilder":
        """Drop the history tables so the builder pickles small."""
        self._tables = None
        self._declared_xi = None
        return self

    def features_for_upcoming(
        self,
        team1_id: int,
        team2_id: int,
        fmt: str,
        gender: str,
        venue_id: Optional[int],
        date: pd.Timestamp,
        xi1: Optional[List[int]] = None,
        xi2: Optional[List[int]] = None,
    ) -> dict:
        """Features for a fixture after the last processed date. Leaves state unchanged."""
        pool = (fmt, gender)
        c1, c2 = self.canonical(team1_id), self.canonical(team2_id)
        d = pd.Timestamp(date)
        if d.tzinfo is not None:
            d = d.tz_convert("UTC").tz_localize(None)
        r = {"match_type": fmt, "gender": gender, "team1_id": team1_id, "team2_id": team2_id,
             "venue_id": venue_id, "date": d.normalize()}
        t1 = self.last_tiered_elo.get((pool, c1), np.nan)
        t2 = self.last_tiered_elo.get((pool, c2), np.nan)
        return self._features(r, t1, t2, list(xi1 or []), list(xi2 or []))


def load_declared_xi(
    tables: Dict[str, pd.DataFrame],
    db_path: str,
    raw_dirs: Iterable[str],
    cache: Optional[Dict[Tuple[int, int], List[int]]] = None,
) -> Dict[Tuple[int, int], List[int]]:
    """Declared playing XI per (match, team) from the Cricsheet JSON `info.players`.

    player_match_stats lists only players who appear in the scorecard, which
    depends on the outcome; the declared XI is what is known at the toss.
    Matches already in `cache` are not re-read.
    """
    import glob
    import json
    from pathlib import Path

    out: Dict[Tuple[int, int], List[int]] = dict(cache or {})
    done = {mid for mid, _ in out}
    todo = tables["matches"][~tables["matches"].match_id.isin(done)]
    if todo.empty:
        return out
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120)
    try:
        reg = dict(conn.execute("SELECT registry_id, player_id FROM players WHERE registry_id IS NOT NULL").fetchall())
    finally:
        conn.close()
    tnames = tables["teams"].set_index("team_id")["name"].to_dict()
    files = {}
    for d in raw_dirs:
        for f in glob.glob(str(Path(d) / "*.json")):
            files.setdefault(Path(f).stem, f)
    for r in todo.itertuples():
        f = files.get(str(r.cricsheet_id))
        if f is None:
            continue
        with open(f) as fh:
            info = json.load(fh)["info"]
        people = info.get("registry", {}).get("people", {})
        for team_name, names in (info.get("players") or {}).items():
            if team_name == tnames.get(r.team1_id):
                tid = r.team1_id
            elif team_name == tnames.get(r.team2_id):
                tid = r.team2_id
            else:
                continue
            pids = [reg.get(people.get(n)) for n in names]
            out[(int(r.match_id), int(tid))] = [p for p in pids if p is not None]
    return out
