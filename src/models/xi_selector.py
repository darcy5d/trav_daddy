"""Pre-toss XI prediction: rank each team's recent players by P(in the XI).

Candidates are every player in the team's last 10 known declared XIs in this
format plus its last 5 in the other format (same gender). A gradient-boosted
classifier scores each candidate from selection recency, how long ago the team
last played, appearances for other teams since, and recent workload; the top 11
form the XI. `scripts/train_xi_selector.py` fits it and writes the per-segment
gate: a format/gender only uses it if it beat the last-match XI on the
time-split holdout.

History is the Cricsheet declared XI (`data/models/match_level/declared_xi.pkl`,
refreshed daily), not player_match_stats, which has no row for players who
neither batted nor bowled. Declared XIs list players in batting order, so a
player's mean index there doubles as a batting-position estimate.
"""
from __future__ import annotations

import os
import pickle
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
ARTIFACT_DIR = REPO / "data" / "models" / "xi_selector"
DECLARED_XI_PATH = REPO / "data" / "models" / "match_level" / "declared_xi.pkl"
FEATURES = [
    "in1", "in2", "in3", "in4", "in5", "n10", "decay", "gap", "days_since_app",
    "team_stale_days", "other_fmt5", "other_team_since", "other_team_14d",
    "apps_any_60d", "career_apps", "mean_pos",
]
N_SAME, N_OTHER = 10, 5


def load_sides(db_path, decl: pd.DataFrame, since: str = "2022-01-01") -> pd.DataFrame:
    """One row per (match, team) with its declared XI, match date and ingest time."""
    c = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    m = pd.read_sql(
        "SELECT match_id, date, created_at, match_type, gender, team1_id, team2_id "
        "FROM matches WHERE date >= ?", c, params=(since,))
    c.close()
    dx = {(int(r.match_id), int(r.team_id)): list(r.players) for r in decl.itertuples()}
    rows = []
    for r in m.itertuples():
        for t in (r.team1_id, r.team2_id):
            xi = dx.get((int(r.match_id), int(t)))
            if xi:
                rows.append((int(r.match_id), int(t), r.date, r.created_at, r.match_type, r.gender, xi))
    s = pd.DataFrame(rows, columns=["match_id", "team_id", "date", "created_at", "fmt", "gender", "xi"])
    s["d"] = pd.to_datetime(s.date)
    s["created"] = pd.to_datetime(s.created_at, utc=True, format="mixed").dt.tz_convert(None)
    return s.sort_values(["d", "match_id"]).reset_index(drop=True)


class History:
    """Declared-XI history indexed for as-of feature lookups.

    `vis` is when each match became known; a match is usable for a target on
    day D only if it was played before D and known by D.
    """

    def __init__(self, sides: pd.DataFrame, vis: Optional[pd.Series] = None):
        s = sides.assign(vis=sides.d if vis is None else vis)
        self.groups = {k: g.sort_values("d", ascending=False) for k, g in s.groupby(["team_id", "gender"])}
        self.apps: Dict[int, List[Tuple]] = defaultdict(list)
        for r in s.itertuples():
            for idx, p in enumerate(r.xi):
                self.apps[p].append((r.d, r.vis, r.team_id, r.gender, idx + 1))

    def known_players(self, team_id: int, gender: str, d: pd.Timestamp, n: int = N_SAME) -> set:
        g = self.groups.get((team_id, gender))
        if g is None:
            return set()
        return set().union(*[set(x) for x in g[(g.d < d) & (g.vis <= d)].xi.head(n)])

    def candidates(self, team_id: int, gender: str, fmt: str, d: pd.Timestamp, extra=()):
        """(features DataFrame, info) for one target side; features empty if no history.

        `extra` adds players (e.g. a squad's newcomers) to the usual candidates.
        """
        g = self.groups.get((team_id, gender))
        if g is None:
            return pd.DataFrame(columns=["player_id"] + FEATURES), {"n_hist": 0, "stale_days": None}
        known = g[(g.d < d) & (g.vis <= d)]
        same = known[known.fmt == fmt]
        if len(same) == 0:
            return pd.DataFrame(columns=["player_id"] + FEATURES), {"n_hist": 0, "stale_days": None}
        other = known[known.fmt != fmt]
        hist = [set(x) for x in same.xi.head(N_SAME)]
        other_sets = [set(x) for x in other.xi.head(N_OTHER)]
        stale = int((d - same.d.iloc[0]).days)
        cand = set().union(*hist, *other_sets, set(extra))
        rows = []
        for p in cand:
            prior = [a for a in self.apps[p] if a[0] < d and a[1] <= d]
            last_team = max((a[0] for a in prior if a[2] == team_id and a[3] == gender), default=None)
            recent_pos = [a[4] for a in sorted(prior, key=lambda a: a[0])[-10:]]
            ins = [int(p in h) for h in hist[:5]] + [0] * (5 - min(5, len(hist)))
            rows.append({
                "player_id": p,
                "in1": ins[0], "in2": ins[1], "in3": ins[2], "in4": ins[3], "in5": ins[4],
                "n10": sum(p in h for h in hist),
                "decay": sum(0.7 ** j for j, h in enumerate(hist) if p in h),
                "gap": next((j for j, h in enumerate(hist) if p in h), N_SAME),
                "days_since_app": (d - last_team).days if last_team is not None else 9999,
                "team_stale_days": stale,
                "other_fmt5": sum(p in o for o in other_sets),
                "other_team_since": sum(1 for a in prior if a[2] != team_id
                                        and last_team is not None and a[0] > last_team),
                "other_team_14d": sum(1 for a in prior if a[2] != team_id and (d - a[0]).days <= 14),
                "apps_any_60d": sum(1 for a in prior if (d - a[0]).days <= 60),
                "career_apps": min(200, sum(1 for a in prior if a[2] == team_id)),
                "mean_pos": float(np.mean(recent_pos)) if recent_pos else 12.0,
            })
        return pd.DataFrame(rows), {"n_hist": len(same), "stale_days": stale}


def pick_xi(cands: pd.DataFrame, probs: np.ndarray, n: int = 11) -> List[int]:
    """Top-n by probability, returned in estimated batting order."""
    c = cands.assign(p=probs).sort_values(["p", "player_id"], ascending=[False, True]).head(n)
    return [int(x) for x in c.sort_values(["mean_pos", "player_id"]).player_id]


@dataclass
class XISelector:
    model: object
    gate: Dict[str, bool]
    trained_through: str
    meta: Dict = field(default_factory=dict)

    def allowed(self, fmt: str, gender: str) -> bool:
        return bool(self.gate.get(f"{fmt}_{gender}", False))

    def predict(self, history: History, team_id: int, gender: str, fmt: str,
                d: pd.Timestamp) -> Tuple[List[int], Dict]:
        cands, info = history.candidates(team_id, gender, fmt, d)
        if len(cands) < 11:
            return [], info
        probs = self.model.predict_proba(cands[FEATURES].astype(float))[:, 1]
        xi = pick_xi(cands, probs)
        info["expected_recall"] = float(np.sort(probs)[::-1][:11].sum() / 11)
        return xi, info

    def predict_with_squad(self, history: History, team_id: int, gender: str, fmt: str,
                           d: pd.Timestamp, squad: List[int]) -> Tuple[List[int], Dict]:
        """Top 11 of the announced squad, ranked by the same model.

        Squad players with no recent XI for this team are scored from their
        other appearances. `expected_recall` is the unrestricted prediction's,
        since the probabilities are not recalibrated for squad membership.
        Returns [] when fewer than 11 squad players can be scored.
        """
        cands, info = history.candidates(team_id, gender, fmt, d, extra=squad)
        if len(cands) < 11:
            return [], info
        probs = self.model.predict_proba(cands[FEATURES].astype(float))[:, 1]
        info["expected_recall"] = float(np.sort(probs)[::-1][:11].sum() / 11)
        keep = cands.player_id.isin(set(squad)).to_numpy()
        info["squad_scored"] = int(keep.sum())
        if keep.sum() < 11:
            return [], info
        return pick_xi(cands[keep], probs[keep]), info


_CACHE: Dict[str, Tuple[float, object]] = {}


def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return -1.0


def load_selector(artifact_dir: Path = ARTIFACT_DIR) -> Optional[XISelector]:
    path = artifact_dir / "selector.pkl"
    mt = _mtime(path)
    if mt < 0:
        return None
    hit = _CACHE.get(str(path))
    if hit and hit[0] == mt:
        return hit[1]
    with open(path, "rb") as fh:
        sel = pickle.load(fh)
    _CACHE[str(path)] = (mt, sel)
    return sel


def load_runtime_history(db_path, decl_path: Path = DECLARED_XI_PATH) -> Optional[History]:
    """History of everything ingested so far (all of it is known at scan time)."""
    key = f"hist:{db_path}:{decl_path}"
    mt = _mtime(decl_path)
    if mt < 0:
        return None
    hit = _CACHE.get(key)
    if hit and hit[0] == mt:
        return hit[1]
    since = (pd.Timestamp.utcnow().tz_localize(None) - pd.Timedelta(days=4 * 365)).strftime("%Y-%m-%d")
    hist = History(load_sides(db_path, pd.read_pickle(decl_path), since=since))
    _CACHE[key] = (mt, hist)
    return hist


def squad_gate(artifact_dir: Path = ARTIFACT_DIR) -> Dict[str, bool]:
    """Segments where the squad-aware selector beat the plain one out of sample
    (written by scripts/eval_xi_squads.py); empty until that test passes."""
    path = artifact_dir / "squad_gate.json"
    mt = _mtime(path)
    if mt < 0:
        return {}
    hit = _CACHE.get(str(path))
    if hit and hit[0] == mt:
        return hit[1]
    import json
    try:
        gate = {k: bool(v) for k, v in json.loads(path.read_text()).get("gate", {}).items()}
    except (OSError, ValueError):
        gate = {}
    _CACHE[str(path)] = (mt, gate)
    return gate


def pretoss_mode() -> str:
    """PAPER_PRETOSS_XI: 'auto' (default; use the selector where its gate passed) or 'last_match'."""
    return os.environ.get("PAPER_PRETOSS_XI", "auto").strip().lower()
