"""Match-level XGBoost win-probability models.

Rows are mirrored (team1<->team2) for training and predictions are averaged
over both orientations, so the arbitrary team order cannot bias the output.
Paired features enter as antisymmetric differences.
"""
from __future__ import annotations

import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

PAIRED = [
    ("tiered_elo", "tiered_elo_pre_1", "tiered_elo_pre_2"),
    ("simple_elo", "simple_elo_1", "simple_elo_2"),
    ("fast_elo", "fast_elo_1", "fast_elo_2"),
    ("elo_mom90", "elo_mom90_1", "elo_mom90_2"),
    ("form_ewm", "form_ewm_1", "form_ewm_2"),
    ("form5", "form5_1", "form5_2"),
    ("n_prior", "n_prior_1", "n_prior_2"),
    ("rest_days", "rest_days_1", "rest_days_2"),
    ("games_30d", "games_30d_1", "games_30d_2"),
    ("country_share", "country_share_1", "country_share_2"),
    ("is_home", "is_home_1", "is_home_2"),
    ("venue_games", "venue_games_1", "venue_games_2"),
    ("tier", "tier_1", "tier_2"),
    ("xi_bat_elo", "xi_bat_elo_1", "xi_bat_elo_2"),
    ("xi_bowl_elo", "xi_bowl_elo_1", "xi_bowl_elo_2"),
    ("xi_runs_form", "xi_runs_form_1", "xi_runs_form_2"),
    ("xi_sr_form", "xi_sr_form_1", "xi_sr_form_2"),
    ("xi_bwk_form", "xi_bwk_form_1", "xi_bwk_form_2"),
    ("xi_econ_form", "xi_econ_form_1", "xi_econ_form_2"),
    ("xi_exp", "xi_exp_1", "xi_exp_2"),
    ("xi_debut", "xi_debut_1", "xi_debut_2"),
    ("xi_overlap", "xi_overlap_1", "xi_overlap_2"),
]

_TEAM = ["tiered_elo", "simple_elo", "fast_elo", "elo_mom90", "form_ewm", "form5",
         "n_prior", "rest_days", "games_30d", "tier", "h2h"]
_VENUE = ["country_share", "is_home", "venue_games"]
_XI = ["xi_bat_elo", "xi_bowl_elo", "xi_runs_form", "xi_sr_form", "xi_bwk_form",
       "xi_econ_form", "xi_exp", "xi_debut", "xi_overlap"]
FEATURE_SETS = {
    "team_venue": _TEAM + _VENUE,
    "full": _TEAM + _VENUE + _XI,
}
XGB_PARAMS = {"half_life": 1460, "max_depth": 3, "n_estimators": 300, "lr": 0.03, "min_child_weight": 20}
MODEL_FEATURES = {"team_venue_xgb": "team_venue", "full_xgb": "full"}


def design(df: pd.DataFrame, groups: List[str], flip: bool) -> pd.DataFrame:
    out = {}
    s = -1.0 if flip else 1.0
    for name, a, b in PAIRED:
        if name not in groups:
            continue
        scale = 200.0 if "elo" in name and "mom" not in name and "xi" not in name else 1.0
        if name.startswith("xi_") and "elo" in name:
            scale = 100.0
        out[f"d_{name}"] = s * (df[a] - df[b]) / scale
    if "h2h" in groups:
        out["d_h2h"] = s * (df["h2h_rate_1"] - 0.5) * np.minimum(df["h2h_n"], 10) / 10
    if "is_home" in groups:
        for c in ("venue_n", "venue_bat1_rate", "venue_par_rel"):
            out[c] = df[c]
    for c in ("is_t20", "is_female", "intl", "franchise"):
        out[c] = df[c]
    out["elo_x_t20"] = out.get("d_tiered_elo", 0) * df["is_t20"]
    out["elo_x_female"] = out.get("d_tiered_elo", 0) * df["is_female"]
    return pd.DataFrame(out, index=df.index)


def recency_weight(dates: pd.Series, ref: pd.Timestamp, half_life_days: float) -> np.ndarray:
    age = (ref - dates).dt.days.clip(lower=0).values
    return 0.5 ** (age / half_life_days)


def trainable(df: pd.DataFrame) -> pd.DataFrame:
    d = df[df.y.notna() & df.tiered_elo_pre_1.notna() & df.tiered_elo_pre_2.notna()].copy()
    d["y"] = d.y.astype(int)
    return d


# Predeclared time split. These dates are not moved after seeing which leagues pass.
HOLDOUT_START = "2026-04-26"
CALIBRATION_DAYS = 90
DECISION_HOURS_BEFORE = 1.0
GATE_RULE = {
    "min_holdout": 20,
    "min_history": 100,
    "min_conf": 0.90,
    "holdout_start": HOLDOUT_START,
    "calibration_days": CALIBRATION_DAYS,
    "decision_hours_before": DECISION_HOURS_BEFORE,
    "n_bootstrap": 1000,
    "bootstrap_seed": 0,
}
# Scored with the same rule as every other prefix. A pass is not installed anywhere.
ALWAYS_SCORED_PREFIXES = ("cricnpl", "cricwbbl")
# Comment in the old constant. Not a prefix the scanner uses, and not WBBL.
_UNUSED_PREFIX = "crict20wbbl"


def time_split_bounds(
    holdout_start: str = HOLDOUT_START,
    calibration_days: int = CALIBRATION_DAYS,
) -> tuple:
    """Return (calibration_start, holdout_start) as naive timestamps."""
    holdout = pd.Timestamp(holdout_start)
    calibration = holdout - pd.Timedelta(days=int(calibration_days))
    return calibration, holdout


def split_trainable(
    df: pd.DataFrame,
    holdout_start,
    calibration_start,
) -> tuple:
    """Cut trainable rows into fit, calibration, and gate blocks.

    Fit is date < calibration_start. Calibration is
    calibration_start <= date < holdout_start. Gate is date >= holdout_start.
    A date sits entirely on one side of a cut.
    """
    d = trainable(df)
    d = d[d.date.notna()].copy()
    d["date"] = pd.to_datetime(d["date"])
    cal_start = pd.Timestamp(calibration_start)
    holdout = pd.Timestamp(holdout_start)
    fit_df = d[d.date < cal_start].copy()
    cal_df = d[(d.date >= cal_start) & (d.date < holdout)].copy()
    gate_df = d[d.date >= holdout].copy()
    return fit_df, cal_df, gate_df


def fit_block_ref(fit_df: pd.DataFrame) -> pd.Timestamp:
    """Recency zero-point: the day after the last fit-block date."""
    return pd.Timestamp(fit_df.date.max()) + pd.Timedelta(days=1)


def time_split_output_allowed(out: Path, artifact_dir: Optional[Path] = None) -> bool:
    """False when --out is the live artifact directory."""
    live = ARTIFACT_DIR if artifact_dir is None else artifact_dir
    return Path(out).resolve() != Path(live).resolve()


def has_both_declared_xi(df: pd.DataFrame) -> pd.Series:
    """True where both sides have a non-empty declared XI. No predicted XI is filled."""
    if "xi_n_1" not in df.columns or "xi_n_2" not in df.columns:
        return pd.Series(False, index=df.index)
    n1 = pd.to_numeric(df["xi_n_1"], errors="coerce")
    n2 = pd.to_numeric(df["xi_n_2"], errors="coerce")
    return (n1.fillna(0) > 0) & (n2.fillna(0) > 0)


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2))


def _log_loss(p: np.ndarray, y: np.ndarray) -> float:
    pp = np.clip(np.asarray(p, dtype=float), 1e-6, 1.0 - 1e-6)
    yy = np.asarray(y, dtype=float)
    return float(-np.mean(yy * np.log(pp) + (1.0 - yy) * np.log(1.0 - pp)))


def _bootstrap_share(model_p: np.ndarray, price_p: np.ndarray, y: np.ndarray, n: int, seed: int) -> float:
    """Share of resamples where model Brier is strictly below the price Brier."""
    model_p = np.asarray(model_p, dtype=float)
    price_p = np.asarray(price_p, dtype=float)
    y = np.asarray(y, dtype=float)
    n_obs = int(len(y))
    if n_obs == 0 or n <= 0:
        return 0.0
    rng = np.random.default_rng(int(seed))
    wins = 0
    for _ in range(int(n)):
        idx = rng.integers(0, n_obs, size=n_obs)
        if _brier(model_p[idx], y[idx]) < _brier(price_p[idx], y[idx]):
            wins += 1
    return wins / float(n)


def score_league_gate(
    observations,
    *,
    models=None,
    always_prefixes=ALWAYS_SCORED_PREFIXES,
    rule: Optional[dict] = None,
) -> dict:
    """League gate from a time split. Same pass rule for every prefix.

    Each observation is a dict with model, prefix, and kind ('history' or
    'holdout'). Holdout rows also carry y, model_prob, and price. A holdout
    row with a missing price is dropped. `crict20wbbl` is ignored. `crictnpl`
    is not treated as `cricnpl`.
    """
    rule = dict(GATE_RULE if rule is None else rule)
    model_names = list(models or ("team_venue_xgb", "full_xgb"))
    prefixes_for = {name: set(always_prefixes or ()) for name in model_names}
    history = {name: {} for name in model_names}
    holdout = {name: {} for name in model_names}

    for obs in observations or []:
        model = obs.get("model")
        prefix = obs.get("prefix")
        if model not in history or not prefix or prefix == _UNUSED_PREFIX:
            continue
        prefixes_for[model].add(prefix)
        kind = obs.get("kind")
        if kind == "history":
            history[model][prefix] = history[model].get(prefix, 0) + 1
            continue
        if kind != "holdout":
            continue
        price = obs.get("price")
        prob = obs.get("model_prob")
        y = obs.get("y")
        if price is None or prob is None or y is None:
            continue
        try:
            price_f = float(price)
            prob_f = float(prob)
            y_f = float(y)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(price_f) or not np.isfinite(prob_f) or y_f not in (0.0, 1.0):
            continue
        holdout[model].setdefault(prefix, []).append((prob_f, price_f, y_f))

    allowed = {}
    table = {}
    n_boot = int(rule.get("n_bootstrap", 1000))
    seed = int(rule.get("bootstrap_seed", 0))
    min_holdout = int(rule["min_holdout"])
    min_history = int(rule["min_history"])
    min_conf = float(rule["min_conf"])
    for model in model_names:
        allowed[model] = {}
        table[model] = {}
        for prefix in sorted(prefixes_for[model]):
            rows = holdout[model].get(prefix, [])
            n_holdout = len(rows)
            n_history = int(history[model].get(prefix, 0))
            if rows:
                mp = np.array([r[0] for r in rows])
                pp = np.array([r[1] for r in rows])
                yy = np.array([r[2] for r in rows])
                model_brier = _brier(mp, yy)
                price_brier = _brier(pp, yy)
                share = _bootstrap_share(mp, pp, yy, n_boot, seed)
                ll_model = _log_loss(mp, yy)
                ll_price = _log_loss(pp, yy)
            else:
                model_brier = price_brier = ll_model = ll_price = None
                share = 0.0
            passed = bool(
                n_holdout >= min_holdout
                and n_history >= min_history
                and share >= min_conf
            )
            allowed[model][prefix] = passed
            table[model][prefix] = {
                "n_holdout": n_holdout,
                "n_history": n_history,
                "model_brier": None if model_brier is None else round(model_brier, 6),
                "price_brier": None if price_brier is None else round(price_brier, 6),
                "bootstrap_share": round(share, 6),
                "log_loss_model": None if ll_model is None else round(ll_model, 6),
                "log_loss_price": None if ll_price is None else round(ll_price, 6),
                "allowed": passed,
            }
    return {"rule": rule, "allowed": allowed, "prefixes": table}


def fit(train: pd.DataFrame, features: str, ref: pd.Timestamp, params: Optional[dict] = None):
    import xgboost as xgb

    params = {**XGB_PARAMS, **(params or {})}
    groups = FEATURE_SETS[features]
    X = pd.concat([design(train, groups, False), design(train, groups, True)])
    y = np.concatenate([train.y.values, 1 - train.y.values])
    w = recency_weight(train.date, ref, params["half_life"])
    w = np.concatenate([w, w])
    clf = xgb.XGBClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        learning_rate=params["lr"], subsample=params.get("subsample", 0.8),
        colsample_bytree=params.get("colsample", 0.8),
        min_child_weight=params["min_child_weight"], reg_lambda=params.get("reg_lambda", 5.0),
        tree_method="hist", n_jobs=4, verbosity=0,
    )
    clf.fit(X, y, sample_weight=w)
    return clf


def predict(clf, df: pd.DataFrame, features: str) -> np.ndarray:
    groups = FEATURE_SETS[features]
    pa = clf.predict_proba(design(df, groups, False))[:, 1]
    pb = clf.predict_proba(design(df, groups, True))[:, 1]
    return 0.5 * (pa + (1 - pb))


@dataclass
class MatchLevelPredictor:
    """Loaded artifacts: fitted models, feature state, and the per-league gate."""

    models: Dict[str, object]
    builder: object
    gate: Dict[str, Dict[str, bool]] = field(default_factory=dict)
    trained_through: Optional[str] = None

    def allowed(self, model_name: str, prefix: Optional[str]) -> bool:
        return bool(prefix) and bool(self.gate.get(model_name, {}).get(prefix, False))

    def predict_fixture(self, model_name: str, *, team1_id: int, team2_id: int, fmt: str,
                        gender: str, venue_id: Optional[int], date,
                        xi1: Optional[List[int]] = None, xi2: Optional[List[int]] = None) -> Optional[float]:
        """P(team1 wins), or None when the features needed are missing."""
        features = MODEL_FEATURES[model_name]
        if features == "full" and (not xi1 or not xi2):
            return None
        row = self.builder.features_for_upcoming(team1_id, team2_id, fmt, gender, venue_id,
                                                 pd.Timestamp(date), xi1, xi2)
        if np.isnan(row["tiered_elo_pre_1"]) or np.isnan(row["tiered_elo_pre_2"]):
            return None
        return float(predict(self.models[model_name], pd.DataFrame([row]), features)[0])


ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "data" / "models" / "match_level"

_CACHE: Dict[str, tuple] = {}


def load_predictor(artifact_dir: Path = ARTIFACT_DIR) -> Optional[MatchLevelPredictor]:
    """Load (and cache by mtime) the artifacts written by refresh_match_level_model.py."""
    path = Path(artifact_dir) / "predictor.pkl"
    if not path.exists():
        return None
    mtime = path.stat().st_mtime
    hit = _CACHE.get(str(path))
    if hit and hit[0] == mtime:
        return hit[1]
    with open(path, "rb") as fh:
        pred = pickle.load(fh)
    gate_path = Path(artifact_dir) / "league_gate.json"
    if gate_path.exists():
        g = json.loads(gate_path.read_text())
        pred.gate = g.get("allowed", pred.gate)
    _CACHE[str(path)] = (mtime, pred)
    return pred
