"""Polymarket cricket market probability extractor (Wave 5 Phase 1).

Takes a `simulate_matches()` result dict from
`src.models.vectorized_nn_sim_v2.V2Simulator` and emits per-market
probability dicts that mirror the Polymarket cricket market structure.

Polymarket lists ~6 cricket market types per fixture; only 4 carry
meaningful cricket-skill signal:

- **Moneyline** (2-way): natural V2 simulator output.
- **Team Top Batter** (3-way: team1 / draw / team2): which side has the
  match's top run-scorer (or a tie). Computed from the per-batter run
  distribution returned by the V2 simulator since Wave 5 Phase 1.
- **Most Sixes** (3-way: team1 / draw / team2): which side hits more
  sixes (or a tie). Computed from per-team six counters.
- **Toss Match Double** (4-way): cross product of toss winner and match
  winner. Toss is genuinely 50/50 and independent of cricket model.

Output schema (deliberately stable so the Polymarket compare service
and the Bulk Predict UI can rely on it):

    {
        "moneyline": {"team1": float, "team2": float},
        "top_batter": {
            "team1_higher": float,
            "draw": float,
            "team2_higher": float,
            "team1_top_batter_distribution": {player_id: prob, ...},
            "team2_top_batter_distribution": {player_id: prob, ...},
        },
        "most_sixes": {
            "team1": float,
            "draw": float,
            "team2": float,
            "team1_avg_sixes": float,
            "team2_avg_sixes": float,
        },
        "toss_match_double": {
            "toss_team1_match_team1": float,
            "toss_team1_match_team2": float,
            "toss_team2_match_team1": float,
            "toss_team2_match_team2": float,
        },
    }

All probabilities sum to 1.0 within their market (within float epsilon).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _max_per_match(arr: np.ndarray) -> np.ndarray:
    """Per-match top run scorer count (handles n_batters=0 safely)."""
    if arr.size == 0:
        return np.zeros((arr.shape[0],), dtype=arr.dtype)
    return arr.max(axis=1)


def _argmax_per_match(arr: np.ndarray) -> np.ndarray:
    """Index of per-match top run scorer column."""
    if arr.shape[1] == 0:
        return np.zeros((arr.shape[0],), dtype=np.int64)
    return arr.argmax(axis=1)


def _top_batter_distribution(
    batter_runs: np.ndarray,
    batter_ids: List[int],
) -> Dict[int, float]:
    """Per-batter probability of being THIS team's top scorer.

    Tie-breaking: if multiple batters tie for top, the probability is
    split evenly across them (so the marginal probabilities sum to 1.0
    across the team's batting list).
    """
    if batter_runs.size == 0 or len(batter_ids) == 0:
        return {}
    n_matches, n_batters = batter_runs.shape
    if n_batters == 0:
        return {}
    max_runs = batter_runs.max(axis=1, keepdims=True)
    top_mask = (batter_runs == max_runs) & (max_runs > 0)
    top_count_per_match = top_mask.sum(axis=1)
    # Avoid div-by-zero on the (rare) all-zero matches; in those cases
    # we skip the match (no batter scored anything; usually a 0-run
    # all-out which is essentially impossible in T20 / ODI).
    contribution = np.zeros_like(batter_runs, dtype=np.float64)
    nonzero = top_count_per_match > 0
    if nonzero.any():
        weights = 1.0 / top_count_per_match[nonzero].astype(np.float64)
        contribution[nonzero] = top_mask[nonzero] * weights[:, None]
    per_batter_prob = contribution.sum(axis=0) / max(1, n_matches)
    return {
        int(batter_ids[i]): float(per_batter_prob[i])
        for i in range(min(n_batters, len(batter_ids)))
    }


def _three_way_compare(team1_arr: np.ndarray, team2_arr: np.ndarray) -> Dict[str, float]:
    """Compare two per-match arrays and return team1_higher / draw / team2_higher probs."""
    if team1_arr.size == 0 or team2_arr.size == 0:
        return {"team1_higher": 0.0, "draw": 1.0, "team2_higher": 0.0}
    n_matches = min(team1_arr.shape[0], team2_arr.shape[0])
    t1 = team1_arr[:n_matches]
    t2 = team2_arr[:n_matches]
    team1_wins = float((t1 > t2).mean())
    team2_wins = float((t2 > t1).mean())
    draw = float((t1 == t2).mean())
    return {
        "team1_higher": team1_wins,
        "draw": draw,
        "team2_higher": team2_wins,
    }


def derive_polymarket_market_probs(
    sim_result: Dict[str, Any],
    toss_field_prob: float = 0.5,
) -> Dict[str, Any]:
    """Convert a V2 simulator result into per-market Polymarket probabilities.

    Args:
        sim_result: Output dict from `V2Simulator.simulate_matches(...)`.
            Must contain `team1_win_prob`, `team1_player_runs`,
            `team2_player_runs`, `team1_sixes`, `team2_sixes`,
            `team1_batter_ids`, `team2_batter_ids`.
        toss_field_prob: Probability that the toss winner chooses to field.
            Used only for downstream toss-aware logic; for the
            `toss_match_double` market itself we treat toss winner as
            50/50 since the team that wins the toss is genuinely random.

    Returns:
        Dict with 4 sub-dicts: moneyline, top_batter, most_sixes,
        toss_match_double. Each market's outcome probabilities sum to 1.0
        within float epsilon.
    """
    moneyline_t1 = float(sim_result.get("team1_win_prob", 0.5))
    moneyline_t2 = 1.0 - moneyline_t1

    t1_player_runs: np.ndarray = sim_result.get(
        "team1_player_runs", np.zeros((0, 0), dtype=np.int32)
    )
    t2_player_runs: np.ndarray = sim_result.get(
        "team2_player_runs", np.zeros((0, 0), dtype=np.int32)
    )
    t1_sixes: np.ndarray = sim_result.get("team1_sixes", np.zeros((0,), dtype=np.int32))
    t2_sixes: np.ndarray = sim_result.get("team2_sixes", np.zeros((0,), dtype=np.int32))
    t1_batter_ids: List[int] = list(sim_result.get("team1_batter_ids", []))
    t2_batter_ids: List[int] = list(sim_result.get("team2_batter_ids", []))

    # ---------- Top Batter ----------
    t1_top = _max_per_match(t1_player_runs)
    t2_top = _max_per_match(t2_player_runs)
    top_batter = _three_way_compare(t1_top, t2_top)
    top_batter["team1_top_batter_distribution"] = _top_batter_distribution(
        t1_player_runs, t1_batter_ids
    )
    top_batter["team2_top_batter_distribution"] = _top_batter_distribution(
        t2_player_runs, t2_batter_ids
    )

    # ---------- Most Sixes ----------
    # Use simpler team1/draw/team2 keys (rather than team1_higher etc) since
    # the Polymarket "Most Sixes" market literally lists team names as the
    # outcomes; "team1_higher" reads awkwardly in that context.
    sixes_compare = _three_way_compare(t1_sixes, t2_sixes)
    most_sixes = {
        "team1": sixes_compare["team1_higher"],
        "draw": sixes_compare["draw"],
        "team2": sixes_compare["team2_higher"],
        "team1_avg_sixes": float(t1_sixes.mean()) if t1_sixes.size else 0.0,
        "team2_avg_sixes": float(t2_sixes.mean()) if t2_sixes.size else 0.0,
    }

    # ---------- Toss Match Double ----------
    # Toss winner is independent of match winner (genuinely 50/50).
    # The toss_field_prob arg is reserved for future "if win toss, choose
    # to field/bat affects match prob" logic; for the cross-product market
    # the cleaner default is independence.
    toss_match_double = {
        "toss_team1_match_team1": 0.5 * moneyline_t1,
        "toss_team1_match_team2": 0.5 * moneyline_t2,
        "toss_team2_match_team1": 0.5 * moneyline_t1,
        "toss_team2_match_team2": 0.5 * moneyline_t2,
    }

    return {
        "moneyline": {"team1": moneyline_t1, "team2": moneyline_t2},
        "top_batter": top_batter,
        "most_sixes": most_sixes,
        "toss_match_double": toss_match_double,
    }


def _normalize_three_way(probs: Dict[str, float]) -> Dict[str, float]:
    """Normalise a three-way market dict in-place; safe for floating drift."""
    keys = [k for k in ("team1_higher", "draw", "team2_higher") if k in probs]
    s = sum(probs[k] for k in keys)
    if s <= 0:
        return probs
    for k in keys:
        probs[k] = probs[k] / s
    return probs


def market_summary_for_ui(market_probs: Dict[str, Any]) -> Dict[str, Any]:
    """Compact dict suitable for embedding in JSON API responses.

    Drops the per-batter distributions (potentially many keys) and just
    keeps the headline 3-way probability. Bulk Predict UI uses this
    shape; the full per-batter distribution stays available via the
    Single Predict endpoint.
    """
    return {
        "moneyline": dict(market_probs.get("moneyline", {})),
        "top_batter": {
            k: market_probs.get("top_batter", {}).get(k, 0.0)
            for k in ("team1_higher", "draw", "team2_higher")
        },
        "most_sixes": {
            k: market_probs.get("most_sixes", {}).get(k, 0.0)
            for k in ("team1", "draw", "team2")
        },
        "toss_match_double": dict(market_probs.get("toss_match_double", {})),
    }
