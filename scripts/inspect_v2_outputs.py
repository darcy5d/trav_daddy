#!/usr/bin/env python3
"""
Wave 4.5 Phase 1 - Deep diagnostic for the v2 model's per-ball softmax.

Walks N recent IPL matches' actual XIs ball-by-ball, runs the v2 model on
each ball's actual state features, and aggregates four key analyses into
a Markdown report:

  1. Per-class predicted vs actual marginal rates by phase (PP/middle/death).
     Confirms the model has learned phase-conditional behaviour.

  2. Per-class predicted vs actual marginal rates by era bucket
     (era=-1 ~ 2016 vs era=0 ~ 2026). If the model's predicted boundary
     rate is the same across eras, the era feature is being ignored.

  3. Counterfactual: identical state, era flipped. For 100 randomly
     sampled balls, runs the model twice with era=-1 and era=0 (everything
     else identical) and reports the average shift in P(4) + P(6).
     Direct test of era sensitivity.

  4. Per-class actual marginals computed from the v2 training data, sliced
     by era bucket. Shows what the model SHOULD have learned. If actual
     2026 boundary rate is 16% but the model predicts 9%, that's the
     under-prediction quantified.

This report is the input to Phase 2 hypothesis selection.

Run:
    python scripts/inspect_v2_outputs.py
    python scripts/inspect_v2_outputs.py --n-matches 10 --since 2025-01-01
    python scripts/inspect_v2_outputs.py --output-dir data/diagnostics
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Helpers - per-ball state reconstruction from a real match
# ============================================================================


def _phase_from_over(over: int, fmt: str) -> str:
    """T20: PP 0-5, middle 6-14, death 15+. ODI: PP 0-9, middle 10-39, death 40+."""
    if fmt.upper() == "ODI":
        if over < 10: return "powerplay"
        if over < 40: return "middle"
        return "death"
    if over < 6: return "powerplay"
    if over < 15: return "middle"
    return "death"


def _era_bucket(year: int) -> str:
    """Coarse era bucket. Old = pre-2020 (par scores ~150-170), new = 2024+."""
    if year < 2020: return "old (<2020)"
    if year < 2024: return "mid (2020-23)"
    return "new (2024+)"


def _classify_actual(d: dict) -> int:
    """Replicate ball_training_data_v2.classify_outcome on a deliveries row.

    Class indices: 0=dot, 1=1, 2=2, 3=3, 4=4, 5=6, 6=wicket, 7=wide, 8=noball.
    Note class 5 is six runs and class 6 is wicket - easy to confuse.
    """
    is_wicket = bool(d["is_wicket"])
    wides = d["extras_wides"] or 0
    noballs = d["extras_noballs"] or 0
    if is_wicket: return 6  # LABEL_WICKET
    if wides > 0: return 7  # LABEL_WIDE
    if noballs > 0: return 8  # LABEL_NOBALL
    r = d["runs_batter"] or 0
    if r in (0, 1, 2, 3, 4): return r
    if r == 6: return 5     # LABEL_SIX
    if r == 5: return 4     # rare 5-runs fold into 4-class
    if r >= 7: return 5     # very rare 7+ runs fold into 6-class
    return 0


# ============================================================================
# Run model over a real match's actual ball-by-ball trajectory
# ============================================================================


def trace_match_balls(simulator, conn, match_id: int) -> List[Dict]:
    """For one historical match, replay each ball's actual state through the
    v2 model and return per-ball records.

    Each record has:
      match_id, innings, over, ball, phase, era_year, era_norm,
      actual_class, predicted_softmax (9 floats), state_vector (22 floats),
      batter_id, bowler_id, venue_id, batting_team_id, bowling_team_id.
    """
    import tensorflow as tf
    from src.utils.format_constants import balls_for_format
    from src.features.ball_training_data import normalize_elo
    from src.features.ball_training_data_v2 import (
        ERA_REFERENCE_YEAR, ERA_NORM_DIVISOR, FORMAT_ID,
    )
    from src.data.franchise_resolver import get_resolver

    cur = conn.cursor()

    # Match metadata
    cur.execute(
        """
        SELECT m.match_id, m.match_type, m.gender, m.date, m.venue_id,
               m.team1_id, m.team2_id
        FROM matches m WHERE m.match_id = ?
        """,
        (match_id,),
    )
    m = cur.fetchone()
    if not m:
        return []
    fmt = m["match_type"]
    gender = m["gender"]
    if fmt != simulator.format_type or gender != simulator.gender:
        logger.warning(f"match {match_id} is {fmt}/{gender}; skipping (sim is {simulator.format_type}/{simulator.gender})")
        return []
    date_str = str(m["date"])[:10]
    year = int(date_str[:4]) if date_str[:4].isdigit() else ERA_REFERENCE_YEAR
    era_norm = (year - ERA_REFERENCE_YEAR) / ERA_NORM_DIVISOR

    # Pull all deliveries in chronological order with team lookups
    cur.execute(
        """
        SELECT d.delivery_id, d.over_number, d.ball_number, d.batter_id,
               d.bowler_id, d.runs_batter, d.runs_extras, d.extras_wides,
               d.extras_noballs, d.is_wicket,
               i.innings_number, i.batting_team_id, i.bowling_team_id
        FROM deliveries d
        JOIN innings i ON d.innings_id = i.innings_id
        WHERE i.match_id = ?
        ORDER BY i.innings_number, d.over_number, d.ball_number
        """,
        (match_id,),
    )
    deliveries = [dict(r) for r in cur.fetchall()]
    if not deliveries:
        return []

    # First-innings total for required-rate computation
    first_innings_total = None
    for d in deliveries:
        if d["innings_number"] == 1:
            first_innings_total = (first_innings_total or 0)
            first_innings_total += (d["runs_batter"] or 0) + (d["runs_extras"] or 0)

    venue_features = (
        simulator.venue_stats.get_venue_features(m["venue_id"])
        if simulator.venue_stats and m["venue_id"]
        else np.array([1.0, 0.55, 0.05, 0.0], dtype=np.float32)
    )
    resolver = get_resolver()

    records: List[Dict] = []

    # Walk balls maintaining innings state
    cur_innings = None
    runs = wkts = legal_balls = 0
    for d in deliveries:
        innings = d["innings_number"]
        if innings != cur_innings:
            cur_innings = innings
            runs = wkts = legal_balls = 0
        bat_team = d["batting_team_id"]
        bowl_team = d["bowling_team_id"]
        bat_team_canonical = resolver.canonical(bat_team) or bat_team
        bowl_team_canonical = resolver.canonical(bowl_team) or bowl_team
        bat_team_elo = simulator._team_elo(bat_team)
        bowl_team_elo = simulator._team_elo(bowl_team)
        batter_elo = simulator._player_batting(d["batter_id"])
        bowler_elo = simulator._player_bowling(d["bowler_id"])

        target = first_innings_total + 1 if (innings == 2 and first_innings_total) else 0
        balls_remaining = balls_for_format(fmt) - legal_balls
        balls_remaining = max(1, balls_remaining)
        runs_needed = max(0, target - runs)
        required_rate = (runs_needed * 6.0 / balls_remaining) if (innings == 2 and target) else 0.0

        phase = _phase_from_over(d["over_number"], fmt)
        is_pp = 1.0 if phase == "powerplay" else 0.0
        is_mid = 1.0 if phase == "middle" else 0.0
        is_death = 1.0 if phase == "death" else 0.0

        # Build state vector matching CONTINUOUS_COLUMNS order
        state = np.array([
            float(FORMAT_ID.get(fmt, 0)),
            float(innings),
            float(d["over_number"]),
            float(d["ball_number"]),
            float(legal_balls),
            float(runs),
            float(wkts),
            float(target if innings == 2 else 0),
            float(required_rate),
            era_norm,
            is_pp, is_mid, is_death,
            float(venue_features[0]), float(venue_features[1]),
            float(venue_features[2]), float(venue_features[3]),
            float(normalize_elo(bat_team_elo)),
            float(normalize_elo(bowl_team_elo)),
            float((bat_team_elo - bowl_team_elo) / 200.0),
            float(normalize_elo(batter_elo)),
            float(normalize_elo(bowler_elo)),
        ], dtype=np.float32).reshape(1, -1)

        inputs = {
            "state": tf.constant(state),
            "batter_id": tf.constant(np.array([simulator._vocab_id("batter", d["batter_id"])], dtype=np.int32)),
            "bowler_id": tf.constant(np.array([simulator._vocab_id("bowler", d["bowler_id"])], dtype=np.int32)),
            "venue_id": tf.constant(np.array([simulator._vocab_id("venue", m["venue_id"])], dtype=np.int32)),
            "batting_team_id": tf.constant(np.array([simulator._vocab_id("team", bat_team_canonical)], dtype=np.int32)),
            "bowling_team_id": tf.constant(np.array([simulator._vocab_id("team", bowl_team_canonical)], dtype=np.int32)),
        }
        try:
            outs = simulator._predict_compiled(inputs, training=False)
        except Exception:
            outs = simulator._model(inputs, training=False)
        proba = outs[simulator.ball_head_name].numpy()[0]

        actual_class = _classify_actual(d)
        records.append({
            "match_id": match_id,
            "innings": innings,
            "over": d["over_number"],
            "ball": d["ball_number"],
            "phase": phase,
            "era_year": year,
            "era_norm": era_norm,
            "actual_class": actual_class,
            "predicted_softmax": proba,
            "state": state[0].copy(),
            "batter_id": d["batter_id"],
            "bowler_id": d["bowler_id"],
            "venue_id": m["venue_id"],
            "batting_team_id": bat_team,
            "bowling_team_id": bowl_team,
        })

        # Update innings state for next ball
        runs += (d["runs_batter"] or 0) + (d["runs_extras"] or 0)
        if d["is_wicket"]:
            wkts += 1
        if (d["extras_wides"] or 0) == 0 and (d["extras_noballs"] or 0) == 0:
            legal_balls += 1

    return records


# ============================================================================
# Aggregations and counterfactuals
# ============================================================================


LABELS = ["dot", "1", "2", "3", "4", "6", "wicket", "wide", "noball"]


def marginals_by(records: List[Dict], key: str) -> Dict[str, Dict]:
    """Per-class predicted vs actual marginal rates, grouped by some attribute."""
    grouped_pred = defaultdict(list)
    grouped_act = defaultdict(list)
    for r in records:
        bucket = r[key] if not callable(key) else key(r)
        grouped_pred[bucket].append(r["predicted_softmax"])
        grouped_act[bucket].append(r["actual_class"])
    out = {}
    for bucket in grouped_pred:
        pred = np.array(grouped_pred[bucket])
        act = np.array(grouped_act[bucket])
        n = len(pred)
        out[bucket] = {
            "n": n,
            "predicted_marginal": pred.mean(axis=0),
            "actual_marginal": np.bincount(act, minlength=9) / n,
        }
    return out


def counterfactual_era_flip(simulator, records: List[Dict], n_samples: int = 100, seed: int = 42) -> Dict:
    """For n_samples random balls, run the model with era=-1 and era=0 on
    otherwise-identical state. Report average shift in P(4) + P(6).
    """
    import tensorflow as tf
    rng = np.random.default_rng(seed)
    n = min(n_samples, len(records))
    indices = rng.choice(len(records), size=n, replace=False)

    shifts_p4 = []
    shifts_p6 = []
    shifts_pdot = []
    shifts_pwkt = []
    shifts_total_boundary = []

    for idx in indices:
        r = records[idx]
        state_old = r["state"].copy()
        state_new = r["state"].copy()
        state_old[9] = -1.0  # era_norm 2016
        state_new[9] = 0.0   # era_norm 2026
        cur_batter = simulator._vocab_id("batter", r["batter_id"])
        cur_bowler = simulator._vocab_id("bowler", r["bowler_id"])
        venue_v = simulator._vocab_id("venue", r["venue_id"])
        from src.data.franchise_resolver import get_resolver
        resolver = get_resolver()
        bat_t = simulator._vocab_id("team", resolver.canonical(r["batting_team_id"]) or r["batting_team_id"])
        bowl_t = simulator._vocab_id("team", resolver.canonical(r["bowling_team_id"]) or r["bowling_team_id"])

        def _pred(state):
            inputs = {
                "state": tf.constant(state.reshape(1, -1)),
                "batter_id": tf.constant(np.array([cur_batter], dtype=np.int32)),
                "bowler_id": tf.constant(np.array([cur_bowler], dtype=np.int32)),
                "venue_id": tf.constant(np.array([venue_v], dtype=np.int32)),
                "batting_team_id": tf.constant(np.array([bat_t], dtype=np.int32)),
                "bowling_team_id": tf.constant(np.array([bowl_t], dtype=np.int32)),
            }
            try:
                outs = simulator._predict_compiled(inputs, training=False)
            except Exception:
                outs = simulator._model(inputs, training=False)
            return outs[simulator.ball_head_name].numpy()[0]

        p_old = _pred(state_old)
        p_new = _pred(state_new)
        shifts_p4.append(p_new[4] - p_old[4])
        shifts_p6.append(p_new[5] - p_old[5])
        shifts_pdot.append(p_new[0] - p_old[0])
        shifts_pwkt.append(p_new[6] - p_old[6])
        shifts_total_boundary.append((p_new[4] + p_new[5]) - (p_old[4] + p_old[5]))

    return {
        "n_samples": n,
        "mean_p4_shift": float(np.mean(shifts_p4)),
        "mean_p6_shift": float(np.mean(shifts_p6)),
        "mean_pdot_shift": float(np.mean(shifts_pdot)),
        "mean_pwkt_shift": float(np.mean(shifts_pwkt)),
        "mean_total_boundary_shift": float(np.mean(shifts_total_boundary)),
    }


def actual_marginals_from_training(npz_path: Path, gender: str) -> Dict:
    """Compute per-class actual marginal from training data, sliced by era bucket
    AND by format. This is what the model SHOULD have learned.
    """
    if not npz_path.exists():
        return {"error": f"training data not found at {npz_path}"}
    data = np.load(npz_path, allow_pickle=True)
    y = data["y"]
    X = data["X"]
    era = X[:, 9]  # era_year_norm column index in CONTINUOUS_COLUMNS
    fmt_id = X[:, 0]
    sw = data["sample_weight"]

    # Era buckets in training-data terms:
    #   era < -0.5: old (~2014-2020)
    #   -0.5 <= era < -0.1: mid (~2021-2024)
    #   era >= -0.1: new (~2025-2026)
    era_buckets = np.where(era < -0.5, "old (<2020)", np.where(era < -0.1, "mid (2020-23)", "new (2024+)"))

    out: Dict[str, Dict] = {}
    for fmt_id_val, fmt_name in [(0, "T20"), (1, "ODI")]:
        for bucket_name in ["old (<2020)", "mid (2020-23)", "new (2024+)"]:
            mask = (fmt_id == fmt_id_val) & (era_buckets == bucket_name)
            if mask.sum() == 0:
                continue
            sub_y = y[mask]
            sub_sw = sw[mask]
            # Sample-weighted marginal (matches what the model effectively trains on)
            marginal = np.zeros(9, dtype=np.float64)
            for c in range(9):
                marginal[c] = (sub_sw[sub_y == c].sum())
            marginal = marginal / max(marginal.sum(), 1e-9)
            # Unweighted marginal
            unw = np.bincount(sub_y, minlength=9) / max(len(sub_y), 1)
            key = f"{fmt_name} / {gender} / {bucket_name}"
            out[key] = {
                "n": int(mask.sum()),
                "weighted_actual_marginal": marginal,
                "unweighted_actual_marginal": unw,
            }
    return out


# ============================================================================
# Markdown report rendering
# ============================================================================


def _fmt_marginal_table(rows: List[Tuple[str, np.ndarray, np.ndarray, int]]) -> str:
    """rows: list of (bucket_label, predicted_marginal, actual_marginal, n)."""
    lines = []
    header = "| bucket | n | " + " | ".join(f"{lab} pred" for lab in LABELS) + " |"
    lines.append(header)
    lines.append("| " + " | ".join(["---"] * (len(LABELS) + 2)) + " |")
    for label, pred, _, n in rows:
        cells = [label, f"{n:,}"] + [f"{p*100:.1f}%" for p in pred]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    header = "| bucket | n | " + " | ".join(f"{lab} actual" for lab in LABELS) + " |"
    lines.append(header)
    lines.append("| " + " | ".join(["---"] * (len(LABELS) + 2)) + " |")
    for label, _, act, n in rows:
        cells = [label, f"{n:,}"] + [f"{p*100:.1f}%" for p in act]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_report(
    sim_label: str,
    by_phase: Dict[str, Dict],
    by_era: Dict[str, Dict],
    counterfactual: Dict,
    training_marginals: Dict,
    n_matches: int,
    n_balls: int,
) -> str:
    md = []
    md.append(f"# V2 Diagnostic Report — {sim_label}")
    md.append("")
    md.append(f"- Matches inspected: **{n_matches}**")
    md.append(f"- Balls inspected: **{n_balls:,}**")
    md.append("")

    md.append("## 1. Per-class predicted vs actual marginal rates by phase")
    md.append("")
    md.append("If predicted boundary rate is much lower than actual in any phase, the model is systematically conservative. Death-overs gap is the most diagnostic.")
    md.append("")
    rows = []
    for phase in ["powerplay", "middle", "death"]:
        if phase in by_phase:
            d = by_phase[phase]
            rows.append((phase, d["predicted_marginal"], d["actual_marginal"], d["n"]))
    md.append(_fmt_marginal_table(rows))
    md.append("")

    md.append("## 2. Per-class predicted vs actual marginal rates by era bucket")
    md.append("")
    md.append("If the model's predicted boundary rate is the same across eras, the era feature isn't being leveraged. Compare 'new (2024+)' actuals vs predictions to see if the model recognises modern par scores.")
    md.append("")
    rows = []
    for bucket in ["old (<2020)", "mid (2020-23)", "new (2024+)"]:
        if bucket in by_era:
            d = by_era[bucket]
            rows.append((bucket, d["predicted_marginal"], d["actual_marginal"], d["n"]))
    md.append(_fmt_marginal_table(rows))
    md.append("")

    md.append("## 3. Counterfactual: identical state, era flipped")
    md.append("")
    md.append(f"For {counterfactual['n_samples']} randomly sampled balls, ran the model twice with era_norm=-1.0 (~2016) vs era_norm=0.0 (~2026), all other features identical. Reports the AVERAGE shift in per-class probability when ONLY the era column changes.")
    md.append("")
    md.append("| metric | value |")
    md.append("| --- | --- |")
    md.append(f"| mean P(4) shift   (new - old) | **{counterfactual['mean_p4_shift']*100:+.3f} pp** |")
    md.append(f"| mean P(6) shift   (new - old) | **{counterfactual['mean_p6_shift']*100:+.3f} pp** |")
    md.append(f"| mean P(boundary) shift (new - old) | **{counterfactual['mean_total_boundary_shift']*100:+.3f} pp** |")
    md.append(f"| mean P(dot) shift (new - old)   | **{counterfactual['mean_pdot_shift']*100:+.3f} pp** |")
    md.append(f"| mean P(wicket) shift (new - old) | **{counterfactual['mean_pwkt_shift']*100:+.3f} pp** |")
    md.append("")
    md.append("**Interpretation guide**: a healthy era-aware model should show meaningful (>0.5pp) positive shift on P(4)+P(6) and negative shift on P(dot) when flipping to the new era. If the shift is < 0.2pp, the era feature is essentially ignored.")
    md.append("")

    md.append("## 4. Actual training-data marginals by era bucket (what the model SHOULD have learned)")
    md.append("")
    md.append("Sample-weighted (recency-decayed) marginals are what the model effectively saw during training. Unweighted marginals are the raw frequencies. Compare to the predicted marginals in section 2 to spot under-prediction.")
    md.append("")
    md.append("| split | n | " + " | ".join(LABELS) + " |")
    md.append("| --- | --- | " + " | ".join(["---"] * len(LABELS)) + " |")
    for key in sorted(training_marginals.keys()):
        d = training_marginals[key]
        if "error" in d:
            continue
        n = d["n"]
        marginal = d["weighted_actual_marginal"]
        cells = [f"{key} (weighted)", f"{n:,}"] + [f"{p*100:.1f}%" for p in marginal]
        md.append("| " + " | ".join(cells) + " |")
    md.append("")
    for key in sorted(training_marginals.keys()):
        d = training_marginals[key]
        if "error" in d:
            continue
        n = d["n"]
        marginal = d["unweighted_actual_marginal"]
        cells = [f"{key} (unweighted)", f"{n:,}"] + [f"{p*100:.1f}%" for p in marginal]
        md.append("| " + " | ".join(cells) + " |")
    md.append("")

    md.append("## Hypothesis prompts")
    md.append("")
    md.append("Based on the above, the Phase 2 step should pick which knob(s) to tune in Phase 3. Quick decision rules:")
    md.append("")
    md.append("- **If counterfactual era shift is < 0.5pp**: era feature is broken. Tightening recency half-life is the highest-priority knob - it forces more gradient signal toward modern data without depending on the era column being learned.")
    md.append("- **If actual 'new (2024+)' boundary rate is much higher than predicted**: confirms score under-prediction is concentrated in modern matches. Either the model has not learned the era shift OR the per-over auxiliary loss is dragging boundary probabilities down.")
    md.append("- **If by-phase predicted boundary rate is uniform but actual is U-shaped (high in PP and death)**: the per-over auxiliary loss is over-smoothing across the over. Drop weight_over_loss to 0.1.")
    md.append("- **If all eras show a similar predicted-vs-actual gap on classes 4 and 6**: targeted boundary upweighting (class weights 4=1.5, 6=1.5) is the right knob.")
    return "\n".join(md)


# ============================================================================
# Main
# ============================================================================


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-matches", type=int, default=5)
    parser.add_argument("--since", default="2025-01-01")
    parser.add_argument("--tournament-pattern", default="%Indian Premier League%")
    parser.add_argument("--format", default="T20", choices=["T20", "ODI"])
    parser.add_argument("--gender", default="male", choices=["male", "female"])
    parser.add_argument("--counterfactual-samples", type=int, default=200)
    parser.add_argument("--output-dir", default="data/diagnostics")
    parser.add_argument("--label", default=None)
    args = parser.parse_args()

    from src.models.vectorized_nn_sim_v2 import V2Simulator, V2SimulatorConfig
    from src.data.database import get_connection, get_db_connection
    from src.models.backtest import load_holdout_matches

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or f"{args.format.lower()}_{args.gender}_n{args.n_matches}"

    # Build simulator (loads model + vocabs + venue + calibration)
    sim = V2Simulator(V2SimulatorConfig(format_type=args.format, gender=args.gender))
    sim._ensure_model_loaded()

    # Pick recent matches
    with get_db_connection() as conn:
        matches = load_holdout_matches(
            conn,
            formats=(args.format,),
            genders=(args.gender,),
            since_date=args.since,
            tournament_pattern=args.tournament_pattern,
            limit=args.n_matches,
        )
    if not matches:
        logger.error("No matches found")
        return 1
    logger.info(f"Inspecting {len(matches)} matches; sample: {[m.match_id for m in matches[:5]]}")

    # Walk balls + run model
    all_records: List[Dict] = []
    with get_db_connection() as conn:
        for m in matches:
            recs = trace_match_balls(sim, conn, m.match_id)
            all_records.extend(recs)
            logger.info(f"  match {m.match_id}: {len(recs)} balls")

    if not all_records:
        logger.error("No ball records produced")
        return 1
    logger.info(f"Total balls inspected: {len(all_records):,}")

    # Aggregate
    by_phase = marginals_by(all_records, "phase")
    # Era bucket from year (not era_norm) for grouping
    for r in all_records:
        r["era_bucket"] = _era_bucket(r["era_year"])
    by_era = marginals_by(all_records, "era_bucket")

    # Counterfactual era flip
    cf = counterfactual_era_flip(sim, all_records, n_samples=args.counterfactual_samples)

    # Training-data marginals (both genders if both files exist)
    training_marginals: Dict[str, Dict] = {}
    for g in ["male", "female"]:
        path = Path("data/processed") / f"ball_training_v2_{g}.npz"
        training_marginals.update(actual_marginals_from_training(path, g))

    # Render
    sim_label = f"V2 ({args.format}/{args.gender}) - {len(all_records):,} balls across {len(matches)} matches"
    md = render_report(
        sim_label=sim_label,
        by_phase=by_phase,
        by_era=by_era,
        counterfactual=cf,
        training_marginals=training_marginals,
        n_matches=len(matches),
        n_balls=len(all_records),
    )

    out_path = out_dir / f"v2_inspect_{label}.md"
    out_path.write_text(md)
    logger.info(f"Report written to {out_path}")

    # Also print to stdout
    print()
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
