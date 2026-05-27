#!/usr/bin/env python3
"""Probe today's Polymarket fixtures with V2 + V3.

Pulls each team's most recent XI from player_match_stats, runs N=300 sims
per match for both simulators, prints model_prob vs market_price vs edge.

Designed to be a quick "what does the model say about today's games?" check.
Run as the matches are about to start; output is informational only.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.database import get_connection, get_db_connection


def get_recent_xi(conn, team_id: int, fmt: str, gender: str, n_recent_matches: int = 3):
    """Return (batters, bowlers) for the team's last N matches.

    batters: 11 player_ids most-frequently selected, in batting position.
    bowlers: 5 player_ids by overs bowled.
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT pms.match_id, pms.player_id, pms.batting_position,
               pms.balls_faced, pms.overs_bowled
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.team_id = ? AND m.match_type = ? AND m.gender = ?
          AND m.winner_id IS NOT NULL
        ORDER BY m.date DESC
        LIMIT 200
        """,
        (team_id, fmt, gender),
    )
    rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return [], []
    # Pick most-recent match (by match_id appearing first in the ordered query)
    last_matches = []
    seen_matches = set()
    for r in rows:
        if r["match_id"] not in seen_matches:
            seen_matches.add(r["match_id"])
            last_matches.append(r["match_id"])
        if len(last_matches) >= n_recent_matches:
            break

    cur.execute(
        f"""
        SELECT pms.match_id, pms.player_id, pms.batting_position,
               pms.balls_faced, pms.overs_bowled, m.date
        FROM player_match_stats pms
        JOIN matches m ON m.match_id = pms.match_id
        WHERE pms.team_id = ? AND pms.match_id IN ({",".join("?" * len(last_matches))})
        """,
        [team_id] + last_matches,
    )
    rows = [dict(r) for r in cur.fetchall()]
    if not rows:
        return [], []

    # Frequency-weighted XI: count appearances + sort by avg balls/overs
    from collections import defaultdict
    appearances = defaultdict(int)
    avg_pos = defaultdict(list)
    avg_balls = defaultdict(list)
    avg_overs = defaultdict(list)
    for r in rows:
        pid = r["player_id"]
        appearances[pid] += 1
        if r["batting_position"] is not None:
            avg_pos[pid].append(r["batting_position"])
        avg_balls[pid].append(r["balls_faced"] or 0)
        avg_overs[pid].append(r["overs_bowled"] or 0)

    # Most recent match's XI as a starting point
    cur.execute(
        """
        SELECT pms.player_id, pms.batting_position, pms.balls_faced, pms.overs_bowled
        FROM player_match_stats pms
        WHERE pms.team_id = ? AND pms.match_id = ?
        """,
        (team_id, last_matches[0]),
    )
    most_recent = [dict(r) for r in cur.fetchall()]
    most_recent.sort(key=lambda r: (
        r["batting_position"] if r["batting_position"] is not None else 99,
        -(r["balls_faced"] or 0),
    ))
    batters = [r["player_id"] for r in most_recent][:11]
    if len(batters) < 11:
        # Pad with most-appeared players from the other 2 matches
        seen = set(batters)
        extra = sorted(appearances.keys(), key=lambda p: -appearances[p])
        for p in extra:
            if p not in seen:
                batters.append(p)
                seen.add(p)
            if len(batters) >= 11:
                break
    most_recent.sort(key=lambda r: -(r["overs_bowled"] or 0))
    bowlers = [r["player_id"] for r in most_recent if (r["overs_bowled"] or 0) > 0][:5]
    if len(bowlers) < 5:
        seen = set(bowlers)
        for r in most_recent:
            if r["player_id"] not in seen:
                bowlers.append(r["player_id"])
                seen.add(r["player_id"])
            if len(bowlers) >= 5:
                break
    return batters[:11], bowlers[:5]


def get_venue_for_team(conn, team_id: int, fmt: str, gender: str):
    """Pick the team's most-played venue as a stand-in if Polymarket
    doesn't tell us the venue. Just used to feed venue features."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT m.venue_id, COUNT(*) AS n
        FROM matches m
        WHERE (m.team1_id = ? OR m.team2_id = ?)
          AND m.match_type = ? AND m.gender = ?
          AND m.venue_id IS NOT NULL
        GROUP BY m.venue_id
        ORDER BY n DESC
        LIMIT 1
        """,
        (team_id, team_id, fmt, gender),
    )
    row = cur.fetchone()
    return row["venue_id"] if row else None


def run_match(simulator_v2, simulator_v3, fixture):
    name = fixture["name"]
    fmt = fixture["fmt"]
    gen = fixture["gender"]
    t1 = fixture["t1_id"]
    t2 = fixture["t2_id"]
    market_t1 = fixture["market_t1_pct"]
    market_t2 = fixture["market_t2_pct"]
    n_sims = fixture.get("n_sims", 300)

    with get_db_connection() as conn:
        t1_bat, t1_bowl = get_recent_xi(conn, t1, fmt, gen)
        t2_bat, t2_bowl = get_recent_xi(conn, t2, fmt, gen)
        venue_id = get_venue_for_team(conn, t1, fmt, gen)

    if len(t1_bat) < 11 or len(t1_bowl) < 5 or len(t2_bat) < 11 or len(t2_bowl) < 5:
        print(f"  [SKIP] {name}: insufficient lineup data "
              f"(t1 bat={len(t1_bat)} bowl={len(t1_bowl)}, t2 bat={len(t2_bat)} bowl={len(t2_bowl)})")
        return None

    # V2 sim
    v2_result = simulator_v2.simulate_matches(
        n_sims, t1_bat, t1_bowl, t2_bat, t2_bowl,
        venue_id=venue_id, team1_id=t1, team2_id=t2,
        use_toss=True, toss_field_prob=0.65, seed=42,
    )
    v2_t1 = float(v2_result["team1_win_prob"]) * 100

    # V3 sim - both modes
    v3_marg = simulator_v3.simulate_matches(
        n_sims, t1_bat, t1_bowl, t2_bat, t2_bowl,
        venue_id=venue_id, team1_id=t1, team2_id=t2,
        use_toss=True, toss_field_prob=0.65, seed=42,
    )
    v3_marg_t1 = float(v3_marg["team1_win_prob"]) * 100

    # Compute edges
    edge_v2_t1 = v2_t1 - market_t1
    edge_v2_t2 = (100 - v2_t1) - market_t2
    edge_v3_t1 = v3_marg_t1 - market_t1
    edge_v3_t2 = (100 - v3_marg_t1) - market_t2

    print(f"\n  === {name} ({fmt} {gen}) ===")
    print(f"    Market: t1={market_t1:.1f}c   t2={market_t2:.1f}c   (sum={market_t1+market_t2:.1f}, spread={market_t1+market_t2-100:.1f}pp)")
    print(f"    V2:     t1={v2_t1:.1f}%       t2={100-v2_t1:.1f}%   edges: t1 {edge_v2_t1:+.1f}pp, t2 {edge_v2_t2:+.1f}pp")
    print(f"    V3-marg t1={v3_marg_t1:.1f}%       t2={100-v3_marg_t1:.1f}%   edges: t1 {edge_v3_t1:+.1f}pp, t2 {edge_v3_t2:+.1f}pp")
    print(f"    V2 expected score: t1={v2_result['avg_team1_score']:.1f}  t2={v2_result['avg_team2_score']:.1f}")
    print(f"    Recommendation:")
    best_edge = max(abs(edge_v2_t1), abs(edge_v2_t2), abs(edge_v3_t1), abs(edge_v3_t2))
    if best_edge < 5:
        print(f"      No meaningful edge (max |edge| = {best_edge:.1f}pp). Skip.")
    else:
        if abs(edge_v2_t1) == best_edge:
            print(f"      V2 says BACK t1 ({edge_v2_t1:+.1f}pp edge)")
        if abs(edge_v2_t2) == best_edge:
            print(f"      V2 says BACK t2 ({edge_v2_t2:+.1f}pp edge)")
        if abs(edge_v3_t1) == best_edge:
            print(f"      V3 says BACK t1 ({edge_v3_t1:+.1f}pp edge)")
        if abs(edge_v3_t2) == best_edge:
            print(f"      V3 says BACK t2 ({edge_v3_t2:+.1f}pp edge)")
    return {"v2_t1": v2_t1, "v3_t1": v3_marg_t1}


def main() -> int:
    # Fixtures from the screenshot. Skipping USA-Nepal (USA not in DB) and
    # Hyderabad Kingsmen vs Rawalpindiz (likely brand-new league, thin history).
    FIXTURES = [
        {
            "name": "Chennai Super Kings vs Gujarat Titans (IPL)",
            "fmt": "T20", "gender": "male",
            "t1_id": 112, "t2_id": 234,
            "market_t1_pct": 15.0, "market_t2_pct": 86.0,
        },
        {
            "name": "Lucknow Super Giants vs Kolkata Knight Riders (IPL)",
            "fmt": "T20", "gender": "male",
            "t1_id": 233, "t2_id": 114,
            "market_t1_pct": 52.0, "market_t2_pct": 49.0,
        },
        {
            "name": "Islamabad United vs Multan Sultans (PSL)",
            "fmt": "T20", "gender": "male",
            "t1_id": 88, "t2_id": 91,
            "market_t1_pct": 49.0, "market_t2_pct": 52.0,
        },
        {
            "name": "Rwanda vs Vanuatu (Intl)",
            "fmt": "T20", "gender": "male",
            "t1_id": 205, "t2_id": 120,
            "market_t1_pct": 99.9, "market_t2_pct": 0.1,
        },
        {
            "name": "Hyderabad Kingsmen vs Rawalpindiz (PSL spinoff?)",
            "fmt": "T20", "gender": "male",
            "t1_id": 372, "t2_id": 373,
            "market_t1_pct": 94.8, "market_t2_pct": 6.6,
        },
    ]

    print("Loading V2 + V3 simulators...")
    from src.models.vectorized_nn_sim_v2 import V2Simulator, V2SimulatorConfig
    from src.models.vectorized_nn_sim_v3 import V3Simulator, V3SimulatorConfig
    sim_v2 = V2Simulator(V2SimulatorConfig(format_type="T20", gender="male"))
    sim_v3 = V3Simulator(V3SimulatorConfig(format_type="T20", gender="male"))

    print(f"\nProbing {len(FIXTURES)} live Polymarket fixtures...")
    for fx in FIXTURES:
        try:
            run_match(sim_v2, sim_v3, fx)
        except Exception as exc:
            print(f"  [ERROR] {fx['name']}: {exc}")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
