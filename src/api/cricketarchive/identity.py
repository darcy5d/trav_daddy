"""Player-identity resolution for CricketArchive ball-by-ball.

Commentary uses bare surnames ("Tector to Tilak Varma"). When two players in a
match share a surname (e.g. TH Tector and HT Tector) a surname is ambiguous.
The scorecard cards, however, carry the unique CA id + initials for every
player. This module reconciles the commentary against the scorecard:

* Batters: a strike-rotation state machine walks the commentary seeded by the
  batting order, so each delivery's striker maps to a specific batting-card row
  (hence a specific ca_id + initials). Non-striker becomes exact.
* Bowlers: each over has one bowler; ambiguous surnames are distributed to
  same-surname candidates by their bowling-card over quota.
* Verification: resolved per-batter balls-faced and per-bowler wickets are
  checked against the scorecard. UNAMBIGUOUS players are always trusted;
  AMBIGUOUS players are only trusted when their checksum matches - otherwise the
  id is set to NULL with resolution_status='ambiguous_unverified' (we never
  store a same-surname guess we could not verify).

Pure functions over the dataclasses in ``models`` - no I/O, no DB.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .models import CAInnings, CADelivery


def _surname(name: Optional[str]) -> Optional[str]:
    return name.split()[-1] if name else name


def _batters_crossed(d: CADelivery) -> bool:
    """Did the batters physically cross (swap ends) on this delivery?

    Crossing happens on an ODD number of runs actually RUN between the wickets -
    boundaries are not run. Runs run = batter runs (unless a boundary) + byes +
    leg-byes + runs run off a wide (wide total minus its 1-run penalty). Runs off
    a no-ball that come from the bat are already in runs_batter.
    """
    if d.is_boundary_four or d.is_boundary_six:
        ran = 0
    else:
        ran = d.runs_batter
    ran += d.extras_byes + d.extras_legbyes
    if d.extras_wides:
        ran += max(0, d.extras_wides - 1)
    return ran % 2 == 1


def _over_to_balls(overs: Optional[str]) -> Optional[int]:
    """'4' -> 24, '3.4' -> 22 (legal balls)."""
    if not overs:
        return None
    overs = overs.strip()
    if "." in overs:
        full, _, part = overs.partition(".")
        if full.isdigit() and part.isdigit():
            return int(full) * 6 + int(part)
        return None
    return int(overs) * 6 if overs.isdigit() else None


def resolve_innings(innings: CAInnings,
                    deliveries: List[CADelivery]) -> Tuple[List[CADelivery], dict]:
    """Resolve batter/non-striker/bowler identities on `deliveries` in place.

    Returns (deliveries, report). report carries per-player checksums and an
    overall `identity_verified` bool.
    """
    batted = [b for b in innings.batting
              if (b.dismissal or "").strip() != "did not bat" and b.ca_id]
    bowlers = [bw for bw in innings.bowling if bw.ca_id]

    # surname -> candidate rows (per role)
    bat_by_surname: Dict[str, list] = defaultdict(list)
    for b in batted:
        bat_by_surname[_surname(b.name)].append(b)
    bowl_by_surname: Dict[str, list] = defaultdict(list)
    for bw in bowlers:
        bowl_by_surname[_surname(bw.name)].append(bw)

    ambiguous_bat = {s for s, lst in bat_by_surname.items() if len(lst) > 1}
    ambiguous_bowl = {s for s, lst in bowl_by_surname.items() if len(lst) > 1}

    # ---- batter strike-rotation state machine ----
    roster = batted  # crease-entry order
    striker = 0
    non_striker = 1 if len(roster) > 1 else None
    next_idx = 2
    current_over: Optional[int] = None
    faced: Dict[str, int] = defaultdict(int)

    def entry(idx: Optional[int]):
        return roster[idx] if (idx is not None and 0 <= idx < len(roster)) else None

    for d in deliveries:
        if current_over is None:
            current_over = d.over_number
        elif d.over_number != current_over:           # end of over -> swap
            striker, non_striker = non_striker, striker
            current_over = d.over_number

        s, ns = entry(striker), entry(non_striker)
        sur = _surname(d.batter)

        # choose striker: trust tracked striker if surname matches, else fall
        # back to a unique candidate, else unresolved
        chosen = None
        if s is not None and _surname(s.name) == sur:
            chosen = s
        else:
            cands = bat_by_surname.get(sur, [])
            if len(cands) == 1:
                chosen = cands[0]
        if chosen is not None:
            d.batter_ca_id = chosen.ca_id
            d.batter_initials = chosen.name
            if not d.extras_wides:
                faced[chosen.ca_id] += 1
        else:
            d.resolution_status = "unresolved"

        if ns is not None:
            d.non_striker = _surname(ns.name)
            d.non_striker_ca_id = ns.ca_id
            d.non_striker_initials = ns.name

        # --- rotation updates AFTER assigning this ball ---
        if _batters_crossed(d):                        # odd runs run -> cross
            striker, non_striker = non_striker, striker

        if d.is_wicket:                                # bring in next batter
            out_sur = _surname(d.dismissed_player)
            cur_s, cur_ns = entry(striker), entry(non_striker)
            if (cur_ns is not None and _surname(cur_ns.name) == out_sur
                    and not (cur_s is not None and _surname(cur_s.name) == out_sur)):
                non_striker = next_idx
            else:
                striker = next_idx
            next_idx += 1

    # ---- bowler assignment (per over) ----
    overs_map: Dict[int, list] = defaultdict(list)
    for d in deliveries:
        overs_map[d.over_number].append(d)

    bowl_quota = {bw.ca_id: (_over_to_balls(bw.overs) or 0) for bw in bowlers}
    bowl_assigned: Dict[str, int] = defaultdict(int)
    prev_bowler_id: Optional[str] = None
    for over_no in sorted(overs_map):
        over_dels = overs_map[over_no]
        sur = _surname(over_dels[0].bowler)
        cands = bowl_by_surname.get(sur, [])
        chosen = None
        if len(cands) == 1:
            chosen = cands[0]
        elif len(cands) > 1:
            # distribute: candidate with most remaining quota, not last over's bowler
            ranked = sorted(
                cands,
                key=lambda bw: (bw.ca_id == prev_bowler_id,
                                -(bowl_quota.get(bw.ca_id, 0) - bowl_assigned[bw.ca_id])),
            )
            chosen = ranked[0]
        for d in over_dels:
            if chosen is not None:
                d.bowler_ca_id = chosen.ca_id
                d.bowler_initials = chosen.name
                if not (d.extras_wides or d.extras_noballs):
                    bowl_assigned[chosen.ca_id] += 1
        prev_bowler_id = chosen.ca_id if chosen is not None else prev_bowler_id

    # ---- dismissed-player + verification ----
    expected_balls = {b.ca_id: b.balls for b in batted if b.balls is not None}
    verified_bat = set()
    for b in batted:
        if _surname(b.name) not in ambiguous_bat:
            verified_bat.add(b.ca_id)
        elif b.ca_id in expected_balls and faced.get(b.ca_id, 0) == expected_balls[b.ca_id]:
            verified_bat.add(b.ca_id)

    # bowler verification by wickets (card wkts exclude run outs)
    resolved_bowl_wkts: Dict[str, int] = defaultdict(int)
    for d in deliveries:
        if d.is_wicket and d.bowler_ca_id and (d.wicket_type or "") != "run out":
            resolved_bowl_wkts[d.bowler_ca_id] += 1
    verified_bowl = set()
    for bw in bowlers:
        if _surname(bw.name) not in ambiguous_bowl:
            verified_bowl.add(bw.ca_id)
        elif resolved_bowl_wkts.get(bw.ca_id, 0) == (bw.wickets or 0):
            verified_bowl.add(bw.ca_id)

    # finalize per-delivery status + null unverified ambiguous guesses
    bat_caid_to_initials = {b.ca_id: b.name for b in batted}
    for d in deliveries:
        # batter
        if d.batter_ca_id:
            if _surname(d.batter_initials) not in ambiguous_bat:
                d.resolution_status = "unambiguous"
            elif d.batter_ca_id in verified_bat:
                d.resolution_status = "verified"
            else:
                d.batter_ca_id = None
                d.batter_initials = None
                d.resolution_status = "ambiguous_unverified"
        # non-striker (mirror verification)
        if d.non_striker_ca_id and d.non_striker_ca_id not in verified_bat:
            d.non_striker_ca_id = None
            d.non_striker_initials = None
        # bowler
        if d.bowler_ca_id and d.bowler_ca_id not in verified_bowl:
            d.bowler_ca_id = None
            d.bowler_initials = None
        # dismissed player -> resolve via batting roster surname
        if d.is_wicket and d.dismissed_player:
            cands = bat_by_surname.get(_surname(d.dismissed_player), [])
            if len(cands) == 1:
                d.dismissed_player_ca_id = cands[0].ca_id
                d.dismissed_player_initials = cands[0].name

    batter_ok = all(faced.get(b.ca_id, 0) == expected_balls.get(b.ca_id)
                    for b in batted if b.ca_id in expected_balls)
    bowler_ok = all(resolved_bowl_wkts.get(bw.ca_id, 0) == (bw.wickets or 0) for bw in bowlers)

    report = {
        "identity_verified": bool(batter_ok and bowler_ok),
        "batter_checksum_ok": bool(batter_ok),
        "bowler_checksum_ok": bool(bowler_ok),
        "ambiguous_batters": sorted(ambiguous_bat),
        "ambiguous_bowlers": sorted(ambiguous_bowl),
        "batter_balls": {bat_caid_to_initials.get(cid, cid): {"resolved": faced.get(cid, 0),
                                                              "expected": expected_balls.get(cid)}
                         for cid in expected_balls},
        "unresolved_deliveries": sum(1 for d in deliveries if d.resolution_status == "unresolved"),
    }
    return deliveries, report
