"""Synthetic ball-by-ball generation from a scorecard.

The premise (Thread B): for matches where CA has only a scorecard (no ball-by-ball),
can we generate a plausible delivery sequence consistent with the scorecard
aggregates? This is an INVERSE problem - the scorecard pins the marginals
(each batter's runs/balls/4s/6s, each bowler's runs/wkts/overs, fall of wickets)
exactly; only the intra-innings SEQUENCING is unknown.

IMPORTANT: synthetic data cannot add information beyond the scorecard. By
construction it reproduces player/innings marginals exactly; the open question
(answered by the validation harness) is how faithfully it reproduces intra-innings
DYNAMICS (powerplay/death scoring, wicket timing) that the scorecard does not
encode. So this is useful for marginal/feature priors and regularisation, not for
fabricating phase signal.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class SynthDelivery:
    over_number: int
    ball_number: int
    batter_ca_id: Optional[str]
    bowler_ca_id: Optional[str]
    runs_batter: int
    is_wicket: bool = False


def _nonboundary_outcomes(rem_balls: int, rem_runs: int) -> List[int]:
    """Multiset of {0,1,2,3} of length rem_balls summing to rem_runs, with a
    realistic dot/single/two/three shape (rather than all-singles)."""
    if rem_balls <= 0:
        return []
    rem_runs = max(0, min(rem_runs, 3 * rem_balls))
    # shape prior calibrated to real CA paired data (within non-boundary balls:
    # ~47% singles, ~8.5% twos, ~0.5% threes, rest dots); singles then absorb the
    # exact-sum constraint below.
    threes = min(int(round(0.005 * rem_balls)), rem_balls)
    twos = min(int(round(0.085 * rem_balls)), rem_balls - threes)
    ones = rem_runs - 3 * threes - 2 * twos
    # repair if ones out of range
    while ones < 0 and twos > 0:
        twos -= 1; ones = rem_runs - 3 * threes - 2 * twos
    while ones < 0 and threes > 0:
        threes -= 1; ones = rem_runs - 3 * threes - 2 * twos
    ones = max(0, ones)
    scoring = ones + twos + threes
    if scoring > rem_balls:                       # too many scoring balls: collapse threes/twos
        overflow = scoring - rem_balls
        take = min(overflow, twos); twos -= take; ones += 2 * take; overflow -= take
        take = min(overflow, threes); threes -= take; ones += 3 * take
        ones = max(0, ones - max(0, (ones + twos + threes) - rem_balls))
    dots = max(0, rem_balls - (ones + twos + threes))
    vals = [0] * dots + [1] * ones + [2] * twos + [3] * threes
    # final exact-sum repair (rounding): nudge values without changing length
    cur = sum(vals)
    i = 0
    while cur < rem_runs and i < len(vals):
        if vals[i] < 3:
            add = min(3 - vals[i], rem_runs - cur); vals[i] += add; cur += add
        i += 1
    i = 0
    while cur > rem_runs and i < len(vals):
        if vals[i] > 0:
            sub = min(vals[i], cur - rem_runs); vals[i] -= sub; cur -= sub
        i += 1
    return vals[:rem_balls]


def _batter_bag(runs: int, balls: int, fours: int, sixes: int, rng: random.Random) -> List[int]:
    """A multiset of per-ball run outcomes summing to `runs` over `balls`."""
    fours = max(0, min(fours, balls))
    sixes = max(0, min(sixes, balls - fours))
    bag = [4] * fours + [6] * sixes
    rem_balls = balls - fours - sixes
    rem_runs = runs - 4 * fours - 6 * sixes
    bag += _nonboundary_outcomes(rem_balls, rem_runs)
    rng.shuffle(bag)
    return bag


def _parse_fow_overs(fow: str) -> List[float]:
    """Fall-of-wickets text -> sorted list of over-floats (e.g. 1.5 -> 1.833)."""
    overs = []
    for m in re.finditer(r"(\d+)\.(\d+)\s*ov", fow or ""):
        overs.append(int(m.group(1)) + int(m.group(2)) / 6.0)
    return sorted(overs)


def _over_assignment(bowlers: List[Tuple[str, int]], n_overs: int) -> List[Optional[str]]:
    """Assign a bowler to each over (no consecutive overs) matching over quotas."""
    quota = {bid: max(1, round(balls / 6)) for bid, balls in bowlers if bid}
    seq: List[Optional[str]] = []
    prev = None
    for _ in range(n_overs):
        cands = sorted((b for b in quota if quota[b] > 0 and b != prev),
                       key=lambda b: -quota[b])
        if not cands:
            cands = [b for b in quota if quota[b] > 0] or list(quota) or [None]
        pick = cands[0]
        seq.append(pick)
        if pick in quota:
            quota[pick] -= 1
        prev = pick
    return seq


def generate_innings(batting: List[dict], bowling: List[dict], fow: str,
                     seed: int = 0) -> List[SynthDelivery]:
    """Generate a synthetic legal-delivery sequence for one innings.

    batting: ordered dicts with ca_id, runs, balls, fours, sixes (position order).
    bowling: dicts with ca_id, balls (legal balls bowled).
    """
    rng = random.Random(seed)
    batters = [b for b in batting if (b.get("balls") or 0) > 0]
    if not batters:
        return []
    bags = {i: _batter_bag(b.get("runs", 0) or 0, b.get("balls", 0) or 0,
                           b.get("fours", 0) or 0, b.get("sixes", 0) or 0, rng)
            for i, b in enumerate(batters)}
    dismissed = {i: bool(b.get("dismissed", True)) for i, b in enumerate(batters)}
    total_balls = sum(len(v) for v in bags.values())
    n_overs = max(1, (total_balls + 5) // 6)
    over_seq = _over_assignment([(bw.get("ca_id"), bw.get("balls", 0) or 0) for bw in bowling], n_overs)

    deliveries: List[SynthDelivery] = []
    striker, non_striker, next_idx = 0, (1 if len(batters) > 1 else None), 2
    legal = 0

    def has_balls(idx):
        return idx is not None and idx < len(batters) and bags.get(idx)

    def pick_with_balls(exclude):
        for i in range(len(batters)):
            if i != exclude and bags.get(i):
                return i
        return None

    while legal < total_balls:
        if not has_balls(striker):                # current striker faced all their balls
            striker = pick_with_balls(non_striker)
            if striker is None:
                break
        over = legal // 6
        ball_in_over = legal % 6 + 1
        bowler = over_seq[over] if over < len(over_seq) else (over_seq[-1] if over_seq else None)
        runs = bags[striker].pop()
        d = SynthDelivery(over, ball_in_over, batters[striker].get("ca_id"), bowler, runs)

        # a dismissed batter's wicket falls on their LAST faced ball (no discarding,
        # so synthetic innings length == sum of scorecard balls faced)
        if not bags[striker] and dismissed[striker]:
            d.is_wicket = True
            deliveries.append(d); legal += 1
            nxt = next_idx if has_balls(next_idx) else pick_with_balls(non_striker)
            if nxt is not None:
                striker = nxt if nxt != next_idx else next_idx
                if nxt == next_idx:
                    next_idx += 1
            continue

        deliveries.append(d); legal += 1
        if runs % 2 == 1:
            striker, non_striker = non_striker, striker
        if legal % 6 == 0:                         # end of over: change strike
            striker, non_striker = non_striker, striker

    return deliveries
