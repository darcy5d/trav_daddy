"""
Format-aware constants for cricket match simulation and feature engineering.

Centralises every place we used to hardcode T20-isms (20 overs, 120 balls,
phase boundaries at 6/15) so that ODI matches actually run as 50-over games
with their own (10/40) phase split rather than silently being simulated as
T20s. Phase boundaries match the official ICC powerplay rules (T20:
mandatory PP1 in overs 0-6, "middle" 6-15, "death" 15-20; ODI: PP1 0-10,
PP2 10-40, PP3 40-50).

Used by:
  - src/models/vectorized_nn_sim.py  (simulator over count + phase encoding)
  - src/features/ball_training_data.py  (training-time required-rate calc
    + phase one-hot)
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


# Default overs per innings keyed by canonical match_type. ODIs default to
# 50; T20s to 20. Anything else falls back to 20 with a warning at the
# call site - we don't have a model for Tests yet.
OVERS_BY_FORMAT = {
    "T20": 20,
    "ODI": 50,
}


def overs_for_format(format_type: str) -> int:
    """Return the canonical max overs per innings for a format string."""
    return OVERS_BY_FORMAT.get((format_type or "T20").upper(), 20)


def balls_for_format(format_type: str) -> int:
    """Return the canonical max LEGAL balls per innings.

    Note: this is the cap the simulator uses for termination of a clean
    (no-extras) innings. v1 simulators don't model wides/noballs (those land
    in Wave 4 / model v2), so legal balls == total balls today.
    """
    return overs_for_format(format_type) * 6


def phase_for_over(format_type: str, over: int) -> Tuple[int, int, int]:
    """Return one-hot (powerplay, middle, death) for the given over.

    T20: PP 0-5, middle 6-14, death 15-19.
    ODI: PP1 0-9, middle (PP2) 10-39, death (PP3) 40-49.
    Falls back to T20 boundaries for unknown formats.
    """
    fmt = (format_type or "T20").upper()
    if fmt == "ODI":
        if over < 10:
            return (1, 0, 0)
        if over < 40:
            return (0, 1, 0)
        return (0, 0, 1)
    # T20 / default
    if over < 6:
        return (1, 0, 0)
    if over < 15:
        return (0, 1, 0)
    return (0, 0, 1)


# Pre-computed numpy arrays for the simulator hot loop (avoid per-ball
# tuple-to-array conversion). Indexed by (format, phase) so the simulator
# can pick the right one O(1).
def phase_arrays(format_type: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """Return (PP_array, MIDDLE_array, DEATH_array, mid_over_threshold, death_over_threshold)
    for the simulator's hot-loop phase selection.

    Returns 5-tuple so the simulator can do `if over < mid_t: pp elif over < death_t: mid else: death`
    without allocating a tuple per ball.
    """
    pp = np.array([1, 0, 0], dtype=np.float32)
    mid = np.array([0, 1, 0], dtype=np.float32)
    death = np.array([0, 0, 1], dtype=np.float32)
    fmt = (format_type or "T20").upper()
    if fmt == "ODI":
        return pp, mid, death, 10, 40
    return pp, mid, death, 6, 15
