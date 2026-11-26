"""Strike rotation and chase target. The repair is on by default.

These tests do not construct V2Simulator or V3Simulator, so they do not load
Keras weights, and they do not run a paper or live scan.
"""
from pathlib import Path

import numpy as np

from src.features.ball_training_data_v2 import LABEL_DOT, LABEL_SINGLE, LABEL_THREE, LABEL_WICKET, LABEL_WIDE
from src.models.vectorized_nn_sim_v2 import (
    V2SimulatorConfig,
    chase_feature_columns,
    feature_target_value,
    strike_after_ball,
    strike_after_over,
)
from src.models.vectorized_nn_sim_v3 import V3SimulatorConfig

ROOT = Path(__file__).resolve().parents[1]
SCAN_FILES = (
    "config.py",
    "scripts/paper_bet_scan.py",
    "scripts/paper_bet_post_toss_scan.py",
    "scripts/live_bet_scan.py",
    "scripts/live_bet_post_toss_scan.py",
)


def _ends(rotate, outcome, is_wicket, wkts, over_complete=False):
    striker, non_striker, next_in = strike_after_ball(
        np.array([0]), np.array([1]), np.array([2]),
        np.array([outcome]), np.array([is_wicket]), np.array([wkts]),
        10, rotate,
    )
    striker, non_striker = strike_after_over(
        striker, non_striker, np.array([outcome]), np.array([over_complete]), rotate,
    )
    return int(striker[0]), int(non_striker[0]), int(next_in[0])


def test_repair_defaults_on_and_scans_use_that_default():
    v2 = V2SimulatorConfig(format_type="T20", gender="male")
    v3 = V3SimulatorConfig(format_type="T20", gender="male")
    assert v2.rotate_strike is True and v2.own_chase_target is True
    assert v3.rotate_strike is True and v3.own_chase_target is True
    for rel in SCAN_FILES:
        text = (ROOT / rel).read_text()
        assert "rotate_strike" not in text
        assert "own_chase_target" not in text


def test_off_path_keeps_the_opener_and_skips_the_non_striker():
    assert _ends(False, LABEL_SINGLE, False, 0) == (0, 1, 2)
    assert _ends(False, LABEL_WIDE, False, 0) == (0, 1, 2)
    # After the first wicket, wickets is already 1, so the striker becomes 2.
    assert _ends(False, LABEL_WICKET, True, 1) == (2, 1, 2)


def test_on_path_rotates_on_a_single_and_brings_in_the_next_batter():
    assert _ends(True, LABEL_SINGLE, False, 0) == (1, 0, 2)
    assert _ends(True, LABEL_THREE, False, 0) == (1, 0, 2)
    assert _ends(True, LABEL_WIDE, False, 0) == (0, 1, 2)
    assert _ends(True, LABEL_DOT, False, 0, over_complete=True) == (1, 0, 2)
    # A single on the last ball swaps once, and the end of the over does not swap again.
    assert _ends(True, LABEL_SINGLE, False, 0, over_complete=True) == (1, 0, 2)
    # Wicket: the non-striker stays, and the new batter is next_in (2), not wickets+1.
    assert _ends(True, LABEL_WICKET, True, 1) == (2, 1, 3)


def test_chase_target_is_shared_until_the_flag_is_on():
    targets = np.array([100, 180], dtype=np.int32)
    assert feature_target_value(targets, 2, False) == 180
    assert list(feature_target_value(targets, 2, True)) == [100, 180]
    assert feature_target_value(targets, 1, False) == 0

    shared_col, shared_rate = chase_feature_columns(180, 2, [0, 0], [0, 0], 120)
    own_col, own_rate = chase_feature_columns(targets, 2, [0, 0], [0, 0], 120)
    assert shared_col[0] == shared_col[1] == 180
    assert shared_rate[0] == shared_rate[1] == 180 * 6 / 120
    assert list(own_col) == [100, 180]
    assert own_rate[0] == 100 * 6 / 120
    assert own_rate[1] == 180 * 6 / 120
