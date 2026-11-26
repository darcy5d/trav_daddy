"""Sample-size wait: n80, the rate window, and what the strategies table shows."""

from datetime import datetime, timedelta, timezone

from src.integrations.polymarket.paper_sample_size import (
    FALLBACK_RATE_START,
    RECENT_RATE_START,
    sample_size_wait,
)

NOW = datetime(2026, 11, 20, 4, 0, tzinfo=timezone.utc)


def _row(pnl, size, settled_at):
    return {
        "pnl_realised_usdc": pnl,
        "size_usdc": size,
        "settled_at": settled_at.isoformat() if isinstance(settled_at, datetime) else settled_at,
    }


def _spread(n, start):
    """n settled bets whose simple returns are 0 and 0.10, so variance is defined."""
    rows = []
    for i in range(n):
        pnl = 0.0 if i % 2 == 0 else 1.0
        rows.append(_row(pnl, 10.0, start + timedelta(hours=i)))
    return rows


def test_one_sided_p_is_the_student_t_tail_against_alpha():
    # Two returns, 0 and 0.10. mean 0.05, se 0.05, t = 1 on 1 df.
    # Cauchy: P(T > 1) = 0.5 - atan(1)/pi = 0.25.
    rows = [
        _row(0.0, 10.0, RECENT_RATE_START),
        _row(1.0, 10.0, RECENT_RATE_START + timedelta(days=1)),
    ]
    out = sample_size_wait(rows, enabled=True, n_bets=2, now=NOW)
    assert out["df"] == 1
    assert abs(out["t"] - 1.0) < 1e-9
    assert abs(out["p_value"] - 0.25) < 1e-8
    assert out["alpha"] == 0.05
    assert out["p_label"] == "0.250 · α 0.05"
    assert "α = 0.05" in out["p_title"]


def test_tiny_p_is_shown_as_less_than_one_thousandth():
    # Eight nearly identical wins. The t-statistic is huge and the p-value is tiny.
    start = RECENT_RATE_START
    rows = [_row(1.0 + i * 0.01, 10.0, start + timedelta(days=i)) for i in range(8)]
    out = sample_size_wait(rows, enabled=True, n_bets=8, now=NOW)
    assert out["p_value"] < 0.001
    assert out["p_label"] == "< 0.001 · α 0.05"
    assert "below α = 0.05" in out["p_title"]


def test_negative_mean_puts_the_one_sided_p_above_one_half():
    rows = [
        _row(-1.0, 10.0, RECENT_RATE_START),
        _row(0.0, 10.0, RECENT_RATE_START + timedelta(days=1)),
    ]
    out = sample_size_wait(rows, enabled=True, n_bets=2, now=NOW)
    assert abs(out["t"] - (-1.0)) < 1e-9
    assert abs(out["p_value"] - 0.75) < 1e-8
    assert out["p_label"] == "0.750 · α 0.05"


def test_dash_and_blank_have_no_t_statistic():
    one = sample_size_wait([_row(3.0, 10.0, RECENT_RATE_START)], enabled=True, n_bets=1, now=NOW)
    assert one["p_value"] is None
    assert one["p_label"] == "--"
    blank = sample_size_wait([], enabled=False, n_bets=0, now=NOW)
    assert blank["p_label"] == ""
    assert blank["alpha"] is None


def test_evidence_classes_do_not_promote_live():
    # p = 0.25, two bets. No certainty, and the gate fraction sits under the wait.
    thin = sample_size_wait(
        [
            _row(0.0, 10.0, RECENT_RATE_START),
            _row(1.0, 10.0, RECENT_RATE_START + timedelta(days=1)),
        ],
        enabled=True, n_bets=2, now=NOW,
    )
    assert thin["evidence_label"] == "No certainty"
    assert thin["gate_label"] == "2/13 · 80%"
    assert "live" in thin["evidence_title"].lower()

    losing = sample_size_wait(
        [
            _row(-1.0, 10.0, RECENT_RATE_START),
            _row(0.0, 10.0, RECENT_RATE_START + timedelta(days=1)),
        ],
        enabled=True, n_bets=2, now=NOW,
    )
    assert losing["evidence_key"] == "dangerous"

    # Mean +0.15, large variance, 30 bets: p under 0.05 and n80 well above 30.
    start = RECENT_RATE_START
    mixed = []
    for i in range(30):
        pnl = 4.0 if i % 2 == 0 else -1.0
        mixed.append(_row(pnl, 10.0, start + timedelta(hours=i)))
    robust = sample_size_wait(mixed, enabled=True, n_bets=30, now=NOW)
    assert robust["p_value"] < 0.05
    assert robust["n"] < robust["n80"]
    assert robust["evidence_label"] == "Robust"
    assert "does not turn live betting on" in robust["evidence_title"]

    tight = [_row(1.0 + i * 0.0001, 10.0, start + timedelta(days=i)) for i in range(8)]
    confident = sample_size_wait(tight, enabled=True, n_bets=8, now=NOW)
    assert confident["state"] == "met"
    assert confident["evidence_label"] == "Confident"


def test_n80_uses_unbiased_variance_and_ceil():
    # returns 0 and 0.10. mean 0.05. s^2 = 0.005. 6.1826 * 0.005 / 0.05^2 = 12.3652.
    rows = [
        _row(0.0, 10.0, RECENT_RATE_START),
        _row(1.0, 10.0, RECENT_RATE_START + timedelta(days=1)),
    ]
    # Pad the recent window so the rate is defined, without changing the two-return case.
    # This test only checks n80 on a hand-built variance, so keep n at 2 and accept the fallback rate.
    out = sample_size_wait(rows, enabled=True, n_bets=2, now=NOW)
    assert out["n"] == 2
    assert out["n80"] == 13
    assert out["bets_needed"] == 11
    assert out["state"] == "wait"


def test_simple_return_uses_size_not_a_zero_fill():
    # Cashout rows keep the stake in size_usdc. A 10 profit on a 40 stake is 0.25.
    # Two identical returns have no variance.
    rows = [
        _row(10.0, 40.0, RECENT_RATE_START),
        _row(5.0, 20.0, RECENT_RATE_START + timedelta(days=1)),
    ]
    out = sample_size_wait(rows, enabled=True, n_bets=2, now=NOW)
    assert out["state"] == "dash"
    assert out["label"] == "--"
    assert out["n80"] is None


def test_one_bet_is_a_dash():
    rows = [_row(3.0, 10.0, RECENT_RATE_START)]
    out = sample_size_wait(rows, enabled=True, n_bets=1, now=NOW)
    assert out["state"] == "dash"
    assert out["label"] == "--"
    assert out["days"] is None
    assert out["eta"] is None


def test_retired_with_zero_bets_stays_blank():
    out = sample_size_wait([], enabled=False, n_bets=0, now=NOW)
    assert out["state"] == "blank"
    assert out["label"] == ""
    assert out["n80"] is None


def test_enabled_with_zero_bets_is_a_dash():
    out = sample_size_wait([], enabled=True, n_bets=0, now=NOW)
    assert out["state"] == "dash"
    assert out["label"] == "--"


def test_bar_met_does_not_call_the_strategy_ready():
    # Tiny variance: n80 is small, so eight similar wins already clear it.
    start = RECENT_RATE_START
    rows = [_row(1.0 + i * 0.0001, 10.0, start + timedelta(days=i)) for i in range(8)]
    out = sample_size_wait(rows, enabled=True, n_bets=8, now=NOW)
    assert out["state"] == "met"
    assert out["bets_needed"] == 0
    assert out["label"] == "Sample-size bar met"
    blob = (out["label"] + " " + out["title"]).lower()
    assert "production" not in blob
    assert "ready" not in blob
    assert "live" not in blob


def test_recent_window_when_at_least_five_settled_bets():
    rows = _spread(6, RECENT_RATE_START)
    out = sample_size_wait(rows, enabled=True, n_bets=6, now=NOW)
    assert out["rate_window"] == "since 14 Nov 2026 09:03 UTC"
    assert out["rate_bets"] == 6
    assert out["days"] is not None
    assert out["eta"] is None
    assert out["label"] == f"{out['days']} days · {out['bets_needed']} bets"
    assert "2026" not in out["label"]
    assert "2027" not in out["label"]


def test_fallback_window_when_recent_slice_is_under_five():
    # Four bets sit before 14 Nov, inside the Melbourne window, and one sits after.
    early = [
        _row(0.0 if i % 2 == 0 else 1.0, 10.0, FALLBACK_RATE_START + timedelta(days=1, hours=i))
        for i in range(4)
    ]
    late = [_row(0.0, 10.0, RECENT_RATE_START + timedelta(hours=1))]
    out = sample_size_wait(early + late, enabled=True, n_bets=5, now=NOW)
    assert out["rate_window"] == "since 8 Nov 2026 00:00 Melbourne"
    assert out["rate_bets"] == 5
    assert out["days"] >= 1
    assert out["eta"] is None
    assert out["label"] == f"{out['days']} days · {out['bets_needed']} bets"


def test_zero_rate_shows_bets_still_needed_and_no_date():
    # Both bets settled before either rate window, so the rate in both windows is 0.
    ancient = datetime(2026, 11, 1, tzinfo=timezone.utc)
    rows = [
        _row(0.0, 10.0, ancient),
        _row(1.0, 10.0, ancient + timedelta(hours=1)),
    ]
    out = sample_size_wait(rows, enabled=True, n_bets=2, now=NOW)
    assert out["state"] == "wait"
    assert out["n80"] == 13
    assert out["bets_needed"] == 11
    assert out["days"] is None
    assert out["eta"] is None
    assert out["label"] == "11 bets"
    assert "2026" not in out["label"]
    assert "Nov" not in out["label"]
    assert "Dec" not in out["label"]


def test_days_round_up():
    # 6 bets from the recent start, spaced so the rate is just under an integer day count.
    # Need a known bets_needed. Use the 0/0.10 pair repeated.
    # n=6, variance of three 0s and three 0.1s:
    # mean = 0.05, s^2 = (3*0.05^2 + 3*0.05^2) / 5 = 0.003, n80 = ceil(6.1826 * 0.003 / 0.0025) = ceil(7.41912) = 8
    # bets needed = 2.
    start = RECENT_RATE_START
    rows = _spread(6, start)
    out = sample_size_wait(rows, enabled=True, n_bets=6, now=NOW)
    assert out["n"] == 6
    assert out["n80"] == 8
    assert out["bets_needed"] == 2
    elapsed = (NOW - RECENT_RATE_START).total_seconds() / 86400.0
    rate = 6 / elapsed
    expected_days = max(1, int(__import__("math").ceil(2 / rate - 1e-9)))
    assert out["days"] == expected_days
    day_word = "day" if expected_days == 1 else "days"
    assert out["label"] == f"{expected_days} {day_word} · 2 bets"
    assert out["eta"] is None
