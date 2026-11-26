"""How long a paper strategy's settled sample takes to clear its standard errors.

Same definition as the 14 Nov 2026 note. A settled paper bet is one simple
return, ``pnl_realised_usdc / size_usdc``. ``n80`` is the settled-bet count at
which a one-sided 5% test has 80% power to detect a true mean return of 0.05:

    n80 = ceil(6.1826 * s^2 / 0.05^2)

``s^2`` is the unbiased variance of those simple returns. This is a sample-size
wait only. It does not say a strategy is ready to bet live.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from decimal import Decimal, ROUND_CEILING
from typing import Mapping, Optional, Sequence
from zoneinfo import ZoneInfo

# (z_0.95 + z_0.80)^2, with z_0.95 = 1.64485 and z_0.80 = 0.84162.
POWER_CONSTANT = Decimal("6.1826")
DETECTABLE_MEAN = Decimal("0.05")
# Significance level of the one-sided test the sample-size wait is sized for.
SIGNIFICANCE_LEVEL = 0.05
POWER = 0.80
# A p-value from fewer bets than this stays "No certainty", even if it is under α.
MIN_CLASS_BETS = 11
RECENT_RATE_MIN_BETS = 5
RECENT_RATE_START = datetime(2026, 11, 14, 9, 3, tzinfo=timezone.utc)
MELBOURNE = ZoneInfo("Australia/Melbourne")
# Midnight at the start of 8 Nov 2026 in Melbourne (AEDT, UTC+11).
FALLBACK_RATE_START = datetime(2026, 11, 8, 0, 0, tzinfo=MELBOURNE)

_RECENT_WINDOW = "since 14 Nov 2026 09:03 UTC"
_FALLBACK_WINDOW = "since 8 Nov 2026 00:00 Melbourne"


def sample_size_wait(
    rows: Sequence[Mapping],
    *,
    enabled: bool,
    n_bets: int,
    now: datetime,
) -> dict:
    """Sample-size wait for one strategy's settled rows in the displayed cohort.

    ``rows`` are settled paper bets: ``pnl_realised_usdc``, ``size_usdc``,
    ``settled_at``. Retired strategies with no bets stay blank. One usable
    return, or a zero variance, has no ``n80`` and is a dash. When the settled
    count is already at ``n80``, the sample-size bar is met.
    """
    if not enabled and int(n_bets or 0) <= 0:
        return _result(state="blank", n=0)

    now_utc = _as_utc(now)
    returns, settled_at = _usable_returns(rows)
    n = len(returns)
    variance = _unbiased_variance(returns)
    test = _mean_test(returns, variance)
    if variance is None:
        return _result(
            state="dash",
            n=n,
            title="One settled bet, or no variance in simple returns, so there is no n80.",
            label="--",
            test=test,
            dash_p=True,
        )

    n80 = _n80(variance)
    bets_needed = max(0, n80 - n)
    if bets_needed == 0:
        return _result(
            state="met",
            n=n,
            n80=n80,
            bets_needed=0,
            title=f"{n} settled bets. n80 is {n80}. Sample-size bar met.",
            label="Sample-size bar met",
            test=test,
        )

    count, rate, window = _settled_rate(settled_at, now_utc)
    if rate <= 0:
        bet_word = _plural(bets_needed, "bet")
        return _result(
            state="wait",
            n=n,
            n80=n80,
            bets_needed=bets_needed,
            rate_per_day=0.0,
            rate_window=window,
            rate_bets=count,
            title=(
                f"{n} settled bets. n80 is {n80}. {bets_needed} bets still needed. "
                "Settled-bet rate is 0, so there is no day count."
            ),
            label=f"{bets_needed} {bet_word}",
            test=test,
        )

    days = max(1, math.ceil((bets_needed / rate) - 1e-9))
    day_word = _plural(days, "day")
    bet_word = _plural(bets_needed, "bet")
    return _result(
        state="wait",
        n=n,
        n80=n80,
        bets_needed=bets_needed,
        days=days,
        rate_per_day=rate,
        rate_window=window,
        rate_bets=count,
        title=(
            f"{n} settled bets. n80 is {n80}. {bets_needed} bets still needed. "
            f"{rate:.2f} settled bets/day {window}."
        ),
        label=f"{days} {day_word} · {bets_needed} {bet_word}",
        test=test,
    )


def _mean_test(values: Sequence[float], variance: Optional[float]):
    """One-sided Student-t test of a zero mean against a positive mean.

    Returns ``(t, df, p)`` or None when the variance is undefined. ``p`` is
    P(T_df > t), the current one-sided p-value. The sample-size wait targets
    this same test at ``SIGNIFICANCE_LEVEL``.
    """
    n = len(values)
    if variance is None or n < 2:
        return None
    mean = sum(values) / n
    standard_error = math.sqrt(variance / n)
    if standard_error <= 0.0 or not math.isfinite(standard_error):
        return None
    t = mean / standard_error
    if not math.isfinite(t):
        return None
    return t, n - 1, _student_t_upper_tail(t, n - 1)


def _student_t_upper_tail(t: float, df: int) -> float:
    """P(T_df > t). At df = 1 this is the Cauchy tail."""
    if df < 1:
        raise ValueError("df must be at least 1")
    x = df / (df + t * t)
    two_tailed = _regularized_incomplete_beta(df / 2.0, 0.5, x)
    upper = two_tailed / 2.0
    if t >= 0.0:
        return upper
    return 1.0 - upper


def _regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """I_x(a, b), Numerical Recipes continued fraction."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    log_beta = (
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log(1.0 - x)
    )
    front = math.exp(log_beta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float) -> float:
    max_iter = 200
    eps = 3e-14
    tiny = 1e-300
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _format_p(p: float) -> str:
    """Three decimals, or '< 0.001' once the value is smaller than that."""
    if p < 0.001:
        return "< 0.001"
    return f"{p:.3f}"


def _p_fields(test, *, dash: bool) -> dict:
    alpha = SIGNIFICANCE_LEVEL
    if dash:
        return {
            "p_value": None,
            "t": None,
            "df": None,
            "alpha": alpha,
            "p_label": "--",
            "p_title": (
                "Fewer than two settled returns, so there is no t-statistic. "
                "The target significance level is α = 0.05."
            ),
        }
    if test is None:
        return {
            "p_value": None,
            "t": None,
            "df": None,
            "alpha": None,
            "p_label": "",
            "p_title": "",
        }
    t, df, p = test
    shown = _format_p(p)
    side = "below" if p < alpha else "above"
    exact = f"{p:.2e}" if p < 0.001 else f"{p:.4f}"
    return {
        "p_value": p,
        "t": round(t, 4),
        "df": df,
        "alpha": alpha,
        "p_label": f"{shown} · α {alpha:.2f}",
        "p_title": (
            f"One-sided p-value {exact} (t = {t:.2f} on {df} df). "
            f"It is {side} α = 0.05, so the test "
            f"{'rejects' if p < alpha else 'does not reject'} a mean return of zero. "
            "That comparison is not the sample-size bar."
        ),
    }


def _n80(variance: float) -> int:
    raw = POWER_CONSTANT * Decimal(variance) / (DETECTABLE_MEAN ** 2)
    return int(raw.to_integral_value(rounding=ROUND_CEILING))


def _unbiased_variance(values: Sequence[float]) -> Optional[float]:
    n = len(values)
    if n < 2:
        return None
    mean = sum(values) / n
    var = sum((value - mean) ** 2 for value in values) / (n - 1)
    if not math.isfinite(var) or var <= 0.0:
        return None
    return var


def _usable_returns(rows: Sequence[Mapping]) -> tuple[list[float], list[datetime]]:
    returns: list[float] = []
    settled_at: list[datetime] = []
    for row in rows:
        pnl = _finite(row.get("pnl_realised_usdc"))
        size = _finite(row.get("size_usdc"))
        if pnl is None or size is None or size <= 0.0:
            continue
        returns.append(pnl / size)
        when = _parse_time(row.get("settled_at"))
        if when is not None:
            settled_at.append(when)
    return returns, settled_at


def _settled_rate(
    settled_at: Sequence[datetime], now: datetime
) -> tuple[int, float, str]:
    """Bets per day since 14 Nov 09:03 UTC, or since 8 Nov Melbourne if thinner than 5."""
    recent_start = RECENT_RATE_START.astimezone(timezone.utc)
    fallback_start = FALLBACK_RATE_START.astimezone(timezone.utc)
    recent = _count_between(settled_at, recent_start, now)
    if recent >= RECENT_RATE_MIN_BETS:
        return recent, _per_day(recent, recent_start, now), _RECENT_WINDOW
    fallback = _count_between(settled_at, fallback_start, now)
    return fallback, _per_day(fallback, fallback_start, now), _FALLBACK_WINDOW


def _count_between(times: Sequence[datetime], start: datetime, end: datetime) -> int:
    return sum(1 for when in times if start <= when <= end)


def _per_day(count: int, start: datetime, end: datetime) -> float:
    elapsed_days = (end - start).total_seconds() / 86400.0
    if count <= 0 or elapsed_days <= 0.0:
        return 0.0
    return count / elapsed_days


def _parse_time(raw) -> Optional[datetime]:
    if isinstance(raw, datetime):
        return _as_utc(raw)
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    text = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    return _as_utc(parsed)


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _finite(value) -> Optional[float]:
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number):
        return None
    return number


def _plural(count: int, word: str) -> str:
    return word if count == 1 else word + "s"


def _result(
    *,
    state: str,
    n: int,
    n80: Optional[int] = None,
    bets_needed: Optional[int] = None,
    days: Optional[int] = None,
    eta: Optional[str] = None,
    rate_per_day: Optional[float] = None,
    rate_window: Optional[str] = None,
    rate_bets: Optional[int] = None,
    title: str = "",
    label: str = "",
    test=None,
    dash_p: bool = False,
) -> dict:
    return {
        "state": state,
        "n": n,
        "n80": n80,
        "bets_needed": bets_needed,
        "days": days,
        "eta": eta,
        "rate_per_day": None if rate_per_day is None else round(rate_per_day, 4),
        "rate_window": rate_window,
        "rate_bets": rate_bets,
        "title": title,
        "label": label,
        **_p_fields(test, dash=dash_p),
        **_evidence(state, n, n80, test),
    }


def _evidence(state: str, n: int, n80: Optional[int], test) -> dict:
    """Four display classes. None of them turns live betting on.

    No certainty: no t-test, fewer than 11 bets, or p at or above 0.05.
    Dangerous: the mean simple return is negative.
    Robust: p < 0.05, at least 11 bets, and still short of n80.
    Confident: p < 0.05 and the settled count has reached n80.
    """
    if state == "blank":
        return {"evidence_key": "", "evidence_label": "", "evidence_title": "", "gate_label": ""}
    gate = ""
    if n80:
        gate = f"{n}/{n80} · {int(POWER * 100)}%"
    if test is None or state == "dash":
        return {
            "evidence_key": "none",
            "evidence_label": "No certainty",
            "evidence_title": "No t-test yet. Not a live signal.",
            "gate_label": gate,
        }
    t, df, p = test
    if t < 0:
        return {
            "evidence_key": "dangerous",
            "evidence_label": "Dangerous",
            "evidence_title": (
                "Mean simple return is negative. "
                "Not a live signal."
            ),
            "gate_label": gate,
        }
    significant = p < SIGNIFICANCE_LEVEL
    bar_met = n80 is not None and n >= n80
    if significant and bar_met:
        return {
            "evidence_key": "confident",
            "evidence_label": "Confident",
            "evidence_title": (
                f"p < 0.05 and {n} settled bets have reached n80 ({n80}). "
                "The 5% test and the 80% power bar are both met. "
                "This does not turn live betting on."
            ),
            "gate_label": gate,
        }
    if significant and n >= MIN_CLASS_BETS:
        return {
            "evidence_key": "robust",
            "evidence_label": "Robust",
            "evidence_title": (
                f"p < 0.05 on {n} bets, still short of n80 ({n80}). "
                "The mean rejects zero. The sample-size gate does not. "
                "This does not turn live betting on."
            ),
            "gate_label": gate,
        }
    thin = significant and n < MIN_CLASS_BETS
    why = (
        f"p is under 0.05 but from only {n} bets."
        if thin else
        "p is not under 0.05."
    )
    return {
        "evidence_key": "none",
        "evidence_label": "No certainty",
        "evidence_title": why + " Not a live signal.",
        "gate_label": gate,
    }
