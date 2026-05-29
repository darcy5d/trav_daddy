"""Wave 5.7: paper-trading strategy configurations.

Each strategy maintains its own paper bankroll in bet_ledger via
strategy_label and is filtered/sized independently. The same scan can
trigger a bet for multiple strategies on the same fixture.

Why multiple strategies:
- Wave 5.6 showed only ONE cell had real edge (V2 ODI men T-3h moneyline
  with 88% win rate but n=9). We want to forward-test that AND parallel
  hypotheses to see which generalises in genuine out-of-sample.

Strategy templates (default starting bankroll $1000 each):
    v2_odi_t3h_3pp        -- the historical winner (V2 / ODI men / T-3h / 3pp+ edge)
    v2_any_t3h_5pp        -- broader: V2 any-tournament T-3h with high 5pp threshold (RETIRED)
    v3_marg_t3h_3pp       -- V3 marginalised / any / T-3h / 3pp+
    consensus_5pp         -- V2 AND V3 agree on same side, both >= 5pp edge

Strategies are pure data; the scanner imports STRATEGIES and iterates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PaperStrategy:
    """Configuration for a single paper-trading strategy."""

    name: str
    description: str

    # Filters
    enabled_market_types: List[str] = field(default_factory=lambda: ["moneyline"])
    enabled_tournament_prefixes: Optional[List[str]] = None  # None = all
    enabled_formats: Optional[List[str]] = None              # None = all
    enabled_genders: Optional[List[str]] = None              # None = all
    min_market_price: float = 0.08
    max_market_price: float = 0.92
    min_edge_pp: float = 3.0      # percentage points (model_prob - market_price) * 100
    # model_prob bounds: skip bets outside this range (None = no bound).
    # Used to exclude the coin-flip zone [0.45, 0.55] where calibration shows
    # no edge, and to prevent chasing markets where model is overconfident.
    min_model_prob: Optional[float] = None
    max_model_prob: Optional[float] = None
    # Fill-gap guard: skip if (model_prob - market_price)*100 exceeds this.
    # Catches cases where the market has moved sharply on information the model
    # doesn't have (e.g. toss outcome for V2 strategies, late XI news).
    # None = no limit.
    max_model_minus_fill_pp: Optional[float] = None

    # Model
    model_version: str = "v2"          # "v2" or "v3"
    toss_mode: str = "marginalised"     # "marginalised" or "pinned" (V3 only)
    require_consensus_with: Optional[str] = None  # name of another strategy that must agree
    # Post-toss real-bet eligibility. Set False for V2-based strategies that
    # cannot condition on the toss outcome (V2 sim ignores toss kwargs).
    # Paper bets are always placed regardless of this flag.
    post_toss_eligible: bool = True

    # Entry timing
    lookback_hours: float = 3.0         # bet within this window before scheduled start
    lookback_hours_min: float = 0.5     # lower bound (don't bet too close)

    # Sizing
    starting_bankroll_usdc: float = 1000.0
    kelly_mult: float = 0.5             # 0.5 = half-Kelly
    kelly_fraction_cap: float = 0.25    # never stake > 25% of bankroll
    min_stake_usdc: float = 5.0
    max_stake_usdc: float = 100.0

    # In-game cashout
    # Manual per-strategy override for the cashout return-ratio threshold.
    # If set, this overrides the global tiered lookup in cashout.tiered_cashout_threshold().
    # None (default) = use the tiered logic based on fill_price (recommended).
    cashout_return_threshold: Optional[float] = None

    # Lifecycle
    enabled: bool = True


# ---------------- Default strategy roster ----------------
# 4 parallel strategies, each starting at $1000 paper bankroll.
# Tweak / disable / add new ones here without touching the scan code.

STRATEGIES: List[PaperStrategy] = [
    # The historical-winner cell from Wave 5.6: V2 / ODI men / 3pp edge.
    # Lookback window kept WIDE (T-48h to T-1h) so daily/hourly cron can
    # always catch the match - paper bets are idempotent so each fixture
    # gets exactly one bet per strategy. Analysis can group by actual
    # T-{hours_to_kickoff} bucket post-hoc using proposed_at vs kickoff.
    PaperStrategy(
        name="v2_odi_3pp",
        description=(
            "V2 / ODI men / moneyline / 3pp edge. RETIRED: -$157 paper PnL "
            "on 7 bets (43% win rate), only 2 live bets placed in months. "
            "ODI fixtures are too rare on Polymarket for meaningful edge. "
            "Bankroll reallocated to v3_marg_3pp and consensus_5pp."
        ),
        enabled_market_types=["moneyline"],
        enabled_formats=["ODI"],
        enabled_genders=["male"],
        min_edge_pp=3.0,
        model_version="v2",
        lookback_hours=48.0,
        lookback_hours_min=0.0,
        starting_bankroll_usdc=1000.0,
        kelly_mult=0.5,
        enabled=False,
    ),
    PaperStrategy(
        name="v2_any_5pp",
        description=(
            "V2 any tournament + moneyline + high 5pp edge threshold. "
            "RETIRED: 25% win rate, no demonstrated edge. "
            "Also excluded from post-toss real bets (V2 is toss-blind)."
        ),
        enabled_market_types=["moneyline"],
        min_edge_pp=5.0,
        model_version="v2",
        lookback_hours=48.0,
        lookback_hours_min=0.0,
        starting_bankroll_usdc=1000.0,
        kelly_mult=0.5,
        post_toss_eligible=False,
        enabled=False,
    ),
    PaperStrategy(
        name="v3_marg_3pp",
        description=(
            "V3 (marginalised toss) any tournament + moneyline + 3pp edge. "
            "Tests V3-vs-V2 lift hypothesised in Wave 5.5. "
            "Post-toss eligible: uses toss_pinned=True in live_bet_post_toss_scan."
        ),
        enabled_market_types=["moneyline"],
        min_edge_pp=3.0,
        model_version="v3",
        toss_mode="marginalised",
        lookback_hours=48.0,
        lookback_hours_min=0.0,
        starting_bankroll_usdc=1000.0,
        kelly_mult=0.75,
        kelly_fraction_cap=0.30,
        min_model_prob=0.50,
        post_toss_eligible=True,
        max_model_minus_fill_pp=20.0,
    ),
    PaperStrategy(
        name="consensus_5pp",
        description=(
            "V2 AND V3 both pick the same side with at least 5pp edge each. "
            "Robust signal filter; best ROI of all strategies (-20%). "
            "Excluded from post-toss real bets: V2 half is toss-blind and "
            "contaminates the consensus signal post-toss."
        ),
        enabled_market_types=["moneyline"],
        min_edge_pp=5.0,
        model_version="consensus",
        lookback_hours=48.0,
        lookback_hours_min=0.0,
        starting_bankroll_usdc=1000.0,
        kelly_mult=0.75,
        kelly_fraction_cap=0.30,
        min_model_prob=0.52,
        post_toss_eligible=False,
    ),
    # Diagnostic: low-edge wide-window quarter-Kelly to capture max data.
    PaperStrategy(
        name="v2_diag_2pp",
        description=(
            "DIAGNOSTIC: V2 any tournament with very low 2pp threshold and "
            "quarter-Kelly. Captures more data points for calibration; "
            "ROI signal expected to be noisier but bet count higher. "
            "Excluded from post-toss real bets (V2 is toss-blind)."
        ),
        enabled_market_types=["moneyline"],
        min_edge_pp=2.0,
        model_version="v2",
        lookback_hours=96.0,
        lookback_hours_min=0.0,
        starting_bankroll_usdc=1000.0,
        kelly_mult=0.25,
        max_stake_usdc=50.0,
        post_toss_eligible=False,
    ),
]


def get_strategy(name: str) -> Optional[PaperStrategy]:
    for s in STRATEGIES:
        if s.name == name:
            return s
    return None


def get_enabled_strategies() -> List[PaperStrategy]:
    return [s for s in STRATEGIES if s.enabled]


def kelly_stake_usdc(
    model_prob: float,
    market_price: float,
    bankroll_usdc: float,
    strategy: PaperStrategy,
) -> float:
    """Half-Kelly sizing capped by strategy's min/max stake.

    Standard Kelly:  f* = (b*p - q) / b
        where b = (1 / market_price) - 1   (decimal-odds payoff per $1 stake)
              p = model_prob (true probability)
              q = 1 - p
    For binary markets where market_price = implied prob this simplifies to:
        f* = (model_prob - market_price) / (1 - market_price)
    """
    if market_price <= 0 or market_price >= 1:
        return 0.0
    if model_prob <= 0 or model_prob >= 1:
        return 0.0
    f_star = (model_prob - market_price) / (1.0 - market_price)
    f_star = max(0.0, min(f_star, 1.0))
    f_capped = min(f_star * strategy.kelly_mult, strategy.kelly_fraction_cap)
    raw_stake = f_capped * bankroll_usdc
    if raw_stake < strategy.min_stake_usdc:
        return 0.0
    return min(raw_stake, strategy.max_stake_usdc, bankroll_usdc * 0.25)


def get_strategy_bankroll(strategy_name: str, conn) -> float:
    """Compute the current paper bankroll for a strategy.

    Sum of:
        starting_bankroll
        + sum(pnl_realised_usdc) for settled paper bets in this strategy

    Open positions are NOT subtracted (we use bankroll_at_proposal at bet time
    to size; settlement updates bankroll going forward).
    """
    strat = get_strategy(strategy_name)
    if strat is None:
        return 0.0
    cur = conn.cursor()
    cur.execute(
        """
        SELECT COALESCE(SUM(pnl_realised_usdc), 0.0)
        FROM bet_ledger
        WHERE bet_kind = 'paper'
          AND strategy_label = ?
          AND status = 'settled'
        """,
        (strategy_name,),
    )
    row = cur.fetchone()
    realised = float(row[0]) if row and row[0] is not None else 0.0
    return strat.starting_bankroll_usdc + realised
