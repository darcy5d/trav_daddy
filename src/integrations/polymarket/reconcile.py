"""Wave 5 Phase 6b + Wave 5.7: Bet ledger reconciliation.

For each bet in `placed`/`filled` status whose match has completed, fetch
the resolved Polymarket market and compute realised P&L. Update
`settled_at`, `settle_outcome`, `pnl_realised_usdc`, `status='settled'`.

Resolution detection (in priority order):
  1. Gamma /markets/{id} returns `closed: true` + `outcomePrices`.
     This is the canonical resolution signal - clean, deterministic,
     sets outcomePrices to exact ["0","1"] or ["1","0"].
  2. CLOB /prices-history shows last_price <= SETTLED_PRICE_LOW or
     >= SETTLED_PRICE_HIGH AND enough real time has passed since
     scheduled kickoff for the match to definitely be over (per-format
     duration). Fallback for markets where Gamma is lagging or the
     market_id wasn't recorded.

The kickoff gate exists because thin-volume markets (women's
internationals, U19, lower-tier T20s) routinely see in-play price
spikes past 0.99 the moment one side gets on top. Without the kickoff
buffer the reconciler would mark a still-live match as settled. See
the May 2026 Pakistan vs Zimbabwe Women's ODI incident where Pakistan
went 114-0 in the first innings and the moneyline jumped to 0.99
within minutes.

Idempotent - already-settled rows are skipped.

Usage:

    from src.integrations.polymarket.reconcile import reconcile_pending_bets
    summary = reconcile_pending_bets()
"""

from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import requests

from src.integrations.polymarket import PolymarketClient
from src.integrations.odds.polymarket_compare import (
    POLYMARKET_TAKER_FEE, PolymarketComparisonService,
)

logger = logging.getLogger(__name__)

GAMMA_MARKETS_BASE = "https://gamma-api.polymarket.com/markets"

# Path 2 (price-history fallback) trigger thresholds. Tightened from the
# original 0.98 / 0.02 after observing in-play moneylines on thin women's
# ODI markets reach 0.99+ during first-innings dominance.
SETTLED_PRICE_HIGH = 0.995
SETTLED_PRICE_LOW = 0.005

# Sanity floor: the bet must be at least this many hours old before Path 2
# can fire. Cheap defense against same-day mistakes.
MIN_HOURS_SINCE_PROPOSAL_FOR_PRICE_FALLBACK = 6.0

# Primary defense: how long after scheduled kickoff before Path 2 will trust
# a settled-looking price as actual resolution. Tuned to comfortably exceed
# real-world match duration:
#   T20:  ~3.5h match -> 4h cushion
#   ODI:  ~7-8h match -> 8h cushion
#   TEST: 5 days; we don't bet test moneyline at scale, but 24h covers day 1
MIN_HOURS_SINCE_KICKOFF_BY_FORMAT = {
    "T20": 4.0,
    "ODI": 8.0,
    "TEST": 24.0,
}
# Used when format inference fails (e.g. legacy bet with no parseable slug).
# Defaulting to ODI keeps us on the safe side for the longest realistic
# format we actually bet on.
DEFAULT_MIN_HOURS_SINCE_KICKOFF = 8.0

# Conservative fallback when we can't anchor to ANY kickoff time at all
# (no kickoff_at column AND no parseable date in fixture_key). Tightens
# the proposal-age gate so degenerate inputs don't silently roll back to
# the loose 6h floor.
MIN_HOURS_SINCE_PROPOSAL_FALLBACK_NO_KICKOFF = 24.0


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _fetch_unsettled_bets(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM bet_ledger
        WHERE status IN ('placed', 'filled')
        ORDER BY placed_at ASC
        """
    )
    return cur.fetchall()


def _last_price_from_history(history: List[Dict[str, Any]]) -> Optional[float]:
    """Return the very last price in a /prices-history series."""
    if not history:
        return None
    last = history[-1]
    if isinstance(last, dict) and "p" in last:
        return float(last["p"])
    return None


def _hours_since(iso_ts: Optional[str]) -> Optional[float]:
    if not iso_ts:
        return None
    try:
        d = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None
    now = datetime.now(timezone.utc)
    return (now - d).total_seconds() / 3600.0


def _fetch_gamma_market(market_id: str) -> Optional[Dict[str, Any]]:
    """Fetch /markets/<market_id> from Gamma. Returns the raw dict or None
    on any network/parse failure. Cached at the call site so we hit Gamma
    at most once per bet per reconcile pass.
    """
    if not market_id:
        return None
    try:
        r = requests.get(f"{GAMMA_MARKETS_BASE}/{market_id}", timeout=10)
        if not r.ok:
            return None
        m = r.json()
    except Exception as exc:
        logger.debug(f"  Gamma /markets/{market_id} fetch failed: {exc}")
        return None
    return m if isinstance(m, dict) else None


def _resolve_via_gamma(
    market: Dict[str, Any],
    side_label: str,
) -> Optional[Tuple[int, float]]:
    """Given a fetched Gamma market dict, return (settle_outcome, resolved_price)
    if the market is canonically resolved, else None.

    settle_outcome: 1 if OUR side won, 0 if it lost.
    resolved_price: outcomePrices entry for OUR side (used as the
        effective resolution price for P&L verification).

    Returns None if:
      - the market isn't `closed: true`
      - we can't match `side_label` to any outcome
      - the response is malformed
    """
    if not market or not side_label:
        return None
    if not market.get("closed"):
        return None

    outcomes_raw = market.get("outcomes")
    prices_raw = market.get("outcomePrices")

    def _parse(value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if isinstance(value, str):
            try:
                p = json.loads(value)
                if isinstance(p, list):
                    return p
            except json.JSONDecodeError:
                pass
        return []

    outcomes = _parse(outcomes_raw)
    prices = _parse(prices_raw)
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None

    # Find which outcome matches our side_label
    matched_idx: Optional[int] = None
    for idx, out_label in enumerate(outcomes):
        if PolymarketComparisonService.label_matches_team(str(out_label), side_label):
            matched_idx = idx
            break
    if matched_idx is None:
        # Substring fallback (handles 'Rawalpindiz' vs 'Rawalpindi Pindiz')
        side_norm = side_label.lower()
        for idx, out_label in enumerate(outcomes):
            if str(out_label).lower() == side_norm:
                matched_idx = idx
                break
    if matched_idx is None:
        return None

    try:
        resolved_price = float(prices[matched_idx])
    except (TypeError, ValueError):
        return None
    settle_outcome = 1 if resolved_price >= 0.5 else 0
    return settle_outcome, resolved_price


def _derive_kickoff_iso_from_fixture_key(fixture_key: Optional[str]) -> Optional[str]:
    """Best-effort kickoff ISO when the bet row's `kickoff_at` is NULL.

    Polymarket cricket slugs trail with the match date (YYYY-MM-DD).
    Mirror the frontend convention (`live_betting.html#kickoffForBet`)
    of treating that date as 14:00 UTC kickoff. The exact time is wrong
    for many fixtures (BBL plays earlier, Asia evening matches later)
    but the only consumer is a "is enough time elapsed?" gate, so erring
    LATE simply means we wait slightly longer to settle — the failure
    mode is a delayed reconcile, not a premature one.
    """
    if not fixture_key:
        return None
    m = re.search(r"(\d{4}-\d{2}-\d{2})$", fixture_key)
    if not m:
        return None
    return f"{m.group(1)}T14:00:00+00:00"


def _format_from_bet(
    fixture_key: Optional[str],
    gamma_market: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Infer match format ('T20' / 'ODI' / 'TEST') for the kickoff-buffer
    duration heuristic.

    Tries the slug prefix first via the canonical
    `TOURNAMENT_PREFIX_MAP`. For ambiguous prefixes (`crint` covers
    everything from women's T20s to men's ODI World Cup matches), falls
    back to inferring from the Gamma market question text using the
    same heuristic the upcoming-fixture pipeline uses for routing.
    """
    if not fixture_key:
        return None
    prefix = fixture_key.split("-")[0]
    # Local import: keep `reconcile` importable in lightweight tooling
    # without dragging the whole upcoming-fixtures stack.
    from src.integrations.polymarket.upcoming import (
        TOURNAMENT_PREFIX_MAP,
        _infer_format_gender_from_title,
    )

    entry = TOURNAMENT_PREFIX_MAP.get(prefix)
    if isinstance(entry, tuple):
        return entry[0]

    if gamma_market:
        question = gamma_market.get("question") or ""
        if question:
            try:
                fmt, _gender, _name = _infer_format_gender_from_title(question, prefix)
                return fmt
            except Exception as exc:
                logger.debug(f"  format inference from title failed: {exc}")
    return None


def _compute_pnl_for_settled_bet(
    fill_price: Optional[float],
    fill_size_usdc: Optional[float],
    settle_outcome: float,
    fee_pct: float = POLYMARKET_TAKER_FEE,
) -> Optional[float]:
    """Net USD P&L (signed) for a filled bet whose outcome is now known."""
    if fill_price is None or fill_size_usdc is None:
        return None
    if fill_price <= 0:
        return None
    shares = fill_size_usdc / fill_price
    gross_payout = shares * float(settle_outcome)
    fee = gross_payout * fee_pct  # fee on proceeds, not on stake (no fee on a loss)
    return round(gross_payout - fill_size_usdc - fee, 4)


def _finalize_stale_proposed_bets(conn: sqlite3.Connection) -> int:
    """Mark stale `proposed` bet_ledger rows as cancelled/errored.

    Two failure modes accumulate forever otherwise:

    1. **Never posted**: a real bet was inserted as 'proposed' but no FOK order
       and no TWAP plan ever attached (DB write failed, daemon died, etc).
       After kickoff (or 6h since proposal as a fallback) there is no way the
       order will execute — mark errored with error_category='never_posted'.

    2. **Plan finalised but bet not**: TWAP plan ended (cancelled/completed)
       but the linked bet_ledger row is still 'proposed'. Call
       finalize_plan_from_chunks to push the chunk-driven outcome onto the bet.

    Returns count of bets touched.
    """
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='bet_ledger'")
        if not cur.fetchone():
            return 0
    except Exception:
        return 0

    now = datetime.now(timezone.utc)
    now_iso = now.isoformat()
    cutoff_kickoff_iso = (now - timedelta(minutes=10)).isoformat()
    cutoff_proposal_iso = (now - timedelta(hours=6)).isoformat()

    n_touched = 0

    # Path A: proposed bets that never produced any chunk and never got an
    # order_id. The bet is dead; mark errored.
    cur.execute(
        """
        SELECT bl.bet_id, bl.fixture_key, bl.kickoff_at, bl.proposed_at
        FROM bet_ledger bl
        LEFT JOIN order_plans op ON op.bet_ledger_id = bl.bet_id
        WHERE bl.status = 'proposed'
          AND COALESCE(bl.bet_kind, 'real') = 'real'
          AND bl.polymarket_order_id IS NULL
          AND op.plan_id IS NULL
          AND (
                (bl.kickoff_at IS NOT NULL AND bl.kickoff_at < ?)
             OR (bl.kickoff_at IS NULL AND bl.proposed_at < ?)
          )
        """,
        (cutoff_kickoff_iso, cutoff_proposal_iso),
    )
    orphan_rows = cur.fetchall()
    for row in orphan_rows:
        cur.execute(
            """
            UPDATE bet_ledger
            SET status = 'errored',
                error_category = COALESCE(error_category, 'never_posted'),
                error_message = COALESCE(error_message, 'never_posted: no order_id and no order_plan'),
                cancelled_at = COALESCE(cancelled_at, ?),
                reconciled_at = ?
            WHERE bet_id = ? AND status = 'proposed'
            """,
            (now_iso, now_iso, row["bet_id"]),
        )
        if cur.rowcount:
            n_touched += 1
            logger.info(
                f"Stale proposed bet_id={row['bet_id']} fixture={row['fixture_key']} "
                f"-> errored (never_posted)"
            )

    # Path B: proposed bets whose plan is already cancelled/completed —
    # re-run finalize_plan_from_chunks to push the right state onto the bet.
    try:
        from src.integrations.polymarket.clob_fills import finalize_plan_from_chunks
    except Exception as exc:
        logger.warning(f"clob_fills import failed: {exc}")
        finalize_plan_from_chunks = None  # type: ignore

    if finalize_plan_from_chunks is not None:
        cur.execute(
            """
            SELECT op.plan_id, bl.bet_id
            FROM order_plans op
            JOIN bet_ledger bl ON bl.bet_id = op.bet_ledger_id
            WHERE bl.status = 'proposed'
              AND op.status IN ('cancelled', 'completed')
            """
        )
        for row in cur.fetchall():
            try:
                result = finalize_plan_from_chunks(conn, cur, row["plan_id"])
                if result.get("bet_updated"):
                    n_touched += 1
                    # Tag cancel_reason on bets that ended up cancelled.
                    cur.execute(
                        """
                        UPDATE bet_ledger
                        SET cancel_reason = COALESCE(cancel_reason, 'twap_no_fill'),
                            cancelled_at = COALESCE(cancelled_at, ?),
                            reconciled_at = ?
                        WHERE bet_id = ? AND status = 'cancelled'
                        """,
                        (now_iso, now_iso, row["bet_id"]),
                    )
            except Exception as exc:
                logger.warning(f"finalize_plan_from_chunks(plan={row['plan_id']}) failed: {exc}")

    if n_touched:
        conn.commit()
        logger.info(f"Finalised {n_touched} stale proposed bet(s)")
    return n_touched


def _resync_missed_maker_fills(conn: sqlite3.Connection) -> int:
    """Promote cancelled bets whose TWAP chunks actually filled as MAKER on-chain.

    Kickoff cancel / reprice can mark a plan cancelled (or overwrite the live
    order_chunks.polymarket_order_id) before CLOB maker fills are indexed.

    Strategy:
      1. Fetch all CLOB trades and index fills using order_history as the
         master "did we place this" set (so repriced ids that no longer live
         on order_chunks are still recognised as ours).
      2. Apply maker fills directly to order_chunks rows when their id still
         matches (covers placed/cancelled/pending chunks).
      3. For each *historical* order id that has an on-chain fill AND we have
         a chunk_id in order_history (which survives reprice/resize), update
         that chunk_id even though its live polymarket_order_id has moved on.
      4. Re-finalise plans for bets stuck in cancelled/proposed/placed.

    Returns count of bets promoted to filled.
    """
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='order_chunks'")
        if not cur.fetchone():
            return 0
    except Exception:
        return 0

    try:
        from src.integrations.polymarket.clob_fills import (
            apply_fill_to_chunk,
            fetch_all_clob_trades,
            finalize_plan_from_chunks,
            index_fills_by_order_id,
            sync_placed_chunk_fills,
        )
        from src.integrations.polymarket.order_audit import (
            all_known_order_ids,
            record_order_filled,
        )
        pm = PolymarketClient()
        trades = fetch_all_clob_trades(pm)
        known_ids = all_known_order_ids(conn)
        fills = index_fills_by_order_id(trades, known_order_ids=known_ids)
        n_chunks = sync_placed_chunk_fills(conn, cur, fills)

        # Follow the reprice chain: any historical order id that filled
        # but whose live chunk has a different polymarket_order_id today.
        n_chained = 0
        try:
            for oid, fill in fills.items():
                if float(fill.get("fill_usdc") or 0) <= 0:
                    continue
                cur.execute(
                    """
                    SELECT chunk_id, final_status
                    FROM order_history
                    WHERE polymarket_order_id = ? AND chunk_id IS NOT NULL
                    """,
                    (oid,),
                )
                hrow = cur.fetchone()
                if hrow is None:
                    continue
                chunk_id = hrow["chunk_id"]
                cur.execute(
                    "SELECT polymarket_order_id, status FROM order_chunks WHERE chunk_id = ?",
                    (chunk_id,),
                )
                crow = cur.fetchone()
                if crow is None:
                    continue
                # If the live chunk still points at this oid, sync_placed_chunk_fills
                # already handled it. Only act when the chunk has moved on.
                if str(crow["polymarket_order_id"] or "") == oid:
                    continue
                if (crow["status"] or "").lower() == "filled":
                    continue
                if apply_fill_to_chunk(conn, cur, chunk_id, fill):
                    n_chained += 1
                    record_order_filled(
                        oid,
                        fill_usdc=float(fill.get("fill_usdc") or 0),
                        fill_price=fill.get("avg_fill_price"),
                        conn=conn,
                    )
        except Exception as exc:
            logger.warning(f"reprice-chain fill follow-through failed: {exc}")

        if n_chained:
            logger.info(f"Reprice-chain: applied {n_chained} historic fill(s) to live chunks")
    except Exception as exc:
        logger.warning(f"MAKER fill resync failed: {exc}")
        return 0

    # Plans needing finalize:
    #   (a) any cancelled/proposed/placed bet whose chunks have any fill
    #       (promote to filled)
    #   (b) any filled bet whose chunks' total fill exceeds bet_ledger.fill_size_usdc
    #       by more than the partial-fill noise threshold (top up the bet)
    cur.execute(
        """
        SELECT DISTINCT op.plan_id
        FROM order_plans op
        JOIN bet_ledger bl ON bl.bet_id = op.bet_ledger_id
        JOIN order_chunks oc ON oc.plan_id = op.plan_id
        WHERE COALESCE(bl.bet_kind, 'real') = 'real'
          AND oc.status = 'filled'
          AND (
                bl.status IN ('cancelled', 'proposed', 'placed')
             OR (
                bl.status = 'filled'
                AND (
                    SELECT COALESCE(SUM(oc2.fill_size_usdc), 0)
                    FROM order_chunks oc2
                    WHERE oc2.plan_id = op.plan_id AND oc2.status = 'filled'
                ) - COALESCE(bl.fill_size_usdc, 0) > 0.02
             )
          )
        """
    )
    n_promoted = 0
    for row in cur.fetchall():
        result = finalize_plan_from_chunks(conn, cur, row["plan_id"])
        if result.get("bet_updated"):
            n_promoted += 1

    if n_chunks or n_promoted:
        logger.info(
            "Resynced %d MAKER chunk fill(s); promoted %d cancelled bet(s)",
            n_chunks,
            n_promoted,
        )
    return n_promoted


def _reconcile_stale_twap_plans(conn: sqlite3.Connection) -> int:
    """Finalize orphaned TWAP plans whose kickoff has passed.

    If a plan is still 'pending' or 'executing' but the kickoff was > 2h ago,
    the daemon must have missed it. Aggregate whatever filled and update
    the bet_ledger row accordingly.

    Returns the number of plans finalized.
    """
    cur = conn.cursor()
    try:
        cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='order_plans'")
        if not cur.fetchone():
            return 0
    except Exception:
        return 0

    now_iso = datetime.now(timezone.utc).isoformat()
    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()

    cur.execute(
        """
        SELECT plan_id, bet_ledger_id FROM order_plans
        WHERE status IN ('pending', 'executing')
          AND kickoff_at IS NOT NULL
          AND kickoff_at < ?
        """,
        (two_hours_ago,),
    )
    stale_plans = cur.fetchall()
    n_finalized = 0

    for plan in stale_plans:
        plan_id = plan["plan_id"]
        bet_id = plan["bet_ledger_id"]

        # Sync MAKER fills from CLOB before cancelling — TWAP limits fill as maker.
        try:
            from src.integrations.polymarket import PolymarketClient
            from src.integrations.polymarket.clob_fills import (
                fetch_all_clob_trades,
                finalize_plan_from_chunks,
                index_fills_by_order_id,
                sync_placed_chunk_fills,
            )
            pm = PolymarketClient()
            trades = fetch_all_clob_trades(pm)
            fills = index_fills_by_order_id(trades)
            sync_placed_chunk_fills(conn, cur, fills, plan_id=plan_id)
            finalize_plan_from_chunks(conn, cur, plan_id)
            n_finalized += 1
            continue
        except Exception as exc:
            logger.warning(f"TWAP plan #{plan_id}: CLOB fill sync failed, falling back: {exc}")

        # Fallback: cancel unfilled chunks without CLOB sync (legacy path)
        cur.execute(
            "UPDATE order_chunks SET status = 'cancelled' WHERE plan_id = ? AND status IN ('pending', 'placed')",
            (plan_id,),
        )

        # Aggregate filled chunks
        cur.execute(
            """
            SELECT COUNT(*) as cnt,
                   COALESCE(SUM(fill_size_usdc), 0) as total_usdc,
                   COALESCE(SUM(fill_price * fill_size_usdc), 0) as weighted_price
            FROM order_chunks
            WHERE plan_id = ? AND status = 'filled'
            """,
            (plan_id,),
        )
        agg = cur.fetchone()
        filled_count = agg["cnt"] if agg else 0
        total_filled_usdc = agg["total_usdc"] if agg else 0
        weighted_price_sum = agg["weighted_price"] if agg else 0

        avg_fill = (weighted_price_sum / total_filled_usdc) if total_filled_usdc > 0 else None
        final_status = "completed" if filled_count > 0 else "cancelled"

        cur.execute(
            """
            UPDATE order_plans
            SET status = ?, chunks_filled = ?, filled_size_usdc = ?, avg_fill_price = ?, updated_at = ?
            WHERE plan_id = ?
            """,
            (final_status, filled_count, total_filled_usdc, avg_fill, now_iso, plan_id),
        )

        # Update bet_ledger
        if bet_id:
            if total_filled_usdc > 0:
                cur.execute(
                    """
                    UPDATE bet_ledger
                    SET status = 'filled', filled_at = ?, fill_price = ?, fill_size_usdc = ?
                    WHERE bet_id = ? AND status IN ('proposed', 'cancelled', 'placed')
                    """,
                    (now_iso, avg_fill, total_filled_usdc, bet_id),
                )
            else:
                cur.execute(
                    "UPDATE bet_ledger SET status = 'cancelled' WHERE bet_id = ? AND status = 'proposed'",
                    (bet_id,),
                )

        n_finalized += 1

    if n_finalized > 0:
        conn.commit()
        logger.info(f"Reconciled {n_finalized} stale TWAP plans (kickoff passed)")

    return n_finalized


def reconcile_pending_bets(
    conn: Optional[sqlite3.Connection] = None,
    poly_client: Optional[PolymarketClient] = None,
) -> Dict[str, Any]:
    """Walk all unsettled bets and try to mark them settled.

    Returns:
        {
            "n_checked": int,
            "n_settled": int,
            "n_still_pending": int,
            "n_twap_finalized": int,
            "errors": [(bet_id, message), ...]
        }
    """
    own_conn = False
    if conn is None:
        from src.data.database import get_connection
        conn = get_connection()
        own_conn = True

    if poly_client is None:
        poly_client = PolymarketClient()

    # Pre-pass: hygiene. Mark stale `proposed` rows that will never execute,
    # and propagate plan outcomes to bets stuck in 'proposed'. Then recover
    # missed MAKER fills, then finalise stale TWAP plans so bet_ledger rows
    # become 'filled' and eligible for settlement below.
    n_stale_proposed = 0
    n_maker_resynced = 0
    n_twap_finalized = 0
    try:
        n_stale_proposed = _finalize_stale_proposed_bets(conn)
    except Exception as exc:
        logger.warning(f"Stale-proposed finalisation failed: {exc}")
    try:
        n_maker_resynced = _resync_missed_maker_fills(conn)
    except Exception as exc:
        logger.warning(f"MAKER fill resync failed: {exc}")
    try:
        n_twap_finalized = _reconcile_stale_twap_plans(conn)
    except Exception as exc:
        logger.warning(f"TWAP plan reconciliation failed: {exc}")

    bets = _fetch_unsettled_bets(conn)
    n_checked = len(bets)
    n_settled = 0
    errors: List[Any] = []

    for bet in bets:
        market_id = bet["polymarket_market_id"]
        token_id = bet["polymarket_token_id"]
        side_label = bet["side_label"] or ""

        settle_outcome: Optional[int] = None
        resolution_method: Optional[str] = None

        # Fetch Gamma market once per bet — used by Path 1 for resolution
        # AND by Path 2 for format inference (so we can pick the right
        # kickoff-buffer duration for ambiguous `crint`-prefix slugs).
        gamma_market = _fetch_gamma_market(str(market_id)) if market_id else None

        # Path 1: Gamma `closed: true` - canonical resolution signal
        if gamma_market is not None:
            gamma_result = _resolve_via_gamma(gamma_market, side_label)
            if gamma_result is not None:
                settle_outcome, _resolved_price = gamma_result
                resolution_method = "gamma"

        # Path 2: prices-history fallback. Two stacked time gates protect
        # against in-play price spikes settling a still-live match:
        #   (a) proposal-age sanity floor (cheap, format-agnostic)
        #   (b) kickoff-age buffer sized to format duration (primary defense)
        if settle_outcome is None and token_id:
            try:
                resp = poly_client.get_prices_history(
                    token_id=token_id, interval="all", fidelity=60,
                )
                history = resp.get("history") if isinstance(resp, dict) else resp
            except Exception as exc:
                errors.append((bet["bet_id"], f"prices-history failed: {exc}"))
                continue
            last_price = _last_price_from_history(history or [])
            if last_price is None:
                continue

            hours_since_proposal = _hours_since(bet["proposed_at"])
            proposal_buffer_ok = (
                hours_since_proposal is None
                or hours_since_proposal >= MIN_HOURS_SINCE_PROPOSAL_FOR_PRICE_FALLBACK
            )
            if not proposal_buffer_ok:
                continue

            kickoff_at = bet["kickoff_at"] if "kickoff_at" in bet.keys() else None
            if not kickoff_at:
                kickoff_at = _derive_kickoff_iso_from_fixture_key(bet["fixture_key"])
            hours_since_kickoff = _hours_since(kickoff_at)

            if hours_since_kickoff is not None:
                fmt = _format_from_bet(bet["fixture_key"], gamma_market)
                min_hours_kickoff = MIN_HOURS_SINCE_KICKOFF_BY_FORMAT.get(
                    (fmt or "").upper(), DEFAULT_MIN_HOURS_SINCE_KICKOFF
                )
                if hours_since_kickoff < min_hours_kickoff:
                    logger.debug(
                        f"  bet_id={bet['bet_id']} skipping price-history fallback: "
                        f"only {hours_since_kickoff:.2f}h since kickoff, need "
                        f">= {min_hours_kickoff:.1f}h for {fmt or 'unknown'} format"
                    )
                    continue
            else:
                # No kickoff to anchor to (no kickoff_at AND no parseable
                # date in fixture_key). Tighten the proposal-age gate so
                # we don't silently fall back to the loose 6h floor.
                if (
                    hours_since_proposal is None
                    or hours_since_proposal < MIN_HOURS_SINCE_PROPOSAL_FALLBACK_NO_KICKOFF
                ):
                    logger.debug(
                        f"  bet_id={bet['bet_id']} skipping price-history fallback: "
                        f"no kickoff anchor and only {hours_since_proposal}h since proposal "
                        f"(need >= {MIN_HOURS_SINCE_PROPOSAL_FALLBACK_NO_KICKOFF:.1f}h)"
                    )
                    continue

            if last_price >= SETTLED_PRICE_HIGH:
                settle_outcome = 1
                resolution_method = "price-history-high"
            elif last_price <= SETTLED_PRICE_LOW:
                settle_outcome = 0
                resolution_method = "price-history-low"

        if settle_outcome is None:
            continue

        pnl = _compute_pnl_for_settled_bet(
            fill_price=bet["fill_price"],
            fill_size_usdc=bet["fill_size_usdc"],
            settle_outcome=float(settle_outcome),
        )

        try:
            # For paper bets, snapshot the LIVE strategy bankroll AT
            # SETTLEMENT TIME (starting + cumsum of all pnl already
            # settled for this strategy + this bet's pnl). We can't just
            # use bankroll_at_proposal because if multiple bets are
            # placed when the bankroll is $1000 and they settle in
            # arbitrary order, each one would have stale bankroll_at_proposal.
            bet_kind = bet["bet_kind"] if "bet_kind" in bet.keys() else "real"
            bankroll_after = None
            if bet_kind == "paper" and pnl is not None:
                strategy_label = bet["strategy_label"]
                if strategy_label:
                    cur2 = conn.cursor()
                    cur2.execute(
                        """
                        SELECT COALESCE(SUM(pnl_realised_usdc), 0.0)
                        FROM bet_ledger
                        WHERE bet_kind = 'paper'
                          AND strategy_label = ?
                          AND status = 'settled'
                          AND bet_id != ?
                        """,
                        (strategy_label, bet["bet_id"]),
                    )
                    prior_pnl = float(cur2.fetchone()[0] or 0.0)
                    from src.integrations.polymarket.paper_strategies import get_strategy
                    strat = get_strategy(strategy_label)
                    starting = strat.starting_bankroll_usdc if strat else 1000.0
                    bankroll_after = starting + prior_pnl + float(pnl)

            now_iso = _utc_now_iso()
            conn.execute(
                """
                UPDATE bet_ledger
                SET status = 'settled',
                    settled_at = ?,
                    settle_outcome = ?,
                    pnl_realised_usdc = ?,
                    bankroll_after_settle = COALESCE(?, bankroll_after_settle),
                    reconciled_at = ?
                WHERE bet_id = ?
                """,
                (now_iso, int(settle_outcome), pnl, bankroll_after, now_iso, bet["bet_id"]),
            )
            conn.commit()
            n_settled += 1
            tag = "[PAPER]" if bet_kind == "paper" else ""
            logger.info(
                f"Reconciled bet_id={bet['bet_id']} {tag} via={resolution_method} "
                f"market={bet['market_type']} side={bet['side_label']} "
                f"settle={settle_outcome} pnl=${pnl}"
            )
        except Exception as exc:
            errors.append((bet["bet_id"], f"DB update failed: {exc}"))

    if own_conn:
        conn.close()

    return {
        "n_checked": n_checked,
        "n_settled": n_settled,
        "n_still_pending": n_checked - n_settled,
        "n_twap_finalized": n_twap_finalized,
        "n_maker_resynced": n_maker_resynced,
        "n_stale_proposed_finalized": n_stale_proposed,
        "errors": errors,
    }
