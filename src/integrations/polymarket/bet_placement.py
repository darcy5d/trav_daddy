"""Wave 5 Phase 6: end-to-end bet placement orchestration.

`place_bet()` is the single entry point used by every Flask route that
writes to Polymarket. Sequence:

    1. risk_gate.can_place_bet(...) -> reject if any cap exceeded.
    2. Insert "proposed" row in bet_ledger.
    3. PolymarketClient.place_market_order(...) -> CLOB write.
    4. Update ledger row to "placed" (or "filled" if synchronous fill).
    5. On exception: ledger row gets status='errored' and error_message.

Wave 5.9 adds `place_bet_twap()` for thin markets: instead of an immediate
FOK order, it writes an order_plan + pre-computed chunk schedule to the DB.
The post-toss daemon picks up pending plans and executes them over time.

The function is intentionally serial (no async); CLOB calls are
~100-500ms each and we want strict ordering so risk-gate decisions
read consistent state.
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.risk_gate import can_place_bet
from src.integrations.polymarket.order_audit import (
    record_order_error,
    record_order_filled,
    record_order_placed,
)
from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE

logger = logging.getLogger(__name__)

POLYMARKET_MIN_ORDER_USDC = 5.0
POLYMARKET_MIN_SHARES = 5.0  # Polymarket enforces >= 5 shares per order


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _uncapped_kelly_stake(
    bankroll_at_proposal: Optional[float],
    model_prob: Optional[float],
    market_price_at_proposal: Optional[float],
    strategy_label: Optional[str],
) -> Optional[float]:
    """Numeric uncapped Kelly stake (before the kelly_fraction_cap) for the
    risk dashboard's "risk trimmed by the cap" metric. Reuses the same
    derivation as the display layer so the persisted value matches the UI.
    Returns None when inputs are insufficient (kept non-fatal)."""
    try:
        from src.integrations.polymarket.sizing import compute_sizing_context

        ctx = compute_sizing_context({
            "bankroll_at_proposal": bankroll_at_proposal,
            "model_prob": model_prob,
            "market_price_at_proposal": market_price_at_proposal,
            "strategy_label": strategy_label,
        })
        return ctx.get("kelly_uncapped_stake") if ctx else None
    except Exception:
        return None


def place_bet(
    *,
    fixture_key: str,
    match_id: Optional[int],
    market_type: str,
    polymarket_market_id: str,
    polymarket_token_id: str,
    side_label: str,
    model_prob: float,
    market_price_at_proposal: float,
    side: str,
    size_usdc: float,
    requested_mode: str = "manual",
    poly_client: Optional[PolymarketClient] = None,
    strategy_label: Optional[str] = None,
    bankroll_at_proposal: Optional[float] = None,
    phase: Optional[str] = None,
    xi_signature: Optional[str] = None,
    toss_winner_team_id: Optional[int] = None,
    toss_chose_to: Optional[str] = None,
    kickoff_at: Optional[str] = None,
    model_snapshot: Optional[str] = None,
) -> Dict[str, Any]:
    """Run risk gate + place market order + insert/update bet ledger row.

    Returns: a dict with keys `success`, `bet_id`, `status`, `reason`, and
    `placement_response` (the raw CLOB API response).

    Wave 5.8: strategy_label threads through to the risk gate (enforces
    per-strategy cap + BETTING_LIVE_STRATEGIES whitelist) and is written to
    bet_ledger alongside bet_kind='real' so paper/live can be compared in SQL.
    The paper-parity metadata (phase/xi_signature/toss_*) is optional.
    """
    edge_pp = (model_prob - market_price_at_proposal) * 100.0

    # Step 1: risk gate
    decision = can_place_bet(
        size_usdc, market_type, edge_pp, requested_mode,
        strategy_label=strategy_label,
        kickoff_at=kickoff_at,
        fixture_key=fixture_key,
    )
    if not decision.allowed:
        return {
            "success": False,
            "bet_id": None,
            "status": "rejected",
            "reason": decision.reason,
            "placement_response": None,
            "cap_remaining_today": decision.cap_remaining_today,
            "cap_remaining_deposit": decision.cap_remaining_deposit,
        }

    from src.data.database import get_connection

    proposed_at = _utc_now_iso()
    fees_est = round(size_usdc * POLYMARKET_TAKER_FEE, 4)
    kelly_uncapped_stake = _uncapped_kelly_stake(
        bankroll_at_proposal, model_prob, market_price_at_proposal, strategy_label
    )

    # Step 2: insert proposed row
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO bet_ledger (
                proposed_at, match_id, fixture_key, market_type,
                polymarket_market_id, polymarket_token_id,
                side_label, model_prob, market_price_at_proposal, edge_pp,
                side, size_usdc, fees_estimated_usdc,
                status, mode, bet_kind, strategy_label,
                bankroll_at_proposal, phase, xi_signature,
                toss_winner_team_id, toss_chose_to, kickoff_at, model_snapshot,
                kelly_uncapped_stake
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, 'real', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposed_at, match_id, fixture_key, market_type,
                polymarket_market_id, polymarket_token_id,
                side_label, model_prob, market_price_at_proposal, edge_pp,
                side.upper(), size_usdc, fees_est,
                requested_mode, strategy_label,
                bankroll_at_proposal, phase, xi_signature,
                toss_winner_team_id, toss_chose_to, kickoff_at, model_snapshot,
                kelly_uncapped_stake,
            ),
        )
        bet_id = cur.lastrowid
        conn.commit()
    except Exception as exc:
        conn.close()
        logger.error(f"Failed to insert proposed bet: {exc}")
        return {
            "success": False,
            "bet_id": None,
            "status": "errored",
            "reason": f"DB insert failed: {exc}",
            "placement_response": None,
        }

    # Step 3: place order
    if poly_client is None:
        poly_client = PolymarketClient()

    placement_response: Optional[Dict[str, Any]] = None
    try:
        placement_response = poly_client.place_market_order(
            token_id=polymarket_token_id,
            side=side,
            amount_usdc=size_usdc,
            order_type="FOK",
        )
    except Exception as exc:
        logger.error(f"CLOB order failed: {exc}")
        error_category = _classify_order_error(exc)
        cur.execute(
            """
            UPDATE bet_ledger
            SET status = 'errored', error_message = ?, error_category = ?
            WHERE bet_id = ?
            """,
            (str(exc), error_category, bet_id),
        )
        conn.commit()
        record_order_error(
            None,
            bet_id=bet_id,
            token_id=polymarket_token_id,
            side=side.upper(),
            order_kind="fok",
            error_category=error_category,
            error_message=str(exc),
            conn=conn,
        )
        conn.close()
        return {
            "success": False,
            "bet_id": bet_id,
            "status": "errored",
            "reason": f"CLOB order failed: {exc}",
            "placement_response": None,
        }

    # Step 4: update ledger row
    placed_at = _utc_now_iso()
    order_id = (placement_response or {}).get("orderID") or (placement_response or {}).get("orderId")
    fill_price = None
    fill_size_usdc = None
    new_status = "placed"
    # FOK orders either fully fill instantly or are cancelled. Detect via
    # `status` / `success` and parse fill details.
    # Wave 5.8: v2 CLOB response shape is {'makingAmount': str, 'takingAmount': str,
    # 'status': 'matched', 'success': bool, 'orderID': str, 'transactionsHashes': [...]}.
    # For BUY: makingAmount = USDC spent, takingAmount = outcome tokens received.
    # fill_price (USDC per share) = makingAmount / takingAmount.
    # The older 'fillPrice'/'fillAmount' fields are v1-era aliases; keep them
    # as fallbacks so local tests with synthetic v1-shaped dicts still pass.
    if isinstance(placement_response, dict):
        if (
            placement_response.get("status") == "matched"
            or placement_response.get("filled")
            or placement_response.get("success")
        ):
            new_status = "filled"
            making = placement_response.get("makingAmount")
            taking = placement_response.get("takingAmount")
            try:
                if making is not None and taking is not None:
                    fill_size_usdc = float(making)
                    taking_f = float(taking)
                    if taking_f > 0:
                        fill_price = fill_size_usdc / taking_f
            except (TypeError, ValueError):
                pass
            # Fall back to legacy v1 field names if v2 fields aren't present
            if fill_price is None:
                fill_price = placement_response.get("fillPrice") or placement_response.get("avg_price")
            if fill_size_usdc is None:
                fill_size_usdc = placement_response.get("fillAmount") or size_usdc

    cur.execute(
        """
        UPDATE bet_ledger
        SET status = ?,
            placed_at = ?,
            polymarket_order_id = ?,
            filled_at = CASE WHEN ?='filled' THEN ? ELSE NULL END,
            fill_price = ?,
            fill_size_usdc = ?
        WHERE bet_id = ?
        """,
        (
            new_status,
            placed_at,
            order_id,
            new_status, placed_at,
            fill_price,
            fill_size_usdc,
            bet_id,
        ),
    )
    conn.commit()

    if order_id:
        record_order_placed(
            order_id,
            bet_id=bet_id,
            token_id=polymarket_token_id,
            side=side.upper(),
            order_kind="fok",
            size_usdc=size_usdc,
            posted_at=placed_at,
            conn=conn,
        )
        if new_status == "filled":
            record_order_filled(
                order_id,
                fill_usdc=fill_size_usdc if fill_size_usdc is not None else size_usdc,
                fill_price=fill_price,
                filled_at=placed_at,
                conn=conn,
            )
    conn.close()

    return {
        "success": True,
        "bet_id": bet_id,
        "status": new_status,
        "reason": "placed" if new_status == "placed" else "filled",
        "placement_response": placement_response,
    }


def _classify_order_error(exc: Exception) -> str:
    """Map a CLOB exception to a coarse error_category tag used by reconcile.

    Tags are stable so dashboards can group by them: order_version_mismatch,
    unfillable, balance, signature, net, unknown.
    """
    msg = str(exc).lower()
    if "order_version_mismatch" in msg or "version mismatch" in msg:
        return "order_version_mismatch"
    if "couldn't be fully" in msg or "not enough" in msg or "unfillable" in msg or "no match" in msg:
        return "unfillable"
    if "insufficient" in msg or "balance" in msg or "allowance" in msg:
        return "balance"
    if "signature" in msg or "signer" in msg or "auth" in msg:
        return "signature"
    if "timeout" in msg or "connect" in msg or "network" in msg or "503" in msg or "502" in msg:
        return "net"
    return "unknown"


def reduce_position(
    *,
    strategy_label: str,
    fixture_key: str,
    side_label: str,
    polymarket_token_id: str,
    reduce_usdc: float,
    current_price: float,
    poly_client: Optional[PolymarketClient] = None,
    conn: Optional[Any] = None,
    dry_run: bool = False,
    reason: str = "rebalance",
    phase: str = "pre_toss",
) -> Dict[str, Any]:
    """De-risk an existing live position by SELLing part (or all) of it.

    Used by the XI-aware rebalancer when an updated lineup drops the model
    edge: instead of holding stale exposure we sell shares at the current
    price. `reduce_usdc` is expressed in ENTRY-COST terms (how much of the
    originally-staked USDC to take off the table), so it composes cleanly
    with the exposure math the risk gate uses.

    Mechanics:
      * Gather filled BUY rows for (strategy, fixture, side, token).
      * Sell a pro-rata fraction `f = reduce_usdc / total_entry_cost` of the
        held shares at `current_price` via a SELL limit order (real only).
      * Decrement each BUY row's `fill_size_usdc` by its share of the cut so
        open exposure (and its eventual settlement pnl) shrinks correctly;
        fully-consumed rows are marked settled.
      * Insert one SELL adjustment row carrying the realized pnl.

    Returns a dict: success, shares_sold, proceeds_usdc, entry_cost_removed,
    realized_pnl_usdc, sell_bet_id, order_response, is_simulated.
    """
    from src.data.database import get_connection

    own_conn = False
    if conn is None:
        conn = get_connection()
        own_conn = True

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT bet_id, fill_price, fill_size_usdc
            FROM bet_ledger
            WHERE COALESCE(bet_kind, 'real') = 'real'
              AND status = 'filled'
              AND side = 'BUY'
              AND strategy_label = ?
              AND fixture_key = ?
              AND side_label = ?
              AND polymarket_token_id = ?
              AND fill_price IS NOT NULL
              AND fill_size_usdc IS NOT NULL
              AND fill_size_usdc > 0
            ORDER BY filled_at ASC
            """,
            (strategy_label, fixture_key, side_label, polymarket_token_id),
        )
        rows = cur.fetchall()
        total_entry = sum(float(r["fill_size_usdc"]) for r in rows)
        if total_entry <= 0:
            return {
                "success": False,
                "reason": "no-open-position-to-reduce",
                "shares_sold": 0.0,
            }
        if current_price <= 0:
            return {"success": False, "reason": "invalid-current-price"}

        reduce_usdc = min(float(reduce_usdc), total_entry)
        if reduce_usdc <= 0:
            return {"success": False, "reason": "non-positive-reduce-amount"}

        f = reduce_usdc / total_entry
        total_shares = sum(float(r["fill_size_usdc"]) / float(r["fill_price"]) for r in rows)
        shares_to_sell = round(total_shares * f, 4)
        if shares_to_sell <= 0:
            return {"success": False, "reason": "rounded-to-zero-shares"}

        proceeds = shares_to_sell * current_price
        fee = proceeds * POLYMARKET_TAKER_FEE
        realized_pnl = round(proceeds - reduce_usdc - fee, 4)

        # dry_run is a pure preview: never touches the CLOB or the ledger.
        if dry_run:
            logger.info(
                f"Rebalance SELL DRY-RUN (no order/no DB): {strategy_label} {fixture_key} "
                f"{side_label} shares={shares_to_sell:.4f} entry_removed=${reduce_usdc:.2f} "
                f"pnl=${realized_pnl:+.2f}"
            )
            return {
                "success": True,
                "dry_run": True,
                "is_simulated": True,
                "shares_sold": shares_to_sell,
                "proceeds_usdc": round(proceeds, 4),
                "entry_cost_removed": round(reduce_usdc, 4),
                "realized_pnl_usdc": realized_pnl,
                "sell_bet_id": None,
                "order_response": None,
                "reason": reason,
            }

        # --- Place the SELL order ---
        order_response: Optional[Dict[str, Any]] = None
        if poly_client is None:
            poly_client = PolymarketClient()
        try:
            order_response = poly_client.place_limit_order(
                token_id=polymarket_token_id,
                side="SELL",
                price=round(current_price, 4),
                size_shares=shares_to_sell,
            )
            logger.info(
                f"Rebalance SELL placed: {strategy_label} {fixture_key} {side_label} "
                f"shares={shares_to_sell:.4f} @ {current_price:.4f} "
                f"entry_removed=${reduce_usdc:.2f} pnl=${realized_pnl:+.2f}"
            )
        except Exception as exc:
            logger.error(f"Rebalance SELL failed ({strategy_label}/{fixture_key}): {exc}")
            return {"success": False, "reason": f"SELL order failed: {exc}"}

        now = _utc_now_iso()

        # --- Decrement BUY rows' open stake pro-rata ---
        for r in rows:
            row_cut = float(r["fill_size_usdc"]) * f
            new_fill = round(float(r["fill_size_usdc"]) - row_cut, 6)
            if new_fill <= 1e-6:
                cur.execute(
                    """
                    UPDATE bet_ledger
                    SET fill_size_usdc = 0,
                        status = 'settled',
                        settled_at = ?,
                        pnl_realised_usdc = COALESCE(pnl_realised_usdc, 0.0)
                    WHERE bet_id = ?
                    """,
                    (now, r["bet_id"]),
                )
            else:
                cur.execute(
                    "UPDATE bet_ledger SET fill_size_usdc = ? WHERE bet_id = ?",
                    (new_fill, r["bet_id"]),
                )

        # --- Insert the SELL adjustment row (carries the realized pnl) ---
        # mode must satisfy CHECK(mode IN ('manual','auto')); model_prob/edge_pp
        # are NOT NULL so we stamp the sell price / 0 edge. Rebalance sells are
        # identified by side='SELL' AND cashout_triggered_at IS NULL.
        cur.execute(
            """
            INSERT INTO bet_ledger (
                proposed_at, placed_at, filled_at, settled_at,
                fixture_key, market_type,
                polymarket_market_id, polymarket_token_id,
                side_label, model_prob, market_price_at_proposal, edge_pp,
                side, size_usdc, fees_estimated_usdc,
                fill_price, fill_size_usdc,
                status, mode, bet_kind, strategy_label, phase,
                pnl_realised_usdc, cancel_reason
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'SELL', ?, ?, ?, ?, 'settled', 'auto', 'real', ?, ?, ?, 'rebalance')
            """,
            (
                now, now, now, now,
                fixture_key, "moneyline",
                "", polymarket_token_id,
                side_label, current_price, current_price, 0.0,
                reduce_usdc, round(fee, 4),
                current_price, proceeds,
                strategy_label, phase,
                realized_pnl,
            ),
        )
        sell_bet_id = cur.lastrowid
        conn.commit()

        order_id = (order_response or {}).get("orderID") or (order_response or {}).get("orderId")
        if order_id:
            try:
                record_order_placed(
                    order_id,
                    bet_id=sell_bet_id,
                    token_id=polymarket_token_id,
                    side="SELL",
                    order_kind="rebalance_sell",
                    limit_price=round(current_price, 4),
                    size_usdc=proceeds,
                    size_shares=shares_to_sell,
                    posted_at=now,
                    conn=conn,
                )
                record_order_filled(
                    order_id,
                    fill_usdc=proceeds,
                    fill_price=current_price,
                    filled_at=now,
                    conn=conn,
                )
            except Exception as exc:
                logger.warning(f"Audit record for rebalance SELL {order_id} failed: {exc}")

        return {
            "success": True,
            "shares_sold": shares_to_sell,
            "proceeds_usdc": round(proceeds, 4),
            "entry_cost_removed": round(reduce_usdc, 4),
            "realized_pnl_usdc": realized_pnl,
            "sell_bet_id": sell_bet_id,
            "order_response": order_response,
            "is_simulated": False,
            "dry_run": False,
            "reason": reason,
        }
    finally:
        if own_conn:
            conn.close()


def place_bet_twap(
    *,
    fixture_key: str,
    match_id: Optional[int],
    market_type: str,
    polymarket_market_id: str,
    polymarket_token_id: str,
    side_label: str,
    model_prob: float,
    market_price_at_proposal: float,
    side: str,
    size_usdc: float,
    max_acceptable_price: float,
    base_price: float,
    best_ask_size: float,
    requested_mode: str = "auto",
    strategy_label: Optional[str] = None,
    bankroll_at_proposal: Optional[float] = None,
    phase: Optional[str] = None,
    xi_signature: Optional[str] = None,
    toss_winner_team_id: Optional[int] = None,
    toss_chose_to: Optional[str] = None,
    kickoff_at: Optional[str] = None,
    model_snapshot: Optional[str] = None,
    price_step_pp: Optional[float] = None,
) -> Dict[str, Any]:
    """Write an order plan to the DB for TWAP execution by the daemon.

    Instead of placing a FOK market order immediately, this function:
      1. Runs the risk gate (same as place_bet).
      2. Inserts a 'proposed' row in bet_ledger (bet is tracked but not yet on-chain).
      3. Computes chunk sizing and price escalation schedule.
      4. Writes to order_plans + order_chunks tables.

    The post-toss daemon picks up pending plans each tick and places
    limit orders chunk-by-chunk.

    Returns: dict with success, bet_id, plan_id, chunks_total.
    """
    import os
    edge_pp = (model_prob - market_price_at_proposal) * 100.0

    # Step 1: risk gate
    decision = can_place_bet(
        size_usdc, market_type, edge_pp, requested_mode,
        strategy_label=strategy_label,
        kickoff_at=kickoff_at,
        fixture_key=fixture_key,
    )
    if not decision.allowed:
        return {
            "success": False,
            "bet_id": None,
            "plan_id": None,
            "status": "rejected",
            "reason": decision.reason,
        }

    from src.data.database import get_connection

    proposed_at = _utc_now_iso()
    fees_est = round(size_usdc * POLYMARKET_TAKER_FEE, 4)
    kelly_uncapped_stake = _uncapped_kelly_stake(
        bankroll_at_proposal, model_prob, market_price_at_proposal, strategy_label
    )

    # Step 2: insert proposed bet_ledger row (status='proposed', not placed yet)
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO bet_ledger (
                proposed_at, match_id, fixture_key, market_type,
                polymarket_market_id, polymarket_token_id,
                side_label, model_prob, market_price_at_proposal, edge_pp,
                side, size_usdc, fees_estimated_usdc,
                status, mode, bet_kind, strategy_label,
                bankroll_at_proposal, phase, xi_signature,
                toss_winner_team_id, toss_chose_to, kickoff_at, model_snapshot,
                kelly_uncapped_stake
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, 'real', ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposed_at, match_id, fixture_key, market_type,
                polymarket_market_id, polymarket_token_id,
                side_label, model_prob, market_price_at_proposal, edge_pp,
                side.upper(), size_usdc, fees_est,
                requested_mode, strategy_label,
                bankroll_at_proposal, phase, xi_signature,
                toss_winner_team_id, toss_chose_to, kickoff_at, model_snapshot,
                kelly_uncapped_stake,
            ),
        )
        bet_id = cur.lastrowid
        conn.commit()
    except Exception as exc:
        conn.close()
        logger.error(f"Failed to insert proposed TWAP bet: {exc}")
        return {
            "success": False,
            "bet_id": None,
            "plan_id": None,
            "status": "errored",
            "reason": f"DB insert failed: {exc}",
        }

    # Step 3: compute chunk sizing
    if price_step_pp is None:
        price_step_pp = float(os.getenv("TWAP_PRICE_STEP_PP", "2"))

    chunk_size = min(size_usdc, best_ask_size * 0.5 * base_price)
    chunk_size = max(chunk_size, POLYMARKET_MIN_ORDER_USDC)
    chunks_total = max(1, math.ceil(size_usdc / chunk_size))
    chunk_size = round(size_usdc / chunks_total, 4)

    # Derive price_step dynamically: walk from base_price to max_acceptable_price
    price_range = max_acceptable_price - base_price
    if chunks_total > 1 and price_range > 0:
        computed_step_pp = (price_range * 100.0) / chunks_total
    else:
        computed_step_pp = price_step_pp

    # Step 4: write order_plan
    try:
        cur.execute(
            """
            INSERT INTO order_plans (
                bet_ledger_id, fixture_key, strategy_label, token_id, side,
                total_size_usdc, chunk_size_usdc, chunks_total,
                max_acceptable_price, base_price, price_step_pp,
                kickoff_at, model_prob, market_price_at_plan,
                status, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending', ?)
            """,
            (
                bet_id, fixture_key, strategy_label or "",
                polymarket_token_id, side.upper(),
                size_usdc, chunk_size, chunks_total,
                max_acceptable_price, base_price, computed_step_pp,
                kickoff_at, model_prob, market_price_at_proposal,
                proposed_at,
            ),
        )
        plan_id = cur.lastrowid

        # Step 5: pre-compute chunk schedule
        for i in range(chunks_total):
            limit_price = base_price + ((i + 1) * computed_step_pp / 100.0)
            limit_price = min(limit_price, max_acceptable_price)
            limit_price = round(limit_price, 4)
            this_chunk_usdc = chunk_size if i < chunks_total - 1 else round(size_usdc - chunk_size * (chunks_total - 1), 4)
            size_shares = max(round(this_chunk_usdc / limit_price, 4), POLYMARKET_MIN_SHARES) if limit_price > 0 else 0

            cur.execute(
                """
                INSERT INTO order_chunks (
                    plan_id, chunk_index, limit_price, size_usdc, size_shares, status
                ) VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (plan_id, i, limit_price, this_chunk_usdc, size_shares),
            )

        conn.commit()
    except Exception as exc:
        conn.rollback()
        conn.close()
        logger.error(f"Failed to write TWAP order plan: {exc}")
        return {
            "success": False,
            "bet_id": bet_id,
            "plan_id": None,
            "status": "errored",
            "reason": f"Order plan write failed: {exc}",
        }

    conn.close()

    logger.info(
        f"TWAP plan #{plan_id} created: {chunks_total} chunks x ${chunk_size:.2f} "
        f"for {side_label} @ base={base_price:.4f} -> max={max_acceptable_price:.4f} "
        f"step={computed_step_pp:.2f}pp"
    )

    return {
        "success": True,
        "bet_id": bet_id,
        "plan_id": plan_id,
        "status": "twap_queued",
        "reason": f"TWAP plan queued ({chunks_total} chunks)",
        "chunks_total": chunks_total,
        "chunk_size_usdc": chunk_size,
        "price_step_pp": computed_step_pp,
    }
