"""Wave 5 Phase 6: end-to-end bet placement orchestration.

`place_bet()` is the single entry point used by every Flask route that
writes to Polymarket. Sequence:

    1. risk_gate.can_place_bet(...) -> reject if any cap exceeded.
    2. Insert "proposed" row in bet_ledger.
    3. PolymarketClient.place_market_order(...) -> CLOB write.
    4. Update ledger row to "placed" (or "filled" if synchronous fill).
    5. On exception: ledger row gets status='errored' and error_message.

The function is intentionally serial (no async); CLOB calls are
~100-500ms each and we want strict ordering so risk-gate decisions
read consistent state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from src.integrations.polymarket import PolymarketClient
from src.integrations.polymarket.risk_gate import can_place_bet
from src.integrations.odds.polymarket_compare import POLYMARKET_TAKER_FEE

logger = logging.getLogger(__name__)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


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
                toss_winner_team_id, toss_chose_to, kickoff_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'proposed', ?, 'real', ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                proposed_at, match_id, fixture_key, market_type,
                polymarket_market_id, polymarket_token_id,
                side_label, model_prob, market_price_at_proposal, edge_pp,
                side.upper(), size_usdc, fees_est,
                requested_mode, strategy_label,
                bankroll_at_proposal, phase, xi_signature,
                toss_winner_team_id, toss_chose_to, kickoff_at,
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
        cur.execute(
            """
            UPDATE bet_ledger
            SET status = 'errored', error_message = ?
            WHERE bet_id = ?
            """,
            (str(exc), bet_id),
        )
        conn.commit()
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
    conn.close()

    return {
        "success": True,
        "bet_id": bet_id,
        "status": new_status,
        "reason": "placed" if new_status == "placed" else "filled",
        "placement_response": placement_response,
    }
