#!/usr/bin/env python3
"""Full two-way reconciliation between Polymarket CLOB trades and bet_ledger.

Fetches all paginated trades from the CLOB API, matches BUY trades to
bet_ledger.polymarket_order_id, identifies ghost bets (on-chain but missing
from DB) and unexecuted bets (in DB but not on-chain), and produces a
per-fixture + per-strategy P&L comparison report.

Usage:
    venv311/bin/python scripts/reconcile_polymarket.py --dry-run
    venv311/bin/python scripts/reconcile_polymarket.py
    venv311/bin/python scripts/reconcile_polymarket.py --fix-ghosts
    venv311/bin/python scripts/reconcile_polymarket.py --since-days 30 --output /tmp/reconcile.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "paper_trading"
END_CURSOR = "LTE="


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


FILL_MATCH_TOLERANCE = 0.05  # 5% — match orphan DB row to a TAKER trade by stake


def _trade_usdc(trade: Dict[str, Any]) -> float:
    try:
        price = float(trade.get("price") or 0)
        size = float(trade.get("size") or 0)
    except (TypeError, ValueError):
        return 0.0
    return price * size if price > 0 and size > 0 else 0.0


def _is_our_taker_trade(trade: Dict[str, Any]) -> bool:
    """True when we initiated the order (bot market orders are always TAKER).

    MAKER-side rows in /data/trades are us providing liquidity to someone
    else's order — not our cricket bets.  Indexing them as ghost buys produced
    false $2k+ stakes (e.g. lei-wor $2484 MAKER fill).
    """
    return (trade.get("trader_side") or "").upper() == "TAKER"


def _is_our_taker_buy(trade: Dict[str, Any]) -> bool:
    return _is_our_taker_trade(trade) and (trade.get("side") or "").upper() == "BUY"


def _is_our_taker_sell(trade: Dict[str, Any]) -> bool:
    return _is_our_taker_trade(trade) and (trade.get("side") or "").upper() == "SELL"


def _trade_timestamp_iso(trade: Dict[str, Any]) -> str:
    raw = trade.get("match_time") or trade.get("last_update")
    if raw is None:
        return _utc_now_iso()
    try:
        ts = int(raw)
        return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
    except (TypeError, ValueError):
        return str(raw)


def _fetch_all_trades(sdk, since_days: Optional[int] = None) -> List[Dict[str, Any]]:
    from py_clob_client_v2 import TradeParams

    params = TradeParams()
    if since_days is not None and since_days > 0:
        cutoff = datetime.now(timezone.utc) - timedelta(days=since_days)
        params.after = int(cutoff.timestamp())

    all_trades: List[Dict[str, Any]] = []
    cursor: Optional[str] = None
    page_num = 0
    while True:
        page = sdk.get_trades_paginated(params, next_cursor=cursor)
        batch = page.get("data") or page.get("trades") or []
        if isinstance(batch, list):
            all_trades.extend(t for t in batch if isinstance(t, dict))
        cursor = page.get("next_cursor")
        page_num += 1
        logger.info(f"  page {page_num}: +{len(batch)} trades (total {len(all_trades)})")
        if not cursor or cursor == END_CURSOR:
            break
    return all_trades


def _build_trade_maps(
    trades: List[Dict[str, Any]],
    known_order_ids: Optional[set] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, List[Dict[str, Any]]], Dict[str, Dict[str, Any]]]:
    """Index TAKER trades by order id; group trades by asset; index our fills."""
    from src.integrations.polymarket.clob_fills import index_fills_by_order_id

    fills_by_order = index_fills_by_order_id(trades, known_order_ids=known_order_ids)
    by_order: Dict[str, Dict[str, Any]] = {}
    by_asset: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for trade in trades:
        tid = trade.get("taker_order_id")
        if tid and _is_our_taker_trade(trade) and tid not in by_order:
            by_order[tid] = trade
        asset_id = trade.get("asset_id")
        if asset_id:
            by_asset[str(asset_id)].append(trade)
    return by_order, dict(by_asset), fills_by_order


def _pm_buy_total_on_token(
    trades_by_order: Dict[str, Dict[str, Any]],
    fills_by_order: Dict[str, Dict[str, Any]],
    token_id: str,
    tracked_order_ids: Optional[set] = None,
) -> float:
    """Sum on-chain BUY stake for a token (TAKER + our MAKER limit orders)."""
    total = sum(
        _trade_usdc(t)
        for t in trades_by_order.values()
        if _is_our_taker_buy(t) and str(t.get("asset_id") or "") == token_id
    )
    for oid, fill in fills_by_order.items():
        if tracked_order_ids is not None and oid not in tracked_order_ids:
            continue
        if str(fill.get("asset_id") or "") != token_id:
            continue
        if "maker" in (fill.get("roles") or []):
            total += float(fill.get("fill_usdc") or 0)
    return total


def _db_fill_total_on_token(
    bets: List[Dict[str, Any]],
    token_id: str,
    *,
    exclude_reconcile_ghost: bool = True,
) -> float:
    total = 0.0
    for bet in bets:
        if str(bet.get("polymarket_token_id") or "") != token_id:
            continue
        if exclude_reconcile_ghost and bet.get("strategy_label") == "RECONCILE_GHOST":
            continue
        total += float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0)
    return total


def _taker_buy_total_on_token(trades_by_order: Dict[str, Dict[str, Any]], token_id: str) -> float:
    return sum(
        _trade_usdc(t)
        for t in trades_by_order.values()
        if _is_our_taker_buy(t) and str(t.get("asset_id") or "") == token_id
    )


def _load_twap_orders_by_bet(conn) -> Dict[int, List[str]]:
    """Map bet_ledger_id -> list of TWAP chunk polymarket_order_ids."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT op.bet_ledger_id, oc.polymarket_order_id
        FROM order_chunks oc
        JOIN order_plans op ON op.plan_id = oc.plan_id
        WHERE op.bet_ledger_id IS NOT NULL
          AND oc.polymarket_order_id IS NOT NULL
        """
    )
    out: Dict[int, List[str]] = defaultdict(list)
    for row in cur.fetchall():
        out[int(row["bet_ledger_id"])].append(str(row["polymarket_order_id"]))
    return dict(out)


def _link_orphan_bets(
    conn,
    bets: List[Dict[str, Any]],
    trades_by_order: Dict[str, Dict[str, Any]],
    matched_order_ids: set,
    dry_run: bool,
) -> List[Dict[str, Any]]:
    """Attach polymarket_order_id to real rows missing it.

    1:1 match by token + fill_size (within tolerance).
    Aggregated row: one DB bet whose fill equals the sum of all unmatched
    TAKER buys on that token (e.g. gla-glo consensus row holding 4 fills).
    """
    links: List[Dict[str, Any]] = []
    orphans = [
        b for b in bets
        if not (b.get("polymarket_order_id") or "").strip()
        and b.get("polymarket_token_id")
        and b.get("strategy_label") != "RECONCILE_GHOST"
    ]
    if not orphans:
        return links

    by_token: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for oid, trade in trades_by_order.items():
        if oid in matched_order_ids or not _is_our_taker_buy(trade):
            continue
        by_token[str(trade["asset_id"])].append(trade)

    linked_bet_ids: set = set()

    for token, token_trades in by_token.items():
        available = sorted(token_trades, key=_trade_usdc)
        token_orphans = [
            b for b in orphans
            if str(b.get("polymarket_token_id")) == token and b["bet_id"] not in linked_bet_ids
        ]
        token_orphans.sort(
            key=lambda b: float(b.get("fill_size_usdc") or b.get("size_usdc") or 0)
        )

        used_oids: set = set()

        # Pass 1: 1:1 stake match
        for bet in token_orphans:
            bet_fill = float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0)
            if bet_fill <= 0:
                continue
            best_trade = None
            best_diff = float("inf")
            for trade in available:
                oid = trade["taker_order_id"]
                if oid in used_oids:
                    continue
                trade_usdc = _trade_usdc(trade)
                diff = abs(trade_usdc - bet_fill)
                if diff / bet_fill <= FILL_MATCH_TOLERANCE and diff < best_diff:
                    best_trade = trade
                    best_diff = diff
            if best_trade is None:
                continue
            oid = best_trade["taker_order_id"]
            used_oids.add(oid)
            matched_order_ids.add(oid)
            linked_bet_ids.add(bet["bet_id"])
            link = {
                "bet_id": bet["bet_id"],
                "order_id": oid,
                "fill_size_usdc": round(_trade_usdc(best_trade), 4),
                "method": "1:1_fill_match",
            }
            links.append(link)
            if dry_run:
                logger.info(
                    f"  [DRY] link bet_id={bet['bet_id']} → order={oid[:16]}... "
                    f"fill=${link['fill_size_usdc']:.2f}"
                )
            else:
                conn.execute(
                    """
                    UPDATE bet_ledger
                    SET polymarket_order_id = ?,
                        fill_price = ?,
                        fill_size_usdc = ?
                    WHERE bet_id = ?
                    """,
                    (
                        oid,
                        float(best_trade.get("price") or 0),
                        _trade_usdc(best_trade),
                        bet["bet_id"],
                    ),
                )

        # Pass 2: aggregated row — DB fill ≈ sum of remaining TAKER buys
        remaining = [t for t in available if t["taker_order_id"] not in used_oids]
        if not remaining:
            continue
        pm_sum = sum(_trade_usdc(t) for t in remaining)
        for bet in token_orphans:
            if bet["bet_id"] in linked_bet_ids:
                continue
            bet_fill = float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0)
            if pm_sum <= 0 or bet_fill <= 0:
                continue
            if abs(bet_fill - pm_sum) / pm_sum > FILL_MATCH_TOLERANCE:
                continue
            primary = max(remaining, key=_trade_usdc)
            oid = primary["taker_order_id"]
            for trade in remaining:
                matched_order_ids.add(trade["taker_order_id"])
            linked_bet_ids.add(bet["bet_id"])
            link = {
                "bet_id": bet["bet_id"],
                "order_id": oid,
                "fill_size_usdc": round(bet_fill, 4),
                "method": "aggregated_token_match",
                "n_trades_consumed": len(remaining),
            }
            links.append(link)
            if dry_run:
                logger.info(
                    f"  [DRY] link bet_id={bet['bet_id']} → aggregated {len(remaining)} "
                    f"TAKER buys (${pm_sum:.2f}) primary={oid[:16]}..."
                )
            else:
                conn.execute(
                    """
                    UPDATE bet_ledger
                    SET polymarket_order_id = ?
                    WHERE bet_id = ?
                    """,
                    (oid, bet["bet_id"]),
                )
            break

    if links and not dry_run:
        conn.commit()
    return links


def _purge_reconcile_ghosts(conn, dry_run: bool) -> int:
    cur = conn.cursor()
    cur.execute(
        "SELECT COUNT(*) FROM bet_ledger WHERE strategy_label = 'RECONCILE_GHOST'"
    )
    n = int(cur.fetchone()[0])
    if n and not dry_run:
        conn.execute("DELETE FROM bet_ledger WHERE strategy_label = 'RECONCILE_GHOST'")
        conn.commit()
        logger.info(f"Purged {n} RECONCILE_GHOST row(s)")
    elif n:
        logger.info(f"[DRY-RUN] Would purge {n} RECONCILE_GHOST row(s)")
    return n


def _load_known_order_ids(conn) -> set:
    """Order ids we have placed (bet_ledger + TWAP chunks + order_history).

    order_history is the canonical "did we place this" set; it survives
    reprice / resize that overwrite or NULL the live order_chunks.polymarket_order_id.
    """
    try:
        from src.integrations.polymarket.order_audit import all_known_order_ids
        return all_known_order_ids(conn)
    except Exception:
        cur = conn.cursor()
        ids: set = set()
        cur.execute(
            "SELECT polymarket_order_id FROM bet_ledger WHERE polymarket_order_id IS NOT NULL"
        )
        ids.update(str(r["polymarket_order_id"]) for r in cur.fetchall())
        ids.update(_load_chunk_order_ids(conn))
        return ids


def _load_chunk_order_ids(conn) -> set:
    cur = conn.cursor()
    cur.execute(
        "SELECT polymarket_order_id FROM order_chunks WHERE polymarket_order_id IS NOT NULL"
    )
    return {str(r["polymarket_order_id"]) for r in cur.fetchall()}


def _load_history_oids(conn) -> set:
    """Every order id ever placed (from order_history)."""
    cur = conn.cursor()
    try:
        return {
            str(r[0]) for r in cur.execute(
                "SELECT polymarket_order_id FROM order_history WHERE polymarket_order_id IS NOT NULL"
            ).fetchall()
        }
    except Exception:
        return set()


def _load_real_bets(conn) -> List[Dict[str, Any]]:
    cur = conn.cursor()
    cur.execute(
        """
        SELECT *
        FROM bet_ledger
        WHERE COALESCE(bet_kind, 'real') = 'real'
        ORDER BY bet_id ASC
        """
    )
    return [dict(r) for r in cur.fetchall()]


def _load_cricket_scope(conn) -> Tuple[set, set]:
    """Return (known_token_ids, known_market_ids) for cricket fixtures in bet_ledger."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT polymarket_token_id, polymarket_market_id
        FROM bet_ledger
        WHERE fixture_key LIKE 'cric%'
          AND (polymarket_token_id IS NOT NULL OR polymarket_market_id IS NOT NULL)
        """
    )
    tokens: set = set()
    markets: set = set()
    for row in cur.fetchall():
        if row["polymarket_token_id"]:
            tokens.add(str(row["polymarket_token_id"]))
        if row["polymarket_market_id"]:
            markets.add(str(row["polymarket_market_id"]))
    return tokens, markets


def _is_cricket_trade(
    trade: Dict[str, Any],
    known_tokens: set,
    known_markets: set,
) -> bool:
    asset = str(trade.get("asset_id") or "")
    market = str(trade.get("market") or "")
    return asset in known_tokens or market in known_markets


def _load_token_fixture_map(conn) -> Dict[str, Dict[str, Any]]:
    """Map polymarket_token_id -> {fixture_key, market_id, side_label}."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT polymarket_token_id, fixture_key, polymarket_market_id, side_label
        FROM bet_ledger
        WHERE polymarket_token_id IS NOT NULL
          AND polymarket_token_id != ''
        """
    )
    out: Dict[str, Dict[str, Any]] = {}
    for row in cur.fetchall():
        token = str(row["polymarket_token_id"])
        if token not in out:
            out[token] = {
                "fixture_key": row["fixture_key"],
                "polymarket_market_id": row["polymarket_market_id"],
                "side_label": row["side_label"],
            }
    return out


def _attribute_sell_to_buys(
    asset_buys: List[Dict[str, Any]],
    asset_sells: List[Dict[str, Any]],
) -> Dict[str, float]:
    total_usdc_bought = sum(_trade_usdc(t) for t in asset_buys)
    total_sell_proceeds = sum(_trade_usdc(t) for t in asset_sells)
    result: Dict[str, float] = {}
    for buy in asset_buys:
        order_id = buy.get("taker_order_id")
        if not order_id:
            continue
        buy_cost = _trade_usdc(buy)
        weight = buy_cost / total_usdc_bought if total_usdc_bought > 0 else 0.0
        result[order_id] = round((total_sell_proceeds * weight) - buy_cost, 4)
    return result


def _direct_sell_attribution(
    bets: List[Dict[str, Any]],
    sells_by_order: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Map BUY order_id -> {cashout_price, cashout_pnl, sell_trade, cashout_order_id}."""
    out: Dict[str, Dict[str, Any]] = {}
    for bet in bets:
        buy_oid = bet.get("polymarket_order_id")
        cashout_oid = bet.get("cashout_order_id")
        if not buy_oid or not cashout_oid:
            continue
        sell_trade = sells_by_order.get(cashout_oid)
        if not sell_trade:
            continue
        sell_proceeds = _trade_usdc(sell_trade)
        buy_cost = float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0)
        out[buy_oid] = {
            "cashout_order_id": cashout_oid,
            "cashout_price": float(sell_trade.get("price") or 0),
            "cashout_pnl_usdc": round(sell_proceeds - buy_cost, 4),
            "cashout_triggered_at": bet.get("cashout_triggered_at"),
            "sell_trade": sell_trade,
            "method": "direct",
        }
    return out


def _match_bets_to_trades(
    bets: List[Dict[str, Any]],
    trades_by_order: Dict[str, Dict[str, Any]],
    known_tokens: set,
    known_markets: set,
    matched_order_ids: Optional[set] = None,
    fills_by_order: Optional[Dict[str, Dict[str, Any]]] = None,
    twap_orders_by_bet: Optional[Dict[int, List[str]]] = None,
    chunk_order_ids: Optional[set] = None,
    known_order_ids: Optional[set] = None,
    history_lookup_fn=None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    reconciled: List[Dict[str, Any]] = []
    unexecuted: List[Dict[str, Any]] = []
    if matched_order_ids is None:
        matched_order_ids = set()
    fills_by_order = fills_by_order or {}
    twap_orders_by_bet = twap_orders_by_bet or {}
    chunk_order_ids = chunk_order_ids or set()
    known_order_ids = known_order_ids or set()

    def _match_fill(bet: Dict[str, Any], oid: str, fill: Dict[str, Any], via: str) -> None:
        matched_order_ids.add(oid)
        pm_usdc = float(fill.get("fill_usdc") or 0)
        db_usdc = float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0)
        reconciled.append({
            "bet_id": bet["bet_id"],
            "order_id": oid,
            "fixture_key": bet.get("fixture_key"),
            "strategy_label": bet.get("strategy_label"),
            "side_label": bet.get("side_label"),
            "db_stake_usdc": round(db_usdc, 4),
            "pm_stake_usdc": round(pm_usdc, 4),
            "stake_gap_usdc": round(pm_usdc - db_usdc, 4),
            "db_pnl_usdc": bet.get("pnl_realised_usdc"),
            "status": bet.get("status"),
            "match_via": via,
        })

    for bet in bets:
        if bet.get("strategy_label") == "RECONCILE_GHOST":
            continue
        oid = (bet.get("polymarket_order_id") or "").strip()
        matched = False

        if oid:
            trade = trades_by_order.get(oid)
            if trade:
                _match_fill(bet, oid, {"fill_usdc": _trade_usdc(trade)}, "taker_trade")
                matched = True
            elif oid in fills_by_order:
                _match_fill(bet, oid, fills_by_order[oid], "maker_fill")
                matched = True

        if not matched:
            for chunk_oid in twap_orders_by_bet.get(int(bet["bet_id"]), []):
                fill = fills_by_order.get(chunk_oid)
                if fill and float(fill.get("fill_usdc") or 0) > 0:
                    _match_fill(bet, chunk_oid, fill, "twap_maker_fill")
                    matched = True

        if matched:
            continue

        if not oid and not twap_orders_by_bet.get(int(bet["bet_id"])):
            unexecuted.append({
                "bet_id": bet["bet_id"],
                "order_id": None,
                "fixture_key": bet.get("fixture_key"),
                "strategy_label": bet.get("strategy_label"),
                "side_label": bet.get("side_label"),
                "db_stake_usdc": round(
                    float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0), 4
                ),
                "status": bet.get("status"),
                "reason": "missing_order_id",
            })
            continue
        if oid and oid not in fills_by_order:
            unexecuted.append({
                "bet_id": bet["bet_id"],
                "order_id": oid,
                "fixture_key": bet.get("fixture_key"),
                "strategy_label": bet.get("strategy_label"),
                "side_label": bet.get("side_label"),
                "db_stake_usdc": round(
                    float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0), 4
                ),
                "status": bet.get("status"),
                "reason": "no_clob_trade",
            })
            continue
        if twap_orders_by_bet.get(int(bet["bet_id"])) and not matched:
            unexecuted.append({
                "bet_id": bet["bet_id"],
                "order_id": None,
                "fixture_key": bet.get("fixture_key"),
                "strategy_label": bet.get("strategy_label"),
                "side_label": bet.get("side_label"),
                "db_stake_usdc": round(
                    float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0), 4
                ),
                "status": bet.get("status"),
                "reason": "twap_no_fill",
            })

    # Tokens whose DB fills already cover all our on-chain buys — skip ghosts
    tracked_order_ids = set(fills_by_order.keys())
    fully_booked_tokens: set = set()
    for token in known_tokens:
        db_total = _db_fill_total_on_token(bets, token)
        pm_total = _pm_buy_total_on_token(trades_by_order, fills_by_order, token, tracked_order_ids)
        if pm_total > 0 and db_total >= pm_total - 0.05:
            fully_booked_tokens.add(token)

    def _resolved_via_history(oid: str, fill: Dict[str, Any]) -> bool:
        """If order_history knows this oid, attribute the fill to its bet."""
        if history_lookup_fn is None:
            return False
        try:
            info = history_lookup_fn(oid)
        except Exception:
            return False
        if not info:
            return False
        bet_id = info.get("bet_id")
        # Find the bet row in the local list to attribute properly.
        bet_row = next((b for b in bets if int(b["bet_id"]) == int(bet_id or -1)), None)
        if bet_row is None:
            return False
        _match_fill(bet_row, oid, fill, "order_history")
        return True

    ghosts: List[Dict[str, Any]] = []
    for oid, fill in fills_by_order.items():
        if oid in matched_order_ids or oid in chunk_order_ids or oid in known_order_ids:
            continue
        if "maker" not in (fill.get("roles") or []):
            continue
        asset_id = str(fill.get("asset_id") or "")
        if not asset_id or asset_id not in known_tokens:
            continue
        if asset_id in fully_booked_tokens:
            continue
        pm_usdc = float(fill.get("fill_usdc") or 0)
        if pm_usdc <= 0:
            continue
        if _resolved_via_history(oid, fill):
            continue
        ghosts.append({
            "order_id": oid,
            "asset_id": fill.get("asset_id"),
            "market_id": None,
            "side_label": fill.get("outcome"),
            "pm_stake_usdc": round(pm_usdc, 4),
            "fill_price": float(fill.get("avg_fill_price") or 0),
            "trade_timestamp": _utc_now_iso(),
            "transaction_hash": None,
            "trade": fill,
        })

    for oid, trade in trades_by_order.items():
        if oid in matched_order_ids or oid in chunk_order_ids or oid in known_order_ids:
            continue
        if not _is_our_taker_buy(trade):
            continue
        if not _is_cricket_trade(trade, known_tokens, known_markets):
            continue
        asset_id = str(trade.get("asset_id") or "")
        if asset_id in fully_booked_tokens:
            continue
        ghosts.append({
            "order_id": oid,
            "asset_id": trade.get("asset_id"),
            "market_id": trade.get("market"),
            "side_label": trade.get("outcome"),
            "pm_stake_usdc": round(_trade_usdc(trade), 4),
            "fill_price": float(trade.get("price") or 0),
            "trade_timestamp": _trade_timestamp_iso(trade),
            "transaction_hash": trade.get("transaction_hash"),
            "trade": trade,
        })

    return reconciled, ghosts, unexecuted


def _compute_asset_pnl(
    asset_id: str,
    trades: List[Dict[str, Any]],
) -> Dict[str, float]:
    buys = [t for t in trades if _is_our_taker_buy(t)]
    sells = [t for t in trades if _is_our_taker_sell(t)]
    pm_invested = sum(_trade_usdc(t) for t in buys)
    pm_proceeds = sum(_trade_usdc(t) for t in sells)
    return {
        "pm_invested": round(pm_invested, 4),
        "pm_proceeds": round(pm_proceeds, 4),
        "pm_pnl": round(pm_proceeds - pm_invested, 4),
        "n_buys": len(buys),
        "n_sells": len(sells),
    }


def _build_fixture_report(
    bets: List[Dict[str, Any]],
    reconciled: List[Dict[str, Any]],
    ghosts: List[Dict[str, Any]],
    unexecuted: List[Dict[str, Any]],
    trades_by_asset: Dict[str, List[Dict[str, Any]]],
    token_fixture_map: Dict[str, Dict[str, Any]],
    sell_attribution: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    fixture_keys: set = set()
    for bet in bets:
        if bet.get("fixture_key"):
            fixture_keys.add(bet["fixture_key"])
    for g in ghosts:
        meta = token_fixture_map.get(str(g.get("asset_id") or ""), {})
        fk = meta.get("fixture_key") or f"unknown_token_{g.get('asset_id', '?')[:12]}"
        g["fixture_key"] = fk
        fixture_keys.add(fk)
    for item in reconciled + unexecuted:
        if item.get("fixture_key"):
            fixture_keys.add(item["fixture_key"])

    rows: List[Dict[str, Any]] = []
    for fk in sorted(fixture_keys):
        db_stake = sum(
            float(b.get("fill_size_usdc") or b.get("size_usdc") or 0)
            for b in bets
            if b.get("fixture_key") == fk and b.get("strategy_label") != "RECONCILE_GHOST"
        )
        db_pnl = sum(
            float(b.get("pnl_realised_usdc") or 0)
            for b in bets
            if b.get("fixture_key") == fk
            and b.get("pnl_realised_usdc") is not None
            and b.get("strategy_label") != "RECONCILE_GHOST"
        )

        asset_ids = {
            str(b["polymarket_token_id"])
            for b in bets
            if b.get("fixture_key") == fk and b.get("polymarket_token_id")
        }
        for g in ghosts:
            if g.get("fixture_key") == fk and g.get("asset_id"):
                asset_ids.add(str(g["asset_id"]))

        pm_stake = 0.0
        pm_pnl = 0.0
        for aid in asset_ids:
            asset_trades = trades_by_asset.get(aid, [])
            stats = _compute_asset_pnl(aid, asset_trades)
            pm_stake += stats["pm_invested"]
            pm_pnl += stats["pm_pnl"]

        n_ghosts = sum(1 for g in ghosts if g.get("fixture_key") == fk)
        n_unexec = sum(1 for u in unexecuted if u.get("fixture_key") == fk)

        rows.append({
            "fixture_key": fk,
            "db_stake_usdc": round(db_stake, 4),
            "pm_stake_usdc": round(pm_stake, 4),
            "stake_gap_usdc": round(pm_stake - db_stake, 4),
            "db_pnl_usdc": round(db_pnl, 4),
            "pm_pnl_usdc": round(pm_pnl, 4),
            "pnl_gap_usdc": round(pm_pnl - db_pnl, 4),
            "n_db_bets": sum(1 for b in bets if b.get("fixture_key") == fk),
            "n_ghost_bets": n_ghosts,
            "n_unexecuted_bets": n_unexec,
        })
    return rows


def _build_strategy_report(
    bets: List[Dict[str, Any]],
    reconciled: List[Dict[str, Any]],
    ghosts: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    by_strategy: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "strategy_label": None,
        "n_db_bets": 0,
        "db_stake_usdc": 0.0,
        "db_pnl_usdc": 0.0,
        "n_reconciled": 0,
        "n_ghost_bets": 0,
        "ghost_stake_usdc": 0.0,
    })

    for bet in bets:
        if bet.get("strategy_label") == "RECONCILE_GHOST":
            continue
        label = bet.get("strategy_label") or "(default)"
        s = by_strategy[label]
        s["strategy_label"] = label
        s["n_db_bets"] += 1
        s["db_stake_usdc"] += float(bet.get("fill_size_usdc") or bet.get("size_usdc") or 0)
        if bet.get("pnl_realised_usdc") is not None:
            s["db_pnl_usdc"] += float(bet["pnl_realised_usdc"])

    for item in reconciled:
        label = item.get("strategy_label") or "(default)"
        by_strategy[label]["n_reconciled"] += 1

    ghost_stake_total = sum(g["pm_stake_usdc"] for g in ghosts)
    ghost_entry = by_strategy["RECONCILE_GHOST (pending)"]
    ghost_entry["strategy_label"] = "RECONCILE_GHOST (pending)"
    ghost_entry["n_ghost_bets"] = len(ghosts)
    ghost_entry["ghost_stake_usdc"] = round(ghost_stake_total, 4)

    rows = []
    for label in sorted(by_strategy.keys()):
        s = by_strategy[label]
        rows.append({
            "strategy_label": s["strategy_label"],
            "n_db_bets": s["n_db_bets"],
            "db_stake_usdc": round(s["db_stake_usdc"], 4),
            "db_pnl_usdc": round(s["db_pnl_usdc"], 4),
            "n_reconciled": s["n_reconciled"],
            "n_ghost_bets": s["n_ghost_bets"],
            "ghost_stake_usdc": round(s["ghost_stake_usdc"], 4),
        })
    return rows


def _attribute_sells(
    bets: List[Dict[str, Any]],
    trades_by_asset: Dict[str, List[Dict[str, Any]]],
    trades_by_order: Dict[str, Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    sells_by_order = {
        oid: t for oid, t in trades_by_order.items()
        if _is_our_taker_sell(t)
    }
    attribution = _direct_sell_attribution(bets, sells_by_order)

    bets_needing_prorata: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for bet in bets:
        if bet.get("cashout_triggered_at") and not bet.get("cashout_order_id"):
            asset = str(bet.get("polymarket_token_id") or "")
            if asset:
                bets_needing_prorata[asset].append(bet)

    for asset_id, asset_trades in trades_by_asset.items():
        buys = [t for t in asset_trades if _is_our_taker_buy(t)]
        sells = [t for t in asset_trades if _is_our_taker_sell(t)]
        if not sells:
            continue

        prorata = _attribute_sell_to_buys(buys, sells)
        for buy in buys:
            oid = buy.get("taker_order_id")
            if not oid or oid in attribution:
                continue
            if oid not in prorata:
                continue
            sell_trade = sells[0] if len(sells) == 1 else None
            attribution[oid] = {
                "cashout_order_id": sell_trade.get("taker_order_id") if sell_trade else None,
                "cashout_price": float(sell_trade.get("price") or 0) if sell_trade else None,
                "cashout_pnl_usdc": prorata[oid],
                "cashout_triggered_at": None,
                "sell_trade": sell_trade,
                "method": "pro_rata",
            }

    return attribution


def _fix_ghosts(
    conn,
    ghosts: List[Dict[str, Any]],
    token_fixture_map: Dict[str, Dict[str, Any]],
    sell_attribution: Dict[str, Dict[str, Any]],
    dry_run: bool,
) -> List[Dict[str, Any]]:
    inserted: List[Dict[str, Any]] = []
    for ghost in ghosts:
        oid = ghost["order_id"]
        asset_id = str(ghost.get("asset_id") or "")
        meta = token_fixture_map.get(asset_id, {})
        fixture_key = ghost.get("fixture_key") or meta.get("fixture_key")
        if not fixture_key:
            fixture_key = f"unknown_token_{asset_id[:12]}"

        sell_info = sell_attribution.get(oid, {})
        pnl = sell_info.get("cashout_pnl_usdc")
        cashout_price = sell_info.get("cashout_price")
        cashout_oid = sell_info.get("cashout_order_id")
        cashout_at = sell_info.get("cashout_triggered_at")
        if cashout_at is None and sell_info.get("sell_trade"):
            cashout_at = _trade_timestamp_iso(sell_info["sell_trade"])

        row = {
            "order_id": oid,
            "fixture_key": fixture_key,
            "pm_stake_usdc": ghost["pm_stake_usdc"],
            "pnl_realised_usdc": pnl,
            "transaction_hash": ghost.get("transaction_hash"),
        }
        inserted.append(row)

        if dry_run:
            logger.info(
                f"  [DRY] ghost order={oid[:16]}... fixture={fixture_key} "
                f"stake=${ghost['pm_stake_usdc']:.2f} pnl={pnl}"
            )
            continue

        conn.execute(
            """
            INSERT INTO bet_ledger (
                proposed_at, fixture_key, market_type,
                polymarket_market_id, polymarket_token_id, polymarket_order_id,
                side_label, model_prob, market_price_at_proposal, edge_pp,
                side, size_usdc, fill_price, fill_size_usdc,
                pnl_realised_usdc, settle_outcome, status, mode, bet_kind, strategy_label,
                filled_at, placed_at
            ) VALUES (
                ?, ?, 'moneyline',
                ?, ?, ?,
                ?, 0.0, ?, 0.0,
                'BUY', ?, ?, ?,
                NULL, NULL, 'filled', 'auto', 'real', 'RECONCILE_GHOST',
                ?, ?
            )
            """,
            (
                ghost["trade_timestamp"],
                fixture_key,
                ghost.get("market_id") or meta.get("polymarket_market_id"),
                asset_id,
                oid,
                ghost.get("side_label") or meta.get("side_label"),
                ghost["fill_price"],
                ghost["pm_stake_usdc"],
                ghost["fill_price"],
                ghost["pm_stake_usdc"],
                ghost["trade_timestamp"],
                ghost["trade_timestamp"],
            ),
        )
        logger.info(
            f"  inserted ghost order={oid[:16]}... fixture={fixture_key} "
            f"stake=${ghost['pm_stake_usdc']:.2f}"
        )

    if not dry_run and inserted:
        conn.commit()
    return inserted


def _settle_reconcile_ghosts(dry_run: bool) -> Dict[str, Any]:
    """Run reconcile_pending_bets so RECONCILE_GHOST rows get pnl_realised_usdc."""
    if dry_run:
        return {"skipped": True, "reason": "dry_run"}
    from src.integrations.polymarket.reconcile import reconcile_pending_bets
    return reconcile_pending_bets()


def reconcile(
    since_days: Optional[int] = None,
    dry_run: bool = False,
    fix_ghosts: bool = False,
    output_path: Optional[Path] = None,
) -> Dict[str, Any]:
    from src.data.database import get_connection, init_cashout_columns
    from src.integrations.polymarket import PolymarketClient

    init_cashout_columns()

    pm = PolymarketClient()
    sdk = pm._get_clob_sdk_client()

    logger.info("Fetching all CLOB trades (paginated)...")
    trades = _fetch_all_trades(sdk, since_days=since_days)
    logger.info(f"Loaded {len(trades)} total trades")

    matched_order_ids: set = set()
    orphan_links: List[Dict[str, Any]] = []
    purged_ghosts = 0
    settle_summary: Optional[Dict[str, Any]] = None
    twap_orders_by_bet: Dict[int, List[str]] = {}
    chunk_order_ids: set = set()
    known_order_ids: set = set()

    with get_connection() as conn:
        if fix_ghosts:
            purged_ghosts = _purge_reconcile_ghosts(conn, dry_run=dry_run)

        bets = _load_real_bets(conn)
        twap_orders_by_bet = _load_twap_orders_by_bet(conn)
        chunk_order_ids = _load_chunk_order_ids(conn)
        known_order_ids = _load_known_order_ids(conn)
        token_fixture_map = _load_token_fixture_map(conn)
        known_tokens, known_markets = _load_cricket_scope(conn)

    trades_by_order, trades_by_asset, fills_by_order = _build_trade_maps(
        trades, known_order_ids=known_order_ids,
    )

    with get_connection() as conn:
        if fix_ghosts:
            logger.info("Linking orphan DB rows to TAKER trades...")
            orphan_links = _link_orphan_bets(
                conn, bets, trades_by_order, matched_order_ids, dry_run=dry_run,
            )
            if orphan_links and not dry_run:
                bets = _load_real_bets(conn)

    logger.info(f"Loaded {len(bets)} real bet_ledger rows")
    logger.info(f"Cricket scope: {len(known_tokens)} tokens, {len(known_markets)} markets")
    if orphan_links:
        logger.info(f"Linked {len(orphan_links)} orphan bet(s) to CLOB order IDs")

    try:
        from src.integrations.polymarket.order_audit import lookup_bet_for_order
        with get_connection() as _conn_lookup:
            # Snapshot a closure that holds its own connection. Sized small so
            # repeated calls inside _match_bets_to_trades are cheap.
            _conn_lookup_handle = _conn_lookup
            history_lookup_fn = lambda oid: lookup_bet_for_order(oid, conn=_conn_lookup_handle)
            reconciled, ghosts, unexecuted = _match_bets_to_trades(
                bets, trades_by_order, known_tokens, known_markets, matched_order_ids,
                fills_by_order=fills_by_order,
                twap_orders_by_bet=twap_orders_by_bet,
                chunk_order_ids=chunk_order_ids,
                known_order_ids=known_order_ids,
                history_lookup_fn=history_lookup_fn,
            )
    except Exception as exc:
        logger.warning(f"history-aware match failed, falling back: {exc}")
        reconciled, ghosts, unexecuted = _match_bets_to_trades(
            bets, trades_by_order, known_tokens, known_markets, matched_order_ids,
            fills_by_order=fills_by_order,
            twap_orders_by_bet=twap_orders_by_bet,
            chunk_order_ids=chunk_order_ids,
            known_order_ids=known_order_ids,
        )
    sell_attribution = _attribute_sells(bets, trades_by_asset, trades_by_order)

    fixture_report = _build_fixture_report(
        bets, reconciled, ghosts, unexecuted,
        trades_by_asset, token_fixture_map, sell_attribution,
    )
    strategy_report = _build_strategy_report(bets, reconciled, ghosts)

    summary: Dict[str, Any] = {
        "generated_at": _utc_now_iso(),
        "since_days": since_days,
        "dry_run": dry_run,
        "n_clob_trades": len(trades),
        "n_db_bets": len(bets),
        "n_reconciled": len(reconciled),
        "n_ghost_bets": len(ghosts),
        "n_unexecuted_bets": len(unexecuted),
        "n_purged_ghosts": purged_ghosts,
        "orphan_links": orphan_links,
        "fixture_report": fixture_report,
        "strategy_report": strategy_report,
        "reconciled": reconciled,
        "ghost_bets": ghosts,
        "unexecuted_bets": unexecuted,
        "sell_attribution": {
            oid: {k: v for k, v in info.items() if k != "sell_trade"}
            for oid, info in sell_attribution.items()
        },
    }

    if fix_ghosts:
        if ghosts:
            logger.info(f"{'[DRY-RUN] ' if dry_run else ''}Fixing {len(ghosts)} ghost bet(s)...")
            with get_connection() as conn:
                summary["ghost_fixes"] = _fix_ghosts(
                    conn, ghosts, token_fixture_map, sell_attribution, dry_run=dry_run,
                )
        else:
            logger.info("No ghost bets to insert")
            summary["ghost_fixes"] = []

        if not dry_run:
            logger.info("Settling RECONCILE_GHOST rows via reconcile_pending_bets...")
            settle_summary = _settle_reconcile_ghosts(dry_run=False)
            summary["settle_summary"] = settle_summary
            logger.info(
                f"Settlement: checked={settle_summary.get('n_checked')} "
                f"settled={settle_summary.get('n_settled')}"
            )

    if output_path is None:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        output_path = OUTPUT_DIR / f"reconcile_polymarket_{ts}.json"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    summary["output_path"] = str(output_path)
    logger.info(f"Wrote report to {output_path}")

    return summary


def _print_console_report(summary: Dict[str, Any]) -> None:
    print()
    print("=" * 90)
    print("POLYMARKET RECONCILIATION")
    print("=" * 90)
    print(f"  CLOB trades fetched:  {summary['n_clob_trades']}")
    print(f"  DB real bets:         {summary['n_db_bets']}")
    print(f"  Reconciled:           {summary['n_reconciled']}")
    print(f"  Ghost bets (PM only): {summary['n_ghost_bets']}")
    print(f"  Unexecuted (DB only): {summary['n_unexecuted_bets']}")
    print()

    print("PER-FIXTURE P&L")
    print("-" * 90)
    print(f"{'Fixture':<40} {'DB Stake':>10} {'PM Stake':>10} {'DB P&L':>10} {'PM P&L':>10} {'Gap':>10}")
    print("-" * 90)
    for row in summary["fixture_report"]:
        fk = row["fixture_key"]
        if len(fk) > 38:
            fk = fk[:35] + "..."
        print(
            f"{fk:<40} "
            f"${row['db_stake_usdc']:>9.2f} "
            f"${row['pm_stake_usdc']:>9.2f} "
            f"${row['db_pnl_usdc']:>9.2f} "
            f"${row['pm_pnl_usdc']:>9.2f} "
            f"${row['pnl_gap_usdc']:>9.2f}"
        )
        if row["n_ghost_bets"] or row["n_unexecuted_bets"]:
            print(
                f"  └─ ghosts={row['n_ghost_bets']}  unexecuted={row['n_unexecuted_bets']}  "
                f"stake_gap=${row['stake_gap_usdc']:+.2f}"
            )

    print()
    print("PER-STRATEGY")
    print("-" * 70)
    print(f"{'Strategy':<25} {'Bets':>6} {'DB Stake':>10} {'DB P&L':>10} {'Reconciled':>11}")
    print("-" * 70)
    for row in summary["strategy_report"]:
        label = row["strategy_label"] or "(default)"
        if len(label) > 23:
            label = label[:20] + "..."
        print(
            f"{label:<25} {row['n_db_bets']:>6} "
            f"${row['db_stake_usdc']:>9.2f} "
            f"${row['db_pnl_usdc']:>9.2f} "
            f"{row['n_reconciled']:>11}"
        )
        if row["n_ghost_bets"]:
            print(f"  └─ pending ghosts: {row['n_ghost_bets']} (${row['ghost_stake_usdc']:.2f})")

    if summary.get("ghost_bets"):
        print()
        print("GHOST BETS (on-chain, missing from DB)")
        print("-" * 90)
        for g in summary["ghost_bets"]:
            fk = g.get("fixture_key") or "?"
            print(
                f"  order={g['order_id'][:20]}...  fixture={fk}  "
                f"stake=${g['pm_stake_usdc']:.2f}  side={g.get('side_label')}  "
                f"tx={str(g.get('transaction_hash', ''))[:18]}..."
            )

    if summary.get("unexecuted_bets"):
        print()
        print("UNEXECUTED / PARTIAL (in DB, not on-chain)")
        print("-" * 90)
        for u in summary["unexecuted_bets"]:
            reason = u.get("reason", "unknown")
            print(
                f"  bet_id={u['bet_id']}  order={str(u.get('order_id') or 'NULL')[:20]}  "
                f"fixture={u.get('fixture_key')}  db_stake=${u['db_stake_usdc']:.2f}  "
                f"status={u.get('status')}  reason={reason}"
            )

    print()
    print(f"Full report: {summary.get('output_path')}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Two-way reconcile Polymarket CLOB trades vs bet_ledger",
    )
    parser.add_argument(
        "--since-days", type=int, default=None,
        help="Only fetch trades from the last N days (default: all available)",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Report only — no DB writes (also applies to --fix-ghosts)",
    )
    parser.add_argument(
        "--fix-ghosts", action="store_true",
        help="Insert untracked real BUY trades as RECONCILE_GHOST bet_ledger rows",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Custom JSON output path",
    )
    args = parser.parse_args()

    summary = reconcile(
        since_days=args.since_days,
        dry_run=args.dry_run,
        fix_ghosts=args.fix_ghosts,
        output_path=args.output,
    )
    _print_console_report(summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
