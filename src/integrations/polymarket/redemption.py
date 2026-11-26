"""Safe discovery and audit plumbing for gasless winner redemption."""

from __future__ import annotations

import json
import logging
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from config import BASE_DIR, BETTING_CONFIG
from src.data.database import init_redemption_attempts
from src.integrations.polymarket.client import PolymarketClient

logger = logging.getLogger(__name__)

RUNNER = BASE_DIR / "scripts" / "redeem_resolved_positions.mjs"
LOCAL_NODE = BASE_DIR / ".tools" / "node" / "bin" / "node"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _value(position: Dict[str, Any]) -> float:
    try:
        return float(position.get("currentValue") or 0.0)
    except (TypeError, ValueError):
        return 0.0


def discover_redeemable_positions(client: PolymarketClient) -> List[Dict[str, Any]]:
    """Return only API-confirmed redeemable positions with usable condition IDs."""
    minimum = float(BETTING_CONFIG["auto_redeem_min_value_usdc"])
    candidates: List[Dict[str, Any]] = []
    for position in client.get_data_api_positions(size_threshold=0.0, limit=500):
        condition_id = str(
            position.get("conditionId") or position.get("condition_id") or ""
        ).strip()
        value = _value(position)
        if not position.get("redeemable") or value < minimum or not condition_id:
            continue
        candidates.append(
            {
                "condition_id": condition_id,
                "token_id": str(position.get("asset") or position.get("token_id") or ""),
                "market_id": str(position.get("market") or position.get("marketId") or ""),
                "title": str(position.get("title") or position.get("question") or ""),
                "neg_risk": bool(position.get("negativeRisk") or position.get("negRisk")),
                "estimated_value_usdc": value,
                # NegRisk redemption requires balances for both outcomes; the
                # winning side is always sufficient for a confirmed position.
                "yes_amount": str(position.get("size") or "0") if position.get("outcome") == "Yes" else "0",
                "no_amount": str(position.get("size") or "0") if position.get("outcome") != "Yes" else "0",
            }
        )
    # A condition can surface once per outcome. Redeem it once while retaining
    # the highest-value position as the audit representative.
    by_condition: Dict[str, Dict[str, Any]] = {}
    for candidate in candidates:
        old = by_condition.get(candidate["condition_id"])
        if old is None or candidate["estimated_value_usdc"] > old["estimated_value_usdc"]:
            by_condition[candidate["condition_id"]] = candidate
    return list(by_condition.values())


def _already_processed(conn, condition_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM redemption_attempts "
        "WHERE condition_id = ? AND status IN ('planned', 'submitted', 'confirmed')",
        (condition_id,),
    ).fetchone()
    return row is not None


def _record(conn, item: Dict[str, Any], status: str, tx_hash: str | None = None,
            error: str | None = None) -> None:
    now = _now()
    conn.execute(
        """
        INSERT INTO redemption_attempts (
            condition_id, token_id, market_id, title, neg_risk,
            estimated_value_usdc, status, transaction_hash, error_message,
            updated_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            item["condition_id"], item["token_id"], item["market_id"],
            item["title"], int(item["neg_risk"]), item["estimated_value_usdc"],
            status, tx_hash, error, now,
        ),
    )


def redeem_winners(conn, *, dry_run: bool) -> Dict[str, Any]:
    """Discover, submit, and audit confirmed winner redemptions.

    No transaction is attempted unless the independent feature flag is enabled.
    The dry-run path does not require Node or wallet credentials.
    """
    init_redemption_attempts()
    candidates = discover_redeemable_positions(PolymarketClient())
    candidates = [
        item for item in candidates
        if not _already_processed(conn, item["condition_id"])
    ][: int(BETTING_CONFIG["auto_redeem_batch_limit"])]

    if not candidates:
        return {"dry_run": dry_run, "n_candidates": 0, "results": []}

    if dry_run:
        return {"dry_run": True, "n_candidates": len(candidates), "results": [
            {**item, "status": "planned"} for item in candidates
        ]}

    if not BETTING_CONFIG["auto_redeem_enabled"]:
        return {
            "dry_run": False,
            "n_candidates": len(candidates),
            "results": [{**item, "status": "skipped", "error": "BETTING_AUTO_REDEEM_ENABLED is off"} for item in candidates],
        }
    if not RUNNER.exists():
        raise RuntimeError(f"Gasless redemption runner missing: {RUNNER}")

    for item in candidates:
        _record(conn, item, "planned")
    conn.commit()

    payload = {
        "dryRun": False,
        "items": [{
            "conditionId": item["condition_id"],
            "negRisk": item["neg_risk"],
            "yesAmount": item["yes_amount"],
            "noAmount": item["no_amount"],
        } for item in candidates],
    }
    try:
        node = str(LOCAL_NODE) if LOCAL_NODE.exists() else "node"
        proc = subprocess.run(
            [node, str(RUNNER)],
            input=json.dumps(payload),
            text=True,
            capture_output=True,
            check=False,
            timeout=int(BETTING_CONFIG["auto_redeem_timeout_seconds"]),
            cwd=str(BASE_DIR),
        )
        response = json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        response = {"ok": False, "error": str(exc), "results": []}

    result_by_condition = {
        str(result.get("conditionId")): result
        for result in response.get("results", [])
    }
    results = []
    for item in candidates:
        result = result_by_condition.get(item["condition_id"], {})
        status = result.get("status", "failed")
        _record(
            conn, item, status,
            tx_hash=result.get("transactionHash"),
            error=result.get("error") or response.get("error"),
        )
        results.append({**item, "status": status, "transaction_hash": result.get("transactionHash")})
    conn.commit()
    return {"dry_run": False, "n_candidates": len(candidates), "results": results}
