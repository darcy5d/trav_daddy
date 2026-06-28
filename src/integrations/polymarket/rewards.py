"""Wave 6 pre-work (W2): liquidity-reward / maker-rebate reconnaissance.

The market-making thesis assumes four income sources, two of which (maker
rebate + liquidity reward) are entirely unverified in this codebase: nothing
fetches reward params, and there is no evidence cricket is in Polymarket's
reward set on any given day. This module answers that empirically.

It is deliberately defensive: Polymarket's reward surface has moved around
(Gamma `clobRewards`, CLOB `/markets/{condition_id}.rewards`, `feeSchedule`),
and reward endpoints sometimes require auth or return nothing. Every probe is
best-effort and returns NULL/unknown rather than raising, so the recon scanner
never crashes on telemetry.

Normalized output (`summarize_market_rewards`):
    {
        "in_reward_set":     1 | 0 | None,   # None = couldn't determine
        "reward_min_size":   float | None,   # min_incentive_size
        "reward_max_spread": float | None,   # max_incentive_spread
        "reward_json":       str | None,     # raw reward blob (json)
        "fee_schedule_json": str | None,     # raw feeSchedule blob (json)
    }
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import requests

from config import POLYMARKET_CONFIG

logger = logging.getLogger(__name__)

_CLOB_BASE = POLYMARKET_CONFIG.get("clob_base_url", "https://clob.polymarket.com").rstrip("/")
_TIMEOUT = 12


def _coerce_json_obj(value: Any) -> Optional[Any]:
    """Gamma sends some nested objects as JSON strings. Decode defensively."""
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (json.JSONDecodeError, ValueError):
            return value
    return value


def _to_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_fee_schedule(market: Dict[str, Any]) -> Optional[Any]:
    """Pull a feeSchedule blob from a Gamma market dict, if present.

    Observed shape (Gamma /markets): feeSchedule={'rate': 0.03,
    'takerOnly': True, 'rebateRate': 0.25}. May be a JSON string.
    """
    if not isinstance(market, dict):
        return None
    for key in ("feeSchedule", "fee_schedule", "fees"):
        if key in market and market[key] not in (None, "", "{}"):
            return _coerce_json_obj(market[key])
    return None


def extract_rewards_from_market(market: Dict[str, Any]) -> Dict[str, Any]:
    """Pull whatever reward fields are embedded in a Gamma market dict.

    Gamma has used several shapes over time:
      * top-level `rewardsMinSize` / `rewardsMaxSpread`
      * a `clobRewards` list of {rewardsDailyRate, assetAddress, ...}
      * a nested `rewards` object {min_size, max_spread, rates: [...]}
    We return a normalized blob plus the raw object for the snapshot log.
    """
    out: Dict[str, Any] = {
        "in_reward_set": None,
        "reward_min_size": None,
        "reward_max_spread": None,
        "raw": None,
    }
    if not isinstance(market, dict):
        return out

    raw_blob: Dict[str, Any] = {}

    rewards_obj = _coerce_json_obj(market.get("rewards"))
    if isinstance(rewards_obj, dict) and rewards_obj:
        raw_blob["rewards"] = rewards_obj
        out["reward_min_size"] = _to_float(
            rewards_obj.get("min_size") or rewards_obj.get("minSize")
        )
        out["reward_max_spread"] = _to_float(
            rewards_obj.get("max_spread") or rewards_obj.get("maxSpread")
        )
        rates = rewards_obj.get("rates") or rewards_obj.get("rewardsDailyRate")
        if rates:
            out["in_reward_set"] = 1

    clob_rewards = _coerce_json_obj(market.get("clobRewards"))
    if isinstance(clob_rewards, list) and clob_rewards:
        raw_blob["clobRewards"] = clob_rewards
        # A non-empty clobRewards list with a positive daily rate => in set.
        for entry in clob_rewards:
            if isinstance(entry, dict):
                rate = _to_float(
                    entry.get("rewardsDailyRate") or entry.get("rewards_daily_rate")
                )
                if rate and rate > 0:
                    out["in_reward_set"] = 1
                    break

    min_size = _to_float(market.get("rewardsMinSize") or market.get("rewards_min_size"))
    max_spread = _to_float(
        market.get("rewardsMaxSpread") or market.get("rewards_max_spread")
    )
    if min_size is not None:
        out["reward_min_size"] = out["reward_min_size"] or min_size
        raw_blob["rewardsMinSize"] = min_size
    if max_spread is not None:
        out["reward_max_spread"] = out["reward_max_spread"] or max_spread
        raw_blob["rewardsMaxSpread"] = max_spread

    # If Gamma explicitly exposes any reward field but no positive rate was
    # found, we can at least say "fields present" (leave in_reward_set as the
    # rate-derived value; None if genuinely silent).
    if raw_blob and out["in_reward_set"] is None:
        # Presence of min_size/max_spread strongly implies the market is at
        # least configured for rewards.
        if out["reward_min_size"] is not None or out["reward_max_spread"] is not None:
            out["in_reward_set"] = 1

    out["raw"] = raw_blob or None
    return out


def fetch_clob_market_rewards(condition_id: str) -> Optional[Dict[str, Any]]:
    """Best-effort: hit the CLOB market endpoint and pull its `rewards` object.

    CLOB `/markets/{condition_id}` returns a market that historically carries
    a `rewards` block: {rates: [...], min_size, max_spread}. Public endpoint;
    returns None on any failure.
    """
    if not condition_id:
        return None
    url = f"{_CLOB_BASE}/markets/{condition_id}"
    try:
        resp = requests.get(url, timeout=_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except (requests.RequestException, ValueError) as exc:
        logger.debug(f"CLOB rewards fetch failed for {condition_id}: {exc}")
        return None
    if not isinstance(data, dict):
        return None
    rewards = data.get("rewards")
    if isinstance(rewards, dict):
        return rewards
    return None


def summarize_market_rewards(
    market: Dict[str, Any],
    condition_id: Optional[str] = None,
    probe_clob: bool = True,
) -> Dict[str, Any]:
    """Combine Gamma-embedded reward fields with an optional CLOB probe.

    Args:
        market: a Gamma market dict (the `raw_market` from upcoming.py).
        condition_id: CLOB condition id (market.conditionId) for the CLOB probe.
        probe_clob: if False, skip the network call (Gamma fields only).

    Returns the normalized dict documented at module level.
    """
    gamma = extract_rewards_from_market(market)
    fee = extract_fee_schedule(market)

    raw_blob: Dict[str, Any] = {}
    if gamma.get("raw"):
        raw_blob["gamma"] = gamma["raw"]

    in_set = gamma.get("in_reward_set")
    min_size = gamma.get("reward_min_size")
    max_spread = gamma.get("reward_max_spread")

    if probe_clob:
        cid = condition_id or (
            market.get("conditionId") or market.get("condition_id")
            if isinstance(market, dict)
            else None
        )
        clob_rewards = fetch_clob_market_rewards(cid) if cid else None
        if clob_rewards:
            raw_blob["clob"] = clob_rewards
            c_min = _to_float(clob_rewards.get("min_size") or clob_rewards.get("minSize"))
            c_max = _to_float(
                clob_rewards.get("max_spread") or clob_rewards.get("maxSpread")
            )
            rates = clob_rewards.get("rates")
            if c_min is not None:
                min_size = min_size or c_min
            if c_max is not None:
                max_spread = max_spread or c_max
            if rates:
                # rates is typically a list of per-asset reward configs; a
                # non-empty list with positive daily rate => in reward set.
                in_set = 1
            elif in_set is None and (c_min is not None or c_max is not None):
                in_set = 1

    return {
        "in_reward_set": in_set,
        "reward_min_size": min_size,
        "reward_max_spread": max_spread,
        "reward_json": json.dumps(raw_blob) if raw_blob else None,
        "fee_schedule_json": json.dumps(fee) if fee is not None else None,
    }
