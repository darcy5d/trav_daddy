"""
Credential readiness helpers for market integrations.

This module never returns full secrets. It only exposes boolean readiness,
masked previews, and missing-field diagnostics so the app can safely report
configuration status before enabling market integrations.
"""

from __future__ import annotations

from typing import Any, Dict, List

from config import POLYMARKET_CONFIG, BETFAIR_CONFIG


def _is_set(value: Any) -> bool:
    """Return True when an env-backed value is present and non-empty."""
    return isinstance(value, str) and bool(value.strip())


def _masked_preview(value: Any, keep_prefix: int = 3, keep_suffix: int = 2) -> str | None:
    """Return a short masked preview of a secret-like value."""
    if not _is_set(value):
        return None
    text = value.strip()
    if len(text) <= keep_prefix + keep_suffix + 1:
        return "*" * len(text)
    return f"{text[:keep_prefix]}...{text[-keep_suffix:]}"


def _missing_fields(config: Dict[str, Any], required_fields: List[str]) -> List[str]:
    """Return required config keys that are missing."""
    return [field for field in required_fields if not _is_set(config.get(field))]


def get_polymarket_credential_status() -> Dict[str, Any]:
    """
    Return Polymarket credential readiness.

    Read-only market data is available without authentication. Trading or other
    authenticated paths require API credentials/private key.
    """
    enabled = bool(POLYMARKET_CONFIG.get("enabled"))
    has_l2 = all(
        _is_set(POLYMARKET_CONFIG.get(field))
        for field in ("api_key", "api_secret", "passphrase")
    )
    has_private_key = _is_set(POLYMARKET_CONFIG.get("private_key"))

    if has_l2 or has_private_key:
        mode = "authenticated_ready"
    else:
        mode = "public_read_only"

    return {
        "enabled": enabled,
        "mode": mode,
        "public_read_ready": True,
        "authenticated_ready": has_l2 or has_private_key,
        "configured": {
            "api_key": _is_set(POLYMARKET_CONFIG.get("api_key")),
            "api_secret": _is_set(POLYMARKET_CONFIG.get("api_secret")),
            "passphrase": _is_set(POLYMARKET_CONFIG.get("passphrase")),
            "private_key": _is_set(POLYMARKET_CONFIG.get("private_key")),
        },
        "masked": {
            "api_key": _masked_preview(POLYMARKET_CONFIG.get("api_key")),
            "private_key": _masked_preview(POLYMARKET_CONFIG.get("private_key"), keep_prefix=4, keep_suffix=4),
        },
        "notes": [
            "Public Gamma/CLOB read endpoints do not require auth.",
            "Authenticated CLOB trading endpoints require API creds and wallet signing.",
        ],
        "endpoints": {
            "api_base_url": POLYMARKET_CONFIG.get("api_base_url"),
            "clob_base_url": POLYMARKET_CONFIG.get("clob_base_url"),
            "chain_id": POLYMARKET_CONFIG.get("chain_id"),
        },
    }


def get_betfair_credential_status() -> Dict[str, Any]:
    """
    Return Betfair credential readiness.

    Exchange API reads require app key + session token (or a way to obtain one).
    """
    enabled = bool(BETFAIR_CONFIG.get("enabled"))
    missing_read_fields = _missing_fields(BETFAIR_CONFIG, ["app_key"])
    has_session_token = _is_set(BETFAIR_CONFIG.get("session_token"))
    has_interactive_login = _is_set(BETFAIR_CONFIG.get("username")) and _is_set(BETFAIR_CONFIG.get("password"))
    has_cert_login = _is_set(BETFAIR_CONFIG.get("cert_file")) and _is_set(BETFAIR_CONFIG.get("key_file"))

    read_ready = len(missing_read_fields) == 0 and (has_session_token or has_interactive_login or has_cert_login)
    if has_session_token:
        auth_path = "session_token"
    elif has_cert_login:
        auth_path = "non_interactive_cert_login"
    elif has_interactive_login:
        auth_path = "interactive_api_login"
    else:
        auth_path = "not_configured"

    missing_login_fields: List[str] = []
    if not has_session_token and not has_interactive_login and not has_cert_login:
        missing_login_fields = ["session_token_or_login_method"]

    return {
        "enabled": enabled,
        "read_ready": read_ready,
        "auth_path": auth_path,
        "configured": {
            "app_key": _is_set(BETFAIR_CONFIG.get("app_key")),
            "session_token": has_session_token,
            "username": _is_set(BETFAIR_CONFIG.get("username")),
            "password": _is_set(BETFAIR_CONFIG.get("password")),
            "cert_file": _is_set(BETFAIR_CONFIG.get("cert_file")),
            "key_file": _is_set(BETFAIR_CONFIG.get("key_file")),
        },
        "masked": {
            "app_key": _masked_preview(BETFAIR_CONFIG.get("app_key")),
            "session_token": _masked_preview(BETFAIR_CONFIG.get("session_token"), keep_prefix=4, keep_suffix=4),
            "username": _masked_preview(BETFAIR_CONFIG.get("username"), keep_prefix=2, keep_suffix=2),
        },
        "missing_required_for_read": missing_read_fields + missing_login_fields,
        "notes": [
            "Read-path needs app key plus a valid session token or login flow.",
            "Live trading and cert-based bot setup can remain deferred for Wave 2 read integration.",
        ],
        "endpoints": {
            "sso_base_url": BETFAIR_CONFIG.get("sso_base_url"),
            "betting_api_base_url": BETFAIR_CONFIG.get("betting_api_base_url"),
        },
    }


def get_market_credentials_status() -> Dict[str, Any]:
    """Return consolidated credential/readiness payload for both providers."""
    polymarket = get_polymarket_credential_status()
    betfair = get_betfair_credential_status()
    ready_polymarket_only = polymarket["public_read_ready"]
    ready_with_betfair = polymarket["public_read_ready"] and betfair["read_ready"]
    return {
        "success": True,
        "wave2_scope": "read_only_market_comparison",
        "focus_provider": "polymarket",
        "providers": {
            "polymarket": polymarket,
            "betfair": betfair,
        },
        "ready_for_wave2_read": ready_polymarket_only,
        "ready_for_wave2_read_polymarket_only": ready_polymarket_only,
        "ready_for_wave2_read_with_betfair": ready_with_betfair,
        "betfair_optional_for_current_wave": True,
        "deferred": [
            "order_placement",
            "automated_trading",
            "ledger_write_path",
        ],
    }

