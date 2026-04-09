"""Betfair session bootstrap and keep-alive utilities."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Optional

import requests

from config import BETFAIR_CONFIG


def _mask(value: Optional[str], keep_prefix: int = 4, keep_suffix: int = 4) -> Optional[str]:
    if not value:
        return None
    text = value.strip()
    if len(text) <= keep_prefix + keep_suffix + 1:
        return "*" * len(text)
    return f"{text[:keep_prefix]}...{text[-keep_suffix:]}"


class BetfairSessionManager:
    """
    Handle Betfair login and session lifecycle for API usage.

    This supports:
    - Existing session token from env
    - Interactive login API (username/password)
    - Non-interactive cert login (if cert paths configured)
    """

    def __init__(self) -> None:
        self._session_token: Optional[str] = BETFAIR_CONFIG.get("session_token")
        self._last_refreshed_at: Optional[datetime] = None
        self._last_login_method: str = "env_session_token" if self._session_token else "none"

    @property
    def sso_base_url(self) -> str:
        return BETFAIR_CONFIG["sso_base_url"].rstrip("/")

    @property
    def login_path(self) -> str:
        return BETFAIR_CONFIG.get("login_path", "/api/login")

    @property
    def cert_login_path(self) -> str:
        return BETFAIR_CONFIG.get("cert_login_path", "/api/certlogin")

    @property
    def keep_alive_path(self) -> str:
        return BETFAIR_CONFIG.get("keep_alive_path", "/api/keepAlive")

    def _headers(self) -> Dict[str, str]:
        headers = {"Accept": "application/json"}
        app_key = BETFAIR_CONFIG.get("app_key")
        if app_key:
            headers["X-Application"] = app_key
        if self._session_token:
            headers["X-Authentication"] = self._session_token
        return headers

    def _extract_token(self, payload: Dict[str, Any]) -> Optional[str]:
        return payload.get("token") or payload.get("sessionToken") or payload.get("session_token")

    def _is_success(self, payload: Dict[str, Any]) -> bool:
        status = str(payload.get("status") or payload.get("loginStatus") or "").upper()
        return status in {"SUCCESS", "LOGIN_SUCCESS"}

    def _login_interactive(self) -> Dict[str, Any]:
        username = BETFAIR_CONFIG.get("username")
        password = BETFAIR_CONFIG.get("password")
        if not username or not password:
            raise ValueError("BETFAIR_USERNAME and BETFAIR_PASSWORD are required for interactive login")

        response = requests.post(
            f"{self.sso_base_url}{self.login_path}",
            data={"username": username, "password": password},
            headers=self._headers(),
            timeout=15,
        )
        response.raise_for_status()
        return response.json()

    def _login_non_interactive(self) -> Dict[str, Any]:
        username = BETFAIR_CONFIG.get("username")
        password = BETFAIR_CONFIG.get("password")
        cert_file = BETFAIR_CONFIG.get("cert_file")
        key_file = BETFAIR_CONFIG.get("key_file")
        if not username or not password:
            raise ValueError("BETFAIR_USERNAME and BETFAIR_PASSWORD are required for cert login")
        if not cert_file or not key_file:
            raise ValueError("BETFAIR_CERT_FILE and BETFAIR_KEY_FILE are required for cert login")

        response = requests.post(
            f"{self.sso_base_url}{self.cert_login_path}",
            data={"username": username, "password": password},
            headers=self._headers(),
            cert=(cert_file, key_file),
            timeout=20,
        )
        response.raise_for_status()
        return response.json()

    def bootstrap(self, force_refresh: bool = False, prefer_cert_login: bool = False) -> Dict[str, Any]:
        """
        Ensure an active session token exists.

        Args:
            force_refresh: ignore current token and re-login
            prefer_cert_login: use cert login when available
        """
        if self._session_token and not force_refresh:
            return {
                "success": True,
                "message": "Session token already available",
                "login_method": self._last_login_method,
                "session_token_masked": _mask(self._session_token),
            }

        # If env token exists and force_refresh is not requested, keep it.
        if BETFAIR_CONFIG.get("session_token") and not force_refresh:
            self._session_token = BETFAIR_CONFIG["session_token"]
            self._last_login_method = "env_session_token"
            self._last_refreshed_at = datetime.now(timezone.utc)
            return {
                "success": True,
                "message": "Loaded session token from configuration",
                "login_method": self._last_login_method,
                "session_token_masked": _mask(self._session_token),
            }

        payload: Dict[str, Any]
        if prefer_cert_login:
            payload = self._login_non_interactive()
            method = "non_interactive_cert_login"
        else:
            try:
                payload = self._login_interactive()
                method = "interactive_api_login"
            except Exception:
                # Fallback to cert login when interactive fails and cert files are configured.
                if BETFAIR_CONFIG.get("cert_file") and BETFAIR_CONFIG.get("key_file"):
                    payload = self._login_non_interactive()
                    method = "non_interactive_cert_login"
                else:
                    raise

        token = self._extract_token(payload)
        if not token or not self._is_success(payload):
            raise RuntimeError(f"Betfair login failed: {payload}")

        self._session_token = token
        self._last_login_method = method
        self._last_refreshed_at = datetime.now(timezone.utc)
        return {
            "success": True,
            "message": "Betfair session bootstrapped",
            "login_method": method,
            "session_token_masked": _mask(token),
        }

    def keep_alive(self) -> Dict[str, Any]:
        """Refresh current Betfair session token validity."""
        if not self._session_token:
            raise ValueError("No active Betfair session token; bootstrap first")

        response = requests.post(
            f"{self.sso_base_url}{self.keep_alive_path}",
            headers=self._headers(),
            timeout=15,
        )
        response.raise_for_status()
        payload = response.json()
        if not self._is_success(payload):
            raise RuntimeError(f"Betfair keepAlive failed: {payload}")

        token = self._extract_token(payload) or self._session_token
        self._session_token = token
        self._last_refreshed_at = datetime.now(timezone.utc)
        return {
            "success": True,
            "message": "Betfair session kept alive",
            "session_token_masked": _mask(self._session_token),
            "last_refreshed_at": self._last_refreshed_at.isoformat(),
        }

    def status(self) -> Dict[str, Any]:
        """Return non-sensitive session status."""
        return {
            "success": True,
            "has_session_token": bool(self._session_token),
            "login_method": self._last_login_method,
            "session_token_masked": _mask(self._session_token),
            "last_refreshed_at": self._last_refreshed_at.isoformat() if self._last_refreshed_at else None,
            "sso_base_url": self.sso_base_url,
        }

