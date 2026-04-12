"""Polymarket API client (Wave 2 read-path + auth scaffolding)."""

from __future__ import annotations

from typing import Any, Dict, Optional

import requests

from config import POLYMARKET_CONFIG


class PolymarketClient:
    """Thin wrapper around Polymarket public endpoints."""

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        clob_base_url: Optional[str] = None,
        timeout_seconds: int = 15,
    ) -> None:
        self.api_base_url = (api_base_url or POLYMARKET_CONFIG["api_base_url"]).rstrip("/")
        self.clob_base_url = (clob_base_url or POLYMARKET_CONFIG["clob_base_url"]).rstrip("/")
        self.timeout_seconds = timeout_seconds

    def _get_json(self, url: str, params: Optional[Dict[str, Any]] = None, headers: Optional[Dict[str, str]] = None) -> Any:
        response = requests.get(url, params=params, headers=headers or {}, timeout=self.timeout_seconds)
        response.raise_for_status()
        return response.json()

    def _build_l2_headers(self) -> Dict[str, str]:
        """
        Build placeholder L2 headers when credentials are configured.

        Note: request signing is intentionally not implemented in Wave 2. This
        exists so the structure is ready when write-path work starts.
        """
        headers: Dict[str, str] = {}
        api_key = POLYMARKET_CONFIG.get("api_key")
        passphrase = POLYMARKET_CONFIG.get("passphrase")
        if api_key:
            headers["POLY_API_KEY"] = api_key
        if passphrase:
            headers["POLY_PASSPHRASE"] = passphrase
        return headers

    def health_check(self) -> Dict[str, Any]:
        """Check that CLOB service is reachable."""
        data = self._get_json(f"{self.clob_base_url}/ok")
        return {"success": True, "service": "polymarket", "ok": bool(data)}

    def get_markets(
        self,
        limit: int = 20,
        active: Optional[bool] = True,
        closed: Optional[bool] = False,
        slug: Optional[str] = None,
    ) -> Any:
        """
        Fetch markets from Gamma API.

        This endpoint is public and does not require API credentials.
        """
        params: Dict[str, Any] = {"limit": max(1, min(limit, 500))}
        if active is not None:
            params["active"] = str(bool(active)).lower()
        if closed is not None:
            params["closed"] = str(bool(closed)).lower()
        if slug:
            params["slug"] = slug
        return self._get_json(f"{self.api_base_url}/markets", params=params)

    def get_markets_by_slug(self, slug: str) -> Any:
        """Fetch markets by exact slug via Gamma API."""
        if not slug:
            raise ValueError("slug is required")
        return self.get_markets(limit=50, active=None, closed=None, slug=slug)

    def get_market(self, market_id: str) -> Any:
        """Fetch a single market from Gamma API by ID."""
        if not market_id:
            raise ValueError("market_id is required")
        return self._get_json(f"{self.api_base_url}/markets/{market_id}")

    def get_clob_order_book(self, token_id: str) -> Any:
        """
        Fetch token order book from CLOB read endpoint.

        For compatibility, this tries both `token_id` and `tokenID` query keys
        because SDK docs refer to tokenID while common REST usage is token_id.
        """
        if not token_id:
            raise ValueError("token_id is required")

        url = f"{self.clob_base_url}/book"
        try:
            return self._get_json(url, params={"token_id": token_id})
        except requests.RequestException:
            return self._get_json(url, params={"tokenID": token_id})

    def get_clob_midpoint(self, token_id: str) -> Any:
        """Fetch midpoint for a token from CLOB read endpoint."""
        if not token_id:
            raise ValueError("token_id is required")

        url = f"{self.clob_base_url}/midpoint"
        try:
            return self._get_json(url, params={"token_id": token_id})
        except requests.RequestException:
            return self._get_json(url, params={"tokenID": token_id})

