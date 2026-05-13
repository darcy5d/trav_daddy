"""Polymarket API client.

Wave 2: read-path + auth scaffolding (Gamma + CLOB read).
Wave 5 Phase 5: prices-history endpoint for historical EV backtest.
Wave 5 Phase 6a: write-path integration via `py-clob-client` SDK.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import requests

from config import POLYMARKET_CONFIG

logger = logging.getLogger(__name__)


class PolymarketClient:
    """Thin wrapper around Polymarket public + CLOB endpoints.

    Read methods (no auth): `get_markets`, `get_market`, `get_clob_order_book`,
    `get_clob_midpoint`, `get_prices_history`.

    Write methods (require POLYGON_PRIVATE_KEY + POLYMARKET_API_* env vars):
    `place_market_order`, `place_limit_order`, `cancel_order`,
    `get_open_orders`, `get_positions`. The SDK client is created lazily
    so tests and read-only callers don't need credentials.
    """

    def __init__(
        self,
        api_base_url: Optional[str] = None,
        clob_base_url: Optional[str] = None,
        timeout_seconds: int = 15,
    ) -> None:
        self.api_base_url = (api_base_url or POLYMARKET_CONFIG["api_base_url"]).rstrip("/")
        self.clob_base_url = (clob_base_url or POLYMARKET_CONFIG["clob_base_url"]).rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._clob_sdk: Optional[Any] = None  # lazy py-clob-client.ClobClient

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

    def get_events(
        self,
        limit: int = 100,
        offset: int = 0,
        tag_slug: Optional[str] = None,
        closed: Optional[bool] = None,
        active: Optional[bool] = None,
    ) -> Any:
        """Wave 5 Phase 5: Gamma /events endpoint (groups multiple markets per fixture).

        For cricket: pass `tag_slug='cricket'` to filter to cricket fixtures.
        Each returned event has a `markets` list with the per-market token ids
        and labels. Pagination via `offset`.
        """
        params: Dict[str, Any] = {"limit": max(1, min(limit, 500)), "offset": max(0, offset)}
        if tag_slug:
            params["tag_slug"] = tag_slug
        if closed is not None:
            params["closed"] = str(bool(closed)).lower()
        if active is not None:
            params["active"] = str(bool(active)).lower()
        return self._get_json(f"{self.api_base_url}/events", params=params)

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

    def get_prices_history(
        self,
        token_id: str,
        start_ts: Optional[int] = None,
        end_ts: Optional[int] = None,
        interval: str = "1h",
        fidelity: int = 5,
    ) -> Any:
        """Wave 5 Phase 5: fetch /prices-history for an outcome token.

        Polymarket CLOB endpoint. Public, no auth required.

        Args:
            token_id: ERC1155 conditional-token id (string).
            start_ts: Unix timestamp (seconds); samples >= this time only.
            end_ts: Unix timestamp (seconds); samples <= this time only.
            interval: One of "max", "all", "1m", "1w", "1d", "6h", "1h".
            fidelity: Sample fidelity in minutes (default 5).

        Returns:
            JSON response. Typical shape: {"history": [{"t": int, "p": float}, ...]}.
        """
        if not token_id:
            raise ValueError("token_id is required")
        params: Dict[str, Any] = {"market": token_id, "fidelity": fidelity}
        if start_ts is not None:
            params["startTs"] = int(start_ts)
        if end_ts is not None:
            params["endTs"] = int(end_ts)
        if interval:
            params["interval"] = interval
        url = f"{self.clob_base_url}/prices-history"
        return self._get_json(url, params=params)

    # ------------------------------------------------------------------
    # Wave 5.9: Order-book analysis helpers for TWAP routing
    # ------------------------------------------------------------------

    def get_book_spread(self, token_id: str) -> Dict[str, Any]:
        """Analyze the order book and return spread/liquidity metrics.

        Returns:
            {
                "bid": float or None,         # best bid price
                "ask": float or None,         # best ask price
                "spread_pp": float or None,   # spread in percentage points (ask - bid) * 100
                "best_bid_size": float,       # total size at best bid level
                "best_ask_size": float,       # total size at best ask level
                "midpoint": float or None,    # (bid + ask) / 2
            }
        """
        book = self.get_clob_order_book(token_id)
        bids = book.get("bids", []) if isinstance(book, dict) else []
        asks = book.get("asks", []) if isinstance(book, dict) else []

        best_bid = float(bids[0]["price"]) if bids else None
        best_ask = float(asks[0]["price"]) if asks else None
        best_bid_size = float(bids[0].get("size", 0)) if bids else 0.0
        best_ask_size = float(asks[0].get("size", 0)) if asks else 0.0

        spread_pp = None
        midpoint = None
        if best_bid is not None and best_ask is not None:
            spread_pp = (best_ask - best_bid) * 100.0
            midpoint = (best_bid + best_ask) / 2.0

        return {
            "bid": best_bid,
            "ask": best_ask,
            "spread_pp": spread_pp,
            "best_bid_size": best_bid_size,
            "best_ask_size": best_ask_size,
            "midpoint": midpoint,
        }

    def get_effective_fill_price(self, token_id: str, size_usdc: float) -> Dict[str, Any]:
        """Walk the ask side of the book to compute the VWAP for a given stake.

        Simulates what a FOK market buy of `size_usdc` would actually pay by
        walking ask levels from best to worst.

        Returns:
            {
                "vwap": float or None,       # volume-weighted average price
                "total_fillable": float,     # max USDC fillable from current asks
                "levels_consumed": int,      # number of ask levels eaten into
                "slippage_pp": float or None, # (vwap - best_ask) * 100
                "asks_below_price": list,    # asks below a given price (for TWAP sizing)
            }
        """
        book = self.get_clob_order_book(token_id)
        asks = book.get("asks", []) if isinstance(book, dict) else []

        if not asks:
            return {
                "vwap": None,
                "total_fillable": 0.0,
                "levels_consumed": 0,
                "slippage_pp": None,
                "asks_below_price": [],
            }

        remaining = float(size_usdc)
        total_cost = 0.0
        total_shares = 0.0
        levels_consumed = 0

        for level in asks:
            price = float(level["price"])
            level_size_shares = float(level.get("size", 0))
            level_cost_capacity = level_size_shares * price

            if remaining <= 0:
                break

            fill_cost = min(remaining, level_cost_capacity)
            fill_shares = fill_cost / price
            total_cost += fill_cost
            total_shares += fill_shares
            remaining -= fill_cost
            levels_consumed += 1

        vwap = (total_cost / total_shares) if total_shares > 0 else None
        best_ask = float(asks[0]["price"]) if asks else None
        slippage_pp = ((vwap - best_ask) * 100.0) if (vwap and best_ask) else None

        return {
            "vwap": round(vwap, 6) if vwap else None,
            "total_fillable": round(total_cost + (size_usdc - remaining) if remaining < 0 else total_cost, 4),
            "levels_consumed": levels_consumed,
            "slippage_pp": round(slippage_pp, 2) if slippage_pp is not None else None,
            "asks_below_price": asks,
        }

    def get_asks_below_price(self, token_id: str, max_price: float) -> List[Dict[str, Any]]:
        """Return all ask levels priced at or below `max_price`.

        Each entry: {"price": float, "size": float (shares)}
        """
        book = self.get_clob_order_book(token_id)
        asks = book.get("asks", []) if isinstance(book, dict) else []
        return [
            {"price": float(a["price"]), "size": float(a.get("size", 0))}
            for a in asks
            if float(a["price"]) <= max_price
        ]

    # ------------------------------------------------------------------
    # Wave 5 Phase 6a: Write-path via py-clob-client
    # ------------------------------------------------------------------

    def _get_clob_sdk_client(self) -> Any:
        """Lazy-init the py-clob-client-v2 SDK with L1 + L2 credentials.

        Wave 5.8: migrated from `py-clob-client` (v1) to `py-clob-client-v2`
        after Polymarket's April 27 2026 protocol upgrade made v1 signed
        orders return `order_version_mismatch`. v2 defaults to the V2
        exchange contract + order schema; otherwise same semantics.

        Raises ValueError if required env vars are missing. Caches the
        client on the instance so subsequent calls are cheap.
        """
        if self._clob_sdk is not None:
            return self._clob_sdk
        try:
            from py_clob_client_v2 import ClobClient, ApiCreds
        except ImportError as exc:
            raise RuntimeError(
                "py-clob-client-v2 is not installed. Run "
                "`pip install py-clob-client-v2` (or add it to requirements.txt)."
            ) from exc

        private_key = POLYMARKET_CONFIG.get("private_key") or ""
        funder = POLYMARKET_CONFIG.get("funder_address") or ""
        chain_id = int(POLYMARKET_CONFIG.get("chain_id", 137))
        signature_type = int(POLYMARKET_CONFIG.get("signature_type", 1))

        if not private_key:
            raise ValueError(
                "POLYGON_PRIVATE_KEY (POLYMARKET_PRIVATE_KEY) is required for "
                "write-path operations. Run scripts/bootstrap_polymarket_wallet.py "
                "to generate one."
            )

        api_key = POLYMARKET_CONFIG.get("api_key") or ""
        api_secret = POLYMARKET_CONFIG.get("api_secret") or ""
        passphrase = POLYMARKET_CONFIG.get("passphrase") or ""

        # v2 ClobClient accepts creds in its constructor (unlike v1 which
        # needed a separate set_api_creds call). Fall back to deriving the
        # creds from the private key if env creds are missing.
        creds: Optional[Any] = None
        if api_key and api_secret and passphrase:
            creds = ApiCreds(api_key=api_key, api_secret=api_secret, api_passphrase=passphrase)

        client = ClobClient(
            host=self.clob_base_url,
            chain_id=chain_id,
            key=private_key,
            creds=creds,
            signature_type=signature_type,
            funder=funder if funder else None,
        )

        if creds is None:
            try:
                creds = client.create_or_derive_api_key()
                client.set_api_creds(creds)
            except Exception as exc:
                logger.warning(f"Failed to derive v2 API creds: {exc}")

        self._clob_sdk = client
        return client

    def place_market_order(
        self,
        token_id: str,
        side: str,
        amount_usdc: float,
        order_type: str = "FOK",
    ) -> Dict[str, Any]:
        """Place a market order (FOK = fill-or-kill, FAK = fill-and-kill).

        Args:
            token_id: ERC1155 outcome token id.
            side: "BUY" or "SELL" (case-insensitive).
            amount_usdc: USD-equivalent stake size.
            order_type: "FOK" (default) or "FAK".

        Returns:
            CLOB API response dict (includes `orderID`, status, fill details).
        """
        client = self._get_clob_sdk_client()
        from py_clob_client_v2 import OrderType, Side, MarketOrderArgsV2
        side_const = Side.BUY if side.upper() == "BUY" else Side.SELL
        order_type_enum = OrderType.FOK if order_type.upper() == "FOK" else OrderType.FAK

        # V2 market orders require the live user_usdc_balance. Query it fresh
        # so the signed order matches on-chain state.
        bal_info = self.get_usdc_balance()
        user_usdc_balance = float(bal_info.get("balance_usdc", 0.0))

        args = MarketOrderArgsV2(
            token_id=token_id,
            amount=float(amount_usdc),
            side=side_const,
            order_type=order_type_enum,
            user_usdc_balance=user_usdc_balance,
        )
        signed = client.create_market_order(args)
        return client.post_order(signed, order_type_enum)

    def place_limit_order(
        self,
        token_id: str,
        side: str,
        price: float,
        size_shares: float,
    ) -> Dict[str, Any]:
        """Place a GTC (good-till-cancelled) limit order.

        Args:
            token_id: ERC1155 outcome token id.
            side: "BUY" or "SELL".
            price: Price per share, in [0, 1].
            size_shares: Number of shares to buy/sell.

        Returns:
            CLOB API response dict.
        """
        client = self._get_clob_sdk_client()
        from py_clob_client_v2 import OrderType, Side, OrderArgs
        side_const = Side.BUY if side.upper() == "BUY" else Side.SELL
        args = OrderArgs(
            token_id=token_id,
            price=float(price),
            size=float(size_shares),
            side=side_const,
        )
        signed = client.create_order(args)
        return client.post_order(signed, OrderType.GTC)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """Cancel an open order by its ID."""
        client = self._get_clob_sdk_client()
        return client.cancel(order_id=order_id)

    def cancel_all_orders(self) -> Dict[str, Any]:
        """Cancel ALL open orders for this wallet (kill-switch helper)."""
        client = self._get_clob_sdk_client()
        return client.cancel_all()

    def get_open_orders(self) -> Any:
        """Return all open orders for this wallet."""
        client = self._get_clob_sdk_client()
        if hasattr(client, "get_open_orders"):
            return client.get_open_orders()
        if hasattr(client, "get_orders"):
            return client.get_orders()
        return []

    def get_positions(self) -> Any:
        """Return current positions (token holdings) for this wallet."""
        client = self._get_clob_sdk_client()
        # Different SDK versions expose this under different names; try both.
        if hasattr(client, "get_positions"):
            return client.get_positions()
        if hasattr(client, "get_orderbook_positions"):
            return client.get_orderbook_positions()
        return []

    def get_token_midpoints(self, token_ids: List[str]) -> Dict[str, float]:
        """Wave 5.8: batched midpoint query for a list of outcome tokens.

        Returns `{token_id: midpoint_price}` as floats. Tokens the CLOB
        doesn't return a price for are omitted. Uses the v2 SDK's batched
        `get_midpoints` to avoid one request per position.
        """
        if not token_ids:
            return {}
        client = self._get_clob_sdk_client()
        # v2 SDK accepts a list of {'token_id': ...} dicts and returns
        # {token_id_str: price_str}. Normalise to float dict.
        params = [{"token_id": t} for t in token_ids if t]
        if not params:
            return {}
        resp = client.get_midpoints(params)
        out: Dict[str, float] = {}
        if isinstance(resp, dict):
            for tid, price_str in resp.items():
                try:
                    out[str(tid)] = float(price_str)
                except (TypeError, ValueError):
                    continue
        return out

    def get_usdc_balance(self) -> Dict[str, Any]:
        """Wave 5.8: live USDC balance + on-chain allowance status for the
        proxy wallet. Uses py-clob-client's get_balance_allowance (L2-authed).

        Returns:
            {
                "balance_usdc": float,       # USDC balance (6 decimals normalised to float)
                "balance_raw": str,          # exact micro-USDC value as string
                "allowances_ok": bool,       # True iff all Polymarket contracts have >0 allowance
                "allowances": dict,          # contract_address -> allowance (raw micro-USDC string)
                "signature_type": int,       # 0/1/2 signature_type used for this wallet
            }
        """
        from py_clob_client_v2 import BalanceAllowanceParams, AssetType

        client = self._get_clob_sdk_client()
        signature_type = int(POLYMARKET_CONFIG.get("signature_type", 1))
        params = BalanceAllowanceParams(
            asset_type=AssetType.COLLATERAL,
            signature_type=signature_type,
        )
        resp = client.get_balance_allowance(params)
        balance_raw = str(resp.get("balance", "0") or "0")
        try:
            balance_usdc = int(balance_raw) / 1_000_000.0
        except (TypeError, ValueError):
            balance_usdc = 0.0
        allowances = resp.get("allowances", {}) or {}
        # Allowances are set to max-uint256 when approvals are healthy; any
        # nonzero value is fine but we guard against all-zero explicitly.
        allowances_ok = len(allowances) > 0 and all(
            int(v or 0) > 0 for v in allowances.values()
        )
        return {
            "balance_usdc": round(balance_usdc, 4),
            "balance_raw": balance_raw,
            "allowances_ok": allowances_ok,
            "allowances": allowances,
            "signature_type": int(POLYMARKET_CONFIG.get("signature_type", 0)),
        }

