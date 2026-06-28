"""Wave 6 follow-up: reconcile real capital flows (deposits / withdrawals) from
the chain, so the capital_flows ledger is grounded in on-chain truth instead of
hand-typed amounts.

The wallet's USDC balance moves for two very different reasons:

  1. *Capital flows* - you fund the wallet (deposit) or pull money out
     (withdrawal). These are plain USDC ERC-20 transfers between your proxy
     wallet and an EXTERNAL address (a CEX, your own EOA, a bridge, ...).
  2. *Trading / protocol* - buying/selling shares, splitting/merging, redeeming
     winnings. These ALSO move USDC, but the counterparty is a Polymarket
     protocol contract and the transfer happens inside a Polymarket protocol
     transaction.

To recover (1) we read every USDC.e transfer touching the proxy wallet and drop
any transfer that belongs to a Polymarket protocol transaction. We identify
protocol transactions two ways (belt and braces):

  * the tx hash appears in the wallet's Polymarket data-api activity feed
    (TRADE / SPLIT / MERGE / REDEEM / REWARD / CONVERSION), and/or
  * the transfer counterparty is a known Polymarket system contract.

Whatever survives is a genuine external capital flow:
  * inbound  (to == wallet)  -> deposit
  * outbound (from == wallet) -> withdrawal

Transfer source: Etherscan V2 (Polygonscan) when an API key is configured -
one call returns the full token-transfer history. Otherwise we fall back to a
keyless, chunked `eth_getLogs` backward scan over a public Polygon RPC (slower,
but needs no key). Either way the goal is the most-recent N capital flows.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests

logger = logging.getLogger(__name__)

# --- Polygon constants --------------------------------------------------------
# Polymarket's CURRENT collateral (since ~March 2026) is its own "Polymarket USD"
# (pUSD) token. Deposits arrive as pUSD MINTS (from the zero address) to the
# proxy; trades/redemptions also mint/move pUSD but show up in the data-api
# protocol feed, which is how we tell them apart.
PUSD_ADDRESS = "0xc011a7e12a19f7b1f670d46f03b03f3342e82dfb"
# Legacy collateral: bridged USDC (USDC.e / PoS). Used pre-pUSD migration.
USDC_E_ADDRESS = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174"
# Native (Circle) USDC - only relevant if the user ever funded the wrong token.
USDC_NATIVE_ADDRESS = "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359"
USDC_DECIMALS = 6

# Tokens scanned by default (current era first, then legacy).
DEFAULT_TOKENS = [PUSD_ADDRESS, USDC_E_ADDRESS]

ZERO_ADDRESS = "0x0000000000000000000000000000000000000000"

# keccak256("Transfer(address,address,uint256)")
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"

DATA_API_BASE = "https://data-api.polymarket.com"

# Public Polygon RPC that serves keyless eth_getLogs in <=10k-block windows.
DEFAULT_RPC_URL = "https://polygon.drpc.org"
DEFAULT_RPC_CHUNK = 9_000  # stay under the 10k free-tier getLogs window

ETHERSCAN_V2_BASE = "https://api.etherscan.io/v2/api"
POLYGON_CHAIN_ID = 137

# Known Polymarket system / operator contracts on Polygon (lowercased). A token
# transfer whose counterparty is one of these is protocol settlement (a trade,
# fee, or exchange leg), never an external capital flow. NOTE: the zero address
# is deliberately NOT here - in the pUSD era a deposit IS a mint from 0x0, so we
# handle 0x0 explicitly in the classifier (mint not in the protocol feed =
# deposit; mint in the feed = redemption payout).
PM_SYSTEM_CONTRACTS: Set[str] = {
    "0x4bfb41d5b3570defd03c39a9a4d8de6bd8b8982e",  # CTF Exchange
    "0xc5d563a36ae78145c45a50134d48a1215220f80a",  # Neg-Risk CTF Exchange
    "0xd91e80cf2e7be2e162c6513ced06f1dd0da35296",  # Neg-Risk Adapter
    "0x4d97dcd97ec945f40cf65f87097ace5ea0476045",  # Conditional Tokens (CTF)
    "0xe111180000d2663c0091e4f400237545b87b996b",  # pUSD operator / settlement
    "0x115f48dc2a731aa16251c6d6e1befc42f92accc9",  # pUSD fee recipient
}

# Polygon block time is ~2.1s; used only to translate a day-lookback into a
# starting block for the keyless scan.
POLYGON_SECONDS_PER_BLOCK = 2.1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_to_iso(unix_ts: int) -> str:
    return datetime.fromtimestamp(int(unix_ts), tz=timezone.utc).isoformat()


def _topic_for_address(addr: str) -> str:
    return "0x" + addr.lower().replace("0x", "").rjust(64, "0")


def _addr_from_topic(topic: str) -> str:
    return "0x" + topic[-40:].lower()


@dataclass
class CapitalFlow:
    """A reconciled external capital movement (deposit or withdrawal)."""

    flow_type: str          # "deposit" | "withdrawal"
    amount_usdc: float
    ts_iso: str
    block: int
    tx_hash: str
    counterparty: str       # the external address funds came from / went to
    token: str              # token contract (USDC.e normally)

    def as_row(self) -> Dict[str, Any]:
        return {
            "flow_type": self.flow_type,
            "amount_usdc": round(self.amount_usdc, 6),
            "ts": self.ts_iso,
            "block": self.block,
            "tx_hash": self.tx_hash,
            "counterparty": self.counterparty,
            "token": self.token,
        }


@dataclass
class RawTransfer:
    block: int
    tx_hash: str
    from_addr: str
    to_addr: str
    amount_usdc: float
    token: str
    ts_unix: Optional[int] = None


# ---------------------------------------------------------------------------
# data-api: the set of Polymarket protocol tx hashes for this wallet
# ---------------------------------------------------------------------------

def fetch_protocol_tx_hashes(
    wallet: str,
    *,
    timeout: int = 25,
    page_limit: int = 500,
    max_pages: int = 60,
) -> Set[str]:
    """Every tx hash where this wallet did a Polymarket protocol action.

    Used to exclude trade/redeem/split USDC movements from capital flows.
    """
    seen: Set[str] = set()
    offset = 0
    for _ in range(max_pages):
        try:
            resp = requests.get(
                f"{DATA_API_BASE}/activity",
                params={"user": wallet, "limit": page_limit, "offset": offset},
                timeout=timeout,
            )
            resp.raise_for_status()
            batch = resp.json()
        except Exception as exc:  # pragma: no cover - network dependent
            logger.warning(f"data-api activity page failed (offset={offset}): {exc}")
            break
        if not isinstance(batch, list) or not batch:
            break
        for item in batch:
            h = (item.get("transactionHash") or "").lower()
            if h:
                seen.add(h)
        if len(batch) < page_limit:
            break
        offset += page_limit
        time.sleep(0.15)
    logger.info(f"Collected {len(seen)} Polymarket protocol tx hashes for {wallet}")
    return seen


# ---------------------------------------------------------------------------
# Transfer sources
# ---------------------------------------------------------------------------

class EtherscanV2Source:
    """One-call full token-transfer history via Etherscan V2 (needs API key)."""

    def __init__(self, api_key: str, timeout: int = 25):
        self.api_key = api_key
        self.timeout = timeout

    def usdc_transfers(self, wallet: str, token: str) -> List[RawTransfer]:
        out: List[RawTransfer] = []
        page = 1
        while True:
            params = {
                "chainid": POLYGON_CHAIN_ID,
                "module": "account",
                "action": "tokentx",
                "contractaddress": token,
                "address": wallet,
                "page": page,
                "offset": 1000,
                "sort": "desc",
                "apikey": self.api_key,
            }
            data = requests.get(ETHERSCAN_V2_BASE, params=params, timeout=self.timeout).json()
            rows = data.get("result")
            if data.get("status") != "1" or not isinstance(rows, list) or not rows:
                if isinstance(rows, str) and rows:
                    logger.debug(f"Etherscan: {rows}")
                break
            for r in rows:
                try:
                    raw = int(r["value"])
                    out.append(RawTransfer(
                        block=int(r["blockNumber"]),
                        tx_hash=r["hash"].lower(),
                        from_addr=r["from"].lower(),
                        to_addr=r["to"].lower(),
                        amount_usdc=raw / (10 ** USDC_DECIMALS),
                        token=token.lower(),
                        ts_unix=int(r["timeStamp"]),
                    ))
                except (KeyError, ValueError):
                    continue
            if len(rows) < 1000:
                break
            page += 1
            time.sleep(0.25)
        return out


class RpcLogSource:
    """Keyless chunked `eth_getLogs` backward scan over a public Polygon RPC.

    Scans newest-first in <=`chunk` block windows and stops once it has
    gathered at least `min_capital_flows` candidate external transfers (so we
    don't crawl the whole chain just to find the last couple of deposits).
    """

    def __init__(
        self,
        rpc_url: str = DEFAULT_RPC_URL,
        chunk: int = DEFAULT_RPC_CHUNK,
        timeout: int = 30,
        max_chunks: int = 800,
    ):
        self.rpc_url = rpc_url
        self.chunk = chunk
        self.timeout = timeout
        self.max_chunks = max_chunks
        self._id = 0
        self._block_ts_cache: Dict[int, int] = {}

    def _rpc(self, method: str, params: List[Any], _retries: int = 4) -> Any:
        """JSON-RPC call with backoff on transient/rate-limit errors.

        Range-limit errors are surfaced (the caller splits the window); only
        rate-limit / network / 5xx blips are retried here.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(_retries):
            self._id += 1
            try:
                resp = requests.post(
                    self.rpc_url,
                    json={"jsonrpc": "2.0", "id": self._id, "method": method, "params": params},
                    timeout=self.timeout,
                )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise requests.RequestException(f"HTTP {resp.status_code}")
                resp.raise_for_status()
                body = resp.json()
                if "error" in body:
                    err = body["error"]
                    msg = str(err).lower()
                    # Rate-limit dressed up as a JSON-RPC error -> retry.
                    if "rate" in msg or "limit exceeded" in msg or "too many" in msg or "capacity" in msg:
                        raise requests.RequestException(str(err))
                    raise RuntimeError(err)
                return body.get("result")
            except requests.RequestException as exc:
                last_exc = exc
                time.sleep(min(8.0, 0.8 * (2 ** attempt)))
        raise RuntimeError(f"RPC {method} failed after {_retries} retries: {last_exc}")

    def latest_block(self) -> int:
        return int(self._rpc("eth_blockNumber", []), 16)

    def _get_logs(self, token: str, topics: List[Any], lo: int, hi: int) -> List[dict]:
        try:
            return self._rpc("eth_getLogs", [{
                "fromBlock": hex(lo), "toBlock": hex(hi),
                "address": token, "topics": topics,
            }]) or []
        except RuntimeError as exc:
            msg = str(exc).lower()
            if hi > lo and ("range" in msg or "limit" in msg or "large" in msg or "too many" in msg):
                mid = (lo + hi) // 2
                return (self._get_logs(token, topics, lo, mid)
                        + self._get_logs(token, topics, mid + 1, hi))
            raise

    def _block_ts(self, block: int) -> Optional[int]:
        if block in self._block_ts_cache:
            return self._block_ts_cache[block]
        try:
            blk = self._rpc("eth_getBlockByNumber", [hex(block), False])
            ts = int(blk["timestamp"], 16)
            self._block_ts_cache[block] = ts
            return ts
        except Exception:
            return None

    def usdc_transfers(
        self,
        wallet: str,
        token: str,
        *,
        min_capital_flows: int = 2,
        protocol_txs: Optional[Set[str]] = None,
        floor_block: int = 0,
    ) -> List[RawTransfer]:
        protocol_txs = protocol_txs or set()
        wtopic = _topic_for_address(wallet)
        wallet_l = wallet.lower()
        latest = self.latest_block()
        hi = latest
        out: List[RawTransfer] = []
        candidate_flows = 0
        chunks = 0

        while hi > floor_block and chunks < self.max_chunks:
            lo = max(floor_block, hi - self.chunk + 1)
            inbound = self._get_logs(token, [TRANSFER_TOPIC, None, wtopic], lo, hi)
            outbound = self._get_logs(token, [TRANSFER_TOPIC, wtopic], lo, hi)
            for lg in inbound + outbound:
                try:
                    frm = _addr_from_topic(lg["topics"][1])
                    to = _addr_from_topic(lg["topics"][2])
                    raw = int(lg["data"], 16)
                    tx = lg["transactionHash"].lower()
                    blk = int(lg["blockNumber"], 16)
                except (KeyError, ValueError, IndexError):
                    continue
                if frm == to:
                    continue
                rt = RawTransfer(
                    block=blk, tx_hash=tx, from_addr=frm, to_addr=to,
                    amount_usdc=raw / (10 ** USDC_DECIMALS), token=token.lower(),
                )
                out.append(rt)
                # Is this a candidate external flow (not protocol)?
                other = frm if to == wallet_l else to
                if tx not in protocol_txs and other not in PM_SYSTEM_CONTRACTS:
                    candidate_flows += 1
            chunks += 1
            if chunks % 10 == 0 or candidate_flows >= min_capital_flows:
                logger.info(
                    f"scan: {chunks} chunks, blocks>={lo}, "
                    f"transfers={len(out)}, candidate_flows={candidate_flows}"
                )
            if candidate_flows >= min_capital_flows:
                break
            hi = lo - 1
            time.sleep(0.15)

        # Attach timestamps for the surviving rows' blocks (cheap; few blocks).
        for rt in out:
            ts = self._block_ts(rt.block)
            if ts is not None:
                rt.ts_unix = ts
        return out


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------

@dataclass
class ReconcileResult:
    wallet: str
    source: str
    deposits: List[CapitalFlow] = field(default_factory=list)
    withdrawals: List[CapitalFlow] = field(default_factory=list)
    n_transfers_scanned: int = 0
    n_protocol_txs: int = 0


def _classify(
    transfers: List[RawTransfer],
    wallet: str,
    protocol_txs: Set[str],
) -> Tuple[List[CapitalFlow], List[CapitalFlow]]:
    """Reduce raw token transfers to genuine external capital flows.

    A transfer is a *capital flow* only if it is NOT part of a Polymarket
    protocol transaction (trade / split / merge / redeem / reward / conversion).
    We detect protocol transactions via the data-api tx-hash set; counterparty
    address is a secondary guard.

    Deposits:
      * pUSD MINT (from 0x0) whose tx is not in the protocol feed  [pUSD era]
      * inbound transfer from an external (non-Polymarket) address  [USDC.e era]
    Withdrawals:
      * burn (to 0x0) not in the protocol feed
      * outbound transfer to an external address                    [USDC.e era]
    Trade proceeds (from the operator/exchange) and redemption mints (in the
    protocol feed) are dropped.
    """
    wallet_l = wallet.lower()
    deposits: List[CapitalFlow] = []
    withdrawals: List[CapitalFlow] = []
    seen_keys: Set[Tuple[str, str, str]] = set()  # (tx, direction, token) dedupe

    for t in transfers:
        if t.amount_usdc <= 0:
            continue
        if t.tx_hash in protocol_txs:
            # Settlement of a Polymarket protocol action (trade / redeem / ...).
            continue
        inbound = t.to_addr == wallet_l
        outbound = t.from_addr == wallet_l
        if inbound == outbound:  # neither or both — not a clean wallet flow
            continue

        counterparty = t.from_addr if inbound else t.to_addr
        # Operator / exchange counterparties are protocol settlement even if the
        # data-api feed somehow missed the tx. (0x0 is handled separately below.)
        if counterparty in PM_SYSTEM_CONTRACTS:
            continue

        direction = "deposit" if inbound else "withdrawal"
        key = (t.tx_hash, direction, t.token)
        if key in seen_keys:
            continue
        seen_keys.add(key)

        flow = CapitalFlow(
            flow_type=direction,
            amount_usdc=t.amount_usdc,
            ts_iso=_ts_to_iso(t.ts_unix) if t.ts_unix else _now_iso(),
            block=t.block,
            tx_hash=t.tx_hash,
            counterparty=counterparty,
            token=t.token,
        )
        (deposits if inbound else withdrawals).append(flow)

    deposits.sort(key=lambda f: f.block, reverse=True)
    withdrawals.sort(key=lambda f: f.block, reverse=True)
    return deposits, withdrawals


def reconcile_capital_flows(
    wallet: str,
    *,
    min_deposits: int = 2,
    lookback_days: int = 120,
    include_native: bool = False,
    etherscan_key: Optional[str] = None,
    rpc_url: Optional[str] = None,
) -> ReconcileResult:
    """Reconcile recent external capital flows for `wallet` from the chain.

    Prefers Etherscan V2 if a key is available; otherwise a keyless RPC scan.
    """
    wallet = wallet.strip()
    if not wallet:
        raise ValueError("wallet address is required")

    etherscan_key = etherscan_key or os.getenv("POLYGONSCAN_API_KEY") or os.getenv("ETHERSCAN_API_KEY")
    tokens = list(DEFAULT_TOKENS) + ([USDC_NATIVE_ADDRESS] if include_native else [])

    protocol_txs = fetch_protocol_tx_hashes(wallet)

    all_transfers: List[RawTransfer] = []
    if etherscan_key:
        source = "etherscan_v2"
        src = EtherscanV2Source(etherscan_key)
        for token in tokens:
            all_transfers.extend(src.usdc_transfers(wallet, token))
    else:
        source = "rpc_getlogs"
        src = RpcLogSource(rpc_url or DEFAULT_RPC_URL)
        latest = src.latest_block()
        floor = max(0, latest - int(lookback_days * 86400 / POLYGON_SECONDS_PER_BLOCK))
        for token in tokens:
            all_transfers.extend(src.usdc_transfers(
                wallet, token,
                min_capital_flows=min_deposits + 2,  # a little headroom
                protocol_txs=protocol_txs,
                floor_block=floor,
            ))

    deposits, withdrawals = _classify(all_transfers, wallet, protocol_txs)
    return ReconcileResult(
        wallet=wallet,
        source=source,
        deposits=deposits,
        withdrawals=withdrawals,
        n_transfers_scanned=len(all_transfers),
        n_protocol_txs=len(protocol_txs),
    )
