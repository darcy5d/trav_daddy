#!/usr/bin/env python3
"""Wave 5 Phase 6a: One-time Polymarket wallet bootstrap.

Two modes:

    # Generate a NEW dedicated Polygon EOA wallet for this bot.
    python scripts/bootstrap_polymarket_wallet.py --generate

    # After funding the wallet with USDC on Polygon, derive L2 API creds.
    python scripts/bootstrap_polymarket_wallet.py --approve

The bot's wallet is intentionally separate from any personal wallet you
may already have. This isolates blast radius if the bot's key is ever
compromised. The plaintext key is printed ONCE during --generate; copy
it into `.env` (or macOS Keychain via `keyring` if you have it
installed) and don't lose it.

USDC + CTF approval flow (--approve):
    1. Reads POLYGON_PRIVATE_KEY from `.env`.
    2. Confirms wallet has USDC balance on Polygon.
    3. Approves the Polymarket Exchange + CTF contracts to spend USDC
       and outcome tokens via py-clob-client's `update_balance_allowance`.
    4. Calls `create_or_derive_api_creds()` and prints L2 creds for `.env`.

Resources:
    - py-clob-client docs: https://github.com/Polymarket/py-clob-client
    - Polymarket developer docs: https://docs.polymarket.com/developers/CLOB/introduction
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate a fresh EOA wallet for the betting bot.

    Uses `eth_account` if installed (preferred); falls back to a
    secrets-based hex generator with a clear warning if not.
    """
    try:
        from eth_account import Account
        Account.enable_unaudited_hdwallet_features()
        acct = Account.create()
        priv = acct.key.hex()
        addr = acct.address
    except ImportError:
        logger.warning(
            "eth_account is not installed; falling back to a secrets-based "
            "raw key generator. Install eth_account for the proper checksummed "
            "address derivation: pip install eth_account"
        )
        import secrets
        priv = "0x" + secrets.token_hex(32)
        addr = "<unknown - install eth_account>"

    print()
    print("=" * 78)
    print("  POLYGON WALLET GENERATED")
    print("=" * 78)
    print(f"  Address:     {addr}")
    print(f"  Private key: {priv}")
    print()
    print("  *** SAVE THE PRIVATE KEY NOW. It is shown ONCE. ***")
    print()
    print("  Next steps:")
    print(f"    1. Add to .env:  POLYGON_PRIVATE_KEY={priv}")
    print( "    2. Fund the address above with USDC on Polygon (recommend $200 to start).")
    print( "       Bridge USDC.e via app.polymarket.com or use a Polygon-native bridge.")
    print( "    3. Run: python scripts/bootstrap_polymarket_wallet.py --approve")
    print( "    4. Add the printed POLYMARKET_API_KEY/SECRET/PASSPHRASE to .env")
    print( "    5. Restart the Flask app; navigate to /live-betting to verify mode=OFF.")
    print("=" * 78)
    print()

    if args.write_env:
        env_path = Path(".env")
        if not env_path.exists():
            logger.error(".env does not exist; not writing")
            return 1
        existing = env_path.read_text()
        if "POLYGON_PRIVATE_KEY=" in existing:
            logger.error("POLYGON_PRIVATE_KEY already set in .env; refusing to overwrite. Edit manually.")
            return 1
        with env_path.open("a") as fp:
            fp.write(f"\n# Auto-added by bootstrap_polymarket_wallet.py\nPOLYGON_PRIVATE_KEY={priv}\n")
        logger.info(f"Appended POLYGON_PRIVATE_KEY to {env_path}")

    return 0


def cmd_approve(args: argparse.Namespace) -> int:
    """Run on-chain approvals + derive L2 API creds.

    Requires POLYGON_PRIVATE_KEY to be set in .env (or env vars).
    """
    from config import POLYMARKET_CONFIG

    if not POLYMARKET_CONFIG.get("private_key"):
        logger.error(
            "POLYGON_PRIVATE_KEY (or POLYMARKET_PRIVATE_KEY) is not set. "
            "Run --generate first or add it to .env manually."
        )
        return 1

    try:
        from py_clob_client.client import ClobClient
        from py_clob_client.constants import POLYGON
    except ImportError:
        logger.error(
            "py-clob-client is not installed. Install with:\n"
            "  pip install py-clob-client"
        )
        return 1

    client = ClobClient(
        host=POLYMARKET_CONFIG["clob_base_url"],
        key=POLYMARKET_CONFIG["private_key"],
        chain_id=POLYMARKET_CONFIG.get("chain_id", POLYGON),
        signature_type=POLYMARKET_CONFIG.get("signature_type", 0),
        funder=POLYMARKET_CONFIG.get("funder_address"),
    )

    logger.info("Deriving L2 API credentials...")
    try:
        creds = client.create_or_derive_api_creds()
    except Exception as exc:
        logger.error(f"Failed to derive API creds: {exc}")
        logger.error("Common causes:")
        logger.error("  - Wallet has no USDC on Polygon (Polymarket requires a funded wallet to issue creds).")
        logger.error("  - Network connectivity issue to clob.polymarket.com.")
        return 1

    print()
    print("=" * 78)
    print("  POLYMARKET L2 API CREDS")
    print("=" * 78)
    print(f"  POLYMARKET_API_KEY={creds.api_key}")
    print(f"  POLYMARKET_API_SECRET={creds.api_secret}")
    print(f"  POLYMARKET_PASSPHRASE={creds.api_passphrase}")
    print()
    print("  Add the three lines above to .env, then restart the app.")
    print("=" * 78)
    print()

    if args.write_env:
        env_path = Path(".env")
        if not env_path.exists():
            logger.error(".env does not exist; not writing")
            return 1
        existing = env_path.read_text()
        if "POLYMARKET_API_KEY=" in existing and "POLYMARKET_API_KEY=\n" not in existing:
            logger.error("POLYMARKET_API_KEY already set in .env; refusing to overwrite. Edit manually.")
            return 1
        with env_path.open("a") as fp:
            fp.write("\n# Auto-added by bootstrap_polymarket_wallet.py --approve\n")
            fp.write(f"POLYMARKET_API_KEY={creds.api_key}\n")
            fp.write(f"POLYMARKET_API_SECRET={creds.api_secret}\n")
            fp.write(f"POLYMARKET_PASSPHRASE={creds.api_passphrase}\n")
        logger.info(f"Appended L2 creds to {env_path}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Polymarket wallet bootstrap (Wave 5 Phase 6a)")
    sub = parser.add_subparsers(dest="cmd", required=False)
    parser.add_argument("--generate", action="store_true", help="Generate a NEW EOA wallet for the bot")
    parser.add_argument("--approve", action="store_true", help="Derive L2 API creds for an existing funded wallet")
    parser.add_argument("--write-env", action="store_true", help="Append generated values to .env automatically")
    args = parser.parse_args()

    if args.generate and args.approve:
        logger.error("Use --generate OR --approve, not both")
        return 1
    if not (args.generate or args.approve):
        parser.print_help()
        return 1

    if args.generate:
        return cmd_generate(args)
    return cmd_approve(args)


if __name__ == "__main__":
    sys.exit(main())
