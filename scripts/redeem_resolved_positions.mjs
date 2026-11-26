#!/usr/bin/env node
/**
 * Gasless Polymarket proxy-wallet redeemer.
 *
 * Reads a JSON payload from stdin:
 *   {"dryRun": true, "items": [{"conditionId":"0x...", "negRisk":false,
 *   "yesAmount":"0", "noAmount":"0"}]}
 *
 * It intentionally does not discover positions itself. The Python caller owns
 * eligibility and the SQLite audit trail; this process only encodes and
 * submits an already-approved redemption plan.
 */
import "dotenv/config";
import { readFileSync } from "node:fs";
import {
  createWalletClient,
  encodeFunctionData,
  http,
  zeroHash,
} from "viem";
import { privateKeyToAccount } from "viem/accounts";
import { polygon } from "viem/chains";
import {
  RelayClient,
  RelayerTxType,
} from "@polymarket/builder-relayer-client";

const CTF = "0x4D97DCd97eC945f40cF65F87097ACe5EA0476045";
const PUSD = "0xC011a7E12a19f7B1f670d46F03B03f3342E82DFB";
const NEG_RISK_ADAPTER = "0xadA2005600Dec949baf300f4C6120000bDB6eAab";

const ctfAbi = [{
  type: "function",
  name: "redeemPositions",
  stateMutability: "nonpayable",
  inputs: [
    { name: "collateralToken", type: "address" },
    { name: "parentCollectionId", type: "bytes32" },
    { name: "conditionId", type: "bytes32" },
    { name: "indexSets", type: "uint256[]" },
  ],
  outputs: [],
}];

const negRiskAbi = [{
  type: "function",
  name: "redeemPositions",
  stateMutability: "nonpayable",
  inputs: [
    { name: "_conditionId", type: "bytes32" },
    { name: "_amounts", type: "uint256[]" },
  ],
  outputs: [],
}];

function emit(value) {
  process.stdout.write(`${JSON.stringify(value)}\n`);
}

function fail(message) {
  emit({ ok: false, error: message });
  process.exitCode = 1;
}

let payload;
try {
  payload = JSON.parse(readFileSync(0, "utf8"));
} catch (error) {
  fail(`invalid redemption payload: ${error.message}`);
  process.exit();
}

if (!Array.isArray(payload.items) || payload.items.length === 0) {
  emit({ ok: true, results: [] });
  process.exit();
}

for (const item of payload.items) {
  if (!/^0x[0-9a-fA-F]{64}$/.test(item.conditionId || "")) {
    fail(`invalid conditionId: ${item.conditionId || "<missing>"}`);
    process.exit();
  }
}

if (payload.dryRun) {
  emit({
    ok: true,
    dryRun: true,
    results: payload.items.map((item) => ({
      conditionId: item.conditionId,
      negRisk: Boolean(item.negRisk),
      status: "planned",
    })),
  });
  process.exit();
}

const privateKey = process.env.POLYGON_PRIVATE_KEY || process.env.POLYMARKET_PRIVATE_KEY;
const relayerUrl = process.env.POLYMARKET_RELAYER_URL;
const rpcUrl = process.env.POLYGON_RPC_URL || "https://polygon-rpc.com";
if (!privateKey || !relayerUrl) {
  fail("POLYGON_PRIVATE_KEY and POLYMARKET_RELAYER_URL are required for live redemption");
  process.exit();
}

try {
  const account = privateKeyToAccount(privateKey);
  const wallet = createWalletClient({ account, chain: polygon, transport: http(rpcUrl) });
  const client = new RelayClient(
    relayerUrl,
    137,
    wallet,
    undefined,
    RelayerTxType.PROXY,
  );

  const results = [];
  for (const item of payload.items) {
    const transaction = item.negRisk
      ? {
          to: NEG_RISK_ADAPTER,
          data: encodeFunctionData({
            abi: negRiskAbi,
            functionName: "redeemPositions",
            args: [
              item.conditionId,
              [BigInt(item.yesAmount || "0"), BigInt(item.noAmount || "0")],
            ],
          }),
          value: "0",
        }
      : {
          to: CTF,
          data: encodeFunctionData({
            abi: ctfAbi,
            functionName: "redeemPositions",
            args: [PUSD, zeroHash, item.conditionId, [1n, 2n]],
          }),
          value: "0",
        };
    try {
      const response = await client.execute([transaction], `redeem ${item.conditionId}`);
      const outcome = await response.wait();
      results.push({
        conditionId: item.conditionId,
        negRisk: Boolean(item.negRisk),
        status: outcome?.transactionHash ? "confirmed" : "failed",
        transactionHash: outcome?.transactionHash || null,
      });
    } catch (error) {
      results.push({
        conditionId: item.conditionId,
        negRisk: Boolean(item.negRisk),
        status: "failed",
        error: error instanceof Error ? error.message : String(error),
      });
    }
  }
  emit({ ok: results.every((result) => result.status === "confirmed"), results });
} catch (error) {
  fail(error instanceof Error ? error.message : String(error));
}
