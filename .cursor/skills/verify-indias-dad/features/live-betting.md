# Live betting

Dashboard for the real-money Polymarket bot: wallet, portfolio, strategies, upcoming fixtures with edges, open/settled bets, calibration. On :5001 the bot is live (`#status-mode` showed `AUTO` on 2026-11-07).

## Sub-features

- Wallet/portfolio header: `#portfolio-value`, `#wallet-balance`, `#wallet-envelope`, `#wallet-funded-pct`, `#wallet-approvals` (`GET /api/betting/wallet` reads on-chain balance — network, read-only).
- Totals `#lb-total-pnl`, `#lb-total-open`; status row `#status-mode`, `#status-todays-bets`, `#status-cap-today`, `#status-settled-total`.
- Bankroll chart `#lb-bankroll-chart`; strategies `#strategies-table tbody tr`; fixtures `#lb-fixtures-body tr`; league policy `#lb-league-policy-table`; open-bet filter chips `#lb-open-filter-chips`; settled `#lb-settled-body`, `#lb-settled-count`; surfaced edges `#surfaced-edges-list`; dashboard `#dash-cum-pl`, `#dash-brier-all`, `#dash-per-market-body`, `#dash-calibration-body`.
- DANGER controls — never click, on any instance: `#kill-switch-btn` (writes `.env`), `#lb-poller-toggle` (starts/stops the auto post-toss poller), the mode control, the post-toss modal (`#lb-fetch-crex-xi`, `#lb-toss-modal-submit` = "Submit Live Post-Toss Scan"), scale-up. drive.py blocks their POSTs, but do not rely on that as a reason to click.

## How to get to it (user POV)

Nav → Live Betting (`/live-betting`). Everything populates from GETs on load.

## Driving it with drive.py

```bash
venv311/bin/python $S/drive.py --run-id live-betting --timeout-ms 60000 goto:/live-betting \
  "waittext:#status-mode~\w{2,}" text:#status-mode text:#wallet-balance text:#portfolio-value \
  "expectcount:#strategies-table tbody tr>=1" "count:#lb-fixtures-body tr" shot:live-betting
```

End state (2026-11-07): mode AUTO, wallet $4.67, 6 strategy rows, 35 fixture rows, zero blocked requests, zero page errors. A non-empty `blocked_requests` in `result.json` after a read-only drive means the page started POSTing on load — that is a finding to report.

## Gotchas

- Values move constantly (crons every 3–30 min); assert shapes/regexes, not exact money values.
- On a scratch instance the wallet card shows "POLYGON_PRIVATE_KEY not set" because scratch blanks credentials — expected.
- Never verify bet placement, cash-out, redemption, or reconcile through the UI. Use `tests/` with mocks at the existing client boundary.
