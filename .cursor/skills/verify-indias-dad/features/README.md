# indias_dad feature map

One file per user-facing page. Each answers: what it is, how a user reaches it, how to drive it with `scripts/drive.py`, and what end state proves it works. All pages share the nav in `app/templates/base.html`; routes are in `app/main.py`.

| Feature | Route | Template | Safe to drive on live? |
|---|---|---|---|
| [Single-match prediction](predict.md) | `/predict` | `predict.html` | Yes (simulate POSTs are compute-only) |
| [Bulk prediction](bulk-predict.md) | `/bulk-predict` | `bulk_predict.html` | Yes, read/compute only |
| [Rankings](rankings.md) | `/rankings` | `rankings.html` | Yes, GET only |
| [Team explorer](team-explorer.md) | `/team-explorer` | `team_explorer.html` | Read view only; proposal buttons are blocked |
| [Training](training.md) | `/training` | `training.html` | Read view only; every button is blocked |
| [Live betting](live-betting.md) | `/live-betting` | `live_betting.html` | Read view only; NEVER click its buttons |

Not yet mapped: `/` (home stats), `/data-explorer`, `/live-risk`, `/paper-trades`, `/healthz` (doctor covers it).

Known baseline (2026-11-07, live :5001): `/predict` logs one pre-existing `[pageerror] Cannot read properties of null (reading 'selectedOptions')`; other mapped pages load with zero page errors. `/rankings` player lists take ~100s+ to load.
