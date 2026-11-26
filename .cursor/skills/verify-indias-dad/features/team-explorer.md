# Team explorer

Franchise unification UI: groups of team records that are the same franchise, fuzzy duplicate candidates, and merge proposals. Reads `GET /api/team-explorer/{franchises,duplicates,proposals}`.

## Sub-features

- Summary grid `#summary-grid`.
- Franchise groupings `#franchise-panel`, count `#franchise-count` (e.g. "387 groups (5 multi-member)").
- Fuzzy merge candidates `#duplicates-table tbody tr`, count `#duplicate-count`, threshold `#similarity`.
- Merge proposals `#proposals-table`, count `#proposal-count`.
- Write actions (BLOCKED by drive.py, do not use on live): stage merge / decision / apply → `POST /api/team-explorer/proposals*`, which rewrite franchise groupings in `cricket.db`.

## How to get to it (user POV)

Nav → Team Explorer (`/team-explorer`); panels load on open.

## Driving it with drive.py

```bash
venv311/bin/python $S/drive.py --run-id team-explorer goto:/team-explorer \
  "waitgone:#franchise-panel .spinner" "expect:#franchise-count~\d+ groups" \
  "expect:#duplicate-count~candidate pairs" "expectcount:#duplicates-table tbody tr>=1" \
  text:#proposal-count shot:team-explorer
```

End state (2026-11-07): 387 groups, 15 candidate pairs at ≥ 82%, 0 pending proposals. To prove a change in proposal writes, use a scratch instance with `--allow-post` for the one route and read the resulting rows back with `sqlite3 -readonly` — and remember scratch shares `cricket.db`, so prefer verifying with unit tests instead.

## Gotchas

- Clicking "Stage Merge" in the UI will show as a `blocked` entry in `network.jsonl`, and the page may show an error toast; that is the guard working.
