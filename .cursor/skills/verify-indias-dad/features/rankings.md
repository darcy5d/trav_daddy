# Rankings

ELO leaderboards for teams, batters, and bowlers, with tier breakdowns. Backed by `GET /api/rankings/{months,teams,batting,bowling,tier-stats}`.

## Sub-features

- Tier stats banner (`#tier-stats-banner`): team counts per tier T1–T5 and pending reviews.
- Team rankings (`#team-rankings`), top batters (`#batting-rankings`), top bowlers (`#bowling-rankings`); each row is `.ranking-row`.
- Filters: `#format-select` (T20), `#gender-select` (`male`/`female`), `#min-matches-select` (5/10/20/30), `#month-select` (historical month; "" = current). Each `onchange` reloads.
- Tier tabs: `#tab-international`, `#tab-regional`, `#tab-domestic`, `#tab-all`; the label shows in `#tier-filter-label`.

## How to get to it (user POV)

Nav bar → Rankings, or open `http://127.0.0.1:5001/rankings`. Lists load on page open; spinners (`.spinner`) disappear when data arrives.

## Driving it with drive.py

```bash
venv311/bin/python $S/drive.py --run-id rankings-check goto:/rankings \
  "waitgone:#team-rankings .spinner" "expectcount:#team-rankings .ranking-row>=10" \
  "expect:#tier-stats-banner~T1" "select:#gender-select=female" \
  "waitgone:#team-rankings .spinner" "expectcount:#team-rankings .ranking-row>=5" shot:rankings
```

Tier tab: `click:#tab-domestic` then `"waitgone:#team-rankings .spinner"` and `text:#team-rankings`. End state that proves it works: rows present in all three lists, first team row has a name, `(N matches)`, and an ELO number (e.g. `England (169 matches) 2118.7`), and changing a filter changes the rendered list.

Player lists (slow path, only when you changed batting/bowling rankings):

```bash
venv311/bin/python $S/drive.py --run-id rankings-players --timeout-ms 300000 goto:/rankings \
  "waitgone:#batting-rankings .spinner" "expectcount:#batting-rankings .ranking-row>=5" \
  "waitgone:#bowling-rankings .spinner" "expectcount:#bowling-rankings .ranking-row>=5" shot:rankings-players
```

## Gotchas

- `/api/rankings/batting` and `/bowling` are slow on live: on 2026-11-07 they took ~100s+ per call, and back-to-back loads queue behind each other (a 120s curl timed out). Screenshots taken right after the team list loads show the player cards still spinning; that is the baseline, not a regression. Don't reload the page repeatedly; each load queues more slow queries on the live process.
- Rows are `div.ranking-row`, not `<tr>`.
- After a filter change the spinner may flash too briefly to catch; follow `waitgone` with an `expect`/`expectcount` on the new content rather than trusting timing.
- Rankings are computed from `cricket.db`, which the 11:00 cron refresh updates; compare shapes, not exact ELO values, across runs.
