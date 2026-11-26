# Bulk prediction

Simulate many upcoming fixtures at once and compare against Polymarket odds. Input comes from `/predict` via `sessionStorage.bulkPredictMatches`.

## Sub-features

- Phase stepper: fetch → `#phase-review` → run → `#phase-results`; progress `#fetch-progress`, `#review-summary`.
- Settings `#bulk-n-simulations`; run `#run-all-btn` (`POST /api/simulate-stream` per match, allow-listed); ETA `#eta-rate`; summary `#bulk-run-summary`.
- Market comparison (`POST /api/integrations/polymarket/compare/multi`, read-only lookups, allow-listed).

## How to get to it (user POV)

On `/predict`, click `#bulk-select-toggle`, tick match cards in the upcoming list, then the bulk bar's launch button (`launchBulkPredict()`), which navigates to `/bulk-predict`. Opening `/bulk-predict` directly shows "Preparing..." with nothing to do.

## Driving it with drive.py

Seed the selection through the real UI in the same run (sessionStorage is per browser session, so it must be one drive.py invocation):

```bash
venv311/bin/python $S/drive.py --run-id bulk --timeout-ms 180000 goto:/predict \
  "wait:#matches-list" "click:#bulk-select-toggle" \
  "click:#matches-list [data-match-url] input[type=checkbox]" "expect:#bulk-select-count~^1 match" \
  "click:#bulk-predict-bar button" "waittext:#review-summary~\w" shot:bulk-review \
  "fill:#bulk-n-simulations=1000" "click:#run-all-btn" "waittext:#bulk-run-summary~\w" shot:bulk-results
```

End state: review summary lists the selected fixture(s), run summary is populated, results phase active, `network.jsonl` shows the simulate-stream POSTs 200 and no blocked requests.

## Gotchas

- This recipe is not yet executed end to end; checkbox and bar-button selectors come from `onBulkCheckChange`/`launchBulkPredict` in `predict.html` — confirm with `count:` first and fix this file if they differ.
- Depends on live CREX fixtures; if `#matches-empty` shows, there is nothing to bulk-select — say so rather than faking `sessionStorage`.
- Direct load of `/bulk-predict` with empty storage is a valid "empty state" check only.
