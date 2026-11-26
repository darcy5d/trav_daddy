# Single-match prediction

Pick two XIs (from upcoming fixtures or manually), choose venue/simulator, run Monte Carlo simulations, see win probabilities and score ranges. Runs `POST /api/simulate-stream` (compute only, allow-listed in drive.py).

## Sub-features

- Upcoming matches card (`#upcoming-matches`, list `#matches-list`, empty `#matches-empty`) with source control `#data-source-control` and filters `#upcoming-gender-filter`, `#upcoming-format-filter`; picking a match fills `#t20-match-info` and both XIs (CREX/ESPN scraping — external network, slow).
- Bulk selection: `#bulk-select-toggle`, checkboxes on match cards, `#bulk-select-count`, `#bulk-predict-bar` → hands off to `/bulk-predict` via `sessionStorage.bulkPredictMatches`.
- Manual team selection per side: toggle `#team{1,2}-manual-section .manual-mode-toggle`, gender `#team{N}-gender-control button[data-value=male|female]`, type `#team{N}-type-control button[data-value=all|international|club]`, team `#team{N}-select`. Selecting a team auto-picks an XI into `#team{N}-selected-list`, count in `#team{N}-selected-count`.
- Settings: `#venue-select`, `#use-toss`, `#simulator-type` (`fast`|`nn`), `#n-simulations`.
- Run: `#run-simulation-btn`; status `#predict-model-status`.
- Results card `#results`: `#result-team{1,2}-name`, `-prob`, `-score`, `-range`, toss text `#toss-stats-text`.
- Polymarket odds card for a selected fixture (`#single-polymarket-presim-card`, GET compare — read-only).

## How to get to it (user POV)

Nav → Predict (`/predict`). Either click an upcoming match, or expand "📝 Manual Team Selection" on both sides and pick teams; then press Run Simulation.

## Driving it with drive.py

Manual path (verified 2026-11-07, ~45s, India 40.2% / Australia 59.8% with fast/1000):

```bash
venv311/bin/python $S/drive.py --run-id predict-manual --timeout-ms 120000 goto:/predict \
  "click:#team1-manual-section .manual-mode-toggle" "click:#team2-manual-section .manual-mode-toggle" \
  "select:#team1-select=India" "select:#team2-select=Australia" \
  "waittext:#team1-selected-count~11/11" "waittext:#team2-selected-count~11/11" \
  "select:#simulator-type=fast" "fill:#n-simulations=1000" dialogs:accept \
  "click:#run-simulation-btn" "wait:#results" "waittext:#result-team1-prob~\d+(\.\d+)?%" \
  text:#result-team1-name text:#result-team1-prob text:#result-team2-prob shot:predict-result
```

End state: `#results` visible, the two probabilities sum to ~100%, names match the chosen teams, and `network.jsonl` shows `POST /api/simulate-stream` 200. For NN changes use `select:#simulator-type=nn` (slower; needs an active model — see training).

## Gotchas

- `select:` waits for the option, so no sleep is needed while `/api/teams` loads (354 options for men's).
- `confirm()` appears if an XI has <11 players or unmatched players; `dialogs:accept` before the run click, and check `console.log` for what was asked.
- Baseline: one `[pageerror] Cannot read properties of null (reading 'selectedOptions')` on load — pre-existing, not your regression.
- If no active model exists for the gender/format, the run is blocked with an alert / HTTP 503.
- Upcoming-match flows scrape CREX/ESPN live; prefer the manual path for deterministic proofs.
