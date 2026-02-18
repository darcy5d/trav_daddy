# Model expansion: format tagging (T20 / ODI / Test + short-form)

Plan for tagging each series and game by format on the prediction tab, and treating short-form variants (T10, 100-ball) as T20 in the model.

---

## Current state

### Prediction tab (predict.html)
- **Gender**: Series and matches are tagged men's (blue box / "M" badge) or women's (red box / "M" badge). Data comes from `series.gender` and the left border colour.
- **Format**: Not shown. CREX API already returns `format` per match (`m.format` → "T20", "ODI"); the template does not display it. `match_type` is shown in parentheses (e.g. "4th Match") and is the match label, not the game format.

### Data sources for upcoming matches
- **CREX (default)**: `/api/crex/upcoming` returns `matches_by_series` with per-match `match_type`, `format` (from `m.format_type`). So format is available for CREX.
- **ESPN**: `/api/espn/upcoming` – need to confirm if format is included in response.
- **API (cricketdata)**: `/api/t20/upcoming` – built for T20 only; no format field in the payload.

### CREX scraper (crex_scraper.py)
- `CREXMatch.format_type`: only set to `'T20'` or `'ODI'` based on text (e.g. `'odi' in text_lower` → ODI, else T20).
- No detection for Test, T10, or 100-ball (Hundred). No “model format” mapping yet.

### Backend simulation
- `POST /api/simulate` accepts `gender` only. Simulator choice is by gender (male/female); format is not in the payload.
- Simulators (fast lookup, NN) are T20-oriented (e.g. `get_nn_simulator(gender)`, `get_fast_simulator(gender)`). ELO/other endpoints already support `format` (e.g. T20, ODI) where relevant.
- **Bulk predict** already shows format (T20/ODI/Test) and passes `format` into the match info; `buildPayload` uses it for display. Simulation itself still uses gender only (no format-specific simulator selection in bulk flow either).

### Bulk predict (bulk_predict.html)
- Already has format tags: `.tag-format-t20`, `.tag-format-odi`, `.tag-format-test` and displays format per row. Good reference for styling and behaviour.

---

## Goals

1. **UI – Prediction tab**
   - Tag each series (and therefore each game) with format: **T20**, **ODI**, **Test**, and optionally **T10**, **100** (Hundred).
   - Show format in a small badge/box (similar to gender), e.g. next to the series title and/or on each match card (consistent with bulk predict where useful).
   - Keep existing gender tagging (blue/red, M/W).

2. **Data – CREX**
   - Detect format from page/slug/text: T20, ODI, Test, T10, Hundred (100-ball).
   - Expose a single “display format” (e.g. "T20", "ODI", "Test", "T10", "100") for UI.
   - For **model/simulation**: treat T10 and 100-ball (and any future short-form) as **T20** (same simulator and logic as T20).

3. **Backend (optional for first phase)**
   - When we later support ODI/Test models: accept `format` (or derived “model format”) in simulate and select the right simulator (T20 vs ODI vs Test). For now, we can keep “model format” = T20 for all short-form (T20, T10, 100) and only use format in the API when we add ODI/Test simulators.

---

## Implementation plan

### Phase 1: UI and data (format tagging visible; short-form → T20 in logic)

| # | Task | Details |
|---|------|---------|
| 1 | **CREX: format detection** | In `_parse_schedule_entry`, extend format detection: Test (e.g. "test" in text/slug), T10 ("t10"), Hundred ("100", "hundred", "the hundred"). Set `format_type` to one of: `T20`, `ODI`, `TEST`, `T10`, `HUNDRED` (or `100`). Keep default `T20` when unknown. |
| 2 | **CREX: model format** | Add a derived field or helper: “model format” = T20 for `T20`, `T10`, `HUNDRED`; = `ODI` for `ODI`; = `TEST` for `TEST`. Use this when we pass format into simulate (Phase 2). For Phase 1, no API change needed if we only display format. |
| 3 | **CREX upstream API** | `/api/crex/upcoming` already returns `format` per match. Ensure we send the **display** format (e.g. "T10", "100") so the frontend can show it; backend can map to model format when needed. |
| 4 | **Predict tab: format badge (series)** | In `predict.html`, for each series in `loadUpcomingMatches`, derive a display format (e.g. first match’s format, or most common). Show a format badge next to the series title (e.g. "T20", "ODI", "Test", "T10", "100") using styles similar to bulk_predict (green T20, orange ODI, purple Test; add styles for T10/100 if desired, or reuse T20 style). |
| 5 | **Predict tab: format badge (match card)** | On each match row/card, show the same format tag (from `m.format`). Ensures each game is clearly tagged. Use same CSS classes as bulk_predict where possible (e.g. `tag-format`, `tag-format-t20`, etc.). |
| 6 | **Predict tab: CSS** | Add or reuse format tag styles (from bulk_predict): `.tag-format-t20`, `.tag-format-odi`, `.tag-format-test`. Add `.tag-format-t10`, `.tag-format-hundred` (or one “short form” style) if we want distinct colours; otherwise treat as T20-style. |
| 7 | **ESPN / API upcoming** | If predict tab can show matches from ESPN or cricketdata API, add a default format (e.g. T20) when `format` is missing so the new format badge always has a value. |

### Phase 2: Backend (use format in simulation)

| # | Task | Details |
|---|------|---------|
| 8 | **Simulate payload** | Add optional `format` (or `model_format`) to `POST /api/simulate`. Frontend sends display format; backend maps T10/Hundred → T20, then selects simulator by (gender, format). |
| 9 | **Simulator selection** | When ODI/Test models exist: select simulator by `format` (T20 vs ODI vs Test). For now, only T20 path exists; short-form (T10, 100) use T20 simulator. |
| 10 | **Predict tab: pass format on Select** | When user selects a match, store `format` (and/or model format) with the match state and include it in the simulate request so the correct model is used. |

### Phase 3 (later): ODI / Test models

- Train or integrate ODI and Test simulators; then wire format in simulator selection (Phase 2 tasks 8–9) and optionally show a note on the prediction tab when format is ODI/Test (e.g. “Using ODI model”).

---

## Display format vs model format (summary)

| Display (UI) | Model (simulation) |
|--------------|--------------------|
| T20         | T20                |
| T10         | T20                |
| 100 / Hundred | T20              |
| ODI         | ODI (when available) |
| Test        | Test (when available) |

T10 and 100-ball are shown as “T10” and “100” (or “Hundred”) but simulated with the T20 model.

---

## Files to touch (Phase 1)

- `src/api/crex_scraper.py` – format detection (Test, T10, Hundred); optional `model_format` or mapping.
- `app/templates/predict.html` – format badge on series and match card; use `m.format`; add or reuse tag CSS.
- `app/main.py` – only if we add a query param for CREX (e.g. include T10/Hundred in “formats” for upcoming). Optional: ensure `/api/t20/upcoming` and ESPN response include a `format` field for consistency (default T20).

---

## Notes

- **CREX get_schedule(formats=...)** currently filters by `format_type in formats`. If we add T10/Hundred, either: (a) when frontend requests “all”, pass no filter or `formats=None`; (b) or pass `['T20','T10','HUNDRED']` for “short form” so T10/100 appear. Default CREX upcoming currently uses `format_type = request.args.get('format', 'T20')` and `formats = [format_type]` – we may want to allow multiple (e.g. `formats=T20,T10,100`) or a single “short form” that includes T20+T10+100.
- **Bulk predict** already has `info.format` and tag classes; we can mirror that behaviour on the single-match prediction tab for consistency.
