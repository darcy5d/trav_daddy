# Next Overhaul — Backlog

A running notepad of bugs, inefficiencies, and improvement ideas to address in the next major refactor. Nothing here is implemented yet. Add freely; clean up when work begins.

---

## Performance / Efficiency

### 1. CREX scraper: skip straight to Playwright for squad data
**File:** `src/api/crex_scraper.py` ~lines 909–924

Currently the scraper first attempts a static HTML parse (BeautifulSoup) to get squad data, then falls back to Playwright if either team's data is missing. Because CREX requires JavaScript to render squads, the static parse almost always fails — meaning the page gets fetched twice in the common case.

**Proposed fix:** Remove the static HTML squad parse and go straight to `fetch_squads_with_playwright()`. One fetch, same result.

---

## Bugs / Data Quality

### 2. Venue list: missing international aliases + Sydney confusion
**File:** `src/data/venue_normalizer.py` ~lines 212–266

The `VENUE_ALIASES` dict only covers Australia, India, England, and New Zealand. Large gaps exist for West Indies, South Africa, Sri Lanka, Pakistan, UAE, Bangladesh, Zimbabwe, etc.

Sydney is a known pain point — three grounds (Sydney Cricket Ground, North Sydney Oval, Cricket Central / formerly Sydney Olympic Park) are easy to confuse. Only one alias variant (`"sydney olympic" -> "Cricket Central"`) is currently handled.

Likely DB duplicates exist where the same ground was stored under slightly different name strings, silently degrading model inputs across all four models.

**Proposed fix:** Audit and expand `VENUE_ALIASES`. Run a fuzzy-duplicate check on the `venues` table. Fix any confirmed duplicates before the next training run.

### 3. Venue hierarchy: no state/province level
**File:** `src/data/venue_normalizer.py`, `docs/DATABASE_SCHEMA.md`, Flask app UI

The `venues` table has `name`, `city`, `country`, `canonical_name`, `region` — but no `state`/`province` column. State ends up either jammed into the `name` string (e.g. `"Sydney Cricket Ground, New South Wales"`) or stripped and lost by the normalizer. The UI ends up displaying a flat `Country → Full-Name-Including-State` list which is hard to navigate.

**Proposed fix:**
- Add a `state` column to the `venues` table
- Populate it during ingestion via an updated normalizer
- `name` should store only the clean ground name; `state` carries the subdivision
- Update any UI dropdowns/lists to group as `Country → State → Ground`

---

## Features / Improvements

### 4. Data exploration tab
**File:** `app/main.py`, `app/routes/`, `app/templates/`

No UI currently exists for inspecting venue data quality. With four models running, bad venue mappings silently corrupt pitch inputs across all of them.

**Proposed feature:** New tab in the Flask app showing:
- Full venue list from the DB with match counts per venue
- Flagged fuzzy-duplicate pairs above a similarity threshold
- Venues in the DB not covered by `VENUE_ALIASES` (mapping gaps)
- Ability to manually confirm or merge canonical names before a training run
- Should group by `Country → State → Ground` once the hierarchy fix (item 3) is done

### 5. Background prefetch on single predict page
**File:** `app/main.py`, `app/routes/`, `app/templates/` (single predict + bulk predict pages)

Currently data is only fetched on demand when the user actively requests a prediction. By the time the user moves from the single predict page to bulk predict and starts clicking through games, each game still has to trigger a cold fetch.

**Proposed fix:** When the single predict page loads, kick off a background fetch (e.g. via a JS `fetch()` to a backend endpoint, or a server-side thread/task) to pre-warm data for upcoming games — squads, venue stats, CREX info, etc. Results cached server-side (or in-memory) so that bulk predict clicks hit the cache instead of fetching from scratch each time.

### 6. Team 1 / Team 2 order swapped in bulk predict
**File:** `src/api/crex_scraper.py` (squad/team parsing), `app/` (bulk predict UI)

When pulling match data — particularly via CREX — the home and away teams occasionally get swapped. Two examples visible in the bulk predict review screen:
- **Victoria vs Western Australia** at WACA Ground (Perth): Western Australia is the home side but Victoria appears as Team 1
- **India vs Australia (Women)** in Adelaide: Australia is the home side but India appears as Team 1

The root cause is likely how `team1`/`team2` are determined during the CREX scrape — the order on the CREX page doesn't reliably map to home/away. The Playwright fix (item 1) may also resolve this if the JS-rendered page presents teams in a more consistent order, but it should be explicitly verified and not assumed.

**Proposed fix:** After the Playwright migration, check whether team order is now stable. If not, add a post-scrape step that infers home/away from venue country vs. team nationality and reorders `team1`/`team2` accordingly.

---
