# Next Overhaul — Backlog

A running notepad of bugs, inefficiencies, and improvement ideas to address in the next major refactor. Some areas are already complete, some are in the current wave, and the rest are deferred. Keep statuses updated as work lands.

## Wave Tracking (branch: `major-rework`)

### Already Completed (pre-Wave 1)
- Team variant display + parent-team training mapping pipeline is in place (A-teams and women variants included where available).
- CREX variant resolution/mapping coverage work has been applied (this is no longer a primary implementation item for this wave).

### In Wave 1 (current implementation scope)
- 1. CREX scraper: Playwright-only squad fetch path
- 2. Venue aliases expansion + duplicate review workflow
- 3. Venue hierarchy: add `state`/`province` support
- 4. Data exploration tab (venue quality tooling)
- 5. Background prefetch for single -> bulk prediction flow
- 6. Team 1 / Team 2 order stabilization (home/away consistency)

### Explicitly Not In Wave 1 (deferred to next waves)
- 7. Live Monte Carlo running-state UI polish
- 8. Polymarket integration (live odds + GUI bet actions)
- 9. Betfair integration + best-odds comparison
- 10. Autonomous runner (discover -> simulate -> bet)
- 11. Bet ledger / audit trail
- 12. Cross-venue true-arb pathway
- 13. Liquidity depth research + sizing analytics
- 14. `cricketdata` R ingest pathway
- 15. Ball outcome class-imbalance model work
- 16. Match-level backtest (simulated vs actual outcomes)
- 17. Cricsheet Register: ingest and wire cross-source player identifiers (CREX, Cricinfo, internal DB)

### Targeted Validation Note (Wave 1)
- Confirm whether separate data is present and properly disaggregated for variants like Australia A / Australia A Women in existing team + ball-by-ball records. If gaps exist, capture as follow-up tasks under next waves.

### Wave 1 Progress Snapshot
- **Done (Sprint 0):** Australia A / variant data audit on current DB completed. Result: no dedicated `Australia A` team rows currently exist; women separation is tracked via `matches.gender` rather than team-name variants.
- **Done (Sprint 1):** CREX squad fetch path updated to Playwright-first for match-detail squad extraction; static-first squad parse path removed from the main flow.
- **Done (Sprint 1):** Team order normalization added using venue-country and venue-domestic hints (including regression coverage for Adelaide and WACA-style swap cases).
- **Done (Sprint 2):** Venue alias coverage expanded and state/province extraction/migration scaffolding added.
- **Done (Sprint 3):** Data explorer tab added with backend payload + UI panels for hierarchy, fuzzy duplicate candidates, and alias-gap review.
- **Done (Sprint 4):** Background prefetch restored for CREX upcoming cards with TTL cache reuse across single-select and bulk-predict flows.
- **Done (Sprint 5):** Final stabilization/docs pass completed (config-driven Flask secret handling, `.env.example` template with deferred market-integration placeholders, README env setup notes).

### In Wave 2 (current implementation scope)
- 7. Bulk predict: polish live Monte Carlo running-state UI
- 8. Polymarket integration (read-path only: model prob vs market implied)
- 9. Betfair integration (read-path only: best-odds comparison foundations) — **parked for now; Polymarket-first execution**

### Wave 2 Progress Snapshot
- **Done (Sprint 0):** Credentials/config readiness scaffold added for market integrations (`config.py` + `.env.example`) with explicit read-path defaults.
- **Done (Sprint 0):** Added integration credential status endpoint (`/api/integrations/credentials-status`) returning masked previews + missing-field diagnostics.
- **Done (Sprint 0):** README updated with Wave 2 market env vars and readiness API documentation.
- **Done (Sprint 1):** Added Polymarket API client + read endpoints (`health`, `markets`, `orderbook`) so read-path wiring can proceed without trading enablement.
- **Done (Sprint 1):** Added Betfair session bootstrap/keep-alive/status endpoints with masked token handling and configurable login paths.
- **Done (Sprint 1.1):** Switched readiness gating to Polymarket-first so Wave 2 can proceed without Betfair credentials; Betfair remains optional diagnostics/scaffolding.
- **Done (Sprint 2):** Added Polymarket fixture comparison service + endpoints (`/api/integrations/polymarket/compare`, `/api/integrations/polymarket/compare/batch`) with linker confidence, quote freshness, and explicit status branches (`ok`, `no_match`, `quote_unavailable`, `stale`).
- **Done (Sprint 2):** Bulk Predict now renders per-row Polymarket market cards after simulation completes; Single Predict adds compact Polymarket comparison panel in results.
- **Done (Sprint 2):** Live-running UI state in Bulk Predict simplified (clearer primary progress, cleaner ETA behavior).
- **Deferred (later wave):** Actual order placement/execution credentials and write-path flows remain intentionally out of scope for this read-only wave.

### Wave 1 Sprint Sequence (execution order)
- **Sprint 0 (Baseline):** scope lock, completed-vs-deferred status check, Australia A disaggregation audit, baseline latency snapshots.
- **Sprint 1 (CREX reliability):** Playwright-only squads + team-order stabilization + regression checks.
- **Sprint 2 (Venue foundations):** alias expansion, duplicate report, `venues.state` migration, normalizer/schema updates.
- **Sprint 3 (Data explorer UI):** venue quality tab with counts, duplicate candidates, alias-gap panels, grouped by country/state/ground.
- **Sprint 4 (Prefetch performance):** background prefetch endpoint, warm-cache usage from single -> bulk flow, TTL/invalidation.
- **Sprint 5 (Stabilize/docs):** cross-feature smoke checks, README updates, final status pass in this backlog doc.

**Dependency notes:**
- Sprint 3 depends on Sprint 2 outputs (`state` and venue quality data feeds).
- Sprint 4 can run partly in parallel with Sprint 3 once Sprint 1 is stable.

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

## Team Variant Training Data Coverage

### Current System (Post-Team Variants Migration)

**Team Display vs Database Integration:**
- **Display Names**: Show full variant distinctions (Hong Kong A, Malaysia A, Australia Women, Australia U19)
- **Training Data**: Use parent team T20 history for predictions (A-teams use main team data, Women's use women's data where available)
- **Database Matching**: All variants validated through existing `match_team_to_db()` pipeline

**Training Data Coverage by Variant Type:**
- **Main Teams**: Full T20 history from Cricsheet (Australia: 298 matches, Hong Kong: 207 matches, Malaysia: 212 matches)
- **Women's Teams**: Separate training data where available in database (Australia Women, India Women, etc.)
- **A-Teams**: NO separate training data - uses parent team T20 history (Hong Kong A uses Hong Kong's 207 T20 matches)
- **U19 Teams**: NO separate training data - uses parent team T20 history (Australia U19 uses Australia's 298 T20 matches)  
- **Special Teams**: NO separate training data - uses parent team or default ELO

**Current Variant Resolution (crex_team_variants table):**
- **133 team variants** mapped from CREX directory with **88.0% database match rate**
- **A-teams: 5/5 (100%)** - Hong Kong A, Malaysia A, etc. all map to parent teams
- **Women's teams: 9/10 (90%)** - Most women's teams have database matches
- **U19 teams: 100%** - Map to parent teams appropriately

### Implications for Match Predictions

**A-Team Matches (e.g., Malaysia A vs Hong Kong A):**
- **Display**: ✅ Correct ("Malaysia A vs Hong Kong A" not "1FN vs 1FO")
- **Training Data**: Uses parent team history (Malaysia's 212 matches, Hong Kong's 207 matches)
- **Prediction Quality**: Good baseline from parent team data, but may not capture development squad differences
- **Warnings**: None (parent teams have sufficient training data)

**Women's Matches:**
- **Display**: ✅ Correct ("Australia Women vs India Women")
- **Training Data**: Uses women's-specific data when available in database
- **Prediction Quality**: High when separate women's training data exists
- **Warnings**: Generated when women's team not in database (falls back to main team data)

**U19 Matches:**
- **Display**: ✅ Correct ("Australia U19 vs England U19") 
- **Training Data**: Uses senior team history (Australia's 298 matches vs England's data)
- **Prediction Quality**: Reasonable baseline, but U19 performance may differ significantly from senior teams
- **Warnings**: None for senior teams with good history, generated for countries with limited data

### Recommendations

**Immediate (Current System):**
- ✅ **Implemented**: Proper team variant display names while preserving database integration
- ✅ **Implemented**: A-teams use parent team training data (sensible baseline)
- ✅ **Implemented**: Comprehensive CREX team directory mapping (133 variants)

**Future Enhancements (If Match Volume Justifies):**
- **A-Team Data Collection**: If A-team matches become frequent, consider collecting separate training data
- **U19 Performance Factors**: Research if U19 teams perform significantly differently than senior teams  
- **Development Squad Modeling**: Potential ELO adjustments for development squad predictions

**Database Expansion Opportunities:**
- 16 unmatched variants identified (mostly domestic/franchise teams) 
- Use `resolver.get_unmatched_variants()` to prioritize database additions
- Monitor prediction requests for high-volume unmatched teams

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

### 7. Bulk predict: polish the live Monte Carlo “running” state
**File:** `app/` (bulk predict templates + JS/CSS), any shared components driving the sim progress UI

While a row is simulating, the result cell shows progress, ETA-style countdown, and thin provisional win bars — but the overall look reads as busy and unfinished compared to the clean “Done” rows with full split bars and stats.

**Proposed fix:** Redesign the in-progress state so it feels intentional and on-brand: clearer hierarchy (primary progress vs. provisional odds), smoother motion (skeleton/shimmer or subtle pulse instead of static clutter), consistent typography with completed rows, and fix oddities like negative or confusing countdown copy. Optionally add a compact “live” indicator so “Running…” matches the visual weight of “Done”.

### 8. Polymarket integration (model prob vs. market price)
**File:** New module e.g. `src/integrations/polymarket/` (or `app/` routes + config), `app/templates/` (bulk/single predict UI)

Bulk predict already produces Monte Carlo win probabilities for fixtures that overlap with real prediction markets. Polymarket lists head-to-head cricket markets (international qualifiers, IPL, PSL, etc.) with implied probabilities from YES/NO prices in cents. Many of those same matches have ball-by-ball coverage in our pipeline — so we can compare **model win probability** to **market-implied odds**, surface edge in the UI, and eventually wire **read-only** market discovery + (optionally) **order placement** via the official Polymarket API / CLOB, subject to auth, jurisdiction, and risk controls.

**Proposed direction (phased):**

- **Match linking:** Fuzzy-map our `team1`/`team2` + league/date to Polymarket market metadata (handle abbreviations vs. full names, e.g. KOL vs Kolkata Knight Riders).
- **Read path:** Poll or subscribe to market prices; show alongside each bulk-predict row: model %, implied market %, and simple “edge” = model minus market (with fee/slippage notes).
- **Write path (later):** If API keys and compliance allow, optional “paper” vs. live execution; never auto-bet without explicit caps and audit logging.
- **With item 9:** Add Betfair Exchange prices so the UI can compare **best available odds** (commission-adjusted) across venues, not only Polymarket.
- **Guardrails:** Treat as experimental capital — calibration (item 16) and class-imbalance fixes (item 15) directly affect whether edges are real.

### 9. Betfair integration + best-odds comparison
**File:** New module e.g. `src/integrations/betfair/` (Betfair Exchange API), shared `src/integrations/odds/` helpers, `app/templates/` (bulk/single predict UI)

Betfair Exchange exposes cricket **Match Odds** (and related markets) with **back** and **lay** prices and liquidity. For the same fixtures we already simulate, we can ingest live or delayed prices (per API/product rules), normalize to implied win probability, and **compare venues**: model vs. Polymarket vs. Betfair (and highlight **best priced side** after a simple commission adjustment on Exchange).

**Proposed direction (phased):**

- **Auth:** App key + session token flow (cert-based login for automated access where allowed); store secrets outside the repo.
- **Match linking:** Map event/market IDs to our `team1`/`team2` + competition + start time; handle naming drift and postponed fixtures.
- **Read path:** Pull current odds (respecting delay rules for in-play if applicable); show a compact row: best back, best lay, mid, and implied % with Betfair commission netted in.
- **Lay side (execution):** Support **laying** runners on Betfair (offer to take backers’ money), not only **backing**. When the book is **thin**, advertised backs can be stale or overpriced — laying can be the natural way to express a model view that a team is **overpriced** (short the outcome), subject to **liability** caps and commission-on-winnings maths. Polymarket-style binary books do not have a first-class lay; this is primarily an **Exchange** capability in our stack.
- **Aggregation:** One UI column or panel: “best book” = max expected value vs. model among connected feeds (Polymarket + Betfair first; more exchanges later if useful) — include whether the actionable quote is **back** or **lay** on Exchange.
- **Write path (later):** Optional place/cancel orders via Exchange API for **both** back and lay with explicit **max liability** on lays, dry-run mode, and audit logs — same compliance bar as item 8.
- **Next:** Item 10 runs the analyser on a schedule and may execute when a configured **risk matrix** says so.

### 10. Autonomous runner: discover → simulate → bet (risk matrix)
**File:** New service e.g. `src/automation/` or `scripts/trading_loop/`, worker + config (YAML/JSON), reuse `app/` APIs or headless pipeline entrypoints; depends on the bet/odds stack in items **8–9** (and whatever UI we call the “bet analyser”).

After Polymarket/Betfair read paths and the combined **edge / best-odds** view exist, the natural step is a **headless loop** that does not require clicking Bulk Predict: poll for upcoming and in-play cricket fixtures we can model, match them to exchange markets, run the Monte Carlo stack, compare to live prices, and **submit orders** only when signals pass a user-defined **risk matrix**. **Placement** uses the same integrations as items **8–9**: either **Polymarket** or **Betfair** (or whichever book offers the best net price), according to config — e.g. enable one or both venues, prefer a primary book, or always route to the best quote that passes the risk checks.

**Proposed building blocks:**

- **Discovery:** Scheduled job that lists candidate matches (fixture feed + market catalog); skip sports/leagues/comps excluded in config.
- **Pipeline:** Fetch squads/venue (reuse CREX path), run sim to fresh win probs, pull latest odds from connected books, compute edge vs. best price.
- **Risk matrix (configurable):** Examples — minimum edge or minimum EV; max stake per bet and per day; **max lay liability** per market/day (Betfair); max concurrent open positions; Kelly capped at _x_%; minimum liquidity; pre-match only vs. allow in-play; bankroll or notional cap; optional “max odds” to avoid longshots; cooldown between bets on same event; optional toggle **allow_lay** vs back-only.
- **Execution:** Paper mode (log intended stakes only) default; live mode gated by env flag + explicit confirmation. Send real orders via **Polymarket** and/or **Betfair** APIs (items 8–9): config can restrict to one venue or **auto-pick** the side with best net price/liquidity that still satisfies the matrix; on **Betfair**, allow **back or lay** when the risk matrix and depth checks pass (e.g. model implies a lower win % than the best **back** — consider **laying** that runner at the top of a thin ladder rather than forcing a Polymarket-only hedge). Full audit log (inputs, probs, prices, venue, **back vs lay**, stake/liability, rationale).
- **Safety:** Global kill switch, alert on API failures, never double-place on retry without idempotency keys.
- **Audit:** Every intended and filled bet should append to the **bet ledger** (item 11); the runner should not rely on ad-hoc logs alone.

**Depends on:** Reliable match↔market linking (items 8–9), sane model calibration — see items **15–16**, realistic **liquidity / depth** assumptions (item 13), and persisted bets for post-mortems (item 11).

### 11. Bet ledger / audit trail (DB or append-only log)
**File:** New tables e.g. `bets` / `bet_legs` in DB (preferred) or rotated structured JSONL; write path from items **8–10** and any manual UI “confirm bet” actions.

We need a **single source of truth** for what was wagered, why, and how it resolved — especially for tuning the risk matrix and proving model calibration vs. realised returns.

**Proposed schema (illustrative):**

- **Identity:** `bet_id`, idempotency / external order id(s), venue (Polymarket vs Betfair), market/ref, outcome/runner, **`order_role`**: `BACK` vs `LAY` (Exchange lays are nullable for Polymarket-only rows), timestamp placed.
- **Sizing & prices:** stake (back) or **lay liability / exposure** (Betfair), currency, quoted odds or price, implied prob at submit, fees/commission assumed, slippage if measured.
- **Model snapshot (pure-play pathway):** `model_win_pct_team` (or both teams) **at time of bet**, sim run id or hash of inputs, optional link to bulk-predict row — **nullable** when the bet is not model-driven.
- **Settlement:** `result` enum (open / won / lost / push / void), `pnl` after fees, `settled_at`, notes (e.g. partial fill, void reason).
- **Pathway:** `strategy_pathway` = `MODEL_EDGE` | `CROSS_VENUE_ARB` | `MANUAL` (etc.) so reporting never conflates arb P&L with discretionary model edge.

**Reporting:** Simple admin or Flask view: filter by date, venue, pathway, outcome; export CSV; charts of model-implied edge vs. realised win rate (item **15** quality gates whether those charts are trustworthy).

### 12. Cross-venue “true arb” flag (independent pathway)
**File:** `src/integrations/odds/` (reuse price normalization from items **8–9**), optional worker — **not** part of the Monte Carlo / autonomous runner’s model-signal path unless explicitly chained.

Sometimes **Polymarket vs Betfair** (or other books later) quotes the **same binary outcome** such that, **net of fees and currency/FX assumptions**, locking both sides yields a **risk-free or bounded-profit box** (subject to fill risk, rule differences, and void rules — “true arb” in practice is rare and fragile).

**Proposed behaviour:**

- **Detection:** On each refresh of linked markets, compute **all-in cost** for a dutch book across venues (commission, Polymarket fee structure, minimum bet size). If guaranteed margin exceeds a configurable epsilon → set `is_true_arb_candidate = true` and store the **legs** (prices, sizes, venues).
- **Separation from model bets:** Own code path, config toggles, and **ledger pathway** (`CROSS_VENUE_ARB`). Do **not** mix arb signals into “model confidence” UI or model-edge attribution; optional separate tab or filter: **Arbitrage** vs **Model / pure play**.
- **Execution (later):** Optional two-leg (or n-leg) executor with atomicity best-effort + alerting if one leg fills; ledger records **both legs** and outcome **independent** of match result (profit from mispricing, not from the model).
- **Caveats:** Document execution risk, rule divergence, and that “arb” flags are **operational hints**, not licensed advice.

### 13. Liquidity depth research & position sizing (Polymarket / Betfair)
**File:** `src/integrations/odds/` analytics (snapshot + time series), optional `scripts/sample_book_depth.py`, feed into item **10** risk matrix and item **11** ledger (record **available liquidity** at decision time where APIs allow).

Directional “punts” are only part of how money is made on exchanges. **Most professional flow** is about **managing liquidity** — who is providing quotes, at what size, adverse selection when the model is wrong, and capture of spread / rebates rather than pure outcome alpha. Our stack should **measure reality** before trusting back-of-envelope stake sizes.

**Proposed investigation:**

- **Depth & impact:** For linked cricket markets on **Polymarket** and **Betfair**, log **visible size** at best and next levels (or full book if available), **24h volume**, and **spread**; estimate **cost to move the line** for stakes we care about.
- **Participation rate:** Cap stakes as a fraction of **top-of-book depth** or **recent volume** so we do not paper-edge ourselves into sizes that cannot fill without sweeping the book.
- **Product rules:** Note delays, minimum size, partial fills, and how “depth” differs pre-match vs in-play (Exchange vs CLOB semantics).
- **Honest strategy framing:** Treat **demonstrable depth** as a gate for automation (item **10**); if markets are **too thin**, favour **paper**, **alert-only**, or **maker-style** behaviour (posting inside spread) — explicit **market-making / quoting** can be a **later** phase once read-only analytics prove it is worthwhile; it is **not** the same as cross-market arb (item **12**). **Thin ladders** are exactly where **laying** on Betfair (item **9**) can sometimes earn a better effective price than chasing a wide back — but **liability** scales badly if the model is wrong, so caps and simulation of settle scenarios belong in the risk matrix.

### 14. Raw ingest: `cricketdata` (R) — Cricinfo + Cricsheet in one toolchain
**Reference:** [cricketdata package vignette](https://cran.r-project.org/web/packages/cricketdata/vignettes/cricketdata_R_pkg.html) (CRAN). **File:** `scripts/` or `data_pipeline/` wrapper, optional `renv` / `Rscript` job; outputs land where the Python training stack already reads (e.g. Parquet/CSV/DB).

The [cricketdata](https://cran.r-project.org/web/packages/cricketdata/vignettes/cricketdata_R_pkg.html) R package pulls **ESPNcricinfo** aggregates (e.g. `fetch_cricinfo()`, `fetch_player_data()`, `fetch_player_meta()`) and **Cricsheet** ball-by-ball / match / player feeds (`fetch_cricsheet()`), returning analysis-ready tibbles — including enriched T20 BBB columns in their examples. That matches a **raw data overhaul** goal: one maintained upstream mapping instead of hand-rolled scrapes for every competition code.

**Integration options (pick one):**

- **Thin wrapper (pragmatic):** Versioned `Rscript` that writes **Parquet or CSV** into our `data/` layout; Python ingest unchanged except for path + schema docs. Easiest to audit; adds R + renv to the build image.
- **`rpy2` (in-process):** Call R from Python — heavier deps, tighter coupling; only if we need incremental pulls from app code.
- **Port the ideas:** Use the package as a **spec** for which Cricinfo/Cricsheet endpoints and competition codes to hit, but implement fetch/normalize in Python — fewer runtimes, more code to own.

**Checks before committing:** Diff sample IPL (or internationals) BBB against our current Cricsheet-derived tables for row counts and column semantics; confirm player-id alignment between Cricinfo and Cricsheet naming; document refresh cadence and licensing (open data from Cricsheet; Cricinfo scraping policy).

### 17. Cricsheet Register — cross-source player identity

**Reference:** [The Cricsheet Register](https://cricsheet.org/register/) (official page; downloads: [`people.csv`](https://cricsheet.org/register/people.csv), [`names.csv`](https://cricsheet.org/register/names.csv)).

Cricsheet maintains an **open registry** that links a single **Cricsheet person identifier** (`identifier` in the CSV) to **external IDs on multiple data sources**, plus **name variants** so the same human can be recognised across sites that spell or abbreviate names differently.

**Why it matters for a DB / pipeline overhaul**

- **Single join key across feeds:** The register maps one person to IDs used elsewhere (e.g. ESPNcricinfo, CricketArchive, BCCI, Big Bash, Pulse, NV Play, Opta, and several others — 12 sources on the page as of exploration). That is the natural backbone for aligning **ball-by-ball JSON** (Cricsheet), **player aggregates / bios** (Cricinfo-style APIs or R `cricketdata`), **domestic league feeds**, and **CREX-derived squads** without relying only on fuzzy name matching.
- **Name disambiguation:** `names.csv` lists alternate spellings and source-specific variants; `people.csv` includes a `unique_name` guaranteed distinct per person — useful when cleaning duplicates and stub players.
- **Upstream usage:** Cricsheet uses the register when merging sources for matches and is **gradually adding the Cricsheet identifier from the Register into released match data**, so our schema should treat `registry_id` (already present on `players` per `DATABASE_SCHEMA.md`) as the stable anchor and **refresh** mappings from these CSVs on ingest.
- **License:** Distributed under [ODC-BY](http://opendatacommons.org/licenses/by/1.0/) — attribute Cricsheet when publishing derivatives.

**Proposed overhaul actions**

- **Ingest:** Versioned download of `people.csv` + `names.csv` (same cadence as other Cricsheet artifacts); store or materialise a lookup table keyed by Cricsheet `identifier` with columns for each `key_*` source.
- **Join strategy:** Prefer **ID-first** resolution (CREX id → Cricinfo id → Cricsheet id via register) before falling back to `name_matcher`-style fuzzy logic; log when only fuzzy path hits.
- **Coverage gaps:** Not every person has every external id — document expected sparsity and keep current stub-registration behaviour where the register has no row yet.
- **Cross-check item 14:** Using the register as ground truth for **Cricinfo ↔ Cricsheet** identity should satisfy the “confirm player-id alignment” check listed under the `cricketdata` R ingest path.

---

## Model / Ball prediction

### 15. Ball outcome model: class imbalance (rare outcomes under-predicted)
**File:** `src/models/ball_prediction_nn.py` (training, loss, `model.fit`)

Evaluation on a chronological holdout (last 20%) shows the 7-way ball outcome models are moderately strong overall (44–55% accuracy vs ~14% random) but **heavily biased toward Dot and Single**. Per-outcome F1 is good for Dot (~0.53–0.70) and Single (~0.28–0.50), but **Two, Three, Four, Six, and Wicket** have F1 near zero — the model rarely predicts these. That’s standard class imbalance: rare classes get drowned out by the majority.

**Impact on simulation:** Match-level run rates can still be plausible (dots/singles dominate), but wicket and boundary rates are under-predicted, so simulated games may be too smooth and chase curves slightly off where wickets matter.

**Proposed fixes (pick one or combine):**

- **Class weights in `model.fit()`**  
  Compute inverse frequency (or sqrt inverse) per class from `y_train` and pass `class_weight` to `model.fit()`. Quick win; no architecture change. Keras supports `class_weight` dict mapping class index → weight.

- **Weighted loss (e.g. `weighted_categorical_crossentropy`)**  
  Same idea but inside a custom loss that multiplies per-sample loss by the class weight. Slightly more control (e.g. extra weight only on Wicket and boundaries).

- **Focal loss**  
  Down-weight easy examples (e.g. dots/singles the model already gets right) so the optimizer focuses more on hard/rare classes. Requires a custom loss and a tuning parameter (e.g. γ).

- **Oversampling / undersampling**  
  Oversample rare classes (Two, Three, Four, Six, Wicket) or undersample Dot/Single in the training data so the batch distribution is less skewed. Can be combined with class weights.

- **Persist evaluation metrics after training**  
  `full_retrain.py` currently saves `accuracy_metrics` only when a `ball_meta_*.csv` with an `f1-score` column exists (from a different pipeline). The ball model’s `evaluate_model()` returns accuracy, log loss, and a per-class classification report. After training (and optionally after any evaluation script), write these into the DB/JSON (e.g. `accuracy_metrics` in `model_versions`) so we can track improvement over time without re-running the analysis script.

**Suggested order:** Implement class weights first; re-run training and `scripts/analyze_model_results.py` to compare F1 on rare outcomes. If still weak, add focal loss or sampling next.

### 16. (Optional) Backtest: simulated vs actual match totals
**File:** New script e.g. `scripts/backtest_ball_model.py`, or extend `scripts/backtest_predictions.py`

We have ball-level evaluation but no automated check that **match-level** simulations (run rates, wicket rates, win probabilities) line up with historical outcomes.

**Proposed feature:** A script that, for a held-out set of completed matches, runs the current ball model through the existing match simulator to produce simulated totals and win probabilities, then compares to actual results (e.g. margin of victory, total runs). Output: aggregate metrics (e.g. calibration of win prob, MAE of total runs). Helps validate that improving ball F1 (item 15) actually improves downstream simulation quality. Complement with **settled** `MODEL_EDGE` rows from the bet ledger (item 11) for real-money calibration, excluding `CROSS_VENUE_ARB` legs.

---
