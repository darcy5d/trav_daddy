# Next Overhaul — Backlog

A running notepad of bugs, inefficiencies, and improvement ideas. Some areas
are already complete, some are in the current wave, and the rest are deferred.
The "Done" sections give a one-line summary; full sprint detail lives in git
history.

## Current state at a glance

| Wave | Status | Theme |
|---|---|---|
| 1 | DONE | CREX scraper hardening + venue tooling + data explorer |
| 2 | DONE | Polymarket / Betfair read paths + Bulk Predict UI polish |
| 3 | DONE | Team franchise unification + ELO data repair |
| 3.5 | DONE | Match-level backtest harness (item 16) |
| **4** | **IN PROGRESS** | **Cricket Model v2 — strategic rewrite (this update)** |
| 5 | PARKING LOT | ELO modelling refinements + market integration write-paths |
| 6 | PARKING LOT | Advanced model features + new data sources |

## Wave 4 (in progress) — Cricket Model v2 strategic rewrite

**Why:** the Wave 3.5 backtest measured concrete model pathologies — top-pick
accuracy 52.3% (barely coin-flip), Brier 0.34 / log-loss 1.03 (worse than
always-50/50), MAE 30.2 runs per innings, calibration deciles where
predicted-95% buckets actually win 33%. Architecture map then surfaced three
structural causes worth fixing in one coordinated pass:

1. ODI is functionally broken (T20 assumptions in shared code paths)
2. Per-ball IID sampling can't reproduce real over-level dynamics (boundary
   clusters, wicket cascades, momentum)
3. Extras are silently dropped, costing 5-10 runs per innings of model
   output and creating an over-length mismatch

Plus the now-known limitations: global per-player histograms with no context,
hand-crafted venue features, no calibration layer, no temporal weighting,
four siloed models that can't share representational signal.

### In scope (this update)

- **Phase 0** — ODI plumbing fixes (`max_overs` defaulting, parallel-worker
  format-type default, ball-training-data 120-balls hardcode, format-aware
  phase boundaries). Independently shippable.
- **Phase 1** — `src/features/ball_training_data_v2.py`: 9-class
  extras-aware labels (dot, 1, 2, 3, 4, 6, wicket, **wide, noball**),
  recency-weighted sample weights (exp half-life ~365 days), era feature,
  joint output across all 4 format/gender combos.
- **Phase 2** — `src/models/cricket_model_v2.py`: multi-task model with
  learned player (24-dim) / venue (16-dim) / team (8-dim) embeddings, plus
  **temporal ELO features kept as inputs** (team1_elo, team2_elo,
  batter_elo, bowler_elo) alongside the embeddings — see "On ELO" below.
  Shared backbone + per-(format, gender) ball-outcome heads + per-over
  auxiliary head. Class-weighted loss for boundary/wicket imbalance.
- **Phase 3** — `src/models/calibration.py`: post-hoc Platt scaling per
  (format, gender) on win-prob outputs, persisted JSON, applied at the
  simulator's match-aggregation step.
- **Phase 4** — `src/models/vectorized_nn_sim_v2.py`: extras as first-class
  citizens (wide / noball don't increment the legal-ball counter, free-hit
  rule), format-aware over count + termination, **budget-bias hierarchical
  sampling** using the per-over head.
- **Phase 5** — `scripts/train_cricket_model_v2.py`: chronological split,
  vocab tables with `<UNK>`, multi-task joint training, early-stopping on
  backtest Brier (not val CCE).
- **Phase 6** — `model_version` flag in `get_nn_simulator` + backtest
  harness, side-by-side A/B per (format, gender), promotion gate
  (Brier + log-loss + accuracy not regressing >1pp on >=200 matches).
- **Phase 7** — Update Wave 4 section to DONE with concrete results.

**On ELO (kept, not replaced):** ELO and team embeddings are complementary,
not competing. The plan keeps the existing tiered ELO system as 4 input
features AND adds learned embeddings. ELO gives us a calibrated
probabilistic signal that's as-of-date queryable, sample-efficient for
sparse data (a women's franchise with 5 matches still has a meaningful
rating from tier baseline + opponent strength), and transfers cross-format.
Embeddings add residual style/matchup signal ELO can't capture (e.g. "good
chasing on flat decks but bad defending on turners"). Throwing ELO away
would also waste all the Wave 3 work (franchise unification, tiered
K-factors, cross-pool asymmetry, the player-batting actual-score formula
fix).

**On the over-level head (included, training and inference):** auxiliary
loss during training pushes per-ball outputs toward realistic over-level
aggregates (Gaussian over `runs_in_over`, Poisson over `wkts_in_over`,
weighted at 0.3 of the per-ball loss). At inference, hierarchical sampling
via budget-bias: for each over, sample `(over_runs_target,
over_wkts_target)` from the per-over head, then walk 6 legal balls with
the per-ball softmax biased by the running over budget. Captures
within-over correlation pure IID can't. Ball-by-ball DNA preserved at
inference.

### Explicitly out of scope (Wave 5 / Wave 6)

- Time-conditional team embeddings (v1 embeddings are static; ELO carries
  the temporal signal).
- Strict rejection sampling for the hierarchical step (v1 uses budget-bias
  as the approximation).
- Wide-runs / noball-runs as separate regression targets (v1 collapses
  them to "+1 run" + folds the residual into byes).
- Pitch / dew / weather data (no upstream feed today).
- Batter-vs-bowler matchup history (LHB vs SLA, RHB vs RM, etc.).
- Per-over head outputting a structured 6-ball sequence rather than just
  `(runs, wkts)`.
- True Bayesian / mixture-model uncertainty.
- Per-over head replacing the per-ball head entirely (option C from
  planning; explicitly rejected because it loses the ball-by-ball DNA).

## Wave 5 (parking lot) — ELO + market integrations follow-ups

ELO modelling improvements that v2 does **not** subsume (because they sit
at the ELO layer, not the simulator):

- **ELO margin-of-victory term.** Replace binary `actual ∈ {0, 0.5, 1}`
  in `update_team_ratings` with a continuous score scaled by margin
  (NRR-equivalent or capped log function). A 100-run win should move more
  rating than a Super Over. v2 uses ELO as an input feature, so improving
  ELO improves v2 too.
- **ELO recency decay / half-life.** Either an exponential decay on
  contribution to `elo_change`, or a rolling-window companion "form ELO"
  the simulator blends with the long-term rating. v2's training-sample
  recency weighting is a different mechanism (training-time only) — this
  fixes the ELO calculation itself.
- **Populate `team_external_ids` for Cricinfo / CREX / Cricsheet at
  scale.** Schema landed Wave 3. Pairs naturally with item 17 (Cricsheet
  Register) on the player side — both should be done together so the Team
  Explorer's external-id chips become real cross-source links.

ELO items **subsumed by Wave 4** (no longer needed as separate work):

- ~~Venue / home advantage in ELO.~~ v2's learned venue embedding can
  encode home advantage in any direction the data supports.
- ~~Team-specific scoring dispersion in the simulator.~~ v2's team
  embedding + per-over head + extras handling addresses this.
- ~~`FIRST_INNINGS_SCORE_BONUS` calibration.~~ v2's calibration layer
  subsumes this; the constant gets removed in Phase 4.

Market integration write-path / autonomous runner (deferred from Wave 2):

- 8b. Polymarket write-path (order placement) — pending API keys + compliance.
- 9b. Betfair write-path including back/lay execution.
- 10. Autonomous runner: discover → simulate → bet (risk matrix).
- 11. Bet ledger / audit trail (DB or append-only).
- 12. Cross-venue true-arb pathway.
- 13. Liquidity depth research + position sizing.

## Wave 6 (parking lot) — Advanced model + new data sources

Bigger architectural / data-pipeline lifts that should land after Wave 4
proves out:

- Strict rejection sampling for hierarchical inference (vs v1 budget-bias).
- Batter-vs-bowler matchup history (LHB vs SLA, RHB vs RM, etc.).
- Pitch / dew / weather data ingestion (requires new external feed).
- Time-conditional team embeddings (period-aware vectors).
- Per-over head outputting a structured 6-ball sequence rather than just
  `(runs, wkts)`.
- True Bayesian / mixture-model uncertainty.
- Conditional / contextual player embeddings (by phase, by venue type,
  by opponent style).
- 14. `cricketdata` R ingest pathway.
- 17. Cricsheet Register — full cross-source player identifier integration.

## Done

### Wave 1 — CREX scraper hardening + venue tooling

(Sprint detail in git history; commits prior to Wave 2.)

- CREX scraper migrated to Playwright-only squad fetch path.
- Venue alias coverage expanded; state/province support added; Data
  Explorer tab shipped with country → state → ground hierarchy + fuzzy
  duplicate candidates + alias-gap review.
- Background prefetch from single-predict to bulk-predict warm-ups.
- Team 1 / Team 2 home/away stabilization.

### Wave 2 — Market read paths + UI polish

(Sprint detail in git history.)

- Polymarket client + read endpoints; per-fixture comparison service;
  Bulk Predict + Single Predict UI cards.
- Betfair session bootstrap / keep-alive (read-only; write deferred to
  Wave 5).
- Polymarket-first readiness gating.
- Bulk Predict live Monte Carlo running-state UI polish.

### Wave 3 — Team franchise unification + ELO data repair

The big one. Diagnosis: live Bulk Predict probabilities were diverging
hard from the IPL ladder (RCB at 2.4% despite being 2nd, KKR at the
tier-3 floor, etc.). Root causes:

1. ~32K team + ~344K player duplicate `(team/player, match)` rows in
   `*_elo_history` from un-guarded re-runs of the calculator (KKR's
   2025-05-25 SRH loss applied 13 times in one session).
2. Franchise rebrand orphans: RCB Bangalore (111) + Bengaluru (283),
   Daredevils (364) + Capitals (115), Kings XI (117) + Punjab Kings
   (196), Pune Supergiant(s) all split across multiple `team_id`s.
3. Player batting ELO actual-score formula compressed everything below
   strike-rate 130 to a guaranteed rating loss (Kohli 1483, Russell
   1320).

Shipped:

- Schema V4 (`schema_v4_franchise.sql`) — `team_groups`,
  `team_external_ids`, `team_merge_proposals`, `teams.franchise_id` +
  `canonical_team_id`. Idempotent migration runs on app startup.
- `scripts/backfill_franchises.py` — explicit IPL franchise unifications
  (idempotent, dry-run).
- `EloCalculatorV3` made franchise-aware + idempotent (INSERT OR IGNORE,
  partial UNIQUE indexes after dedupe). Player-side INSERT also patched
  in the same wave.
- `scripts/dedupe_elo_history.py` — backup → recalc → UNIQUE-index
  install → before/after validation report.
- `src/data/franchise_resolver.py` — process-wide singleton wired into
  `VectorizedNNSimulator.get_team_elo` and
  `pad_order_with_team_extras`.
- `/team-explorer` Flask page + API endpoints + staged-approval
  workflow.
- Bulk Predict per-fixture data-quality + ELO badge.
- Player batting ELO formula fix (centred at 0.5 when runs == expected,
  bounded at [0, 1]).

Validation results: 32,260 + 343,975 duplicates → **0/0**; KKR off the
floor 1350 → 1427; MI 1351 → 1437; PBKS unified → 1487; RCB Bengaluru
(now incl. Bangalore era) → 1535; AD Russell batting 1320 → 1494; V
Kohli 1483 → 1557. Re-running the calculator on a clean DB now adds
0 rows (true idempotency).

**Operator runbook (post-Wave-3, still relevant):**

```bash
# Idempotent; unifies known IPL franchise rebrands
python scripts/backfill_franchises.py

# Backup + full recalc + UNIQUE-index install (5+ min on a large DB)
python scripts/dedupe_elo_history.py

# After any future Team Explorer apply, re-run the recalc:
python scripts/dedupe_elo_history.py --skip-backup
```

### Wave 3.5 — Match-level backtest harness (item 16)

Files: [`scripts/backtest_simulator.py`](../scripts/backtest_simulator.py),
[`src/models/backtest.py`](../src/models/backtest.py).

For a chronological holdout of completed matches, loads each match's
actual XI from `player_match_stats`, looks up team & player ELOs **as
of the match date** (with the simulator's in-memory ELO caches
temporarily overridden per match), runs N simulations, and reports
Brier / log-loss / calibration deciles / MAE of total runs and margin.
Per-match CSV + summary JSON in `data/backtest/`.

**Wave 3.5 baseline (44 IPL T20 men, 2025 season, 500 sims/match):**

| Metric | Model | 50/50 baseline |
| --- | --- | --- |
| Top-pick accuracy | 0.523 | 0.500 |
| Brier score | 0.343 | 0.250 |
| Log loss | 1.034 | 0.693 |
| MAE total runs | 30.2 | – |

This is the calibration alarm + the prerequisite for Wave 4 (every
Wave 4 intervention will be A/B'd against this baseline using
`scripts/backtest_simulator.py --label <variant>` and a JSON diff).

## Numbered backlog (chronological)

| # | Item | Status |
|---|---|---|
| 1 | CREX Playwright squads | DONE (Wave 1) |
| 2 | Venue aliases expansion | DONE (Wave 1) |
| 3 | Venue hierarchy state/province | DONE (Wave 1) |
| 4 | Data exploration tab | DONE (Wave 1) |
| 5 | Background prefetch | DONE (Wave 1) |
| 6 | Team 1/2 order stabilization | DONE (Wave 1) |
| 7 | Bulk predict live UI polish | DONE (Wave 2) |
| 8a | Polymarket read path | DONE (Wave 2) |
| 8b | Polymarket write path | Wave 5 |
| 9a | Betfair read path | DONE (Wave 2) |
| 9b | Betfair write path (back/lay) | Wave 5 |
| 10 | Autonomous runner | Wave 5 |
| 11 | Bet ledger | Wave 5 |
| 12 | Cross-venue true-arb | Wave 5 |
| 13 | Liquidity depth research | Wave 5 |
| 14 | `cricketdata` R ingest | Wave 6 |
| 15 | Ball outcome class imbalance | Wave 4 (class-weighted loss in Phase 2) |
| 16 | Match-level backtest | DONE (Wave 3.5) |
| 17 | Cricsheet Register | Wave 6 (paired with team_external_ids in Wave 5) |
