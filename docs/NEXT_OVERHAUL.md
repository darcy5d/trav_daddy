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
| **4** | **BUILD DONE; AWAITING TRAIN** | **Cricket Model v2 — strategic rewrite** |
| 5 | PARKING LOT | ELO modelling refinements + market integration write-paths |
| 6 | PARKING LOT | Advanced model features + new data sources |

## Wave 4 — Cricket Model v2 strategic rewrite (build complete; awaiting train)

**Status (post-build):** All seven phases of the architecture rewrite have
shipped (commits `1f44a6d` through `5572b1b` on `major-rework`). The full
v2 stack — data layer, model architecture, calibration layer, hierarchical
simulator, training script, A/B harness — is in. **What's left is the
empirical training run + A/B against the Wave 3.5 baseline**, which is a
1-2 hour compute job the operator runs offline. This section will be flipped
to a clean "DONE" with metric deltas once that completes.

**Why this wave existed:** the Wave 3.5 backtest measured concrete model
pathologies — top-pick accuracy 52.3% (barely coin-flip), Brier 0.34 /
log-loss 1.03 (worse than always-50/50), MAE 30.2 runs per innings,
calibration deciles where predicted-95% buckets actually win 33%.
Architecture map then surfaced three structural causes worth fixing in one
coordinated pass:

1. ODI is functionally broken (T20 assumptions in shared code paths)
2. Per-ball IID sampling can't reproduce real over-level dynamics (boundary
   clusters, wicket cascades, momentum)
3. Extras are silently dropped, costing 5-10 runs per innings of model
   output and creating an over-length mismatch

Plus the now-known limitations: global per-player histograms with no context,
hand-crafted venue features, no calibration layer, no temporal weighting,
four siloed models that can't share representational signal.

### Build phases (all shipped)

- **Phase 0 — ODI plumbing fixes (commit `1f44a6d`).** New
  `src/utils/format_constants.py` centralises overs/balls/phase thresholds
  per format. T20 = 20 overs / 120 balls / 6/15 phase split; ODI = 50
  overs / 300 balls / 10/40 phase split. `simulate_matches` and
  `simulate_detailed_match` both derive `max_overs` from `self.format_type`
  when not pinned. `_simulate_innings_vectorized` hot-loop phase selection
  now uses format-aware thresholds. Parallel worker constructs
  `VectorizedNNSimulator(gender=gender, format_type=fmt)` instead of
  defaulting format to T20. `ball_training_data.py` `balls_remaining`
  hardcode now uses `balls_for_format(self.format_type)`.

  **Phase 0 baseline (25 ODI male matches, 2025+, 300 sims/match):**
  top-pick accuracy 0.680, Brier 0.272 (marginal vs 50/50's 0.250),
  log-loss 0.978, MAE total runs 80.2, with One-Day Cup (Aus) at acc
  0.875 / Brier 0.147 — same overconfidence shape as T20.

- **Phase 1 — `src/features/ball_training_data_v2.py` (commit `830c7d7`).**
  9-class outcome label `{dot, 1, 2, 3, 4, 6, wicket, wide, noball}`,
  recency-weighted sample weights (exp half-life default 365 days), era
  feature `(year - 2026) / 10`, native categorical id columns
  (canonical batting/bowling team via `FranchiseResolver`), joint output
  per gender (one npz, format_id is a row column). Per-over (`over_runs`,
  `over_wkts`) targets stamped in a second pass so the trainer doesn't
  need dynamic batch-time aggregation. Bye/legbye runs fold into the
  matching run class (the ball still counts as legal).

- **Phase 2 — `src/models/cricket_model_v2.py` (commit `01b9fba`).**
  Multi-task model with learned player (24-dim) / venue (16-dim) / team
  (8-dim) embeddings PLUS `temporal ELO features kept as inputs`
  (team1_elo, team2_elo, batter_elo, bowler_elo) alongside the embeddings.
  See "On ELO" below for why we kept ELO. Shared backbone (3 × Dense(256)
  with BN + ReLU + Dropout) feeds 8 outputs (4 ball heads × 4 over heads).
  `routed_cce_loss_fn` for class-weighted per-ball CCE; `over_nll_loss_fn`
  for Gaussian (runs) + Poisson (wkts) per-over auxiliary loss.

- **Phase 3 — `src/models/calibration.py` (commit `3a9ba62`).** Per-(format,
  gender) Platt scaling fit by gradient descent on BCE. Decoupled from
  the model so refitting calibration is a 5-second job. NLL-regression
  guard falls back to identity when fit instability would make NLL worse.
  9 unit tests in `tests/test_calibration.py`.

  **Sanity fit on Wave 3.5 + Phase 0 baselines:**
  T20 male `n=44 a=0.050 b=-0.000 NLL 1.034 -> 0.697`
  ODI male `n=25 a=0.203 b=-0.506 NLL 0.978 -> 0.571`
  The T20 fit's `a=0.050` is striking: the optimum is to almost
  fully shrink toward 0.5, dropping NLL to essentially the always-50/50
  baseline. The v1 model's confident predictions carry near-zero
  information once calibrated. Calibration on top of v2 will be much
  more interesting because v2's raw probabilities should already be
  meaningfully better.

- **Phase 4 — `src/models/vectorized_nn_sim_v2.py` (commit `130f42c`).**
  9-class outcomes; wide / noball add 1 run, do NOT increment legal-ball
  counter, do NOT advance the batter; noball sets the next legal ball as
  free hit (wicket prob zeroed and renormalised, then flag clears).
  Innings terminates on legal_balls >= max_overs * 6, with extras
  headroom of `max_overs * 9` for the inner loop. **Budget-bias
  hierarchical sampling** (Option B from planning): each over starts by
  sampling `(over_runs_target, over_wkts_target)` from the per-over head,
  then the per-ball softmax is biased by the running over budget so
  high-budget overs lean toward 4/6 and low-budget overs lean toward
  dot/single. Returns v1-compatible result dict + v2 extras
  (`team1_win_prob_raw`, `calibration_used`). 10 unit tests in
  `tests/test_simulator_v2_extras.py`.

- **Phase 5 — `scripts/train_cricket_model_v2.py` (commit `dbdc83b`).**
  Multi-task joint training. Loads ball_training_v2_{gender}.npz, builds
  vocabs (UNK at index 0, default `min_count=5`), chronological split
  via sample-weight ordering (train = oldest 85%, val = middle 10%, test
  = held-back 5%), class weights from inverse frequency clipped to
  [0.5, 5.0]. **Multi-task routing via per-output sample_weight masks**:
  each row contributes loss only to its (format, gender) head; all four
  heads share backbone gradients. Compiles with `loss_weights = 1.0` for
  the ball head and `0.3` for the per-over auxiliary. EarlyStopping +
  ReduceLROnPlateau + ModelCheckpoint to `data/models/v2/`.

- **Phase 6 — A/B harness wiring (commit `5572b1b`).** New
  `--model-version v1|v2` flag on `scripts/backtest_simulator.py`;
  `_override_simulator_elos` polymorphic over both simulators (v1's
  public attrs vs v2's lazy-loaded underscore attrs); new
  `scripts/compare_backtests.py` for side-by-side diff + promotion-gate
  verdict (Brier improves AND log-loss improves AND accuracy doesn't
  regress > 1pp on >= 50 matches).

- **Phase 7 — this docs update (current commit).**

### Empirical training run (operator's next step)

Roughly 1-2 hours of compute end-to-end:

```bash
# 1. Build the full v2 training data (both genders, both formats)
python scripts/build_ball_training_v2.py --gender both
# expect: ~2-4 GB total across both files; multi-million rows

# 2. Train the v2 model (multi-task joint)
python scripts/train_cricket_model_v2.py --epochs 15
# Watch val_loss; EarlyStopping kicks in after 3 stale epochs

# 3. Fit calibration on the Wave 3.5 baseline backtest
python scripts/fit_calibration.py \
    --backtest-csv data/backtest/backtest_baseline_ipl_2025.csv \
    --backtest-csv data/backtest/backtest_baseline_odi_phase0.csv \
    --output data/models/v2/calibration.json

# 4. Run v2 backtest with same holdout as the V1 baseline
python scripts/backtest_simulator.py \
    --model-version v2 \
    --tournament-pattern '%Indian Premier League%' \
    --since-date 2025-01-01 --limit 50 --n-sims 500 \
    --label v2_ipl_2025

# 5. Compare against the Wave 3.5 baseline + verdict
python scripts/compare_backtests.py \
    --baseline data/backtest/backtest_baseline_ipl_2025_summary.json \
    --candidate data/backtest/backtest_v2_ipl_2025_summary.json
```

This section will be replaced with the actual A/B numbers + a clean
"DONE" header once the run completes.

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
| 15 | Ball outcome class imbalance | DONE (Wave 4 Phase 5: class-weighted loss + 9-class extras) |
| 16 | Match-level backtest | DONE (Wave 3.5) |
| 17 | Cricsheet Register | Wave 6 (paired with team_external_ids in Wave 5) |
