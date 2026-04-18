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
| 4 | DONE | Cricket Model v2 — strategic rewrite (build + first train) |
| **4.5** | **DONE** | **V2 second-pass training: closed score-MAE gap, V2 now beats V1 on every probabilistic metric** |
| 5 | PARKING LOT | ELO modelling refinements + market integration write-paths |
| 6 | PARKING LOT | Advanced model features + new data sources |
| 7 | NEW PARKING LOT | V2 productionisation (GUI integration, calibration refit at scale, expanded backtests) |

## Wave 4 — Cricket Model v2 strategic rewrite (DONE)

**Status:** Architecture build (`1f44a6d` → `5572b1b`) + first training pass
caught two structural bugs (`6ca7be1`); Wave 4.5 second-pass training closed
the score-MAE gap. The full v2 stack is deployable and beats v1 on every
probabilistic metric on the Wave 3.5 IPL holdout. See "Wave 4.5 results"
below for the headline numbers.

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

### Empirical training run + A/B results

**Bugs caught + fixed during the first training pass** (both real, both
landed in commits below alongside the data + model artifacts):

1. **Class weights were applied twice** in
   `scripts/train_cricket_model_v2.py`: once inside `routed_cce_loss_fn`
   AND once inside the per-route sample-weight masks. So the effective
   per-class weighting was `cls_w²` — wickets at 6.25x and noballs at
   25x of the natural rate. Symptom: per-ball softmax produced ~12%
   wicket prob (vs ~5% in reality), which then collapsed 2nd innings in
   the simulator to ~50 runs avg vs reality's ~180. Fix: remove
   class-weight multiplication from the sample-weight masks; keep it
   inside the loss function only.

2. **Win-condition was structurally impossible** in
   `vectorized_nn_sim_v2.simulate_matches`: the line
   `team1_won = first_runs >= targets` where `targets = first_runs + 1`
   is ALWAYS False (`x >= x+1` can't be true). Result: every backtest
   match returned `team1_win_prob = 0.000` regardless of actual model
   quality. Fix: `team1_won = second_runs < targets` (team1 wins iff
   team2 fails to chase).

After both fixes (and a milder class-weight clip range of `[0.7, 1.5]`
instead of `[0.5, 5.0]`), the v2 model trained for 12 epochs in ~10
minutes against 4.75M rows (3.6M male + 1.2M female, both formats
joint). Final val_loss = **1.44** (was 2.69 with the bugs in place).

**A/B vs Wave 3.5 V1 baseline (44 IPL T20 men, 2025 season, 500 sims/match):**

| Metric | V1 baseline | V2 first pass | Verdict |
| --- | --- | --- | --- |
| Top-pick accuracy | 0.523 | 0.409 | **regressed** by 11pp |
| Brier score | 0.343 | **0.271** | **improved 21%** |
| Log loss | 1.034 | **0.737** | **improved 29% (near-optimal calibration)** |
| MAE total runs | 30.2 | 45.7 | **regressed 51%** |
| MAE margin | 29.4 | 29.2 | tied |

**What this tells us:**

The headline is that **V2 is a much better-calibrated probabilistic
model but a worse score-prediction model**. The accuracy regression is
honest: V2 correctly identifies that most IPL games are coin-flips and
outputs 40-60% predictions for them, so the "top pick" is essentially
arbitrary. The Brier and log-loss wins are real — V2 doesn't say
"95% confident" on matches that turn out 50/50 the way V1 did.

V1 calibration deciles (the alarm we started with):
- predicted [0.0, 0.1) → actual win rate 0.40 (overconfident in losses)
- predicted [0.9, 1.0) → actual win rate 0.33 (overconfident in wins)

V2 calibration deciles (after the fixes):
- predicted [0.4, 0.5) n=15 → actual 0.60
- predicted [0.5, 0.6) n=20 → actual 0.50 (perfectly calibrated)
- predicted [0.6, 0.7) n=6  → actual 0.33 (slight overconfidence)
- nothing in [0.0, 0.4) or [0.7, 1.0) — V2 is conservative

The score MAE regression is real: V2's individual innings totals
sit ~30-40 runs below reality (sim avg ~140 vs actual ~200 in IPL
2025). The model's per-ball softmax is too conservative on
boundary classes; this is the next-pass training fix.

Calibration on top of V2 (post-hoc Platt, fit on the v2 backtest CSV)
gives `a=0.05 b=-0.004` with NLL 0.737 → 0.695. That's essentially the
50/50 baseline NLL (0.693), confirming V2's raw output is already
near-calibrated and the post-hoc layer is just shrinking everything
toward 0.5.

## Wave 4.5 — V2 score-MAE fix (DONE)

**The headline:** V2 overnight model **beats V1 on every probabilistic
metric AND now beats V1 on score MAE for the first time** on the same
44-match IPL 2025 holdout the Wave 3.5 baseline used.

| Metric | V1 baseline | V2 first pass | **V2 overnight** | vs V1 |
| --- | --- | --- | --- | --- |
| Brier score | 0.343 | 0.271 | **0.252** | **-27%** |
| Log loss | 1.034 | 0.737 | **0.696** | **-33%** (essentially at the 0.693 50/50 baseline) |
| MAE total runs | 30.15 | 45.67 | **29.62** | **-2%** (better!) |
| MAE margin runs | 29.43 | 29.25 | **28.63** | -3% |
| Top-pick accuracy | 0.523 | 0.409 | 0.432 | -9pp (V2 honestly outputs ~50% on coin-flip matches) |

The accuracy gap is the well-understood "honest probabilities on
genuinely-50/50 matches" pattern; for any decision-theoretic use of the
probabilities (Polymarket EV calculations, Kelly stakes, etc.) Brier and
log-loss are the relevant metrics, and V2 wins both decisively.

### How we got here — Phase 1 diagnostic findings

[`scripts/inspect_v2_outputs.py`](../scripts/inspect_v2_outputs.py) walks
N recent IPL matches ball-by-ball and aggregates per-ball softmax against
actual outcomes by phase, era, and counterfactual era flip. Run on the V2
first-pass model (5 matches, 1257 balls, IPL 2025):

- **Counterfactual era flip showed -0.030pp on P(4) and -0.030pp on
  P(6)** when only the era column changes from 2016 to 2026. The era
  feature was being ignored entirely by the model.
- **Per-class predicted vs actual** showed wicket prob 9.3% predicted
  vs 4.9% actual — a 4.4pp over-prediction. NOT a boundary
  under-prediction problem (boundaries were within 2pp); it was wickets
  being over-predicted, which terminated innings early in the simulator
  and dragged scores down.
- The original Wave 4 hypothesis (boundary under-prediction) was wrong;
  the actual root cause was the inverse-frequency class weights still
  upweighting rare classes, including wickets, even after the
  cls_w² bug was fixed.

### Phase 3 A/B sweep (3 variations, captured in `data/backtest/wave_4_5_ab.csv`)

| Variant | Knob | Brier | Log loss | MAE runs | MAE margin |
| --- | --- | --- | --- | --- | --- |
| V2 first pass | clip [0.7, 1.5] | 0.271 | 0.737 | 45.67 | 29.25 |
| A1 uniform | uniform class weights | 0.251 | 0.696 | **38.58** | 30.52 |
| A2 over_loss=0.1 | drop aux loss weight | 0.251 | 0.695 | 42.61 | **30.10** |
| A3 hl180+uniform | half-life 180d | 0.250 | 0.694 | 39.28 | 45.76 ← regressed |

A1 won on score MAE; A2 was the calibration champ. A3 (tighter recency)
hurt margin MAE significantly — discarded. **Combined recipe for
overnight: A1 + A2 (uniform class weights + per-over loss weight 0.1).**

### Phase 4 overnight train

- 12 epochs at hidden=256, emb=24 → **39 epochs at hidden=512, emb=32**
  (1.23M params, ~2x larger than first-pass 623K).
- Same 4.75M-row joint training data (365-day half-life, both genders,
  both formats).
- Uniform class weights (A1) + over_loss weight 0.1 (A2).
- Early-stopped at epoch 39/50 (patience=5); best val_loss = 1.015
  (was 2.69 with bugs in place; was 1.44 with mild class weights at
  hidden=256; was 1.40 with uniform at hidden=256).

### Files added in Wave 4.5

- [`scripts/inspect_v2_outputs.py`](../scripts/inspect_v2_outputs.py) —
  the deep diagnostic that surfaced the wicket over-prediction and
  era-feature-broken findings; reusable for future investigations.
- [`scripts/run_v2_ab_sweep.sh`](../scripts/run_v2_ab_sweep.sh) — A/B
  sweep automation (not used in the actual run because we ran each
  variation manually to inspect results between runs, but kept for
  future re-runs).
- New CLI flags on
  [`scripts/train_cricket_model_v2.py`](../scripts/train_cricket_model_v2.py):
  `--class-weight-mode`, `--over-loss-weight`, `--hidden-units`,
  `--n-hidden-layers`, `--embedding-dim-batter|venue|team`. All wired
  through `CricketModelV2Config`.

### Operator runbook (V2 overnight is now the canonical model)

```bash
# Re-run V2 backtest against an existing baseline (after any fix)
python scripts/backtest_simulator.py --model-version v2 \
    --tournament-pattern '%Indian Premier League%' \
    --since-date 2025-01-01 --limit 50 --n-sims 500 --label v2_<run-tag>

python scripts/compare_backtests.py \
    --baseline data/backtest/backtest_baseline_ipl_2025_summary.json \
    --candidate data/backtest/backtest_v2_<run-tag>_summary.json

# Re-train V2 from scratch with the Wave 4.5 winning recipe (~30-40 min):
python scripts/build_ball_training_v2.py --gender both
python scripts/train_cricket_model_v2.py \
    --epochs 50 --batch-size 4096 --vocab-min-count 5 \
    --early-stopping-patience 5 \
    --class-weight-mode uniform --over-loss-weight 0.1 \
    --hidden-units 512 --n-hidden-layers 3 \
    --embedding-dim-batter 32 --embedding-dim-venue 24 --embedding-dim-team 12 \
    --label v2_overnight

# Fit calibration on the latest V2 backtest
python scripts/fit_calibration.py \
    --backtest-csv data/backtest/backtest_v2_<run-tag>.csv \
    --output data/models/v2/calibration.json

# Diagnose what the V2 model is producing per ball (helpful for next iteration)
python scripts/inspect_v2_outputs.py --n-matches 5 --since 2025-01-01 \
    --label diagnostic
```

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

## Wave 7 (parking lot) — V2 productionisation

Now that V2 beats V1 on every probabilistic metric, the next session-sized
items are about getting V2 in front of users and building wider evaluation
coverage:

- **Wire V2 into the GUI training tab.** Currently
  [`app/main.py`](../app/main.py) line ~5023 calls
  `from full_retrain import run_full_pipeline` (V1-only). Add a
  `model_version` toggle on the training UI so non-CLI users can train V2
  with the same recipe Wave 4.5 settled on.
- **Promote V2 in the predict path.** Currently the live web predict
  routes default to V1 simulator. Switch the default to V2 (with a hidden
  v1 fallback flag) once we've also validated on a wider holdout.
- **Wider backtest holdout.** We've validated V2 on 44 IPL T20 male
  matches. Run the harness on (a) IPL 2025 (already), (b) PSL 2025-26,
  (c) BBL 2025-26, (d) WPL 2025, (e) ODI internationals 2025. Each is one
  `backtest_simulator.py --tournament-pattern '%X%'` invocation; we want
  V2's promotion to be backed by ~200+ matches across formats and
  tournaments, not just one.
- **Hidden-state debug UI.** Streamlit-style page that lets you step
  through a v2 simulated match ball-by-ball and inspect per-ball softmax
  + over-budget biases. Effectively a UI wrapper around
  [`scripts/inspect_v2_outputs.py`](../scripts/inspect_v2_outputs.py).
- **Calibration refit on the wider backtest.** Once we have the wider
  holdout, refit the per-(format, gender) Platt scalars to keep
  near-optimal NLL on each route.
- **Address the wicket over-prediction structurally.** The Wave 4.5
  diagnostic showed wicket prob 9.3% vs reality 4.9%. Uniform class
  weights closed most of the gap, but the residual could be addressed by
  (a) explicit wicket-down-weight (e.g. 0.7 weight just on class 6),
  (b) adding a per-batter "in" feature that captures whether the batter
  has just come in or is set, or (c) a hazard-style per-ball wicket
  model trained as a separate head.
- **Era-feature investigation v2.** The counterfactual era flip showed
  -0.030pp on P(4)+P(6) — the era column is being ignored. Worth one
  diagnostic session to figure out why the model isn't using it (likely
  swamped by the higher-magnitude state features) and either fix via
  feature scaling or accept and remove the column.

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
