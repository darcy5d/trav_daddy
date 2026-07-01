"""
Configuration settings for the Cricket Match Predictor project.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Database
DATABASE_PATH = BASE_DIR / "cricket.db"
DATABASE_URL = f"sqlite:///{DATABASE_PATH}"

# Data sources - Cricsheet URLs
CRICSHEET_BASE_URL = "https://cricsheet.org/downloads/"
CRICSHEET_HOMEPAGE = "https://cricsheet.org/"
# Page that contains "The most recent matches added to the site are: ..."
CRICSHEET_MATCHES_URL = "https://cricsheet.org/matches/"
DATA_SOURCES = {
    "t20i": f"{CRICSHEET_BASE_URL}t20s_json.zip",           # All T20s (men + women internationals)
    "t20s_female": f"{CRICSHEET_BASE_URL}t20s_female_json.zip",  # Women's T20 internationals
    "all_female": f"{CRICSHEET_BASE_URL}all_female_json.zip",    # ALL women's cricket (T20+ODI+franchise)
    "all_male": f"{CRICSHEET_BASE_URL}all_male_json.zip",        # ALL men's cricket (T20+ODI+franchise - IPL, BBL, PSL, etc.)
    "wpl": f"{CRICSHEET_BASE_URL}wpl_json.zip",             # Women's Premier League (India)
    "odi": f"{CRICSHEET_BASE_URL}odis_json.zip",
}

# Data filtering
MIN_MATCH_DATE = "2019-01-01"  # Only use matches from this date onwards
SUPPORTED_FORMATS = ["T20", "ODI"]

# Cricket Data API (cricketdata.org)
CRICKET_DATA_API_KEY = os.getenv("CRICKET_DATA_API_KEY")
CRICKET_DATA_BASE_URL = "https://api.cricapi.com/v1"

# CricketArchive (subscription site — PERSONAL-USE enrichment only).
# ToS has no anti-automation clause, and robots.txt permits the /Archive/* data
# paths for normal user agents, but the copyright clause forbids redistribution
# and the "detrimental to use" catch-all means we MUST be gentle. Hence: present
# as a normal browser, honour the robots `*` disallow list, rate-limit hard,
# cache every page on disk (fetch each URL once, ever), and never redistribute
# the raw scraped data. Credentials live in .env (gitignored) — never committed.
CRICKETARCHIVE_CONFIG = {
    "enabled": os.getenv("CRICKETARCHIVE_ENABLED", "false").lower() == "true",
    "username": os.getenv("CRICKETARCHIVE_USERNAME"),
    "password": os.getenv("CRICKETARCHIVE_PASSWORD"),
    # Data/content lives on cricketarchive.com; auth/paywall is on my.cricketarchive.com.
    "base_url": os.getenv("CRICKETARCHIVE_BASE_URL", "https://cricketarchive.com"),
    "auth_base_url": os.getenv("CRICKETARCHIVE_AUTH_BASE_URL", "https://my.cricketarchive.com"),
    "login_url": os.getenv("CRICKETARCHIVE_LOGIN_URL", "https://my.cricketarchive.com/"),
    # Politeness controls (be gentle to avoid the ToS "detrimental to use" catch-all).
    # RATE PHILOSOPHY: the per-request delay is the primary politeness mechanism.
    # The daily cap is NOT the governor — it's just a runaway-bug safety net set
    # intentionally high. With 3 parallel workers at 6-12 s/request the effective
    # rate is ~20 req/min (~29,000/day), well within polite territory and proven
    # stable over multi-day runs at ~0.7% 403 rate.
    # NOTE: 2.5-5s rate triggered intermittent 403s early on; 6-12s has been
    # robust across 30k+ requests. The fetcher adds adaptive slowdown on top.
    "min_delay_sec": float(os.getenv("CRICKETARCHIVE_MIN_DELAY", "6.0")),
    "max_delay_sec": float(os.getenv("CRICKETARCHIVE_MAX_DELAY", "12.0")),
    # High cap = effectively no daily limit; rely on the delay to govern rate.
    # Set CRICKETARCHIVE_MAX_PER_DAY in env to a lower value during testing.
    "max_requests_per_day": int(os.getenv("CRICKETARCHIVE_MAX_PER_DAY", "100000")),
    "request_timeout_sec": float(os.getenv("CRICKETARCHIVE_TIMEOUT", "30")),
    # Throttle-resilience: retry 403/429 with exponential backoff, and adaptively
    # slow the base rate when throttled (decays back down on sustained success).
    "max_retries": int(os.getenv("CRICKETARCHIVE_MAX_RETRIES", "4")),
    "backoff_base_sec": float(os.getenv("CRICKETARCHIVE_BACKOFF_BASE", "20")),
    "user_agent": os.getenv(
        "CRICKETARCHIVE_USER_AGENT",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    ),
    # On-disk cache + saved login session (both under the gitignored data/raw/).
    "cache_dir": RAW_DATA_DIR / "cricketarchive",
    "auth_state_path": RAW_DATA_DIR / "cricketarchive" / "auth_state.json",
    # Isolated CA datastore — DELIBERATELY separate from cricket.db so the archive
    # can be harvested, audited and experimented on without touching the trusted
    # production database (gitignored via *.db).
    "archive_db_path": BASE_DIR / "ca_archive.db",
    # Re-use a saved login session for this many hours before forcing re-login.
    "auth_max_age_hours": float(os.getenv("CRICKETARCHIVE_AUTH_MAX_AGE_HOURS", "168")),
}

# Market integrations (Wave 2: read-path readiness; Wave 5 Phase 6a: write-path)
POLYMARKET_CONFIG = {
    "enabled": os.getenv("POLYMARKET_ENABLED", "false").lower() == "true",
    "api_base_url": os.getenv("POLYMARKET_API_BASE_URL", "https://gamma-api.polymarket.com"),
    "clob_base_url": os.getenv("POLYMARKET_CLOB_BASE_URL", "https://clob.polymarket.com"),
    "chain_id": int(os.getenv("POLYMARKET_CHAIN_ID", "137")),
    # Read-only fields (Wave 2). Optional.
    "api_key": os.getenv("POLYMARKET_API_KEY"),
    "api_secret": os.getenv("POLYMARKET_API_SECRET"),
    "passphrase": os.getenv("POLYMARKET_PASSPHRASE"),
    # Write-path (Wave 5 Phase 6a). Required for live betting only.
    "private_key": os.getenv("POLYGON_PRIVATE_KEY") or os.getenv("POLYMARKET_PRIVATE_KEY"),
    "funder_address": os.getenv("POLYGON_FUNDER_ADDR") or os.getenv("POLYMARKET_FUNDER_ADDR"),
    # Polymarket signature_type: 0 = EOA, 1 = POLY_PROXY (Magic), 2 = GNOSIS_SAFE.
    # POLY_PROXY (1) is the Polymarket-default for accounts created via their
    # web UI; bot-managed wallets generated by bootstrap_polymarket_wallet.py
    # are EOAs (signature_type=0). Override via env if your wallet is different.
    "signature_type": int(os.getenv("POLYMARKET_SIGNATURE_TYPE", "0")),
}

# Phase 2 (2026-06-27): per-league LIVE-trading policy — the documented source
# of truth for which competitions may place REAL-money bets. Paper trading runs
# on ALL discovered leagues regardless of this table; it governs live placement
# only. Verdicts come from the data-coverage gate
# (scripts/poly_competition_coverage.py): GO = well-covered; PAPER-ONLY =
# marginal, forward-test first; NO-GO = insufficient data, don't bet blind.
#
# To toggle a league on/off for live, flip its "live" flag (and update "reason").
# Any prefix with live=False is added to live_exclude_prefixes below. Prefixes
# NOT listed here are treated as live-eligible by default (blocklist posture) —
# add new leagues here as the coverage tool surfaces them.
BETTING_LEAGUE_POLICY = {
    # --- GO: well-covered, live-eligible -------------------------------------
    "crint":    {"live": True,  "verdict": "GO",        "reason": "Internationals: 94% team-match, deep history, 99% player-vocab."},
    "cricpsl":  {"live": True,  "verdict": "GO",        "reason": "PSL: 100% team-match, ~114 median matches, 100% vocab."},
    "cricipl":  {"live": True,  "verdict": "GO",        "reason": "IPL: flagship league, fully modelled (not always listed)."},
    "cricbbl":  {"live": True,  "verdict": "GO",        "reason": "Big Bash: fully modelled (not always listed)."},
    # --- PAPER-ONLY: marginal coverage; left live-eligible per option-B ------
    # (flip live=False to enforce paper-confirm-first on these).
    "criclcl":  {"live": False, "verdict": "PAPER-ONLY", "reason": "Legends League: moved to paper-only by operator 2026-06-28 (only 77% team-match; watch fills / resolve team mappings before relisting)."},
    "cricmlc":  {"live": True,  "verdict": "PAPER-ONLY", "reason": "MLC: median 26 matches (<30) and stale (last 2025-07); confirm in-season."},
    # crictbcl = "T20 Brisbane Champions League" (exhibition; e.g. NY Liberty XI
    # vs Melbourne Pirates). Only 1/2 teams resolve and the high match count is a
    # false fuzzy-match (e.g. "mel" -> a Melbourne BBL side), so it is NOT safe
    # for live despite the inflated median. Live-excluded.
    "crictbcl": {"live": False, "verdict": "NO-GO", "reason": "Brisbane Champions League (exhibition): 50% team-match; inflated match count is a false fuzzy-match artifact."},
    # --- Relisted live by operator (2026-06-28) ------------------------------
    # County T20 Blast (men + women) turned back ON for live after the
    # paper-confirm window. Data coverage is fine (the earlier block was a
    # strategy choice, not a data NO-GO). Watch the segment ROI in the rollup.
    "crict20blast":  {"live": True,  "verdict": "GO", "reason": "County T20 Blast: relisted live 2026-06-28 by operator after paper-confirm window (prior block was strategy-choice, not data-coverage)."},
    "crict20blastw": {"live": True,  "verdict": "GO", "reason": "Women's T20 Blast: relisted live 2026-06-28 by operator after paper-confirm window."},
    # --- NO-GO: insufficient data, live-excluded -----------------------------
    "criccoppat10":   {"live": False, "verdict": "NO-GO", "reason": "Coppa il Mondo T10: 18% team-match; T10 novelty format."},
    "cricmaharaja":   {"live": False, "verdict": "NO-GO", "reason": "Maharaja Trophy: 45% team-match."},
    "cricecsbg":      {"live": False, "verdict": "NO-GO", "reason": "ECS Bulgaria: 25% team-match; minor-league."},
    "cricnsk":        {"live": False, "verdict": "NO-GO", "reason": "NSK Trophy State T20: 0% team-match, no history in DB."},
    "crictelangana":  {"live": False, "verdict": "NO-GO", "reason": "Telangana T20: 33% team-match."},
    "cricjcl":        {"live": False, "verdict": "NO-GO", "reason": "Japan Cricket League: 40% team-match."},
    "crickpl":        {"live": False, "verdict": "NO-GO", "reason": "KPL Indo-Nepal T20: 38% team-match."},
    "cricshpageeza":  {"live": False, "verdict": "NO-GO", "reason": "Shpageeza (Afghanistan): 38% team-match."},
    "crict20blastl2w":{"live": False, "verdict": "NO-GO", "reason": "Blast League 2 Women: no match history in DB."},
    "cricapl":        {"live": False, "verdict": "NO-GO", "reason": "Andhra Premier League: 0% team-match, no history."},
}

# Wave 5 Phase 6c: Live betting risk gate config.
# All caps are enforced server-side in src/integrations/polymarket/risk_gate.py;
# the UI can re-check for UX but cannot bypass the server gate.
BETTING_CONFIG = {
    "mode": os.getenv("BETTING_MODE", "OFF").upper(),  # OFF | MANUAL | AUTO
    "max_deposit_usdc": float(os.getenv("BETTING_MAX_DEPOSIT", "200")),
    "max_per_bet_usdc": float(os.getenv("BETTING_MAX_PER_BET", "25")),
    # Floating per-bet cap as a fraction of the strategy's live bankroll.
    # When > 0 and a strategy_label is provided, overrides max_per_bet_usdc.
    # e.g. 0.225 = 22.5% of live bankroll. Grows/shrinks as bankroll moves.
    # Falls back to max_per_bet_usdc for untagged manual bets.
    "max_per_bet_fraction": float(os.getenv("BETTING_MAX_PER_BET_FRACTION", "0")),
    # Wallet-proportional caps (preferred when > 0). Scale with portfolio value
    # (USDC + open positions). Top-ups increase limits automatically.
    "max_deploy_fraction": float(os.getenv("BETTING_MAX_DEPLOY_FRACTION", "0.95")),
    "max_open_fraction_per_strategy": float(os.getenv("BETTING_MAX_OPEN_FRACTION", "0.85")),
    # Per-strategy open cap per UTC kickoff date (strategy slice × fraction).
    # When > 0, replaces flat max_open_fraction_per_strategy for step 6b.
    "max_open_fraction_per_kickoff_day": float(
        os.getenv("BETTING_MAX_OPEN_FRACTION_PER_KICKOFF_DAY", "0")
    ),
    "max_per_day_fraction": float(os.getenv("BETTING_MAX_PER_DAY_FRACTION", "0")),
    "max_loss_per_day_fraction": float(os.getenv("BETTING_MAX_LOSS_PER_DAY_FRACTION", "0")),
    "max_per_day_usdc": float(os.getenv("BETTING_MAX_PER_DAY", "50")),
    "max_loss_per_day_usdc": float(os.getenv("BETTING_MAX_LOSS_PER_DAY", "30")),
    # Wave 5.11: guarded in-play stop-loss. Ships OFF; only the gated,
    # deep-floor config survived the 35-day winners-vs-losers backtest split
    # (floor 0.20 + 2nd-innings gate; ungated/shallow floors and re-entry lose
    # money). When enabled, the cashout scanner SELLs a position whose price
    # has fallen to stop_loss_floor after stop_loss_gate_min minutes from
    # kickoff. No re-entry.
    "stop_loss_enabled": os.getenv("BETTING_STOP_LOSS_ENABLED", "0").strip()
    in ("1", "true", "True", "TRUE"),
    "stop_loss_floor": float(os.getenv("BETTING_STOP_LOSS_FLOOR", "0.20")),
    "stop_loss_gate_min": float(os.getenv("BETTING_STOP_LOSS_GATE_MIN", "105")),
    # Stop-loss exits liquidate progressively: instead of holding for the tight
    # profit-take slippage, a forced stop sweeps every resting bid down to
    # stop_loss_min_exit_price (a hard ruin floor). On a collapsing, thin county
    # book the bid side empties just below mid, so the tight 3c profit floor
    # would refuse to sell ("best-bid-below-floor") and the position rotted to
    # ~0 at settlement. A wide stop slippage takes whatever liquidity exists.
    "stop_loss_min_exit_price": float(
        os.getenv("BETTING_STOP_LOSS_MIN_EXIT_PRICE", "0.01")
    ),
    "auto_min_edge_pp": float(os.getenv("BETTING_AUTO_MIN_EDGE", "5.0")),
    # Low-data-league throttle: associate-nation internationals (crint fixtures
    # where either side is not a Tier-1 Full Member) are sized at this fractional
    # Kelly instead of the strategy's normal kelly_mult. 0.05 = one-twentieth
    # Kelly (~1/10th of the live half-Kelly). County / franchise leagues are
    # unaffected; they keep the strategy's normal kelly_mult.
    "associate_kelly_mult": float(os.getenv("BETTING_ASSOCIATE_KELLY_MULT", "0.05")),
    # Master switch for the associate-league Kelly throttle above. When OFF
    # (default), effective_kelly_mult() is a no-op and associate-nation
    # internationals are sized at the strategy's normal kelly_mult. 60-day
    # ledger review (2026-06-27) showed associate crint fixtures were the most
    # profitable live segment in the clean window (+25% ROI / +$300 held-edge),
    # so the throttle is disabled. Flip to 1 to re-arm the down-sizing.
    "associate_throttle_enabled": os.getenv(
        "BETTING_ASSOCIATE_THROTTLE_ENABLED", "0"
    ).strip() in ("1", "true", "True", "TRUE"),
    # Live-only tournament-prefix exclusion. Blocks LIVE placement only; paper
    # scanners ignore this list so excluded leagues keep accruing paper bets for
    # confirmation. The effective set is the union of:
    #   1. every BETTING_LEAGUE_POLICY entry with live=False (the documented gate
    #      result — NO-GO leagues + deliberate paper-confirms), and
    #   2. any extra prefixes in BETTING_LIVE_EXCLUDE_PREFIXES (ad-hoc override).
    # Edit the league policy table above to toggle a competition on/off.
    "live_exclude_prefixes": sorted(
        {p for p, v in BETTING_LEAGUE_POLICY.items() if not v.get("live", False)}
        | {
            p.strip().lower()
            for p in os.getenv("BETTING_LIVE_EXCLUDE_PREFIXES", "").split(",")
            if p.strip()
        }
    ),
    # Full per-league policy table (verdict + reason + live toggle) for UI/ops.
    "league_policy": BETTING_LEAGUE_POLICY,
    # Exit-health circuit breaker (2026-06-27, post-outage). New LIVE bets are
    # blocked when there are open positions AND the in-play cashout scanner's
    # heartbeat (data/paper_trading/cashout_scan_status.json) is older than this
    # many minutes or reports the CLOB unreachable. "Don't open risk you can't
    # close." 0 disables the gate. The Jun 1-6 outage rode positions to zero
    # because exits were frozen while live_bet_scan kept opening new exposure.
    "exit_health_max_stale_min": float(
        os.getenv("BETTING_EXIT_HEALTH_MAX_STALE_MIN", "15")
    ),
    # Hard ceiling on model_prob used for stake sizing. Prevents a simulator
    # "certainty" (e.g. 1.00 on a thin-data associate game) from driving a
    # full-Kelly stake. Sizing only; the edge gate is computed separately.
    "model_prob_cap": float(os.getenv("BETTING_MODEL_PROB_CAP", "0.95")),
    # Comma-separated list (e.g. "moneyline,most_sixes"). Default to moneyline only;
    # the Wave 5 Phase 5 EV report should expand this list per-tournament.
    "auto_enabled_markets": [
        m.strip() for m in os.getenv("BETTING_AUTO_MARKETS", "moneyline").split(",") if m.strip()
    ],
    "kill_switch": os.getenv("BETTING_KILL_SWITCH", "0").strip() in ("1", "true", "True", "TRUE"),
    # Phase 7 scale-up gate thresholds (used by the dashboard "graduate envelope" flow).
    "scale_up_min_settled_bets": int(os.getenv("BETTING_SCALE_MIN_BETS", "50")),
    "scale_up_max_brier_drift": float(os.getenv("BETTING_SCALE_MAX_BRIER_DRIFT", "0.02")),
    # Wave 5.8: per-strategy caps for the 5-strategy live-betting envelope.
    # Each strategy_label in bet_ledger is independently capped so a single
    # strategy cannot monopolise the total deposit envelope.
    "max_deposit_per_strategy_usdc": float(os.getenv("BETTING_MAX_DEPOSIT_PER_STRATEGY", "100")),
    # Whitelist of strategy_labels allowed to place real (bet_kind='real') orders.
    # Empty list = no strategies are live; remove a name here to silence that
    # strategy without any code changes.
    "live_strategies": [
        s.strip() for s in os.getenv("BETTING_LIVE_STRATEGIES", "").split(",") if s.strip()
    ],
    # XI-aware rebalancing: continuously re-size live exposure toward the
    # fresh Kelly target as updated CREX lineups move the model. Ships OFF;
    # when off, the scanner only places a first bet per fixture/strategy and
    # never sells. Full rebalance (add / reduce-by-sell / exit-and-flip) only
    # runs when this is enabled.
    "rebalance_enabled": os.getenv("BETTING_REBALANCE_ENABLED", "0").strip()
    in ("1", "true", "True", "TRUE"),
    # Only act when the model edge has moved at least this many pp since the
    # bet we hold (cheap guard against churning on simulator noise).
    "rebalance_edge_delta_pp": float(os.getenv("BETTING_REBALANCE_EDGE_DELTA_PP", "1.5")),
    # Minimum |target - current| as a fraction of the target (or current)
    # exposure before we add/reduce. Mirrors the TWAP resize threshold.
    "rebalance_min_delta_frac": float(os.getenv("BETTING_REBALANCE_MIN_DELTA_FRAC", "0.20")),
    # Hard cap on the number of rebalance adjustments per fixture/strategy
    # across the pre-toss life of the bet (prevents runaway churn).
    "rebalance_max_per_fixture": int(os.getenv("BETTING_REBALANCE_MAX_PER_FIXTURE", "6")),
    # Freeze SELLs (de-risking / exits) inside this many minutes before
    # kickoff — by then the lineup is locked and exiting just pays spread.
    "rebalance_freeze_min_before_toss": float(
        os.getenv("BETTING_REBALANCE_FREEZE_MIN_BEFORE_TOSS", "20")
    ),
    # Anti-averaging-down guard. When on, the rebalancer refuses to ADD to a
    # position whose current market price has fallen materially below our
    # average entry price — i.e. the market has moved against the model. This
    # stops the "buy it all the way down" behaviour (Bahrain 0.54 -> 0.26).
    # Reduces / exits / side-flips are unaffected.
    "rebalance_no_average_down": os.getenv(
        "BETTING_REBALANCE_NO_AVERAGE_DOWN", "1"
    ).strip() in ("1", "true", "True", "TRUE"),
    # Block an add when market_price < avg_entry * (1 - this). 0.10 = 10% below.
    "rebalance_max_drawdown_frac": float(
        os.getenv("BETTING_REBALANCE_MAX_DRAWDOWN_FRAC", "0.10")
    ),
    # SELL execution (cashout / stop-loss / rebalance de-risk). The exit is a
    # marketable limit priced this many cents BELOW the reference midpoint so it
    # actually crosses the book; fills are read back and only the matched size
    # is booked. A wider cap exits more reliably into a thin book at the cost of
    # more slippage.
    "cashout_sell_max_slippage_cents": float(
        os.getenv("CASHOUT_SELL_MAX_SLIPPAGE_CENTS", "0.03")
    ),
    # Cancel any resting unfilled remainder after the marketable sweep so we
    # never leave an orphan SELL on the book (retried next scan tick).
    "cashout_sell_cancel_remainder": os.getenv(
        "CASHOUT_SELL_CANCEL_REMAINDER", "1"
    ).strip() in ("1", "true", "True", "TRUE"),
    # Treat fills below this many shares as no-fill (dust / sub-minimum); do not
    # book a cashout and leave the position open to retry.
    "cashout_sell_min_fill_shares": float(
        os.getenv("CASHOUT_SELL_MIN_FILL_SHARES", "5.0")
    ),
}

BETFAIR_CONFIG = {
    "enabled": os.getenv("BETFAIR_ENABLED", "false").lower() == "true",
    "app_key": os.getenv("BETFAIR_APP_KEY"),
    "username": os.getenv("BETFAIR_USERNAME"),
    "password": os.getenv("BETFAIR_PASSWORD"),
    "session_token": os.getenv("BETFAIR_SESSION_TOKEN"),
    # Optional for non-interactive login; not required for interactive/session-token mode.
    "cert_file": os.getenv("BETFAIR_CERT_FILE"),
    "key_file": os.getenv("BETFAIR_KEY_FILE"),
    "sso_base_url": os.getenv("BETFAIR_SSO_BASE_URL", "https://identitysso.betfair.com"),
    "login_path": os.getenv("BETFAIR_LOGIN_PATH", "/api/login"),
    "cert_login_path": os.getenv("BETFAIR_CERT_LOGIN_PATH", "/api/certlogin"),
    "keep_alive_path": os.getenv("BETFAIR_KEEP_ALIVE_PATH", "/api/keepAlive"),
    "betting_api_base_url": os.getenv("BETFAIR_BETTING_API_BASE_URL", "https://api.betfair.com/exchange/betting"),
}

# ELO Configuration
ELO_CONFIG = {
    "initial_rating": 1500,
    "k_factor_team": 32,  # K-factor for team ELO updates
    "k_factor_player_batting": 20,  # K-factor for player batting ELO
    "k_factor_player_bowling": 20,  # K-factor for player bowling ELO
    "rating_floor": 1000,  # Minimum possible rating
    "rating_ceiling": 2500,  # Maximum possible rating
}

# Model Configuration
MODEL_CONFIG = {
    "test_size": 0.2,
    "validation_size": 0.1,
    "random_state": 42,
    "epochs": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
}

# Monte Carlo Simulation
SIMULATION_CONFIG = {
    "num_simulations": 10000,  # Number of match simulations to run
    "confidence_level": 0.95,  # For confidence intervals
    "chunk_size": 1000,  # Simulations per chunk for progress updates
}

# Performance / Parallelism Configuration (Apple M2 Pro optimization)
import multiprocessing
PARALLELISM_CONFIG = {
    "n_cpu_cores": multiprocessing.cpu_count(),
    "tf_inter_op_threads": max(4, multiprocessing.cpu_count() // 2),
    "tf_intra_op_threads": max(4, multiprocessing.cpu_count() // 2),
    "numpy_threads": 4,
    "n_workers": max(2, multiprocessing.cpu_count() - 2),  # For ProcessPoolExecutor
}

# Flask Configuration
class FlaskConfig:
    SECRET_KEY = os.getenv("SECRET_KEY", "dev-secret-key-change-in-production")
    # Secure by default: the Werkzeug debugger allows arbitrary code execution,
    # so debug mode must be explicitly opted into (FLASK_DEBUG=True) for local
    # development. It defaults OFF so an unset/missing .env is never debug.
    DEBUG = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    HOST = os.getenv("FLASK_HOST", "127.0.0.1")
    PORT = int(os.getenv("FLASK_PORT", 5000))


# Logging Configuration
LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {
            "format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
        },
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "formatter": "standard",
            "level": "INFO",
        },
        "file": {
            "class": "logging.FileHandler",
            "filename": BASE_DIR / "app.log",
            "formatter": "standard",
            "level": "DEBUG",
        },
    },
    "root": {
        "handlers": ["console", "file"],
        "level": "INFO",
    },
}

# Feature Engineering Configuration
FEATURE_CONFIG = {
    "recent_form_matches": 10,  # Number of recent matches for form calculation
    "min_matches_for_stats": 5,  # Minimum matches for reliable statistics
    "innings_phases": {
        "powerplay": (0, 6),
        "middle": (6, 15),  # For T20, adjusted for ODI
        "death": (15, 20),
    },
}

# Ball outcome categories
BALL_OUTCOMES = {
    "runs": [0, 1, 2, 3, 4, 6],
    "extras": ["wide", "noball", "bye", "legbye"],
    "wickets": ["bowled", "caught", "lbw", "run out", "stumped", "hit wicket"],
}

# Known ICC teams safety net
# When a team is scraped from CREX but cannot be matched to the database,
# check this list. If the team is a known ICC member, auto-create it with
# the specified tier and default ELO. This prevents phantom "Unknown" teams
# from producing misleading predictions.
KNOWN_ICC_TEAMS = {
    # Full Members (Tier 1)
    'afghanistan': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1700.0, 'default_elo_odi_male': 1650.0, 'note': 'Cricsheet withheld all Afghanistan match data (Nov 2024) due to ICC women\'s cricket policy. ELO is manually estimated.'},
    'australia': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1650.0, 'default_elo_odi_male': 1650.0},
    'bangladesh': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1600.0, 'default_elo_odi_male': 1600.0},
    'england': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1650.0, 'default_elo_odi_male': 1650.0},
    'india': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1650.0, 'default_elo_odi_male': 1650.0},
    'ireland': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1550.0, 'default_elo_odi_male': 1550.0},
    'new zealand': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1650.0, 'default_elo_odi_male': 1650.0},
    'pakistan': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1650.0, 'default_elo_odi_male': 1650.0},
    'south africa': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1650.0, 'default_elo_odi_male': 1650.0},
    'sri lanka': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1600.0, 'default_elo_odi_male': 1600.0},
    'west indies': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1600.0, 'default_elo_odi_male': 1600.0},
    'zimbabwe': {'tier': 1, 'team_type': 'international', 'default_elo_t20_male': 1550.0, 'default_elo_odi_male': 1550.0},
    # Top Associates (Tier 2)
    'canada': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1500.0, 'default_elo_odi_male': 1450.0},
    'hong kong': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1450.0, 'default_elo_odi_male': 1400.0},
    'namibia': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1550.0, 'default_elo_odi_male': 1500.0},
    'nepal': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1550.0, 'default_elo_odi_male': 1500.0},
    'netherlands': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1500.0, 'default_elo_odi_male': 1500.0},
    'oman': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1450.0, 'default_elo_odi_male': 1400.0},
    'papua new guinea': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1450.0, 'default_elo_odi_male': 1400.0},
    'scotland': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1550.0, 'default_elo_odi_male': 1500.0},
    'united arab emirates': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1500.0, 'default_elo_odi_male': 1450.0},
    'united states of america': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1500.0, 'default_elo_odi_male': 1450.0},
    'uganda': {'tier': 2, 'team_type': 'international', 'default_elo_t20_male': 1500.0, 'default_elo_odi_male': 1450.0},
}

