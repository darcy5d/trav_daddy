"""Shared book layout stays on two ledgers.

Paper cashouts must not enter the live rollup, and the shared painter must
not default to the paper book.
"""

from __future__ import annotations

import inspect
import sqlite3
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

from jinja2 import Environment, FileSystemLoader

from src.integrations.polymarket.risk_gate import get_strategy_breakdown
from tests.test_risk_gate import _make_betting_config

REPO = Path(__file__).resolve().parents[1]


def _ledger() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE bet_ledger (
            bet_id INTEGER PRIMARY KEY AUTOINCREMENT,
            proposed_at TEXT NOT NULL,
            filled_at TEXT,
            settled_at TEXT,
            fixture_key TEXT NOT NULL,
            market_type TEXT NOT NULL,
            model_prob REAL NOT NULL,
            market_price_at_proposal REAL NOT NULL,
            edge_pp REAL NOT NULL,
            side TEXT NOT NULL,
            size_usdc REAL NOT NULL,
            fill_size_usdc REAL,
            settle_outcome INTEGER,
            pnl_realised_usdc REAL,
            status TEXT NOT NULL,
            mode TEXT NOT NULL,
            bet_kind TEXT,
            strategy_label TEXT,
            cashout_triggered_at TEXT
        )
        """
    )
    class _Open:
        def cursor(self):
            return conn.cursor()

        def execute(self, *args, **kwargs):
            return conn.execute(*args, **kwargs)

        def commit(self):
            return conn.commit()

        def close(self):
            return None

    return _Open()


def _insert(conn, **kwargs):
    defaults = {
        "proposed_at": datetime.now(timezone.utc).isoformat(),
        "fixture_key": "cricipl-aaa-bbb-2026-11-01",
        "market_type": "moneyline",
        "model_prob": 0.55,
        "market_price_at_proposal": 0.50,
        "edge_pp": 5.0,
        "side": "BUY",
        "size_usdc": 25.0,
        "status": "settled",
        "mode": "manual",
    }
    defaults.update(kwargs)
    cols = ",".join(defaults.keys())
    placeholders = ",".join("?" * len(defaults))
    conn.execute(
        f"INSERT INTO bet_ledger ({cols}) VALUES ({placeholders})",
        list(defaults.values()),
    )
    conn.commit()


def test_live_breakdown_excludes_paper_cashouts_and_cohort():
    source = inspect.getsource(get_strategy_breakdown)
    assert "cohort_filter_sql" not in source
    assert "paper_strategies" not in source
    assert "bet_kind = 'paper'" not in source
    assert "COALESCE(bet_kind, 'real') = 'real'" in source
    assert "1000" not in source

    conn = _ledger()
    now = datetime.now(timezone.utc).isoformat()
    _insert(
        conn, strategy_label="alpha", bet_kind="real", status="settled",
        settled_at=now, pnl_realised_usdc=10.0, size_usdc=20.0, settle_outcome=1,
    )
    _insert(
        conn, strategy_label="alpha", bet_kind="real", status="settled",
        settled_at=now, pnl_realised_usdc=5.0, size_usdc=25.0,
        settle_outcome=None, cashout_triggered_at=now,
    )
    _insert(
        conn, strategy_label="alpha", bet_kind=None, status="settled",
        settled_at=now, pnl_realised_usdc=3.0, size_usdc=10.0, settle_outcome=1,
    )
    _insert(
        conn, strategy_label="alpha", bet_kind="real", status="filled",
        filled_at=now, settled_at=None, fill_size_usdc=15.0, size_usdc=15.0,
        pnl_realised_usdc=None, settle_outcome=None,
    )
    _insert(
        conn, strategy_label="alpha", bet_kind="paper", status="settled",
        settled_at=now, pnl_realised_usdc=7929.0, size_usdc=1000.0,
        settle_outcome=None, cashout_triggered_at=now,
    )
    _insert(
        conn, strategy_label="paper_whale", bet_kind="paper", status="settled",
        settled_at=now, pnl_realised_usdc=7929.0, size_usdc=1000.0,
        settle_outcome=None, cashout_triggered_at=now,
    )
    _insert(
        conn, strategy_label="old", bet_kind="real", status="settled",
        settled_at=now, pnl_realised_usdc=1.0, size_usdc=10.0, settle_outcome=1,
    )

    with patch("config.BETTING_CONFIG", _make_betting_config(live_strategies=["alpha"])), \
         patch("src.data.database.get_connection", return_value=conn), \
         patch("src.integrations.polymarket.live_bankroll.get_portfolio_value", return_value=500.0):
        out = get_strategy_breakdown()

    labels = [row["strategy_label"] for row in out["strategies"]]
    assert labels == ["alpha"]
    assert "paper_whale" not in labels
    assert "old" not in labels
    assert "v2_diag_2pp" not in labels
    alpha = out["strategies"][0]
    assert alpha["cashout_pnl_usdc"] == 5.0
    assert alpha["held_pnl_usdc"] == 13.0
    assert alpha["realised_pnl_usdc"] == 18.0
    assert alpha["settled_stake_usdc"] == 55.0
    assert alpha["open_exposure_usdc"] == 15.0
    assert alpha["n_match_wins"] == 2
    assert alpha["n_wins"] == 3
    assert alpha["n_settled"] == 3
    assert alpha["starting_bankroll_usdc"] == round(500.0 - 18.0, 2)
    assert alpha["roi_pct"] == round(100.0 * 18.0 / (500.0 - 18.0), 2)
    assert alpha["stake_roi_pct"] == round(100.0 * 18.0 / 55.0, 2)
    assert alpha["roi_pct"] != round(100.0 * 18.0 / 1000.0, 2)
    assert "cohort" not in out

    with patch("config.BETTING_CONFIG", _make_betting_config(live_strategies=[])), \
         patch("src.data.database.get_connection", return_value=conn), \
         patch("src.integrations.polymarket.live_bankroll.get_portfolio_value", return_value=500.0):
        empty = get_strategy_breakdown()
    assert empty["strategies"] == []
    assert empty["live_strategies"] == []


def test_shared_partial_has_no_fetch_or_order_controls():
    partial = (REPO / "app/templates/partials/book_layout.html").read_text()
    js = (REPO / "app/static/book_layout.js").read_text()
    paper = (REPO / "app/templates/paper_trades.html").read_text()
    live = (REPO / "app/templates/live_betting.html").read_text()
    for banned in (
        "fetch(", "/api/", "also_live", "kill-switch", "scale-up",
        "scaleUp", "post-toss-scan", "bet_kind", "post_edge", "paper_cohort",
        "?cohort",
    ):
        assert banned not in partial, banned
        assert banned not in js, banned
    assert "function paintBookLayout(book, payload)" in js
    assert "book = 'paper'" not in js
    assert 'book = "paper"' not in js
    shared = [
        "partials/book_header.html",
        "partials/book_layout.html",
        "partials/book_chart.html",
        "partials/book_cards.html",
        "partials/book_upcoming.html",
    ]
    for name in shared:
        assert name in paper, name
        assert name in live, name
        text = (REPO / "app/templates" / name).read_text()
        for banned in (
            "fetch(", "/api/", "also_live", "kill-switch", "scale-up",
            "scaleUp", "post-toss-scan", "bet_kind", "post_edge", "paper_cohort",
        ):
            assert banned not in text, (name, banned)
    def include_order(src):
        import re
        return re.findall(r"partials/(book_[a-z_]+\.html)", src)
    assert include_order(paper) == [
        "book_header.html", "book_layout.html", "book_chart.html",
        "book_cards.html", "book_cards.html", "book_upcoming.html",
    ]
    assert include_order(live) == include_order(paper)
    assert "kill-switch-btn" in live
    assert "scaleUp(" in live
    assert "kill-switch-btn" not in paper
    assert "scaleUp(" not in paper
    assert "paintBookLayout('paper', d)" in paper
    assert "paintBookLayout('live', data)" in live
    assert "/api/paper/strategies" in paper
    assert "/api/betting/strategies" in live
    assert "/api/paper/strategies" not in live
    assert "post_edge" not in live
    assert "/api/paper/reconcile-only" in paper
    assert "/api/paper/reconcile-only" not in live
    assert "/api/paper/reconcile-only" not in js

    env = Environment(loader=FileSystemLoader(str(REPO / "app/templates")))
    env.parse(paper)
    env.parse(live)
    env.parse(partial)


def test_painter_requires_book_and_keeps_cashouts_apart():
    script = textwrap.dedent(
        r"""
        const fs = require('fs');
        const vm = require('vm');
        const context = {};
        context.globalThis = context;
        context.window = context;
        const nodes = {};
        function tile(id) { const n = { id, textContent: '', className: '' }; nodes[id] = n; return n; }
        function tbody(id) {
            const node = { id, rows: [], _html: '' };
            Object.defineProperty(node, 'innerHTML', {
                get() { return node._html; },
                set(v) { node._html = v; if (v === '') node.rows.length = 0; }
            });
            node.appendChild = (tr) => node.rows.push(tr);
            nodes[id] = node;
            return node;
        }
        function root(book) {
            const prefix = book === 'paper' ? 'pt' : 'lb';
            ['total','cashout','held','open','bankroll','stake','settled','win'].forEach((k) => tile(prefix + '-' + k));
            tbody(prefix + '-body');
            const attrs = {
                'data-book': book,
                'data-total-id': prefix + '-total',
                'data-cashout-id': prefix + '-cashout',
                'data-held-id': prefix + '-held',
                'data-open-id': prefix + '-open',
                'data-bankroll-roi-id': prefix + '-bankroll',
                'data-stake-roi-id': prefix + '-stake',
                'data-settled-id': prefix + '-settled',
                'data-win-id': prefix + '-win',
                'data-body-id': prefix + '-body',
                'data-footer-name': book === 'paper' ? 'Paper book' : 'Live book',
                'data-footer-note': 'note',
                'data-empty': 'empty'
            };
            return { getAttribute: (k) => attrs[k] };
        }
        const roots = { paper: root('paper'), live: root('live') };
        context.document = {
            querySelector(sel) {
                const m = String(sel).match(/data-book="([^"]+)"/);
                return m ? roots[m[1]] || null : null;
            },
            getElementById: (id) => nodes[id] || null,
            createElement: () => ({ className: '', innerHTML: '', dataset: {} })
        };
        vm.createContext(context);
        vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
        function fails(fn, pattern) {
            try { fn(); } catch (e) {
                if (!pattern.test(e.message)) throw e;
                return;
            }
            throw new Error('expected throw ' + pattern);
        }
        fails(() => context.paintBookLayout(), /explicit book/);
        fails(() => context.paintBookLayout(undefined, {}), /explicit book/);
        const paperPayload = {
            success: true,
            cohort: 'post_edge_20261108',
            strategies: [{
                name: 's', description: 'd', enabled: true, model_version: 'v3',
                min_edge_pp: 3, starting: 1000, bankroll: 8929, held_pnl: 10,
                cashout_pnl: 7929, total_pnl: 7939, open_size: 4, roi_pct: 793.9,
                stake_roi_pct: 20, n_bets: 2, n_settled: 2, n_won: 1, win_rate: 0.5,
                sample_wait: {
                    label: '12d / 40 bets', title: 'wait',
                    p_label: '0.250 · α 0.05', p_title: 'p',
                    gate_label: '20/60 · 80%',
                    evidence_key: 'robust', evidence_label: 'Robust', evidence_title: 'short of n80'
                }
            }],
            book: { total_pnl: 7939, bankroll_roi_pct: 793.9, stake_roi_pct: 20 }
        };
        fails(() => context.paintBookLayout('live', paperPayload), /paper cohort/);
        context.paintBookLayout('paper', paperPayload);
        if (nodes['pt-cashout'].textContent.indexOf('7,929') < 0) {
            throw new Error('paper cashout missing: ' + nodes['pt-cashout'].textContent);
        }
        const paperRow = nodes['pt-body'].rows[0].innerHTML;
        if (paperRow.indexOf('12d / 40 bets') < 0) throw new Error('sample wait missing on paper');
        const waitAt = paperRow.indexOf('12d / 40 bets');
        const pAt = paperRow.indexOf('0.250 · α 0.05');
        if (pAt < 0 || pAt < waitAt) throw new Error('one-sided p not beside sample wait: ' + paperRow);
        if (paperRow.indexOf('20/60 · 80%') < 0 || paperRow.indexOf('Robust') < 0) {
            throw new Error('sample wait class missing: ' + paperRow);
        }
        const livePayload = {
            live_strategies: ['alpha'],
            strategies: [{
                strategy_label: 'alpha', enabled: true,
                current_bankroll_usdc: 500, starting_bankroll_usdc: 482,
                realised_pnl_usdc: 18, held_pnl_usdc: 13, cashout_pnl_usdc: 5,
                settled_stake_usdc: 55, roi_pct: 3.73, stake_roi_pct: 32.73,
                n_total: 4, n_settled: 3, n_wins: 3, win_rate: 1,
                open_exposure_usdc: 15, n_errored: 0
            }]
        };
        fails(() => context.paintBookLayout('paper', livePayload), /live strategies/);
        context.paintBookLayout('live', livePayload);
        const cash = nodes['lb-cashout'].textContent;
        if (cash.indexOf('7,929') >= 0 || cash.indexOf('7929') >= 0) {
            throw new Error('paper cashout landed on live: ' + cash);
        }
        if (cash !== '$5.00') throw new Error('live cashout ' + cash);
        if (nodes['lb-bankroll'].textContent !== '3.73%') {
            throw new Error('live basis ROI ' + nodes['lb-bankroll'].textContent);
        }
        if (nodes['lb-stake'].textContent !== '32.73%') {
            throw new Error('live stake ROI ' + nodes['lb-stake'].textContent);
        }
        const liveRow = nodes['lb-body'].rows[0].innerHTML;
        if (!liveRow.includes('<td class="book-sample-wait"></td>')) {
            throw new Error('live sample cell not blank: ' + liveRow);
        }
        if (!liveRow.includes('<td class="book-significance"></td>')) {
            throw new Error('live p-value cell not blank: ' + liveRow);
        }
        if (liveRow.indexOf('12d') >= 0) throw new Error('paper sample wait on live');
        console.log('ok');
        """
    )
    proc = subprocess.run(
        ["node", "-e", script, str(REPO / "app/static/book_layout.js")],
        check=False,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
