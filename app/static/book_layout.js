/* Paints the shared book tiles and strategies table.
   book is required: "paper" or "live". This file does not fetch. */
(function (global) {
    'use strict';

    var COLS = 15;

    function paintBookLayout(book, payload) {
        if (book !== 'paper' && book !== 'live') {
            throw new Error('paintBookLayout requires an explicit book argument: "paper" or "live"');
        }
        var root = document.querySelector('.book-layout[data-book="' + book + '"]');
        if (!root) {
            throw new Error('book layout root missing for ' + book);
        }
        if (book === 'paper') {
            paintPaper(root, payload);
        } else {
            paintLive(root, payload);
        }
    }

    function paintPaper(root, payload) {
        if (!payload || payload.success === false) {
            throw new Error((payload && payload.error) || 'failed');
        }
        if (payload.live_strategies) {
            throw new Error('paper layout refused a live strategies payload');
        }
        var strategies = payload.strategies || [];
        var tbody = body(root);
        tbody.innerHTML = '';
        if (!strategies.length) {
            tbody.innerHTML = emptyRow(root);
            clearTiles(root);
            return;
        }
        for (var i = 0; i < strategies.length; i++) {
            tbody.appendChild(paperRow(strategies[i]));
        }
        if (payload.book) {
            tbody.appendChild(paperFooter(root, payload.book));
        }
        var included = strategies.filter(function (s) {
            return s.enabled || (s.n_bets || 0) > 0;
        });
        var held = sumCents(included, 'held_pnl');
        var cashout = sumCents(included, 'cashout_pnl');
        var openStake = sumCents(included, 'open_size');
        var nSettled = included.reduce(function (sum, s) { return sum + (s.n_settled || 0); }, 0);
        var nWon = included.reduce(function (sum, s) { return sum + (s.n_won || 0); }, 0);
        var book = payload.book || {};
        paintTile(root, 'total', book.total_pnl, 'money');
        paintTile(root, 'cashout', cashout, 'money');
        paintTile(root, 'held', held, 'money');
        paintTile(root, 'open', openStake, 'stake');
        paintTile(root, 'bankrollRoi', book.bankroll_roi_pct, 'pct');
        paintTile(root, 'stakeRoi', book.stake_roi_pct, 'pct');
        paintTile(root, 'settled', nSettled, 'count');
        paintTile(root, 'win', nSettled ? (nWon / nSettled) * 100 : null, 'pct1');
    }

    function paintLive(root, payload) {
        if (payload && Object.prototype.hasOwnProperty.call(payload, 'cohort')) {
            throw new Error('live layout refused a paper cohort payload');
        }
        var tbody = body(root);
        if (!payload || payload.success === false) {
            tbody.innerHTML = messageRow('Error loading strategies.');
            clearTiles(root);
            return;
        }
        var strategies = (payload.strategies || []).slice().sort(byBasisRoi);
        if (!strategies.length) {
            tbody.innerHTML = emptyRow(root);
            clearTiles(root);
            return;
        }
        tbody.innerHTML = '';
        for (var i = 0; i < strategies.length; i++) {
            tbody.appendChild(liveRow(strategies[i]));
        }
        var enabled = strategies.filter(function (r) { return r.enabled; });
        var sumPnl = sumNum(enabled, 'realised_pnl_usdc');
        var sumBasis = sumNum(enabled, 'starting_bankroll_usdc');
        var sumStake = sumNum(enabled, 'settled_stake_usdc');
        var nSettled = sumNum(enabled, 'n_settled');
        var nWins = sumNum(enabled, 'n_wins');
        var bankrollRoi = sumBasis > 0 ? (sumPnl / sumBasis) * 100 : null;
        var stakeRoi = sumStake > 0 ? (sumPnl / sumStake) * 100 : null;
        tbody.appendChild(liveFooter(root, sumPnl, bankrollRoi, stakeRoi));
        paintTile(root, 'total', sumPnl, 'money');
        paintTile(root, 'cashout', sumNum(enabled, 'cashout_pnl_usdc'), 'money');
        paintTile(root, 'held', sumNum(enabled, 'held_pnl_usdc'), 'money');
        paintTile(root, 'open', sumNum(enabled, 'open_exposure_usdc'), 'stake');
        paintTile(root, 'bankrollRoi', bankrollRoi, 'pct');
        paintTile(root, 'stakeRoi', stakeRoi, 'pct');
        paintTile(root, 'settled', nSettled, 'count');
        paintTile(root, 'win', nSettled ? (nWins / nSettled) * 100 : null, 'pct1');
    }

    function byBasisRoi(a, b) {
        var aMissing = a.roi_pct === null || a.roi_pct === undefined;
        var bMissing = b.roi_pct === null || b.roi_pct === undefined;
        if (aMissing !== bMissing) return aMissing ? 1 : -1;
        if (a.roi_pct !== b.roi_pct) return b.roi_pct - a.roi_pct;
        return String(a.strategy_label || '').localeCompare(String(b.strategy_label || ''));
    }

    function paperRow(s) {
        var tr = document.createElement('tr');
        if (!s.enabled) tr.className = 'disabled';
        tr.dataset.strategy = s.name || '';
        tr.innerHTML =
            '<td><strong>' + esc(s.name) + '</strong><br><span class="book-muted" style="font-size:0.7rem">' +
                esc(s.description) + '</span></td>' +
            '<td>' + esc(s.model_version) + '</td>' +
            '<td>' + esc(s.min_edge_pp) + 'pp</td>' +
            num(s.bankroll, moneyClass(s.bankroll - s.starting), fmtMoney(s.bankroll)) +
            num(s.held_pnl, moneyClass(s.held_pnl), fmtMoney(s.held_pnl)) +
            num(s.cashout_pnl, moneyClass(s.cashout_pnl), fmtMoney(s.cashout_pnl)) +
            num(s.total_pnl, 'book-total-pnl ' + moneyClass(s.total_pnl), fmtMoney(s.total_pnl)) +
            num(s.roi_pct, 'book-bankroll-roi ' + moneyClass(s.roi_pct || 0), fmtPct(s.roi_pct, 2)) +
            num(s.stake_roi_pct, 'book-stake-roi ' + (s.stake_roi_pct == null ? '' : moneyClass(s.stake_roi_pct)),
                s.stake_roi_pct == null ? '--' : fmtPct(s.stake_roi_pct, 2)) +
            '<td class="num">' + esc(s.n_bets) + '</td>' +
            '<td class="num">' + esc(s.n_settled) + '</td>' +
            '<td class="num">' + (s.win_rate !== null && s.win_rate !== undefined ? fmtPct(s.win_rate * 100, 1) : '--') + '</td>' +
            '<td class="num">' + fmtMoney(s.open_size) + '</td>' +
            sampleWaitCell(s) +
            significanceCell(s);
        return tr;
    }

    function paperFooter(root, book) {
        var tr = document.createElement('tr');
        tr.className = 'book-footer';
        tr.dataset.strategy = 'paper-book';
        var stake = book.stake_roi_pct == null ? '--' : fmtPct(book.stake_roi_pct, 2);
        tr.innerHTML =
            '<td><strong>' + esc(root.getAttribute('data-footer-name')) + '</strong><br>' +
                '<span class="book-muted" style="font-size:0.7rem;font-weight:400">' +
                esc(root.getAttribute('data-footer-note')) + '</span></td>' +
            '<td></td><td></td><td></td><td></td><td></td>' +
            num(book.total_pnl, 'book-total-pnl ' + moneyClass(book.total_pnl), fmtMoney(book.total_pnl)) +
            num(book.bankroll_roi_pct, 'book-bankroll-roi ' + moneyClass(book.bankroll_roi_pct || 0), fmtPct(book.bankroll_roi_pct, 2)) +
            num(book.stake_roi_pct, 'book-stake-roi ' + (book.stake_roi_pct == null ? '' : moneyClass(book.stake_roi_pct)), stake) +
            '<td></td><td></td><td></td><td></td><td class="book-sample-wait"></td>' +
            '<td class="book-significance"></td>';
        return tr;
    }

    function liveRow(r) {
        var tr = document.createElement('tr');
        tr.className = r.enabled ? '' : 'disabled';
        tr.dataset.strategy = r.strategy_label || '';
        var badge = r.enabled
            ? '<span class="book-badge on">LIVE</span>'
            : '<span class="book-badge off">retired</span>';
        var bankrollText = r.enabled ? fmtMoney(r.current_bankroll_usdc) : fmtMoney(0);
        var bankrollCls = r.enabled ? moneyClass((r.current_bankroll_usdc || 0) - (r.starting_bankroll_usdc || 0)) : 'book-muted';
        var roiText = (r.roi_pct === null || r.roi_pct === undefined) ? '--' : fmtPct(r.roi_pct, 2);
        var roiCls = (r.roi_pct === null || r.roi_pct === undefined) ? 'book-muted' : moneyClass(r.roi_pct);
        var stakeText = (r.stake_roi_pct === null || r.stake_roi_pct === undefined) ? '--' : fmtPct(r.stake_roi_pct, 2);
        var stakeCls = (r.stake_roi_pct === null || r.stake_roi_pct === undefined) ? '' : moneyClass(r.stake_roi_pct);
        var winText = (r.win_rate === null || r.win_rate === undefined) ? '--' : fmtPct(r.win_rate * 100, 1);
        var betsText = esc(r.n_total) + (r.n_errored
            ? ' <span class="book-neg" style="font-size:0.7rem">err:' + esc(r.n_errored) + '</span>'
            : '');
        tr.innerHTML =
            '<td><strong>' + esc(r.strategy_label) + '</strong>' + badge + '</td>' +
            '<td class="book-muted">—</td>' +
            '<td class="book-muted">—</td>' +
            num(null, bankrollCls, bankrollText) +
            num(r.held_pnl_usdc, moneyClass(r.held_pnl_usdc), fmtMoney(r.held_pnl_usdc)) +
            num(r.cashout_pnl_usdc, moneyClass(r.cashout_pnl_usdc), fmtMoney(r.cashout_pnl_usdc)) +
            num(r.realised_pnl_usdc, 'book-total-pnl ' + moneyClass(r.realised_pnl_usdc), fmtMoney(r.realised_pnl_usdc)) +
            num(r.roi_pct, 'book-bankroll-roi ' + roiCls, roiText) +
            num(r.stake_roi_pct, 'book-stake-roi ' + stakeCls, stakeText) +
            '<td class="num">' + betsText + '</td>' +
            '<td class="num">' + esc(r.n_settled) + '</td>' +
            '<td class="num">' + winText + '</td>' +
            '<td class="num">' + fmtMoney(r.open_exposure_usdc) + '</td>' +
            '<td class="book-sample-wait"></td>' +
            '<td class="book-significance"></td>';
        return tr;
    }

    function liveFooter(root, totalPnl, bankrollRoi, stakeRoi) {
        var tr = document.createElement('tr');
        tr.className = 'book-footer';
        tr.dataset.strategy = 'live-book';
        var stakeText = stakeRoi == null ? '--' : fmtPct(stakeRoi, 2);
        var bankrollText = bankrollRoi == null ? '--' : fmtPct(bankrollRoi, 2);
        tr.innerHTML =
            '<td><strong>' + esc(root.getAttribute('data-footer-name')) + '</strong><br>' +
                '<span class="book-muted" style="font-size:0.7rem;font-weight:400">' +
                esc(root.getAttribute('data-footer-note')) + '</span></td>' +
            '<td></td><td></td><td></td><td></td><td></td>' +
            num(totalPnl, 'book-total-pnl ' + moneyClass(totalPnl), fmtMoney(totalPnl)) +
            num(bankrollRoi, 'book-bankroll-roi ' + (bankrollRoi == null ? '' : moneyClass(bankrollRoi)), bankrollText) +
            num(stakeRoi, 'book-stake-roi ' + (stakeRoi == null ? '' : moneyClass(stakeRoi)), stakeText) +
            '<td></td><td></td><td></td><td></td><td class="book-sample-wait"></td>' +
            '<td class="book-significance"></td>';
        return tr;
    }

    function sampleWaitCell(s) {
        var w = s.sample_wait;
        if (!w || w.state === 'blank' || !w.label) return '<td class="book-sample-wait"></td>';
        var title = w.title ? ' title="' + esc(w.title) + '"' : '';
        var gate = w.gate_label
            ? '<div class="book-muted">' + esc(w.gate_label) + '</div>' : '';
        var badge = '';
        if (w.evidence_label) {
            var evidenceTitle = w.evidence_title ? ' title="' + esc(w.evidence_title) + '"' : '';
            badge = '<div class="book-evidence book-evidence-' + esc(w.evidence_key) + '"' +
                evidenceTitle + '>' + esc(w.evidence_label) + '</div>';
        }
        return '<td class="book-sample-wait"' + title + '>' + esc(w.label) + gate + badge + '</td>';
    }

    function significanceCell(s) {
        var w = s.sample_wait;
        if (!w || w.state === 'blank' || !w.p_label) return '<td class="book-significance"></td>';
        var title = w.p_title ? ' title="' + esc(w.p_title) + '"' : '';
        return '<td class="book-significance"' + title + '>' + esc(w.p_label) + '</td>';
    }

    function num(value, cls, text) {
        return '<td class="num ' + cls + '">' + text + '</td>';
    }

    function body(root) {
        return document.getElementById(root.getAttribute('data-body-id'));
    }

    function emptyRow(root) {
        return messageRow(root.getAttribute('data-empty') || 'No strategies');
    }

    function messageRow(text) {
        return '<tr><td colspan="' + COLS + '" class="book-muted" style="text-align:center;padding:1rem">' +
            esc(text) + '</td></tr>';
    }

    function clearTiles(root) {
        ['total', 'cashout', 'held', 'open', 'bankrollRoi', 'stakeRoi', 'settled', 'win'].forEach(function (key) {
            paintTile(root, key, null, 'money');
        });
    }

    function paintTile(root, key, value, kind) {
        var map = {
            total: 'data-total-id',
            cashout: 'data-cashout-id',
            held: 'data-held-id',
            open: 'data-open-id',
            bankrollRoi: 'data-bankroll-roi-id',
            stakeRoi: 'data-stake-roi-id',
            settled: 'data-settled-id',
            win: 'data-win-id'
        };
        var el = document.getElementById(root.getAttribute(map[key]));
        if (!el) return;
        if (value === null || value === undefined || Number.isNaN(Number(value))) {
            el.textContent = '--';
            el.className = 'book-tile-value';
            return;
        }
        var n = Number(value);
        if (kind === 'money' || kind === 'stake') el.textContent = fmtMoney(n);
        else if (kind === 'count') el.textContent = Math.round(n).toLocaleString();
        else el.textContent = fmtPct(n, kind === 'pct1' ? 1 : 2);
        var colored = kind === 'money' || kind === 'pct';
        el.className = 'book-tile-value' + (colored ? (n >= 0 ? ' positive' : ' negative') : '');
    }

    function sumCents(rows, key) {
        return rows.reduce(function (sum, row) {
            return sum + Math.round(Number(row[key] || 0) * 100);
        }, 0) / 100;
    }

    function sumNum(rows, key) {
        return rows.reduce(function (sum, row) {
            return sum + (Number(row[key]) || 0);
        }, 0);
    }

    function moneyClass(v) {
        return Number(v) >= 0 ? 'book-pos' : 'book-neg';
    }

    function fmtMoney(v) {
        if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
        var n = Number(v);
        var sign = n < 0 ? '-' : '';
        return sign + '$' + Math.abs(n).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
    }

    function fmtPct(v, places) {
        if (v === null || v === undefined || Number.isNaN(Number(v))) return '--';
        return Number(v).toFixed(places == null ? 1 : places) + '%';
    }

    function esc(s) {
        return String(s == null ? '' : s).replace(/[&<>"']/g, function (c) {
            return {'&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'}[c];
        });
    }

    global.paintBookLayout = paintBookLayout;
})(typeof globalThis !== 'undefined' ? globalThis : window);
