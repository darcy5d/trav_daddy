# Training

Model/data admin page: DB coverage, active models per gender/format, model version history, hyper-parameter tuning results. Reads `GET /api/training/{db-status,models,tune-results,job/<id>}`.

## Sub-features

- DB status (`#db-status-loading` then `#male-matches`, `#female-matches`, `#total-players`, `#total-deliveries`, `#source-freshness-text`).
- Active models (`#active-models-loading` then `#male-model-date`, `#male-model-samples`, `#female-model-*`).
- Version history `#version-table-body tr`, filters `#filter-male` etc.
- Tuned hyper-params table cells `#hp-<fmt>_<gender>-<field>` (e.g. `#hp-t20_male-val_acc`).
- Write actions (BLOCKED, never on live): `#download-btn` (download), `#reset-btn` (reset DB + retrain), retrain, `#tune-btn`, version "activate" — these rewrite `cricket.db` / model files used by the betting crons.

## How to get to it (user POV)

Nav → Training (`/training`).

## Driving it with drive.py

```bash
venv311/bin/python $S/drive.py --run-id training goto:/training \
  "waitgone:#db-status-loading" "expect:#male-matches~\d" "waitgone:#active-models-loading" \
  "expect:#male-model-date~\d{4}" "waitgone:#version-history-loading" \
  "expectcount:#version-table-body tr>=1" shot:training
```

End state (2026-11-07): 10,082 male matches, male model dated Nov 3 2026, 117 version rows.

## Gotchas

- Never verify retrain/tune/reset through this page; they are long jobs on shared data. Use the unit tests in `tests/` or a script against a copied DB.
