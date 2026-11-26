#!/usr/bin/env bash
# Thin wrapper so the catch-up has a shell entry point like the other scripts
# in this directory. The real logic is in ca_topup_catchup.py.
#
# It must be a wrapper, not the implementation: under launchd, /bin/bash is
# denied TCC access to ~/Desktop (where this repo lives), so bash cannot read or
# exec a script from here. `exec` hands the file to the interpreter without bash
# opening it, which works. Running this from an interactive shell works too.
#
# Usage:
#     scripts/ca_topup_catchup.sh            # catch-up: run only if stale
#     scripts/ca_topup_catchup.sh --force    # weekly: run unless it just ran
#     scripts/ca_topup_catchup.sh --dry-run  # print the decision, run nothing

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"

exec "${REPO_ROOT}/venv311/bin/python" \
    "${REPO_ROOT}/scripts/ca_topup_catchup.py" "$@"
