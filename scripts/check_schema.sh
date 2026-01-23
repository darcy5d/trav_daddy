#!/bin/bash
# Quick schema validation check
cd "$(dirname "$0")/.."
venv/bin/python scripts/validate_schema.py
