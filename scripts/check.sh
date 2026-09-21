#!/bin/bash
# Run quality checks: formatting (no changes) and tests
set -e
cd "$(dirname "$0")/.."

echo "==> Checking formatting (black)"
uv run black --check backend main.py

echo "==> Running tests (pytest)"
uv run pytest backend/tests -q --ignore=backend/tests/test_live_smoke.py

echo "All quality checks passed."
