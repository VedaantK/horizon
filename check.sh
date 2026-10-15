#!/usr/bin/env bash
# Full verification: lint + main test suite (includes the hftbacktest env self-test).
set -euo pipefail
cd "$(dirname "$0")"
UV="${UV:-$HOME/.local/bin/uv}"
"$UV" run ruff check src tests
"$UV" run pytest -W ignore::DeprecationWarning "$@"
