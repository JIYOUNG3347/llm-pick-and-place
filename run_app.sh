#!/usr/bin/env bash
# llm-pick-and-place native desktop launcher (PySide6)
# Usage:  bash run_app.sh
#         pip install 'llm-pick-and-place[ui]'   # if PySide6 is missing
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
exec python3 "$ROOT/scripts/launcher.py" "$@"
