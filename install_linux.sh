#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip -q
.venv/bin/pip install -r requirements.txt -q
[ -f .env ] || { cp .env.example .env; chmod 600 .env; }
echo "Done. Fill in .env, then ./start_linux.sh   (demo: ./start_linux.sh --demo)"
