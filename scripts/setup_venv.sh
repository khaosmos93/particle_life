#!/usr/bin/env bash
set -euo pipefail

python3 -m venv .venv
source .venv/bin/activate
pip install -U 'pip<25.3'
pip install pip-tools
pip-compile requirements.in
pip install -r requirements.txt
