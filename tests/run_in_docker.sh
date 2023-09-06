#!/bin/bash
set -xeuo pipefail
if ! hash pip 2>/dev/null && hash apt-get 2>/dev/null; then
	apt-get update
	apt-get install -y --no-install-recommends python-pip
fi
pip install --upgrade pip
git config --global --add safe.directory "$PWD" || true
pip install -e '.[test]'
pytest -sxv tests/unit
sudo -u nobody bash ./tests/start_nomad_server.sh
pytest -sxv tests/integration
