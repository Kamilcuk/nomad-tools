#!/bin/bash
set -xeuo pipefail
sudo -u nobody bash ./tests/start_nomad_server.sh
pytest -sxv tests/integration -p no:cacheprovider "$@"
