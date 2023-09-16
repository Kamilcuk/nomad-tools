#!/bin/bash
set -euo pipefail
sudo=()
if ((UID == 0)); then
	sudo=(sudo -u nobody)
fi
set -x
"${sudo[@]}" bash ./tests/start_nomad_server.sh
pytest -sxv tests/integration -p no:cacheprovider "$@"
